import sys
import math
import torch
import shutil
import logging
import torch.nn.functional as F

from tqdm import tqdm
from torch import no_grad
from copy import deepcopy
from accelerate import Accelerator
from torch.utils.data import DataLoader

from .io import load_cached_similarities, save_similarities_cache
from .utils import prepare_calibration_input, print_gpu_memory, auto_map, CUSTOM_FILE, advance_swinv2_hierarchical_stage
from .utils import get_position_embeddings
from .wrapper import HiddenStatesRecordWrapper
from .utils import is_vision_model, get_vision_model_layers

logger = logging.getLogger(__name__)


def consecutive_block_dropping(args, model, dataloader: DataLoader, accelerator: Accelerator, num_samples: int):
    drop_n = args.drop_n
    if drop_n == 0:
        accelerator.print("drop_n=0: No blocks to drop, returning empty list")
        return []

    similarities = get_block_similarities_consecutive(model, dataloader, accelerator, num_samples, cache_file=args.similarity_cache_file)
    similarities_drop_n = similarities[:, drop_n].view(-1)
    max_similarity, begin_layer_id = torch.max(similarities_drop_n, dim=0)
    accelerator.print(f"similarities_drop_n: {similarities_drop_n}")
    accelerator.print(f"max_similarity: {max_similarity}, begin_layer_id: {begin_layer_id}")

    end_layer_id = begin_layer_id + drop_n
    dropped_layer_list = [i for i in range(begin_layer_id, end_layer_id)]

    accelerator.print(f"Dropped layer: {dropped_layer_list}, max_similarity: {max_similarity}")
    return dropped_layer_list


def discrete_block_dropping(args, model, dataloader: DataLoader, accelerator: Accelerator, num_samples: int):
    drop_n = args.drop_n
    if drop_n == 0:
        accelerator.print("drop_n=0: No blocks to drop, returning empty list")
        return []

    similarities = get_block_similarities(model, dataloader, accelerator, num_samples, cache_file=args.similarity_cache_file)

    similarities_drop_1 = similarities[:, 0].view(-1)
    sorted_similarities, sorted_layer_id = torch.sort(similarities_drop_1, dim=0, descending=True)

    dropped_layer_list = sorted_layer_id[:drop_n].tolist()
    accelerator.print(f"Dropped layer: {dropped_layer_list}, similarities: {sorted_similarities[:drop_n].tolist()}")
    return dropped_layer_list


def post_block_drop(prune_model_save_path, model, tokenizer, reserved_layer_list, accelerator: Accelerator, only_update_config=False):
    unwrapped_model = accelerator.unwrap_model(model)

    if accelerator.is_main_process:
        out_cfg = deepcopy(unwrapped_model.config)
        model_type = getattr(unwrapped_model.config, "model_type", None)

        if model_type in auto_map:
            out_cfg.auto_map = auto_map[model_type]
        else:
            raise ValueError("Unsupported model type!")

        # Get the correct total number of layers
        if model_type == "swinv2":
            total_layers = sum(out_cfg.depths)  # Total transformer blocks across all stages
        else:
            total_layers = out_cfg.num_hidden_layers

        dropped_attn_list = dropped_mlp_list = list(set(list(range(total_layers))) - set(reserved_layer_list))
        out_cfg.drop_mlp_list = [idx for idx, v in enumerate(getattr(unwrapped_model.config, 'drop_mlp_list', [])) if v] + dropped_mlp_list
        out_cfg.drop_attn_list = [idx for idx, v in enumerate(getattr(unwrapped_model.config, 'drop_attn_list', [])) if v] + dropped_attn_list

        accelerator.print(f"Dropped attention list: {dropped_attn_list}")
        accelerator.print(f"Dropped MLP list: {dropped_mlp_list}")

        accelerator.print("Saving...")
        shutil.copy(CUSTOM_FILE[out_cfg.model_type]["config"], prune_model_save_path)
        shutil.copy(CUSTOM_FILE[out_cfg.model_type]["model"], prune_model_save_path)
        if not only_update_config:
            model.save_pretrained(prune_model_save_path)
            tokenizer.save_pretrained(prune_model_save_path)
        out_cfg.save_pretrained(prune_model_save_path)


@no_grad()
def get_block_similarities(model, dataloader: DataLoader, accelerator: Accelerator, num_samples: int, cache_file=None):
    device = accelerator.device
    unwrapped_model = accelerator.unwrap_model(model)
    is_vision = is_vision_model(unwrapped_model)
    is_swinv2 = is_vision and unwrapped_model.config.model_type == "swinv2"

    # Use stage-aware computation for SwinV2
    if is_swinv2:
        return get_swinv2_block_similarities(model, dataloader, accelerator, num_samples, cache_file)

    cached_similarities = load_cached_similarities(cache_file, device, accelerator)
    if cached_similarities is not None:
        similarities = cached_similarities
    else:
        # calculate similarities
        accelerator.print(f"No cached model found. Running model on {num_samples} samples for each device.")
        unwrapped_model = accelerator.unwrap_model(model)  # 🔍 unwrap model first
        unwrapped_model.config.use_cache = False

        if is_vision:
            layers = get_vision_model_layers(unwrapped_model)
            accelerator.print("Using vision model - accessing encoder layers")
        else:
            layers = unwrapped_model.model.layers
            accelerator.print("Using language model - accessing model layers")

        accelerator.print("Getting features...")
        inputs, outputs, attention_mask, position_ids, cache_position = prepare_calibration_input(unwrapped_model, dataloader, num_samples)
        num_layers = unwrapped_model.config.num_hidden_layers
        # Initialize the similarities.
        # Row: each layer
        # Column: similarity to the next n layer
        # Example: [ [0.5],  [0.5],  [0.5],  [0.5],  [0.5],  [0.5]]  # shape(6, 1)
        similarities = torch.full((len(layers), 1), -math.inf, device=device)

        accelerator.print('Starting ...')
        dtype = torch.float32
        for i in tqdm(range(num_layers), desc="Recording hidden states...", disable=not accelerator.is_main_process):
            sys.stderr.flush()
            torch.cuda.empty_cache()
            print_gpu_memory(accelerator)
            layer = layers[i]

            wrapped_layer = HiddenStatesRecordWrapper(layer, record_input=True, record_output=True)  # 🔍 Wrap layer

            # Forward hook for recording hidden states
            def record_states_hook(_, input, output):
                wrapped_layer.record(input[0].data, output[0].data)

            # Get states
            handle = layer.register_forward_hook(record_states_hook)
            for j in range(num_samples):
                if is_vision:
                    model_type = getattr(unwrapped_model.config, "model_type", None)
                    if model_type == "dinov3_vit":
                        pos_emb = getattr(unwrapped_model, 'dinov3_position_embeddings', None)
                        pos_emb_j = pos_emb[j] if pos_emb is not None else None
                        layer_output = layer(inputs[j], position_embeddings=pos_emb_j)
                    else:
                        layer_output = layer(inputs[j])
                    outputs[j] = layer_output[0] if isinstance(layer_output, tuple) else layer_output
                else:
                    inp = inputs[j]
                    if inp.dim() == 2:
                        inp = inp.unsqueeze(0)

                    kwargs = dict(attention_mask=attention_mask[j], position_ids=position_ids[j])
                    if getattr(unwrapped_model.config, "model_type", None) == "llama":
                        kwargs["cache_position"] = cache_position[j]

                    pos_emb = get_position_embeddings(unwrapped_model, inp, position_ids[j])
                    if pos_emb is not None:
                        kwargs["position_embeddings"] = pos_emb
                    outputs[j] = layer(inp, **kwargs)[0]

            handle.remove()

            # Update inputs & outputs
            inputs, outputs = outputs, inputs
            print_gpu_memory(accelerator)

            input_hidden_states = torch.cat(wrapped_layer.input_hidden_states, dim=0).to(dtype).to(device)
            output_hidden_states = torch.cat(wrapped_layer.output_hidden_states, dim=0).to(dtype).to(device)
            cos_sim = F.cosine_similarity(input_hidden_states, output_hidden_states, dim=-1)
            cos_sim = cos_sim.mean()
            cos_sim = accelerator.reduce(cos_sim, reduction="mean")
            similarities[i, 0] = cos_sim
            layer.to("cpu")

        save_similarities_cache(similarities, cache_file, accelerator)

    accelerator.print("similarities\n", similarities)
    return similarities


@no_grad()
def get_swinv2_block_similarities(model, dataloader: DataLoader, accelerator: Accelerator, num_samples: int, cache_file=None):
    device = accelerator.device

    cached_similarities = load_cached_similarities(cache_file, device, accelerator)
    if cached_similarities is not None:
        similarities = cached_similarities
    else:
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.config.use_cache = False

        get_vision_model_layers(unwrapped_model)
        stage_info = unwrapped_model.swinv2_stage_info
        
        total_layers = sum(unwrapped_model.config.depths)
        similarities = torch.full((total_layers, 1), -math.inf, device=device)
        calibration_inputs, _, _, _, _ = prepare_calibration_input(unwrapped_model, dataloader, num_samples)
        accelerator.print(f"Processing {total_layers} blocks across {len(stage_info)} stages for discrete block dropping")
        current_stage_inputs = calibration_inputs

        for stage_idx, stage_data in enumerate(stage_info):
            start_layer = stage_data['start_layer']
            stage_depth = stage_data['depth']
            stage_module = stage_data['stage_module']

            stage_inputs = current_stage_inputs
            batch_size, seq_len, hidden_dim = stage_inputs[0].shape
            hw = int(seq_len ** 0.5)
            input_dimensions = (hw, hw)

            current_hidden_states = stage_inputs
            for local_layer_idx in range(stage_depth):
                global_layer_idx = start_layer + local_layer_idx
                layer = stage_module.blocks[local_layer_idx]

                wrapped_layer = HiddenStatesRecordWrapper(layer, record_input=True, record_output=True)

                def record_block_states_hook(_, input, output):
                    wrapped_layer.record(input[0].data, output[0].data)

                # Register hook and forward pass through this block for all samples
                handle = layer.register_forward_hook(record_block_states_hook)
                next_hidden_states = []
                
                for j in range(num_samples):
                    layer_output = layer(current_hidden_states[j], input_dimensions)
                    next_hidden_states.append(layer_output[0])

                handle.remove()

                dtype = torch.float32
                input_hidden_states = torch.cat(wrapped_layer.input_hidden_states, dim=0).to(dtype).to(device)
                output_hidden_states = torch.cat(wrapped_layer.output_hidden_states, dim=0).to(dtype).to(device)

                cos_sim = F.cosine_similarity(input_hidden_states, output_hidden_states, dim=-1)
                cos_sim = cos_sim.mean()
                cos_sim = accelerator.reduce(cos_sim, reduction="mean")

                similarities[global_layer_idx, 0] = cos_sim
                current_hidden_states = next_hidden_states

            # Prepare inputs for next stage
            if stage_idx < len(stage_info) - 1:
                current_stage_inputs = advance_swinv2_hierarchical_stage(current_stage_inputs, stage_data, input_dimensions, num_samples)

        save_similarities_cache(similarities, cache_file, accelerator)
    
    accelerator.print("Block similarities\n", similarities)
    return similarities


@no_grad()
def get_block_similarities_consecutive(model, dataloader: DataLoader, accelerator: Accelerator, num_samples: int, cache_file=None):
    device = accelerator.device
    unwrapped_model = accelerator.unwrap_model(model)
    is_vision = is_vision_model(unwrapped_model)
    is_swinv2 = is_vision and unwrapped_model.config.model_type == "swinv2"

    if is_swinv2:
        raise NotImplementedError("Consecutive block dropping is not supported for SwinV2 due to its hierarchical stage structure. Use discrete block dropping instead.")

    cached_similarities = load_cached_similarities(cache_file, device, accelerator)
    if cached_similarities is not None:
        similarities = cached_similarities
    else:
        accelerator.print(f"No cached model found. Running model on {num_samples} samples for each device.")
        unwrapped_model.config.use_cache = False

        if is_vision:
            layers = get_vision_model_layers(unwrapped_model)
            accelerator.print("Using vision model - accessing encoder layers")
        else:
            layers = unwrapped_model.model.layers
            accelerator.print("Using language model - accessing model layers")

        accelerator.print("Getting features...")
        inputs, outputs, attention_mask, position_ids, cache_position = prepare_calibration_input(unwrapped_model, dataloader, num_samples)  # 🔍

        # Get layer ids
        num_layers = unwrapped_model.config.num_hidden_layers
        # Initialize the similarities.
        # Row: each layer
        # Column: similarity to the next n layer
        # Example: [[ 0.5,  0.5,  0.5,  0.5,  0.5,  0.5],
        #           [ 0.5,  0.5,  0.5,  0.5,  0.5, -inf],
        #           [ 0.5,  0.5,  0.5,  0.5, -inf, -inf],
        #           [ 0.5,  0.5,  0.5, -inf, -inf, -inf],
        #           [ 0.5,  0.5, -inf, -inf, -inf, -inf],
        #           [ 0.5, -inf, -inf, -inf, -inf, -inf]]  # shape(6, 6)
        similarities = torch.full((len(layers), len(layers)), -math.inf, device=device)

        accelerator.print('Starting ...')
        wrapped_layers = []
        for i in tqdm(range(num_layers), desc="Recording hidden states...", disable=not accelerator.is_main_process):
            sys.stderr.flush()
            torch.cuda.empty_cache()
            print_gpu_memory(accelerator)
            layer = layers[i]

            wrapped_layer = HiddenStatesRecordWrapper(layer, record_input=True, record_output=(i == len(layers) - 1))
            wrapped_layers.append(wrapped_layer)

            # Forward hook for recording hidden states
            def record_states_hook(_, input, output):
                wrapped_layer.record(input[0].data, output[0].data)

            # Get states
            handle = layer.register_forward_hook(record_states_hook)
            for j in range(num_samples):
                if is_vision:
                    model_type = getattr(unwrapped_model.config, "model_type", None)
                    if model_type == "dinov3_vit":
                        pos_emb = getattr(unwrapped_model, 'dinov3_position_embeddings', None)
                        pos_emb_j = pos_emb[j] if pos_emb is not None else None
                        layer_output = layer(inputs[j], position_embeddings=pos_emb_j)
                    else:
                        layer_output = layer(inputs[j])
                    outputs[j] = layer_output[0] if isinstance(layer_output, tuple) else layer_output
                else:
                    inp = inputs[j]
                    if inp.dim() == 2:
                        inp = inp.unsqueeze(0)

                    kwargs = dict(attention_mask=attention_mask[j], position_ids=position_ids[j])
                    if getattr(unwrapped_model.config, "model_type", None) == "llama":
                        kwargs["cache_position"] = cache_position[j]
                    pos_emb = get_position_embeddings(unwrapped_model, inp, position_ids[j])

                    if pos_emb is not None:
                        kwargs["position_embeddings"] = pos_emb
                    outputs[j] = layer(inp, **kwargs)[0]

            handle.remove()

            # Update inputs & outputs
            inputs, outputs = outputs, inputs
            print_gpu_memory(accelerator)

        dtype = torch.float32
        all_hidden_states = []
        for i in tqdm(range(len(layers)), desc="Concatenating hidden states...", disable=not accelerator.is_main_process):
            all_hidden_states.append(torch.cat(wrapped_layers[i].input_hidden_states, dim=0).to(dtype))  # (total_token_num, hidden_size)
        all_hidden_states.append(torch.cat(wrapped_layers[-1].output_hidden_states, dim=0).to(dtype))
        accelerator.print(f'Total {len(all_hidden_states)} hidden states concatenated.')

        for i in tqdm(range(len(all_hidden_states)), desc="Calculating similarities...", disable=not accelerator.is_main_process):
            for j in range(i + 1, len(all_hidden_states)):
                packed_hidden_states_layer_i = all_hidden_states[i].to(device)
                packed_hidden_states_layer_j = all_hidden_states[j].to(device)
                index_gap = j - i

                cos_sim = F.cosine_similarity(packed_hidden_states_layer_i, packed_hidden_states_layer_j, dim=-1)  # (total_token_num)
                cos_sim = cos_sim.mean()
                cos_sim = accelerator.reduce(cos_sim, reduction="mean")  # 🔍 All reduce across devices

                similarities[i, index_gap - 1] = cos_sim


        save_similarities_cache(similarities, cache_file, accelerator)

    accelerator.print("similarities\n", similarities)
    return similarities
