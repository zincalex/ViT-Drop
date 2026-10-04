import sys
import math
import shutil
import logging

import torch
import torch.nn.functional as F
from torch import no_grad
from torch.utils.data import DataLoader

from tqdm import tqdm
from copy import deepcopy
from accelerate import Accelerator

from .io import load_cached_similarities, save_similarities_cache
from .utils import (is_vision_model, get_vision_model_layers, advance_swinv2_hierarchical_stage,
                    print_gpu_memory, prepare_calibration_input,
                    get_position_embeddings, auto_map, CUSTOM_FILE, create_recording_wrappers)
from .diagnostics import compute_layer_output_norm, print_layer_output_norms, run_experiments_on_target_layers

logger = logging.getLogger(__name__)

# Extra Computations Guards
RUN_LAYER_NORM_COMPUTATION = False
RUN_ANALYSIS = False


def calculate_layer_similarity(wrapped_module_pre_norm, wrapped_module, drop_norm, device, accelerator, model_type=None):
    dtype = torch.float32
    if drop_norm:
        if model_type == "swinv2":
            input_hidden_states = torch.cat(wrapped_module_pre_norm.input_hidden_states, dim=0).to(dtype).to(device)
            module_output = torch.cat(wrapped_module.output_hidden_states, dim=0).to(dtype).to(device)

            if input_hidden_states.shape != module_output.shape:
                input_flat = input_hidden_states.view(-1, input_hidden_states.shape[-1])
                module_flat = module_output.view(-1, module_output.shape[-1])
                min_size = min(input_flat.shape[0], module_flat.shape[0])
                accelerator.print(f"Min size: {min_size}")
                input_hidden_states = input_flat[:min_size]
                module_output = module_flat[:min_size]

            output_hidden_states = input_hidden_states + module_output
        else :
            input_hidden_states = torch.cat(wrapped_module_pre_norm.input_hidden_states, dim=0).to(dtype).to(device)
            output_hidden_states = input_hidden_states + torch.cat(wrapped_module.output_hidden_states, dim=0).to(dtype).to(device)
    else :
        input_hidden_states = torch.cat(wrapped_module_pre_norm.output_hidden_states, dim=0).to(dtype).to(device)
        output_hidden_states = torch.cat(wrapped_module.output_hidden_states, dim=0).to(dtype).to(device)

    # Calculate similarity
    cos_sim = F.cosine_similarity(input_hidden_states, output_hidden_states, dim=-1)
    cos_sim = cos_sim.mean()
    cos_sim = accelerator.reduce(cos_sim, reduction="mean")
    
    return cos_sim


def manual_forward_pass(unwrapped_model, inputs, outputs, attention_mask, position_ids, cache_position, layer, num_samples, input_dimensions=None, save_attention=False):
    model_type = getattr(unwrapped_model.config, "model_type", None)
    for j in range(num_samples):

        if model_type == "swinv2":
            layer_output = layer(inputs[j], input_dimensions, output_attentions=save_attention)
            outputs[j] = layer_output[0] if isinstance(layer_output, tuple) else layer_output
        elif model_type == "dinov3_vit":
            pos_emb = getattr(unwrapped_model, 'dinov3_position_embeddings', None)
            pos_emb_j = pos_emb[j] if pos_emb is not None else None
            layer_output = layer(inputs[j], position_embeddings=pos_emb_j)
            outputs[j] = layer_output if not isinstance(layer_output, tuple) else layer_output[0]
        elif model_type in ["vit", "dinov2"]:
            layer_output = layer(inputs[j])
            outputs[j] = layer_output[0] if isinstance(layer_output, tuple) else layer_output
        else:  # LLM models (llama, mistral, etc.)
            inp = inputs[j]
            if inp.dim() == 2:
                inp = inp.unsqueeze(0)

            kwargs = dict(attention_mask=attention_mask[j], position_ids=position_ids[j], output_attentions=save_attention)
            if model_type == "llama":
                kwargs["cache_position"] = cache_position[j]
            pos_emb = get_position_embeddings(unwrapped_model, inp, position_ids[j])

            if pos_emb is not None:
                kwargs["position_embeddings"] = pos_emb
            outputs[j] = layer(inp, **kwargs)[0]


@no_grad()
def get_layer_similarities(model, dataloader: DataLoader, accelerator: Accelerator, num_samples: int, drop_norm: bool, target_layer: str, cache_file=None):
    device = accelerator.device
    unwrapped_model = accelerator.unwrap_model(model)
    model_type = getattr(unwrapped_model.config, "model_type", None)
    is_vision = is_vision_model(unwrapped_model)
    is_swinv2 = is_vision and unwrapped_model.config.model_type == "swinv2"

    if is_vision: layers = get_vision_model_layers(unwrapped_model)
    
    # Use stage-aware computation for SwinV2
    if is_swinv2 : return get_swinv2_stage_similarities(model, dataloader, accelerator, num_samples, drop_norm, target_layer, cache_file)

    cached_similarities = load_cached_similarities(cache_file, device, accelerator)
    if cached_similarities is not None:
        similarities = cached_similarities
    else:
        accelerator.print(f"No cached model found. Running model on {num_samples} samples for each device.")
        unwrapped_model.config.use_cache = False

        if not is_vision:
            layers = unwrapped_model.model.layers
            accelerator.print("Using language model - accessing model layers")

        accelerator.print("Getting features...")
        inputs, outputs, attention_mask, position_ids, cache_position = prepare_calibration_input(unwrapped_model, dataloader, num_samples)
        layer_output_norms = []

        # Get layer ids
        num_layers = unwrapped_model.config.num_hidden_layers
        layer_indices = list(range(num_layers))

        # Initialize the similarities.
        # Row: each layer
        # Column: similarity to the next n layer
        # Example: [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]  # shape(6)
        similarities = torch.full((num_layers,), -math.inf, device=device)
        if hasattr(unwrapped_model.config, f'drop_{target_layer}_list'):
            skipped_layers = [idx for idx, v in enumerate(getattr(unwrapped_model.config, f'drop_{target_layer}_list', [])) if v]
        else:
            skipped_layers = []

        accelerator.print('Starting ...')
        for i in tqdm(range(num_layers), desc="Recording hidden states...", disable=not accelerator.is_main_process):
            if i in skipped_layers:
                similarities[i] = -math.inf
                accelerator.print('Skip the dropped layer: ', i)
                continue
            sys.stderr.flush()
            torch.cuda.empty_cache()
            print_gpu_memory(accelerator)
            layer = layers[i]

            if i in layer_indices:
                if is_vision:
                    if target_layer == 'mlp':
                        if model_type in ("dinov2", "dinov3_vit"):
                            module_pre_norm = layer.norm2
                            module = layer.mlp
                        else:
                            module_pre_norm = layer.layernorm_after
                            module = layer.output  
                    elif target_layer == 'attn':
                        if model_type in ("dinov2", "dinov3_vit"):
                            module_pre_norm = layer.norm1
                            module = layer.attention
                        else:
                            module_pre_norm = layer.layernorm_before
                            module = layer.attention
                    elif target_layer == 'all':
                        raise ValueError("Unsupported target_layer!")
                else:
                    if target_layer == 'mlp':
                        module_pre_norm = layer.post_attention_layernorm
                        module = layer.mlp
                    elif target_layer == 'attn':
                        module_pre_norm = layer.input_layernorm
                        module = layer.self_attn
                    elif target_layer == 'all':
                        raise ValueError("Unsupported target_layer!")
                
                # Create recording wrappers
                wrapped_module_pre_norm, wrapped_module = create_recording_wrappers(module_pre_norm, module, drop_norm)

                # Forward hook for recording hidden states
                def record_module_pre_norm_states_hook(_, input, output):
                    wrapped_module_pre_norm.record(input[0].data, output[0].data)

                if target_layer == 'mlp':
                    def record_module_states_hook(_, input, output):
                        wrapped_module.record(input[0].data, output[0].data)
                elif target_layer == 'attn':
                    def record_module_states_hook(_, input, output):
                        wrapped_module.record(None, output[0].data)
                else:
                    raise ValueError("Unsupported target_layer!")

                handles = []
                handles.append(module_pre_norm.register_forward_hook(record_module_pre_norm_states_hook))
                handles.append(module.register_forward_hook(record_module_states_hook))

                manual_forward_pass(unwrapped_model, inputs, outputs, attention_mask, position_ids, cache_position, layer, num_samples)

                for handle in handles:
                    handle.remove()

                cos_sim = calculate_layer_similarity(wrapped_module_pre_norm, wrapped_module, drop_norm, device, accelerator, model_type)
                accelerator.print(f'layer {i} similarity: {cos_sim.item()}')
                similarities[i] = cos_sim
                
            else:
                manual_forward_pass(unwrapped_model, inputs, outputs, attention_mask, position_ids, cache_position, layer, num_samples)

            if RUN_LAYER_NORM_COMPUTATION :
                avg_layer_norm = compute_layer_output_norm(outputs, num_samples, accelerator, device)
                layer_output_norms.append(avg_layer_norm)

            # Update inputs & outputs
            inputs, outputs = outputs, inputs

        # Save to the cache file
        save_similarities_cache(similarities, cache_file, accelerator)

        if RUN_LAYER_NORM_COMPUTATION:
            print_layer_output_norms(layer_output_norms, accelerator)

    accelerator.print("similarities\n", similarities)
    return similarities


@no_grad()
def get_swinv2_stage_similarities(model, dataloader: DataLoader, accelerator: Accelerator, num_samples: int, drop_norm: bool, target_layer: str, cache_file=None):
    device = accelerator.device
    cached_similarities = load_cached_similarities(cache_file, device, accelerator)
    if cached_similarities is not None:
        similarities = cached_similarities
    else:
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.config.use_cache = False
        model_type = getattr(unwrapped_model.config, "model_type", None)

        stage_info = getattr(unwrapped_model, 'swinv2_stage_info', None)
        total_layers = sum(unwrapped_model.config.depths)  # Get total number of layers across all stages
        similarities = torch.full((total_layers,), -math.inf, device=device)
        calibration_inputs, _, _, _, _ = prepare_calibration_input(unwrapped_model, dataloader, num_samples)

        # Process each stage incrementally
        current_stage_inputs = calibration_inputs
        for stage_idx, stage_data in enumerate(stage_info):
            stage_idx = stage_data['stage_idx']
            start_layer = stage_data['start_layer']
            stage_depth = stage_data['depth']
            stage_module = stage_data['stage_module']

            stage_inputs = current_stage_inputs

            batch_size, seq_len, hidden_dim = stage_inputs[0].shape
            hw = int(seq_len ** 0.5)
            input_dimensions = (hw, hw)

            # Process each layer within this stage
            current_hidden_states = stage_inputs
            for local_layer_idx in range(stage_depth):
                global_layer_idx = start_layer + local_layer_idx
                layer = stage_module.blocks[local_layer_idx]

                if target_layer == 'mlp':
                    module_pre_norm = layer.intermediate
                    module = layer.layernorm_after
                elif target_layer == 'attn':
                    module_pre_norm = layer
                    module = layer.layernorm_before
                else:
                    raise ValueError(f"Unsupported target_layer: {target_layer}")

                wrapped_module_pre_norm, wrapped_module = create_recording_wrappers(module_pre_norm, module, drop_norm)

                def record_module_pre_norm_states_hook(_, input, output):
                    if isinstance(output, tuple):  # this control is needed because mlp or attn block use different output format
                        wrapped_module_pre_norm.record(input[0].data, output[0].data)
                    else:
                        wrapped_module_pre_norm.record(input[0].data, output.data)

                if target_layer == 'mlp':
                    def record_module_states_hook(_, input, output):
                        wrapped_module.record(input[0].data, output.data)
                elif target_layer == 'attn':
                    def record_module_states_hook(_, input, output):
                        wrapped_module.record(None, output[0].data)

                handles = []
                handles.append(module_pre_norm.register_forward_hook(record_module_pre_norm_states_hook))
                handles.append(module.register_forward_hook(record_module_states_hook))

                # Forward pass through this layer for all samples
                next_hidden_states = []
                for j in range(num_samples):
                    layer_output = layer(current_hidden_states[j], input_dimensions)
                    next_hidden_states.append(layer_output[0])

                for handle in handles: handle.remove()

                cos_sim = calculate_layer_similarity(wrapped_module_pre_norm, wrapped_module, drop_norm, device, accelerator, model_type=model_type)
                similarities[global_layer_idx] = cos_sim
                current_hidden_states = next_hidden_states

            # Prepare inputs for next stage (if not last stage)
            if stage_idx < len(stage_info) - 1:
                current_stage_inputs = advance_swinv2_hierarchical_stage(current_stage_inputs, stage_data, input_dimensions, num_samples)

        save_similarities_cache(similarities, cache_file, accelerator)

    accelerator.print("similarities\n", similarities)
    return similarities


def discrete_layer_dropping(args, model, dataloader: DataLoader, accelerator: Accelerator, num_samples: int, tokenizer=None):
    drop_n = args.drop_n

    if drop_n == 0:
        accelerator.print("drop_n=0: No layers to drop, returning empty list")
        return []
    
    if args.target_layer == 'all':
        similarities_attn = get_layer_similarities(model, dataloader, accelerator, num_samples, args.layer_drop_norm, target_layer='attn', cache_file=args.similarity_cache_file.replace("all", "all_attn"))
        similarities_mlp = get_layer_similarities(model, dataloader, accelerator, num_samples, args.layer_drop_norm, target_layer='mlp', cache_file=args.similarity_cache_file.replace("all", "all_mlp"))
        similarities = torch.cat((similarities_attn, similarities_mlp), dim=0)
    else:
        similarities = get_layer_similarities(model, dataloader, accelerator, num_samples, args.layer_drop_norm, target_layer=args.target_layer, cache_file=args.similarity_cache_file)

    sorted_similarities, sorted_layer_id = torch.sort(similarities, dim=0, descending=True)

    if RUN_ANALYSIS :
        run_experiments_on_target_layers(model, dataloader, accelerator, sorted_similarities, sorted_layer_id, args, num_samples, tokenizer)

    dropped_layer_list = sorted_layer_id[:drop_n].tolist()
    accelerator.print(f"Dropped layer: {dropped_layer_list}, similarities: {sorted_similarities[:drop_n].tolist()}")
    return dropped_layer_list


def post_layers_drop(prune_model_save_path, target_layer, model, tokenizer, reserved_layer_list, accelerator: Accelerator, only_update_config=False):
    unwrapped_model = accelerator.unwrap_model(model)  # 🔍 unwrap model first

    if accelerator.is_main_process:
        out_cfg = deepcopy(unwrapped_model.config)
        model_type = getattr(unwrapped_model.config, "model_type", None)

        if model_type in auto_map:
            out_cfg.auto_map = auto_map[model_type]
        else:
            raise ValueError("Unsupported model type!")

        if model_type == "swinv2":
            layers = get_vision_model_layers(unwrapped_model)
            total_layers = len(layers)
        else:
            total_layers = out_cfg.num_hidden_layers
            
        dropped_attn_list = []
        dropped_mlp_list = []
        if target_layer == 'all':
            dropped_layer_list = list(set(list(range(total_layers * 2))) - set(reserved_layer_list))
            for idx in dropped_layer_list:
                if idx >= total_layers:
                    dropped_mlp_list.append(idx - total_layers)
                else:
                    dropped_attn_list.append(idx)
        elif target_layer == 'attn':
            dropped_attn_list = list(set(list(range(total_layers))) - set(reserved_layer_list))
        elif target_layer == 'mlp':
            dropped_mlp_list = list(set(list(range(total_layers))) - set(reserved_layer_list))
        else:
            raise ValueError("Unsupported target_layer!")

        out_cfg.drop_mlp_list = [idx for idx, v in enumerate(getattr(unwrapped_model.config, "drop_mlp_list", [])) if v] + dropped_mlp_list
        out_cfg.drop_attn_list = [idx for idx, v in enumerate(getattr(unwrapped_model.config, "drop_attn_list", [])) if v] + dropped_attn_list

        accelerator.print(f"Dropped attention list: {dropped_attn_list}")
        accelerator.print(f"Dropped MLP list: {dropped_mlp_list}")

        accelerator.print("Saving...")
        shutil.copy(CUSTOM_FILE[out_cfg.model_type]["config"], prune_model_save_path)
        shutil.copy(CUSTOM_FILE[out_cfg.model_type]["model"], prune_model_save_path)
        if not only_update_config:
            model.save_pretrained(prune_model_save_path)
            tokenizer.save_pretrained(prune_model_save_path)
        out_cfg.save_pretrained(prune_model_save_path)
