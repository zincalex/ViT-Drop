import os
import torch

from torch import nn as nn, cuda
from transformers import PretrainedConfig, AutoConfig

from .wrapper import HiddenStatesRecordWrapper


def create_recording_wrappers(module_pre_norm, module, drop_norm):
    if drop_norm:
        wrapped_module_pre_norm = HiddenStatesRecordWrapper(module_pre_norm, record_input=True, record_output=False)
    else:
        wrapped_module_pre_norm = HiddenStatesRecordWrapper(module_pre_norm, record_input=False, record_output=True)
    wrapped_module = HiddenStatesRecordWrapper(module, record_input=False, record_output=True)

    return wrapped_module_pre_norm, wrapped_module


def is_vision_model(model) -> bool:
    if hasattr(model, 'config'):
        config = model.config
    elif isinstance(model, PretrainedConfig):
        config = model
    elif isinstance(model, str):
        config = AutoConfig.from_pretrained(model, trust_remote_code=True)
    else:
        return False

    return hasattr(config, 'image_size') or hasattr(config, 'patch_size')


def get_vision_model_layers(model):
    base_prefix = getattr(model, 'base_model_prefix', None)
    base_model = getattr(model, base_prefix, model)

    if base_prefix == "swinv2":
        layers = []
        stage_info = []
        for stage_idx, stage in enumerate(base_model.encoder.layers):
            stage_start = len(layers)
            layers.extend(stage.blocks)
            stage_end = len(layers)
            stage_info.append({'stage_idx': stage_idx, 'start_layer': stage_start, 'end_layer': stage_end,
                'depth': len(stage.blocks), 'stage_module': stage})

        if not hasattr(model, 'swinv2_stage_info'):
            model.swinv2_stage_info = stage_info
    elif base_prefix == "dinov3_vit":   # DINOv3 no encoder submodule
        layers = base_model.layer
    else:                               # ViT and DINOv2 use encoder.layer directly
        layers = base_model.encoder.layer

    return layers


@torch.no_grad()
def advance_swinv2_hierarchical_stage(stage_inputs, stage_data, input_dimensions, num_samples):
    stage_module = stage_data['stage_module']

    stage_outputs = []
    for j in range(num_samples):  # Forward through all blocks in this stage
        hidden_states = stage_inputs[j]
        for block in stage_module.blocks:
            layer_output = block(hidden_states, input_dimensions)
            hidden_states = layer_output[0]

        if stage_module.downsample is not None:
            hidden_states = stage_module.downsample(hidden_states, input_dimensions)

        stage_outputs.append(hidden_states)

    return stage_outputs


def print_gpu_memory(accelerator):
    if accelerator.is_local_main_process:  # 🔍
        for i in range(cuda.device_count()):
            used_memory = cuda.memory_allocated(0) // 1024 ** 2
            print(f"GPU {i} Used Memory: {used_memory}MB")


def print_gpu_memory_device():
    device = cuda.current_device()
    used_memory = cuda.memory_allocated(device) // 1024 ** 2
    print(f"GPU {device} Used Memory: {used_memory}MB")


def find_modules(module, layers=[], name='') -> dict:
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_modules(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res


def find_linears(module) -> dict:
    res = find_modules(module, [nn.Linear])
    return res


@torch.no_grad()
def check_sparsity(model):
    use_cache = model.config.use_cache
    model.config.use_cache = False

    layers = model.model.layers
    count = 0
    total_params = 0
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_modules(layer)

        sub_count = 0
        sub_params = 0
        for name in subset:
            W = subset[name].weight.data
            count += (W == 0).sum().item()
            total_params += W.numel()

            sub_count += (W == 0).sum().item()
            sub_params += W.numel()

        print(f"layer {i} sparsity {float(sub_count) / sub_params:.6f}")

    model.config.use_cache = use_cache
    return float(count) / total_params


@torch.no_grad()
def check_sparsity_from_state_dict(state_dict):
    # Get corresponding names for each layer
    layer_params = {}
    for name in sorted(list(state_dict.keys())):
        if "layers" in name:
            layer_id = int(name.split(".")[2])
            if layer_id not in layer_params:
                layer_params[layer_id] = [name]
            else:
                layer_params[layer_id].append(name)
    layer_num = max(list(layer_params.keys())) + 1

    # Calculate sparsity
    count = 0
    total_params = 0
    for i in range(layer_num):
        sub_count = 0
        sub_params = 0
        for name in layer_params[i]:
            count += (state_dict[name] == 0).sum().item()
            total_params += state_dict[name].numel()

            sub_count += (state_dict[name] == 0).sum().item()
            sub_params += state_dict[name].numel()

        print(f"layer {i} sparsity {float(sub_count) / sub_params:.6f}")

    return float(count) / total_params


@torch.no_grad()
def prepare_calibration_input(model, dataloader, num_samples=16):
    cache = {"inputs": [], "attention_mask": [], "position_ids": [], "cache_position": [],
             "position_embeddings": []}

    class Catcher(nn.Module):
        def __init__(self, module, is_vision=False, embeddings=None, dinov3_model=None):
            super().__init__()
            self.module = module
            self.self_attn = None
            self.is_vision = is_vision
            self.embeddings = embeddings
            self.dinov3_model = dinov3_model

        def forward(self, *args, **kwargs):
            if self.embeddings is not None: # SwinV2 patch embeddings case
                pixel_values = args[0]
                embeddings, output_dimensions = self.module(pixel_values)
                cache['inputs'].append(embeddings)
            elif self.dinov3_model is not None: # DINOv3: capture hidden_states and position_embeddings
                hidden_states = args[0]
                cache['inputs'].append(hidden_states)
                cache['position_embeddings'].append(kwargs.get('position_embeddings'))
            else: # Standard layer case (vision or language)
                hidden_states = args[0]
                cache['inputs'].append(hidden_states)
                if not self.is_vision:
                    cache['attention_mask'].append(kwargs.get('attention_mask'))
                    cache['position_ids'].append(kwargs.get('position_ids'))
                    cache['cache_position'].append(kwargs.get('cache_position'))
                    raise ValueError

            # Common for all vision models
            cache['attention_mask'].append(None)
            cache['position_ids'].append(None)
            cache['cache_position'].append(None)
            raise ValueError

    def run_calibration_loop(model, dataloader, num_samples, is_vision):
        for index, batch in enumerate(dataloader):
            if index >= num_samples:
                break
            try:
                if is_vision:
                    pixel_values = batch["pixel_values"]
                    model_dtype = next(model.parameters()).dtype
                    if pixel_values.dtype != model_dtype:
                        pixel_values = pixel_values.to(dtype=model_dtype)
                    model(pixel_values=pixel_values)
                else:
                    model(**batch)
            except ValueError:
                pass

    is_vision = is_vision_model(model)
    if is_vision and model.config.model_type == "swinv2":    # Replace patch embeddings
        base_prefix = getattr(model, 'base_model_prefix', None)
        base_model = getattr(model, base_prefix)
        original_patch_embeddings = base_model.embeddings.patch_embeddings
        base_model.embeddings.patch_embeddings = Catcher(original_patch_embeddings,is_vision=is_vision,embeddings=True)
        run_calibration_loop(model, dataloader, num_samples, is_vision)
        base_model.embeddings.patch_embeddings = original_patch_embeddings
    elif is_vision and model.config.model_type == "dinov3_vit": # DINOv3: layers need position_embeddings, so we intercept the first layer
        layers = get_vision_model_layers(model)
        original_layer = layers[0]
        layers[0] = Catcher(original_layer, is_vision=is_vision, dinov3_model=model)
        run_calibration_loop(model, dataloader, num_samples, is_vision)
        layers[0] = original_layer

        if cache['position_embeddings']: # Store position_embeddings on model for later use in forward passes
            model.dinov3_position_embeddings = cache['position_embeddings']
    else:
        layers = model.model.layers if not is_vision else get_vision_model_layers(model)
        original_layer = layers[0]
        layers[0] = Catcher(original_layer, is_vision=is_vision)
        run_calibration_loop(model, dataloader, num_samples, is_vision)
        layers[0] = original_layer

    outputs = [None] * len(cache['inputs'])
    return cache['inputs'], outputs, cache['attention_mask'], cache['position_ids'], cache['cache_position']


def get_position_embeddings(model, hidden_states, position_ids):
    model_base = getattr(model, "model", None)
    rotary_emb = getattr(model_base, "rotary_emb", None) if model_base is not None else None
    if rotary_emb is not None:
        if position_ids is None:
            seq_len = hidden_states.shape[1]
            position_ids = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0)
        cos, sin = rotary_emb(hidden_states, position_ids)
        return cos, sin
    return None


auto_map = {
    "llama": {
                "AutoConfig": "configuration_dropped_llama.LlamaConfig",
                "AutoModelForCausalLM": "modeling_dropped_llama.LlamaForCausalLM"
            }, 
    "mistral": {
                "AutoConfig": "configuration_dropped_mistral.MistralConfig",
                "AutoModelForCausalLM": "modeling_dropped_mistral.MistralForCausalLM"
            },
    "deepseek_v3":
                {
                "AutoConfig": "configuration_deepseek.DeepseekV3Config",
                "AutoModelForCausalLM": "modeling_dropped_deepseek.DeepseekV3ForCausalLM"
            },
    "gemma2": {
                "AutoConfig": "configuration_dropped_gemma2.Gemma2Config",
                "AutoModelForCausalLM": "modeling_dropped_gemma2.Gemma2ForCausalLM"
            },
    "gemma3_text": {
                "AutoConfig": "configuration_dropped_gemma3.Gemma3TextConfig",
                "AutoModelForCausalLM": "modeling_dropped_gemma3.Gemma3ForCausalLM"
            },

    "vit": {
                "AutoConfig": "configuration_dropped_vit.ViTConfig",
                "AutoModelForImageClassification": "modeling_dropped_vit.ViTForImageClassification"
            },
    "dinov2": {
                "AutoConfig": "configuration_dropped_dinov2.Dinov2Config",
                "AutoModelForImageClassification": "modeling_dropped_dinov2.Dinov2ForImageClassification"
            },
    "dinov3_vit": {
                "AutoConfig": "configuration_dropped_dinov3.DINOv3ViTConfig",
                "AutoModel": "modeling_dropped_dinov3.DINOv3ViTModel"
            },
    "swinv2": {
        "AutoConfig": "configuration_dropped_swinv2.Swinv2Config",
        "AutoModelForImageClassification": "modeling_dropped_swinv2.Swinv2ForImageClassification"
    },
}

CUSTOM_FILE ={
    "llama": {
        "config": os.path.join(os.path.dirname(__file__), "models/configuration_dropped_llama.py"),
        "model": os.path.join(os.path.dirname(__file__), "models/modeling_dropped_llama.py")
    },
    "mistral": {
        "config": os.path.join(os.path.dirname(__file__), "models/configuration_dropped_mistral.py"),
        "model": os.path.join(os.path.dirname(__file__), "models/modeling_dropped_mistral.py")
    },
    "deepseek_v3": {
        "config": os.path.join(os.path.dirname(__file__), "models/configuration_deepseek.py"),
        "model": os.path.join(os.path.dirname(__file__), "models/modeling_dropped_deepseek.py")
    }, 
    "gemma2": {
        "config": os.path.join(os.path.dirname(__file__), "models/configuration_dropped_gemma2.py"),
        "model": os.path.join(os.path.dirname(__file__), "models/modeling_dropped_gemma2.py")
    }, 
    "gemma3_text": {
        "config": os.path.join(os.path.dirname(__file__), "models/configuration_dropped_gemma3.py"),
        "model": os.path.join(os.path.dirname(__file__), "models/modeling_dropped_gemma3.py")
    },

    "vit": {
        "config": os.path.join(os.path.dirname(__file__), "models/configuration_dropped_vit.py"),
        "model": os.path.join(os.path.dirname(__file__), "models/modeling_dropped_vit.py")
    },
    "dinov2": {
        "config": os.path.join(os.path.dirname(__file__), "models/configuration_dropped_dinov2.py"),
        "model": os.path.join(os.path.dirname(__file__), "models/modeling_dropped_dinov2.py")
    },
    "dinov3_vit": {
        "config": os.path.join(os.path.dirname(__file__), "models/configuration_dropped_dinov3.py"),
        "model": os.path.join(os.path.dirname(__file__), "models/modeling_dropped_dinov3.py")
    },
    "swinv2": {
        "config": os.path.join(os.path.dirname(__file__), "models/configuration_dropped_swinv2.py"),
        "model": os.path.join(os.path.dirname(__file__), "models/modeling_dropped_swinv2.py")
    }
}
