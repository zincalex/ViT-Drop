import os
import json
import torch
from accelerate import Accelerator

from .utils import check_sparsity_from_state_dict, get_vision_model_layers

def load_json(file_path):
    with open(file_path, "r", encoding="utf8") as f:
        data = json.load(f)
    return data


def save_json(data, file_path, indent=4, **kwargs):
    create_dir(os.path.dirname(file_path))
    with open(file_path, "w", encoding="utf8") as f:
        f.write(f"{json.dumps(data, ensure_ascii=False, indent=indent, **kwargs)}\n")


def create_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def load_cached_similarities(cache_file, device, accelerator):
    if cache_file is not None and os.path.exists(cache_file):
        accelerator.print(f"Loading cached model from {cache_file}")
        similarities = torch.load(cache_file, map_location=device, weights_only=True)
        return similarities
    return None


def save_similarities_cache(similarities, cache_file, accelerator):
    if cache_file is not None and accelerator.is_main_process:
        create_dir(os.path.dirname(cache_file))
        torch.save(similarities.clone().cpu(), cache_file)
        accelerator.print(f"Saving cached similarities to {cache_file}")
    accelerator.wait_for_everyone()


def save_update_state_dict(save_path, accelerator, update_state_dict):
    accelerator.print("Saving state dicts...")
    if accelerator.is_main_process:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(update_state_dict, os.path.join(save_path, "update_state_dict.pt"))
    accelerator.wait_for_everyone()


def save_sparse_model(prune_model_save_path, model, tokenizer, accelerator: Accelerator, update_state_dict, check_sparsity=True):
    if check_sparsity and accelerator.is_main_process:
        accelerator.print("*" * 30)
        accelerator.print("Calculating sparsity for pruned params in the state dict...")
        sparsity_ratio = check_sparsity_from_state_dict(update_state_dict)
        accelerator.print(f"sparsity sanity check {sparsity_ratio:.4f}")
        accelerator.print("*" * 30)
    accelerator.wait_for_everyone()

    accelerator.print("Saving models... (may take minutes)")
    if accelerator.is_main_process:
        if not os.path.exists(prune_model_save_path):
            os.makedirs(prune_model_save_path)
    accelerator.wait_for_everyone()

    save_state_dict = accelerator.get_state_dict(model)

    if save_state_dict is not None:
        accelerator.print(f"State dict stored in CPU on process {accelerator.process_index}")

        for name, param in save_state_dict.items():
            if name in update_state_dict:
                accelerator.print(f"Updating {name} (device = {save_state_dict[name].device})")
                save_state_dict[name] = update_state_dict[name]

        if check_sparsity:
            accelerator.print("*" * 30)
            accelerator.print("Calculating sparsity for all params in the model after update...")
            sparsity_ratio = check_sparsity_from_state_dict(save_state_dict)
            accelerator.print(f"sparsity sanity check {sparsity_ratio:.4f}")
            accelerator.print("*" * 30)

        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            prune_model_save_path,
            is_main_process=accelerator.is_main_process,
            save_function=accelerator.save,
            state_dict=save_state_dict,
        )
        tokenizer.save_pretrained(prune_model_save_path)

    accelerator.wait_for_everyone()
    accelerator.print(f"Model saved to {prune_model_save_path}")


def save_layer_dropped_config(target_layer, prune_model_save_path, model, accelerator: Accelerator, dropped_layer_list):
    if accelerator.is_main_process:
        if not os.path.exists(prune_model_save_path):
            os.makedirs(prune_model_save_path)

        unwrapped_model = accelerator.unwrap_model(model)
        model_type = getattr(unwrapped_model.config, "model_type", None)

        if model_type == "swinv2":  # For SwinV2, get the total number of layers across all stages
            layers = get_vision_model_layers(unwrapped_model)
            total_layers = len(layers)
        else:
            total_layers = unwrapped_model.config.num_hidden_layers
            
        if target_layer == 'all':
            reserved_layer_list = sorted(list(set(range(total_layers * 2)) - set(dropped_layer_list)))
        else:
            reserved_layer_list = sorted(list(set(range(total_layers)) - set(dropped_layer_list)))
        accelerator.print(f"Reserved layers: {reserved_layer_list}")

        save_file = os.path.join(prune_model_save_path, "reserved_layers.json")
        save_json(reserved_layer_list, save_file)

    accelerator.wait_for_everyone()


def save_block_dropped_config(prune_model_save_path, model, accelerator: Accelerator, dropped_layer_list):
    if accelerator.is_main_process:
        if not os.path.exists(prune_model_save_path):
            os.makedirs(prune_model_save_path)

        unwrapped_model = accelerator.unwrap_model(model)
        model_type = getattr(unwrapped_model.config, "model_type", None)
        

        if model_type == "swinv2":   # Handle SwinV2 total layer count
            total_layers = sum(unwrapped_model.config.depths)  # Total layers across all stages
        else:
            total_layers = unwrapped_model.config.num_hidden_layers
            
        reserved_layer_list = sorted(list(set(range(total_layers)) - set(dropped_layer_list)))
        accelerator.print(f"Reserved layers: {reserved_layer_list}")
        
        save_file = os.path.join(prune_model_save_path, "reserved_layers.json")
        save_json(reserved_layer_list, save_file)

        layer_id_mapping = {}
        for new_id, reserved_old_id in enumerate(reserved_layer_list):
            layer_id_mapping[reserved_old_id] = new_id

        save_mapping_file = os.path.join(prune_model_save_path, "layer_mapping.json")
        save_json(layer_id_mapping, save_mapping_file)

    accelerator.wait_for_everyone()
