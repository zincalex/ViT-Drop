import os
import gc
import re
import logging

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from accelerate import Accelerator
from matplotlib.colors import PowerNorm, LogNorm

from .utils import (is_vision_model, prepare_calibration_input, get_position_embeddings,
                    create_recording_wrappers)

logger = logging.getLogger(__name__)

GENERATE_HEATMAP = False
RUN_ATTENTION_ANALYSIS = False
RUN_TOKEN_LEVEL_ANALYSIS = False


def run_experiments_on_target_layers(model, dataloader, accelerator, sorted_similarities, sorted_layer_id, args, num_samples, tokenizer=None):
    if is_vision_model(accelerator.unwrap_model(model)):
        return

    layer_to_examine_index = [0, 3, 7, -1]
    for idx in layer_to_examine_index:
        layer_id = sorted_layer_id[idx].item()

        accelerator.print(f"\nAnalyzing activation triggers for layer {layer_id}")
        unwrapped_model = accelerator.unwrap_model(model)
        if hasattr(unwrapped_model, 'gradient_checkpointing_disable'):
            unwrapped_model.gradient_checkpointing_disable()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()

        fewer_num_samples = 64
        if GENERATE_HEATMAP and args.target_layer == 'attn':
            accelerator.print(f"\nGenerating attention heatmaps for layer {layer_id}")
            generate_attention_heatmaps(model, dataloader, accelerator, layer_id, fewer_num_samples, args)

        if RUN_ATTENTION_ANALYSIS and args.target_layer == 'attn':
            accelerator.print(f"\nRunning attention analysis for layer {layer_id}")
            analyze_attention_layer(model, dataloader, accelerator, layer_id, fewer_num_samples)

        if RUN_TOKEN_LEVEL_ANALYSIS:
            find_layer_activation_triggers_with_sentence_mapping(model, dataloader, accelerator, layer_id,
                target_layer_type=args.target_layer, drop_norm=args.layer_drop_norm,
                num_samples_each_device=num_samples, tokenizer=tokenizer)


def compute_layer_output_norm(outputs, num_samples, accelerator, device):
    layer_norms = []
    for j in range(num_samples):
        norm_val = outputs[j].norm(dim=-1).mean().item()
        layer_norms.append(norm_val)
    avg_layer_norm = sum(layer_norms) / len(layer_norms)
    avg_layer_norm = accelerator.reduce(torch.tensor(avg_layer_norm, device=device), reduction="mean").item()

    return avg_layer_norm


def print_layer_output_norms(layer_output_norms, accelerator):
    if accelerator.is_main_process and len(layer_output_norms) > 0:
        accelerator.print("\n" + "=" * 80)
        accelerator.print("X NORM THROUGH LAYERS")
        accelerator.print("=" * 80)
        for i, norm in enumerate(layer_output_norms):
            if norm < 0:
                accelerator.print(f"Layer {i}: SKIPPED")
            else:
                accelerator.print(f"Layer {i}: {norm:.6f}")
        accelerator.print("=" * 80 + "\n")


def calculate_layer_sim_advanced(wrapped_module_pre_norm, wrapped_module, drop_norm, device, accelerator):
    dtype = torch.float32
    if drop_norm:
        input_hidden_states = torch.cat(wrapped_module_pre_norm.input_hidden_states, dim=0).to(dtype).to(device)
        module_output = torch.cat(wrapped_module.output_hidden_states, dim=0).to(dtype).to(device)

        if input_hidden_states.shape != module_output.shape:  # SwinV2 needs attention output reshape
            input_flat = input_hidden_states.view(-1, input_hidden_states.shape[-1])
            module_flat = module_output.view(-1, module_output.shape[-1])
            min_size = min(input_flat.shape[0], module_flat.shape[0])
            accelerator.print(f"Min size: {min_size}")
            input_hidden_states = input_flat[:min_size]
            module_output = module_flat[:min_size]

        output_hidden_states = input_hidden_states + module_output
    else:
        input_hidden_states = torch.cat(wrapped_module_pre_norm.output_hidden_states, dim=0).to(dtype).to(device)
        output_hidden_states = torch.cat(wrapped_module.output_hidden_states, dim=0).to(dtype).to(device)

    cos_sim = F.cosine_similarity(input_hidden_states, output_hidden_states, dim=-1)
    cos_sim_mean = cos_sim.mean()
    cos_sim_mean = accelerator.reduce(cos_sim_mean, reduction="mean")

    return cos_sim_mean, cos_sim


@torch.no_grad()
def forward_to_target_layer(unwrapped_model, dataloader, target_layer_idx, num_samples):
    from .layer_drop import manual_forward_pass

    layers = unwrapped_model.model.layers
    inputs, outputs, attention_mask, position_ids, cache_position = prepare_calibration_input(unwrapped_model, dataloader, num_samples)

    for i in range(target_layer_idx):
        layer = layers[i]
        manual_forward_pass(unwrapped_model, inputs, outputs, attention_mask, position_ids, cache_position, layer, num_samples)
        inputs, outputs = outputs, inputs

    return layers[target_layer_idx], inputs, outputs, attention_mask, position_ids, cache_position


# --- Attention analysis experiment ---
@torch.no_grad()
def compute_stats(X, ln, delta, accelerator, output):
    X_ln = ln(X)
    Y = X + delta

    cos_x_ln_tensor = F.cosine_similarity(X, X_ln, dim=-1)
    cos_x_ln = cos_x_ln_tensor.mean()
    cos_x_ln = accelerator.reduce(cos_x_ln, reduction="mean").item()

    cos_ln_delta_tensor = F.cosine_similarity(X_ln, delta, dim=-1)
    cos_ln_delta = cos_ln_delta_tensor.mean()
    cos_ln_delta = accelerator.reduce(cos_ln_delta, reduction="mean").item()

    cos_x_delta_tensor = F.cosine_similarity(X, delta, dim=-1)
    cos_x_delta = cos_x_delta_tensor.mean()
    cos_x_delta = accelerator.reduce(cos_x_delta, reduction="mean").item()

    x_norm = X.norm(dim=-1).mean().item()
    y_norm = Y.norm(dim=-1).mean().item()
    o_norm = output.norm(dim=-1).mean().item()
    ln_norm = X_ln.norm(dim=-1).mean().item()
    delta_norm = delta.norm(dim=-1).mean().item()
    diff_x_ln = abs(x_norm - ln_norm)
    diff_ln_delta = abs(ln_norm - delta_norm)
    diff_x_delta = abs(x_norm - delta_norm)
    diff_x_y = abs(x_norm - y_norm)
    diff_output_x = o_norm - x_norm
    diff_output_y = o_norm - y_norm

    x_flat = X.view(-1, X.shape[-1])
    d_flat = delta.view(-1, delta.shape[-1])
    x_unit = x_flat / (x_flat.norm(dim=-1, keepdim=True) + 1e-9)
    alpha = (d_flat * x_unit).sum(dim=-1).mean().item()
    beta = d_flat - alpha * x_unit
    beta_norm = beta.norm(dim=-1).mean().item()

    return {
        "Similarity(X, RMSNorm(X))": cos_x_ln,
        "Similarity(RMSNorm(X), Delta)": cos_ln_delta,
        "Similarity(X, Delta)": cos_x_delta,
        "x_norm": x_norm,
        "y_norm": y_norm,
        "ln_norm": ln_norm,
        "delta_norm": delta_norm,
        "abs_diff_x_ln": diff_x_ln,
        "abs_diff_ln_delta": diff_ln_delta,
        "abs_diff_x_delta": diff_x_delta,
        "abs_diff_x_y": diff_x_y,
        "diff_output_x": diff_output_x,
        "diff_output_y": diff_output_y,
        "alpha": alpha,
        "beta_norm": beta_norm,
    }


@torch.no_grad()
def analyze_attention_layer(model, dataloader, accelerator, target_layer_idx, num_samples):
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.config.use_cache = False

    layer, inputs, outputs, attention_mask, position_ids, cache_position = forward_to_target_layer(
        unwrapped_model, dataloader, target_layer_idx, num_samples)

    ln = getattr(layer, "input_layernorm", None)

    w = ln.weight.detach().cpu()
    accelerator.print(
        f"🔍 [RMSNorm Weights] mean={w.mean().item():.4f}, std={w.std().item():.4f}, "
        f"min={w.min().item():.4f}, max={w.max().item():.4f}"
    )
    accelerator.print(f"🔍 [RMSNorm Weights Preview] {', '.join(f'{v:.3f}' for v in w[:10].tolist())}")

    attention_outputs_captured = []
    delta_stats = []

    for j in range(num_samples):
        attn_out_holder = {}
        attention_internals = {}

        def self_attn_hook(_module, _inp, _out):
            attn_out_holder['tensor'] = _out[0].detach().cpu()
            if len(_out) > 1 and _out[1] is not None:
                attention_internals['weights'] = _out[1].detach().cpu()

        def q_proj_hook(_module, _inp, _out):
            attention_internals['q_proj'] = _out.detach().cpu()

        def k_proj_hook(_module, _inp, _out):
            attention_internals['k_proj'] = _out.detach().cpu()

        def v_proj_hook(_module, _inp, _out):
            attention_internals['v_proj'] = _out.detach().cpu()

        handle_attn = layer.self_attn.register_forward_hook(self_attn_hook)
        handle_q = layer.self_attn.q_proj.register_forward_hook(q_proj_hook)
        handle_k = layer.self_attn.k_proj.register_forward_hook(k_proj_hook)
        handle_v = layer.self_attn.v_proj.register_forward_hook(v_proj_hook)

        if getattr(unwrapped_model.config, "model_type", None) == "llama":
            kwargs = dict(attention_mask=attention_mask[j], position_ids=position_ids[j],
                          cache_position=cache_position[j], output_attentions=True)
        else:
            kwargs = dict(attention_mask=attention_mask[j], position_ids=position_ids[j],
                          output_attentions=True)
        inp = inputs[j]
        if inp.dim() == 2:
            inp = inp.unsqueeze(0)
        pos_emb = get_position_embeddings(unwrapped_model, inp, position_ids[j])
        if pos_emb is not None:
            kwargs["position_embeddings"] = pos_emb

        layer_output = layer(inp, **kwargs)

        handle_attn.remove()
        handle_q.remove()
        handle_k.remove()
        handle_v.remove()

        if 'tensor' in attn_out_holder:
            attention_outputs_captured.append(attn_out_holder['tensor'])

            X = inputs[j].detach().to(accelerator.device)
            attn_tensor = attn_out_holder['tensor'].to(accelerator.device)
            stats = compute_stats(X, ln, attn_tensor, accelerator, layer_output[0].detach().to(accelerator.device))
            delta_stats.append(stats)

            if j == 0 and 'weights' in attention_internals and 'q_proj' in attention_internals:
                compute_actual_attention_contributions(X, attention_internals, layer.self_attn, accelerator, sample_idx=j)
        else:
            accelerator.print(f"🔍 [ERROR] self_attn hook did not capture attn_output for sample {j}")

    if len(delta_stats) > 0:
        keys = delta_stats[0].keys()
        agg = {k: [float(s[k]) for s in delta_stats] for k in keys}
        mean_vals = {k: sum(agg[k]) / len(agg[k]) for k in keys}
        std_vals = {k: (sum((v - mean_vals[k]) ** 2 for v in agg[k]) / len(agg[k])) ** 0.5 for k in keys}
        accelerator.print("\n===== PER-SAMPLE STATISTICS =====")
        for k in keys:
            accelerator.print(f"{k:15s} = {mean_vals[k]:.4e} ± {std_vals[k]:.4e}")

    if len(attention_outputs_captured) > 0:
        frob_norms = torch.stack([torch.norm(t.to(torch.float32), p='fro') for t in attention_outputs_captured])
        accelerator.print(f"🔍 [STATS] Average Frobenius norm of attention outputs (Δ): {frob_norms.mean().item():.4e} (±{frob_norms.std().item():.4e})")

    accelerator.wait_for_everyone()


@torch.no_grad()
def compute_actual_attention_contributions(X, attention_internals, self_attn, accelerator):
    if len(X.shape) == 2:
        X = X.unsqueeze(0)
    elif len(X.shape) == 3:
        pass
    else:
        raise ValueError(f"Unexpected X shape: {X.shape}")

    batch_size, seq_len, d_model = X.shape
    device = next(self_attn.parameters()).device

    attention_weights = attention_internals['weights'].to(device)  # [batch, num_heads, seq_len, seq_len]
    Q_proj = attention_internals['q_proj'].to(device)  # [batch, seq_len, d_model]
    K_proj = attention_internals['k_proj'].to(device)  # [batch, seq_len, d_kv_model]
    V_proj = attention_internals['v_proj'].to(device)  # [batch, seq_len, d_kv_model]

    X = X.to(device)

    num_heads = getattr(self_attn, 'num_heads', None) or self_attn.config.num_attention_heads
    num_kv_heads = getattr(self_attn, 'num_key_value_heads', None) or getattr(self_attn.config, 'num_key_value_heads', num_heads)
    head_dim = getattr(self_attn, 'head_dim', d_model // num_heads)
    o_proj = self_attn.o_proj

    # Reshape projections to multi-head format
    # Take first batch
    X_batch = X[0]
    Q_batch = Q_proj[0]
    K_batch = K_proj[0]
    V_batch = V_proj[0]
    attn_weights_batch = attention_weights[0]

    # Reshape to per-head format
    Q_heads = Q_batch.view(seq_len, num_heads, head_dim).transpose(0, 1)  # [num_heads, seq_len, head_dim]
    K_heads = K_batch.view(seq_len, num_kv_heads, head_dim).transpose(0, 1)  # [num_kv_heads, seq_len, head_dim]
    V_heads = V_batch.view(seq_len, num_kv_heads, head_dim).transpose(0, 1)  # [num_kv_heads, seq_len, head_dim]

    if num_kv_heads != num_heads:
        repeat_factor = num_heads // num_kv_heads
        Q_heads = K_heads.repeat_interleave(repeat_factor, dim=0)
        V_heads = V_heads.repeat_interleave(repeat_factor, dim=0)

    # Analyze random target tokens
    torch.manual_seed(0)
    num_targets = min(5, seq_len)
    if seq_len <= 5:
        target_indices = list(range(seq_len))
    else:
        target_indices = torch.randperm(seq_len)[:num_targets].sort()[0].tolist()

    contribution_stats = {}
    for target_t in target_indices:
        accelerator.print(f"\n🔍 === TARGET TOKEN {target_t} ===")

        X_t = X_batch[target_t]

        # Analyze contributions from each source token s <= t
        source_similarities = []
        source_contributions = []

        for source_s in range(target_t + 1):
            head_contributions = []

            for h in range(num_heads):
                alpha_ts_h = attn_weights_batch[h, target_t, source_s]
                V_s_h = V_heads[h, source_s]
                head_contrib = alpha_ts_h * V_s_h
                head_contributions.append(head_contrib)

            concat_contrib = torch.cat(head_contributions, dim=0)
            c_ts = o_proj(concat_contrib)

            cos_sim = F.cosine_similarity(c_ts.unsqueeze(0), X_t.unsqueeze(0), dim=-1).item()

            contrib_norm = c_ts.norm().item()
            source_similarities.append(cos_sim)
            source_contributions.append(contrib_norm)

            accelerator.print(f"🔍 [STATS] Similarity(X[{target_t}], delta_{{t={target_t}, s={source_s}}}) = {cos_sim:+.4f}")

        mean_similarity = sum(source_similarities) / len(source_similarities)
        std_similarity = (sum((x - mean_similarity) ** 2 for x in source_similarities) / len(source_similarities)) ** 0.5
        max_similarity = max(source_similarities)
        min_similarity = min(source_similarities)
        mean_contribution = sum(source_contributions) / len(source_contributions)

        # Compute total delta for this target token (sum of all contributions)
        total_delta_t = torch.zeros_like(X_t)
        for source_s in range(target_t + 1):
            head_contributions = []
            for h in range(num_heads):
                alpha_ts_h = attn_weights_batch[h, target_t, source_s]
                V_s_h = V_heads[h, source_s]
                head_contrib = alpha_ts_h * V_s_h
                head_contributions.append(head_contrib)
            concat_contrib = torch.cat(head_contributions, dim=0)
            c_ts = o_proj(concat_contrib)
            total_delta_t += c_ts

        delta_x_cos_sim = F.cosine_similarity(total_delta_t.unsqueeze(0), X_t.unsqueeze(0), dim=-1).item()

        accelerator.print(f"🔍 Summary Target {target_t} : "
                        f"mean_cos_sim={mean_similarity:+.4f} ± {std_similarity:.4f}, "
                        f"max_cos_sim={max_similarity:+.4f}, "
                        f"min_cos_sim={min_similarity:+.4f}")
        accelerator.print(f"🔍 Similarity(X[{target_t}], Delta[{target_t}]) = {delta_x_cos_sim:+.6f}")

        contribution_stats[f'target_{target_t}'] = {'source_similarities': source_similarities, 'source_contributions': source_contributions,
            'mean_similarity': mean_similarity, 'std_similarity': std_similarity, 'max_similarity': max_similarity,
            'min_similarity': min_similarity, 'mean_contribution': mean_contribution, 'delta_x_cos_sim': delta_x_cos_sim,
            'num_sources': len(source_similarities)}

    return contribution_stats


# --- Heatmap experiment ---
@torch.no_grad()
def generate_attention_heatmaps(model, dataloader, accelerator, target_layer_idx, heatmap_samples, args):
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.config.use_cache = False

    layer, inputs, outputs, attention_mask, position_ids, cache_position = forward_to_target_layer(
        unwrapped_model, dataloader, target_layer_idx, heatmap_samples)

    attention_weights_captured = []

    for j in range(heatmap_samples):
        if getattr(unwrapped_model.config, "model_type", None) == "llama":
            kwargs = dict(attention_mask=attention_mask[j], position_ids=position_ids[j],
                          cache_position=cache_position[j], output_attentions=True)
        else:
            kwargs = dict(attention_mask=attention_mask[j], position_ids=position_ids[j],
                          output_attentions=True)
        inp = inputs[j]
        if inp.dim() == 2:
            inp = inp.unsqueeze(0)
        pos_emb = get_position_embeddings(unwrapped_model, inp, position_ids[j])
        if pos_emb is not None:
            kwargs["position_embeddings"] = pos_emb

        layer_output = layer(inp, **kwargs)

        if len(layer_output) > 1 and layer_output[1] is not None:
            attention_weights_captured.append(layer_output[1].detach().cpu())
        else:
            accelerator.print(f"🔍 [ERROR] No attention weights returned for sample {j}")

    if accelerator.is_main_process and len(attention_weights_captured) > 0:
        save_dir = os.path.join(args.prune_model_save_path, "attention_heatmaps")
        os.makedirs(save_dir, exist_ok=True)
        for sample_idx in range(len(attention_weights_captured)):
            create_attention_heatmap(attention_weights_captured[sample_idx], sample_idx, target_layer_idx, save_dir, accelerator, unwrapped_model)

    accelerator.wait_for_everyone()


def create_attention_heatmap(attention_weights, sample_idx, layer_idx, save_dir, accelerator, unwrapped_model,
                             norm_mode: str = "gamma", gamma: float = 0.25, clip_hi_pct: float = 99.95):
    # Handle different input shapes
    if attention_weights.dim() == 4:
        attention = attention_weights.squeeze(0).to(torch.float32)
    else:
        attention = attention_weights.to(torch.float32)

    # Create weighted average attention matrix across heads
    H, S1, S2 = map(int, attention.shape)
    view = torch.zeros(S1, S2, dtype=torch.float32)
    head_weights = attention.sum(dim=(1, 2))
    head_weights = head_weights / (head_weights.sum() + 1e-8)
    for h in range(H):
        view += head_weights[h] * attention[h]
    view = view.cpu().numpy()
    view = np.nan_to_num(view, nan=0.0, posinf=0.0, neginf=0.0)
    accelerator.print(f"🔍 [HEATMAP] After weighted averaging across heads: {view.shape}")
    accelerator.print(f"🔍 [HEATMAP] Head weights : {head_weights.cpu().numpy()}")

    r = min(10, view.shape[0]); c = min(10, view.shape[1])
    accelerator.print(f"🔍 [HEATMAP] Top-left {r}x{c} attention values for sample {sample_idx}:")
    accelerator.print(f"{np.array2string(view[:r, :c], precision=4, suppress_small=True)}")

    vmin = 0.0
    vmax_raw = float(view.max())
    vmax = float(np.percentile(view, clip_hi_pct)) if clip_hi_pct is not None else vmax_raw
    if vmax <= vmin:
        vmax = 1.0

    if norm_mode == "gamma":
        norm = PowerNorm(gamma=max(1e-3, float(gamma)), vmin=vmin, vmax=vmax)
    elif norm_mode == "log":
        positive_vals = view[view > 0]
        eps = max(1e-9, float(np.min(positive_vals)) if len(positive_vals) > 0 else 1e-9)
        norm = LogNorm(vmin=eps, vmax=vmax, clip=True)

    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(16, 14))
    ax = sns.heatmap(view, cmap="viridis", norm=norm, vmin=None if norm is not None else vmin,
                     vmax=None if norm is not None else vmax, cbar=True, square=True, xticklabels=False,
                     yticklabels=False)

    S = view.shape[0]
    if S > 10:
        tick_positions = [0, S//4, S//2, 3*S//4, S-1]
        tick_labels = [str(pos) for pos in tick_positions]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels)
        ax.set_yticks(tick_positions)
        ax.set_yticklabels(tick_labels)

    model_name = getattr(unwrapped_model.config, "model_type", "unknown")
    title = f"{model_name.upper()} Attention Heatmap\nLayer {layer_idx}, Sample {sample_idx}"
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Key Position (j)", fontsize=12)
    ax.set_ylabel("Query Position (i)", fontsize=12)

    cbar = ax.collections[0].colorbar
    cbar.set_label('Attention Weight', rotation=270, labelpad=20)

    out_png = os.path.join(save_dir, f"{model_name}_layer_{layer_idx}_sample_{sample_idx}_attention.png")
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()


# --- Token-level experiment ---
@torch.no_grad()
def find_layer_activation_triggers_with_sentence_mapping(model, dataloader, accelerator, target_layer_idx, target_layer_type='attn',
                                                         drop_norm=True, num_samples_each_device=256, tokenizer=None):
    from .layer_drop import manual_forward_pass

    unwrapped_model = accelerator.unwrap_model(model)
    model_name = getattr(unwrapped_model.config, "model_type", "unknown_model")
    percentile_of_interest = 0.0001

    # STEP 1 - Reconstruct original text and build sentence-to-token mappings
    accelerator.print(f"🔍 Analyzing activation triggers for layer {target_layer_idx} ({target_layer_type})")
    all_token_similarities = []
    sample_similarity_data = []
    inputs, outputs, attention_mask, position_ids, cache_position = prepare_calibration_input(unwrapped_model, dataloader, num_samples_each_device)

    # Get original text from the dataloader
    original_texts = []
    original_input_ids = []
    sample_count = 0
    for batch in dataloader:
        if sample_count >= num_samples_each_device: break

        if 'input_ids' in batch and tokenizer:
            batch_input_ids = batch['input_ids']
            batch_texts = []

            for i in range(batch_input_ids.shape[0]):
                decoded_text = tokenizer.decode(batch_input_ids[i].cpu().numpy(), skip_special_tokens=True)
                batch_texts.append(decoded_text)

            for i in range(batch_input_ids.shape[0]):
                if len(original_input_ids) >= num_samples_each_device: break
                original_input_ids.append(batch_input_ids[i].cpu().numpy())

            for i in range(len(batch_texts)):
                if sample_count >= num_samples_each_device: break
                original_texts.append(batch_texts[i])
                sample_count += 1
        else:
            accelerator.print(f"⚠️  No input_ids found in batch - skipping trigger analysis")
            break

    # Create sentence mapping for each sample
    sentence_mappings = []
    accelerator.print(f"📝 Creating sentence mappings for {len(inputs)} samples...")
    for sample_idx in range(len(inputs)):
        if sample_idx < len(original_texts) and sample_idx < len(original_input_ids):
            full_text = original_texts[sample_idx]
            input_ids = original_input_ids[sample_idx]
            token_to_char_map = []
            current_char_pos = 0
            first_content_token_idx = 0

            for token_idx, token_id in enumerate(input_ids):
                token_text = tokenizer.decode([token_id], skip_special_tokens=False)
                if token_text.strip() and token_text in full_text:
                    first_content_token_idx = token_idx
                    break

            for token_idx, token_id in enumerate(input_ids):
                token_text = tokenizer.decode([token_id], skip_special_tokens=False)

                if token_idx < first_content_token_idx or token_text.strip() == "":
                    token_to_char_map.append({'token_idx': token_idx, 'token_text': token_text, 'char_start': -1, 'char_end': -1, 'is_special': True})
                    continue

                clean_token_text = token_text.replace('Ġ', ' ').replace('▁', ' ')

                char_start = full_text.find(clean_token_text, current_char_pos)
                if char_start == -1:
                    char_start = full_text.find(token_text, current_char_pos)
                    clean_token_text = token_text

                if char_start != -1:
                    char_end = char_start + len(clean_token_text)
                    token_to_char_map.append({'token_idx': token_idx,'token_text': token_text, 'char_start': char_start,
                        'char_end': char_end, 'is_special': False})
                    current_char_pos = char_end
                else:
                    token_to_char_map.append({'token_idx': token_idx, 'token_text': token_text, 'char_start': -1, 'char_end': -1,'is_special': True})

            raw_sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', full_text)
            sentences = []
            i = 0
            while i < len(raw_sentences):
                current_sentence = raw_sentences[i].strip()
                while i + 1 < len(raw_sentences) and len(current_sentence) < 50 and len(sentences) < 20:
                    i += 1
                    current_sentence += " " + raw_sentences[i].strip()
                if current_sentence:
                    sentences.append(current_sentence)
                i += 1

            sentence_token_mapping = []
            current_pos = 0
            for sent_idx, sentence in enumerate(sentences):
                sent_start = full_text.find(sentence, current_pos)
                if sent_start != -1:
                    sent_end = sent_start + len(sentence)

                    tokens_in_sentence = []
                    for token_info in token_to_char_map:
                        if token_info['is_special']:
                            continue
                        token_start = token_info['char_start']
                        token_end = token_info['char_end']

                        if (sent_start <= token_start < sent_end or
                            sent_start < token_end <= sent_end or
                            (token_start <= sent_start and token_end >= sent_end)):
                            tokens_in_sentence.append(token_info['token_idx'])

                    if tokens_in_sentence:
                        sentence_token_mapping.append({'sentence': sentence, 'sentence_idx': sent_idx, 'token_start': min(tokens_in_sentence),
                            'token_end': max(tokens_in_sentence), 'char_start': sent_start, 'char_end': sent_end, 'tokens_in_sentence': tokens_in_sentence})
                    current_pos = sent_end

            sentence_mappings.append({'sample_idx': sample_idx, 'full_text': full_text, 'sentences': sentence_token_mapping, 'token_to_char_map': token_to_char_map,
                'first_content_token_idx': first_content_token_idx})
        else:
            sentence_mappings.append(None)


    # STEP 2 - Collect all token-level similarities for threshold computation
    layers = unwrapped_model.model.layers
    current_inputs = inputs
    current_outputs = outputs
    accelerator.print(f"Processing {len(current_inputs)} samples through layer {target_layer_idx}...")
    for layer_idx in range(target_layer_idx + 1):
        layer = layers[layer_idx]
        if layer_idx == target_layer_idx:
            batch_size = 16
            for batch_start in range(0, len(current_inputs), batch_size):
                batch_end = min(batch_start + batch_size, len(current_inputs))

                for sample_idx in range(batch_start, batch_end):
                    torch.cuda.empty_cache()

                    if target_layer_type == 'mlp':
                        module_pre_norm = layer.post_attention_layernorm
                        module = layer.mlp
                    elif target_layer_type == 'attn':
                        module_pre_norm = layer.input_layernorm
                        module = layer.self_attn
                    else:
                        raise ValueError(f"Unsupported target_layer_type: {target_layer_type}")

                    wrapped_module_pre_norm, wrapped_module = create_recording_wrappers(module_pre_norm, module, drop_norm)

                    def record_module_pre_norm_states_hook(_, input, output):
                        wrapped_module_pre_norm.record(input[0].data, output[0].data)

                    if target_layer_type == 'mlp':
                        def record_module_states_hook(_, input, output):
                            wrapped_module.record(input[0].data, output[0].data)
                    elif target_layer_type == 'attn':
                        def record_module_states_hook(_, input, output):
                            wrapped_module.record(None, output[0].data)

                    handles = []
                    handles.append(module_pre_norm.register_forward_hook(record_module_pre_norm_states_hook))
                    handles.append(module.register_forward_hook(record_module_states_hook))

                    single_input = [current_inputs[sample_idx]]
                    single_output = [torch.zeros_like(current_inputs[sample_idx])]
                    single_attention_mask = [attention_mask[sample_idx]] if attention_mask else [None]
                    single_position_ids = [position_ids[sample_idx]] if position_ids else [None]
                    single_cache_position = [cache_position[sample_idx]] if cache_position else [None]

                    manual_forward_pass(unwrapped_model, single_input, single_output, single_attention_mask,
                                        single_position_ids, single_cache_position, layer, 1)

                    for handle in handles: handle.remove()

                    cos_sim_mean, token_similarities_tensor = calculate_layer_sim_advanced(wrapped_module_pre_norm,
                                                                                          wrapped_module, drop_norm,
                                                                                          accelerator.device, accelerator)

                    all_token_similarities.append(token_similarities_tensor)
                    sample_similarity_data.append({'sample_idx': sample_idx, 'mean_similarity': cos_sim_mean.item(),
                                                   'token_similarities': token_similarities_tensor.cpu().numpy().tolist(),
                                                   'sentence_mapping': sentence_mappings[sample_idx]})
                    current_outputs[sample_idx] = single_output[0]

                    del wrapped_module_pre_norm, wrapped_module, single_input, single_output
                    torch.cuda.empty_cache()

                torch.cuda.empty_cache()
        else:
            manual_forward_pass(unwrapped_model, current_inputs, current_outputs, attention_mask, position_ids,
                                cache_position, layer, len(current_inputs))

        current_inputs, current_outputs = current_outputs, current_inputs


    # STEP 3 - Compute layer statistics and threshold
    accelerator.print(f"📊 Computing layer statistics and threshold...")
    all_similarities_tensor = torch.cat(all_token_similarities, dim=0)
    similarity_mean = all_similarities_tensor.mean().item()
    similarity_percentile = torch.quantile(all_similarities_tensor, percentile_of_interest).item()

    threshold = similarity_percentile

    accelerator.print(f"📈 Layer {target_layer_idx} ({target_layer_type}) Statistics:")
    accelerator.print(f"   • Total tokens analyzed: {all_similarities_tensor.shape[0]:,}")
    accelerator.print(f"   • Mean similarity: {similarity_mean:.6f}")
    accelerator.print(f"   • Lower-tail (0.01%): {similarity_percentile:.6f}")
    accelerator.print(f"   • Threshold: {threshold:.6f}")

    # Find top 10 lowest similarities
    all_similarities_flat = all_similarities_tensor.flatten()
    top_10_values, top_10_indices = torch.topk(all_similarities_flat, k=10, largest=False)
    accelerator.print(f"\n TOP 10 TOKENS WITH LOWEST SIMILARITIES:")
    accelerator.print(f"   {'Rank':<4} {'Sample':<6} {'Token':<6} {'Similarity':<12} {'Token Text'}")
    accelerator.print(f"   {'-' * 4} {'-' * 6} {'-' * 6} {'-' * 12} {'-' * 20}")

    tokens_per_sample = 2048
    for rank, (global_idx, similarity) in enumerate(zip(top_10_indices, top_10_values), 1):
        sample_idx = global_idx.item() // tokens_per_sample
        token_idx = global_idx.item() % tokens_per_sample
        token_text = f"[token_{token_idx}]"
        if tokenizer and sample_idx < len(original_input_ids) and original_input_ids[sample_idx] is not None:
            sample_input_ids = original_input_ids[sample_idx]
            if token_idx < len(sample_input_ids):
                token_id = sample_input_ids[token_idx]
                token_text = tokenizer.decode([token_id], skip_special_tokens=False)
                if token_text.startswith('Ġ'):
                    token_text = ' ' + token_text[1:]
                elif token_text.startswith('▁'):
                    token_text = ' ' + token_text[1:]
        token_text_display = token_text[:20] + "..." if len(token_text) > 20 else token_text
        accelerator.print(f"   {rank:<4} {sample_idx:<6} {token_idx:<6} {similarity.item():<12.6f} '{token_text_display}'")


    # STEP 4 - Analyze samples using computed threshold
    accelerator.print(f"\n🎯 Analyzing activation triggers (threshold: {threshold:.6f})...")
    activation_triggers = []
    for sample_data in sample_similarity_data:
        sample_idx = sample_data['sample_idx']
        mean_similarity = sample_data['mean_similarity']
        token_similarities = sample_data['token_similarities']
        sample_mapping = sample_data['sentence_mapping']

        below_threshold_tokens = []
        for token_idx, token_sim in enumerate(token_similarities):
            if token_sim < threshold:
                below_threshold_tokens.append({
                    'token_idx': token_idx,
                    'similarity': token_sim
                })

        if len(below_threshold_tokens) > 0:
            if sample_mapping:
                token_details = []
                if tokenizer and sample_idx < len(original_input_ids) and original_input_ids[sample_idx] is not None:
                    sample_input_ids = original_input_ids[sample_idx]

                    for token_info in below_threshold_tokens:
                        token_idx = token_info['token_idx']
                        similarity = token_info['similarity']
                        if token_idx < len(sample_input_ids):
                            token_id = sample_input_ids[token_idx]
                            token_text = tokenizer.decode([token_id], skip_special_tokens=False)
                            if token_text.startswith('Ġ'):
                                token_text = ' ' + token_text[1:]
                            elif token_text.startswith('▁'):
                                token_text = ' ' + token_text[1:]
                        else:
                            token_text = f"[token_{token_idx}]"

                        token_details.append({'token_idx': token_idx, 'token_text': token_text, 'similarity': similarity})
                else:
                    for token_info in below_threshold_tokens:
                        token_details.append({'token_idx': token_info['token_idx'], 'token_text': f"[token_{token_info['token_idx']}]",
                                              'similarity': token_info['similarity']})

                trigger_info = {'sample_idx': sample_idx, 'mean_similarity': mean_similarity, 'layer_idx': target_layer_idx,
                    'layer_type': target_layer_type, 'full_text': sample_mapping['full_text'], 'sentences': sample_mapping['sentences'],
                    'below_threshold_tokens': token_details, 'num_tokens_below_threshold': len(below_threshold_tokens),
                    'total_tokens': len(token_similarities), 'threshold_used': threshold
                }

                activation_triggers.append(trigger_info)
                accelerator.print(f"🎯 TRIGGER FOUND - Sample {sample_idx}:")
                accelerator.print(f"   • Mean similarity: {mean_similarity:.4f}")
                accelerator.print(f"   • Tokens below threshold: {len(below_threshold_tokens)}/{len(token_similarities)}")
                accelerator.print(f"   • Sentences: {len(sample_mapping['sentences'])}")

                if len(token_details) > 0:
                    accelerator.print(f"   • Example tokens:")
                    for i, token_detail in enumerate(token_details[:3]):
                        accelerator.print(f"     - Token {token_detail['token_idx']}: '{token_detail['token_text']}' ({token_detail['similarity']:.4f})")
                    if len(token_details) > 3:
                        accelerator.print(f"     - ... and {len(token_details) - 3} more tokens")

    local_trigger_count = len(activation_triggers)
    total_triggers_tensor = accelerator.reduce(torch.tensor(local_trigger_count, device=accelerator.device), reduction="sum")
    total_triggers_found = total_triggers_tensor.item()
    total_samples_processed = len(inputs) * accelerator.num_processes

    accelerator.print(f"\n📋 SUMMARY:")
    accelerator.print(f"   • Total samples processed: {total_samples_processed}")
    accelerator.print(f"   • Activation triggers found: {total_triggers_found}")
    accelerator.print(f"   • Trigger rate: {(total_triggers_found / total_samples_processed) * 100:.1f}%")

    base_filename = f"token_level_triggers_{model_name}_layer_drop_{target_layer_type}_layer{target_layer_idx}"
    temp_filepath = f"{base_filename}_gpu{accelerator.process_index}.txt"
    final_filepath = f"{base_filename}.txt"
    with open(temp_filepath, 'w', encoding='utf-8') as f:
        f.write(f"=== GPU {accelerator.process_index} TRIGGERS ({len(activation_triggers)} triggers) ===\n\n")
        for i, trigger in enumerate(activation_triggers):
            f.write(f"TRIGGER #{i + 1} (GPU {accelerator.process_index})\n")
            f.write(f"-" * 40 + "\n")
            f.write(f"Sample Index: {trigger['sample_idx']}\n")
            f.write(f"Mean Similarity Score: {trigger['mean_similarity']:.6f}\n")
            f.write(f"Threshold Used: {trigger.get('threshold_used', threshold):.6f}\n")
            f.write(
                f"Tokens Below Threshold: {trigger.get('num_tokens_below_threshold', 0)}/{trigger.get('total_tokens', 'N/A')}\n")
            f.write(f"Number of Sentences: {len(trigger['sentences'])}\n\n")

            if 'below_threshold_tokens' in trigger and len(trigger['below_threshold_tokens']) > 0:
                f.write(f"TOKENS BELOW THRESHOLD:\n")
                f.write(f"-" * 25 + "\n")
                for j, token_detail in enumerate(trigger['below_threshold_tokens']):
                    f.write(
                        f"  {j + 1:3d}. Token {token_detail['token_idx']:4d}: '{token_detail['token_text']}' (similarity: {token_detail['similarity']:.6f})\n")
                f.write(f"\n")
            else:
                f.write(f"No token details available for this trigger.\n\n")

            f.write(f"SENTENCE BREAKDOWN:\n")
            f.write(f"-" * 20 + "\n")
            for j, sent_info in enumerate(trigger['sentences']):
                f.write(f"  {j + 1}. \"{sent_info['sentence']}\"\n")
                f.write(f"     (Tokens {sent_info['token_start']}-{sent_info['token_end']})\n")
                f.write(f"\n")

            f.write(f"\n" + "=" * 60 + "\n\n")

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        with open(final_filepath, 'w', encoding='utf-8') as final_file:
            final_file.write(f"Layer Activation Triggers Analysis Report (Token-Level Analysis)\n")
            final_file.write(f"=" * 60 + "\n")
            final_file.write(f"Model: {model_name}\n")
            final_file.write(f"Target Layer: {target_layer_idx}\n")
            final_file.write(f"Layer Type: {target_layer_type}\n")
            final_file.write(f"Computed Adaptive Threshold: {threshold:.6f}\n")
            final_file.write(f"Similarity Mean: {similarity_mean:.6f}\n")
            final_file.write(f"Similarity Lower-tail (0.01%): {similarity_percentile:.6f}\n")
            final_file.write(f"Total Samples Processed: {total_samples_processed}\n")
            final_file.write(f"Activation Triggers Found: {total_triggers_found}\n")
            final_file.write(f"Drop Norm: {drop_norm}\n")
            final_file.write(f"Number of GPUs: {accelerator.num_processes}\n")
            final_file.write(f"\nMethod: Token-level similarity analysis with adaptive threshold\n")
            final_file.write(f"Rationale: Detects individual tokens where layer works harder than computed threshold\n")
            final_file.write(f"Detection: Samples with ANY tokens below threshold are flagged as triggers\n")
            final_file.write(f"Threshold Formula: similarity_percentile_001 (0.01% quantile)\n")
            final_file.write(f"\n" + "=" * 60 + "\n\n")

            if total_triggers_found == 0:
                final_file.write("No activation triggers found across all GPUs.\n")
            else:
                final_file.write(f"Showing detailed analysis for all {total_triggers_found} triggers from all GPUs:\n\n")

                for gpu_id in range(accelerator.num_processes):
                    gpu_temp_file = f"{base_filename}_gpu{gpu_id}.txt"
                    if os.path.exists(gpu_temp_file):
                        with open(gpu_temp_file, 'r', encoding='utf-8') as gpu_file:
                            final_file.write(gpu_file.read())
                            final_file.write("\n")
                        os.remove(gpu_temp_file)

        accelerator.print(f"📄 Results saved to: {final_filepath}")
        accelerator.print(f"✅ Analysis complete: {total_triggers_found} triggers found across {accelerator.num_processes} GPUs")
        accelerator.print(f"=" * 35 + "\n")

    return activation_triggers
