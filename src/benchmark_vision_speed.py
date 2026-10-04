import os
import time
import json
import torch
import argparse
import numpy as np
import pandas as pd
import torch.nn as nn
from transformers import AutoImageProcessor, AutoModelForImageClassification, AutoModel, AutoConfig
from transformers.modeling_outputs import ImageClassifierOutput


def count_model_flops(model, image_size, model_path, batch_size=1):
    config = model.config
    drop_attn_list = []
    drop_mlp_list = []
    config_file_path = os.path.join(model_path, "config.json")
    if os.path.exists(config_file_path):
        with open(config_file_path, 'r') as f:
            config_data = json.load(f)
            drop_attn_list = config_data.get('drop_attn_list', [])
            drop_mlp_list = config_data.get('drop_mlp_list', [])

    model_type = getattr(config, 'model_type', '').lower()
    if 'swin' in model_type:
        return count_hierarchical_transformer_flops(config, image_size, drop_attn_list, drop_mlp_list, batch_size)
    else:
        return count_vit_flops(config, image_size, drop_attn_list, drop_mlp_list, batch_size)


def count_hierarchical_transformer_flops(config, image_size, drop_attn_list, drop_mlp_list, batch_size=1):
    depths = getattr(config, 'depths', [2, 2, 6, 2])
    embed_dim = getattr(config, 'embed_dim', 96)
    mlp_ratio = getattr(config, 'mlp_ratio', 4.0)
    window_size = getattr(config, 'window_size', 7)
    patch_size = getattr(config, 'patch_size', 4)
    
    total_flops = 0
    H, W = image_size, image_size
    
    # Patch embedding
    patch_embed_flops = 2 * (H // patch_size) * (W // patch_size) * (patch_size * patch_size * 3) * embed_dim
    total_flops += patch_embed_flops * batch_size
    
    # Calculate FLOPs for each stage
    layer_idx = 0
    for stage_idx, depth in enumerate(depths):
        stage_embed_dim = embed_dim * (2 ** stage_idx)
        H_stage = H // (patch_size * (2 ** stage_idx))
        W_stage = W // (patch_size * (2 ** stage_idx))
        num_tokens = H_stage * W_stage
        
        for _ in range(depth):
            attn_active = layer_idx not in drop_attn_list
            mlp_active = layer_idx not in drop_mlp_list
            
            if attn_active:
                # Window-based attention
                num_windows = (H_stage // window_size) * (W_stage // window_size)
                tokens_per_window = window_size * window_size
                
                # QKV + attention + output projection
                qkv_flops = 2 * tokens_per_window * stage_embed_dim * (3 * stage_embed_dim) * num_windows
                qk_flops = 2 * tokens_per_window * tokens_per_window * stage_embed_dim * num_windows
                sv_flops = 2 * tokens_per_window * tokens_per_window * stage_embed_dim * num_windows
                o_proj_flops = 2 * tokens_per_window * stage_embed_dim * stage_embed_dim * num_windows
                
                total_flops += (qkv_flops + qk_flops + sv_flops + o_proj_flops) * batch_size
            
            if mlp_active:
                mlp_dim = int(stage_embed_dim * mlp_ratio)
                mlp_flops = 2 * num_tokens * stage_embed_dim * mlp_dim
                mlp_flops += 2 * num_tokens * mlp_dim * stage_embed_dim
                total_flops += mlp_flops * batch_size
            
            layer_idx += 1
    
    # Classification head
    final_embed_dim = embed_dim * (2 ** (len(depths) - 1))
    head_flops = 2 * final_embed_dim * getattr(config, 'num_labels', 1000) * batch_size
    total_flops += head_flops
    
    return total_flops / 1e9


def count_vit_flops(config, image_size, drop_attn_list, drop_mlp_list, batch_size=1):
    num_layers = getattr(config, 'num_hidden_layers', 0)
    hidden_size = getattr(config, 'hidden_size', 0)
    mlp_ratio = getattr(config, 'mlp_ratio', 4)
    patch_size = getattr(config, 'patch_size', 16)
    
    num_patches = (image_size // patch_size) ** 2
    seq_len = num_patches + 1  # +1 for CLS token
    
    # Attention FLOPs per layer
    qkv_flops = 3 * 2 * seq_len * hidden_size * hidden_size
    o_proj_flops = 2 * seq_len * hidden_size * hidden_size
    qk_flops = 2 * seq_len * seq_len * hidden_size
    sv_flops = 2 * seq_len * seq_len * hidden_size
    attn_flops = qkv_flops + o_proj_flops + qk_flops + sv_flops
    
    # MLP FLOPs per layer
    mlp_dim = int(hidden_size * mlp_ratio)
    mlp_flops = 2 * seq_len * hidden_size * mlp_dim
    mlp_flops += 2 * seq_len * mlp_dim * hidden_size
    
    # Count active layers
    num_active_attn = num_layers - len(drop_attn_list)
    num_active_mlp = num_layers - len(drop_mlp_list)
    
    # Total layer FLOPs
    total_flops = (num_active_attn * attn_flops + num_active_mlp * mlp_flops) * batch_size
    
    # Embedding and head
    embedding_flops = 2 * num_patches * (patch_size * patch_size * 3) * hidden_size * batch_size
    head_flops = 2 * seq_len * hidden_size * getattr(config, 'num_labels', 1000) * batch_size
    total_flops += embedding_flops + head_flops
    
    return total_flops / 1e9


class PowerMonitor:
    """Samples GPU power draw via NVML in a background thread.

    Degrades gracefully: if pynvml is missing or NVML fails, power metrics are
    simply omitted. NVML indexes physical GPUs, so the first entry of
    CUDA_VISIBLE_DEVICES is used to find the device being benchmarked.
    """

    def __init__(self, interval=0.05):
        self.interval = interval
        self.samples = []
        self._stop = None
        self._thread = None
        self.mode = None
        self.gpu_index = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
        try:
            import pynvml
            pynvml.nvmlInit()
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(int(self.gpu_index))
            self.pynvml = pynvml
            self.mode = "nvml"
        except Exception:
            # Fallback: poll nvidia-smi. Heavier (one subprocess per sample),
            # so use a coarser interval.
            import subprocess
            try:
                subprocess.check_output(
                    ["nvidia-smi", "-i", self.gpu_index,
                     "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
                    timeout=5)
                self.mode = "smi"
                self.interval = max(self.interval, 0.2)
                print("[power] pynvml not available, falling back to nvidia-smi polling")
            except Exception as ex:
                print(f"[power] no NVML and no nvidia-smi, power metrics disabled: {ex}")
        self.available = self.mode is not None

    def _sample(self):
        if self.mode == "nvml":
            return self.pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0  # mW -> W
        import subprocess
        out = subprocess.check_output(
            ["nvidia-smi", "-i", self.gpu_index,
             "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
            timeout=5)
        return float(out.decode().strip().splitlines()[0])

    def measure_idle(self, duration=2.0):
        """Average power over `duration` seconds with the GPU idle."""
        if not self.available:
            return None
        time.sleep(0.5)
        samples = []
        end = time.time() + duration
        while time.time() < end:
            samples.append(self._sample())
            time.sleep(self.interval)
        return float(np.mean(samples)) if samples else None

    def start(self):
        if not self.available:
            return
        import threading
        self.samples = []
        self._stop = threading.Event()

        def loop():
            while not self._stop.is_set():
                try:
                    self.samples.append(self._sample())
                except Exception:
                    break
                time.sleep(self.interval)

        self._thread = threading.Thread(target=loop, daemon=True)
        self._thread.start()

    def stop(self):
        if not self.available or self._stop is None:
            return None
        self._stop.set()
        self._thread.join(timeout=2.0)
        return float(np.mean(self.samples)) if self.samples else None


def warmup(model, device):
    warm_up = torch.randn((4096, 4096), device=device, dtype=torch.float16)
    torch.mm(warm_up, warm_up)
    torch.cuda.synchronize()


def benchmark_inference(model, pixel_values, num_iterations=100):
    model.eval()
    times = []
    with torch.no_grad():
        for _ in range(num_iterations):
            torch.cuda.synchronize()
            start = time.time()
            _ = model(pixel_values)
            torch.cuda.synchronize()
            times.append(time.time() - start)
    return times


def run_benchmark(model, processor, batch_size, image_size, num_iterations, device, num_runs=10,
                  power_monitor=None, idle_power=None):
    model_dtype = next(model.parameters()).dtype
    dummy_images = torch.randn(batch_size, 3, image_size, image_size, device=device, dtype=model_dtype)
    pixel_values = dummy_images

    warmup(model, device)
    with torch.no_grad():
        for _ in range(5):
            _ = model(pixel_values)
    torch.cuda.synchronize()

    all_throughputs = []
    all_latencies = []
    all_memories = []
    if power_monitor is not None:
        power_monitor.start()
    for run_idx in range(num_runs):
        torch.cuda.reset_peak_memory_stats()
        try:
            times = benchmark_inference(model, pixel_values, num_iterations)
            successful = True
        except RuntimeError as ex:
            if 'out of memory' in str(ex).lower():
                torch.cuda.empty_cache()
                successful = False
                times = []
            else:
                raise
        
        if not successful:
            if power_monitor is not None:
                power_monitor.stop()
            return None

        memory_used = torch.cuda.max_memory_allocated() / (1024 ** 3)
        mean_time = np.mean(times)
        throughput = batch_size / mean_time
        all_throughputs.append(throughput)
        all_latencies.append(mean_time * 1000)
        all_memories.append(memory_used)

    avg_power = power_monitor.stop() if power_monitor is not None else None

    avg_throughput = np.mean(all_throughputs)
    std_throughput = np.std(all_throughputs)
    avg_latency = np.mean(all_latencies)
    std_latency = np.std(all_latencies)
    avg_memory = np.mean(all_memories)
    memory_pct = avg_memory / (torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)) * 100

    stats = {"Batch Size": batch_size, "Latency (ms)": avg_latency, "Latency Std (ms)": std_latency, "Throughput (imgs/s)": avg_throughput,
        "Throughput Std (imgs/s)": std_throughput, "Memory (GB)": avg_memory, "Memory (%)": memory_pct}

    if avg_power is not None:
        stats["Power (W)"] = avg_power
        # Net power (idle-subtracted, as in LPViT) and energy per image in joules:
        # W / (imgs/s) = J/img
        net_power = avg_power - idle_power if idle_power is not None else avg_power
        stats["Net Power (W)"] = net_power
        stats["Energy/Image (J)"] = net_power / avg_throughput if avg_throughput > 0 else None
    return stats


class DINOv3ForImageClassification(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.dinov3_vit = backbone
        self.config = backbone.config
        self.base_model_prefix = "dinov3_vit"
        self.classifier = nn.Linear(backbone.config.hidden_size, getattr(backbone.config, 'num_labels', 1000))

    def forward(self, pixel_values, **kwargs):
        outputs = self.dinov3_vit(pixel_values, **kwargs)
        logits = self.classifier(outputs.pooler_output)
        return ImageClassifierOutput(logits=logits)


def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    config_kwargs = {"trust_remote_code": True, "cache_dir": None}
    processor = AutoImageProcessor.from_pretrained(args.model_path, **config_kwargs)
    config = AutoConfig.from_pretrained(args.model_path, **config_kwargs)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model_type = getattr(config, "model_type", None)

    if model_type == "dinov3_vit":
        backbone = AutoModel.from_pretrained(args.model_path, config=config, torch_dtype=dtype,
            low_cpu_mem_usage=True, **config_kwargs)
        model = DINOv3ForImageClassification(backbone)
    else:
        model = AutoModelForImageClassification.from_pretrained(args.model_path, config=config, torch_dtype=dtype,
            low_cpu_mem_usage=True, ignore_mismatched_sizes=True, **config_kwargs)

    model = model.to(device)
    model = model.to(dtype)
    model.eval()

    batch_sizes = [int(b) for b in args.batch_sizes.split(",")] if args.batch_sizes else [1, 4, 8, 16, 32]
    image_sizes = [int(s) for s in args.image_sizes.split(",")] if args.image_sizes else [getattr(config, 'image_size', 224)]

    model_gflops = count_model_flops(model, image_sizes[0], args.model_path, batch_size=1)

    # Power monitoring (NVML); idle power is measured once, before any load.
    power_monitor = PowerMonitor()
    idle_power = power_monitor.measure_idle() if power_monitor.available else None
    if idle_power is not None:
        print(f"[power] Idle power: {idle_power:.1f} W")

    # Run benchmarks
    all_stats = []
    for image_size in image_sizes:
        for batch_size in batch_sizes:
            stats = run_benchmark(model, processor, batch_size, image_size, args.num_iterations, device,
                                  num_runs=args.num_runs,
                                  power_monitor=power_monitor if power_monitor.available else None,
                                  idle_power=idle_power)
            if stats is None:
                break
            all_stats.append(stats)
    
    if len(all_stats) == 0:
        print("ERROR: No successful benchmarks.")
        return
    
    df = pd.DataFrame(all_stats)
    df['FLOPs (G)'] = model_gflops

    if args.save_file:
        model_name = os.path.basename(args.save_file).replace("_speed.csv", "")
    else:
        model_name = os.path.basename(args.model_path)

    baseline_df = None
    baseline_flops = None
    if args.baseline_file and os.path.exists(args.baseline_file):
        baseline_df = pd.read_csv(args.baseline_file)
        baseline_dict = dict(zip(baseline_df['Batch Size'], baseline_df['Throughput (imgs/s)']))
        df['Speedup'] = df.apply(lambda row: row['Throughput (imgs/s)'] / baseline_dict.get(row['Batch Size'], 1.0), axis=1)

        if 'FLOPs (G)' in baseline_df.columns:
            baseline_flops = baseline_df['FLOPs (G)'].iloc[0]
            df['FLOPs Remained (%)'] = (df['FLOPs (G)'] / baseline_flops) * 100
            flops_reduction_factor = baseline_flops / df['FLOPs (G)']
            df['Conversion Efficiency'] = df['Speedup'] / flops_reduction_factor

    if args.save_file:
        if os.path.dirname(args.save_file):
            os.makedirs(os.path.dirname(args.save_file), exist_ok=True)
        df.to_csv(args.save_file, index=False)

        json_payload = {
            "model_name": model_name,
            "model_path": args.model_path,
            "gpu": torch.cuda.get_device_name(),
            "image_size": image_sizes[0],
            "flops_g": model_gflops,
            "baseline_file": args.baseline_file,
            "baseline_flops_g": baseline_flops,
            "idle_power_w": idle_power,
            "num_runs": args.num_runs,
            "num_iterations": args.num_iterations,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "results": df.to_dict(orient="records"),
        }
        json_file = args.save_file.replace(".csv", ".json")
        with open(json_file, "w") as f:
            json.dump(json_payload, f, indent=2)
        print(f"[log] Metrics JSON written to {json_file}")

    print(f"{'='*80}")
    print(f"RESULTS: {model_name}")
    print(f"Image size: {image_sizes[0]}x{image_sizes[0]}")
    print(f"FLOPs (G): {model_gflops:.2f}")
    if baseline_flops is not None:
        flops_remained_pct = (model_gflops / baseline_flops) * 100
        print(f"FLOPs Remained (%): {flops_remained_pct:.1f}%")
    print(f"{'='*80}")
    
    if baseline_df is not None:
        for _, row in df.iterrows():
            extra = ""
            if 'Conversion Efficiency' in df.columns:
                extra += f" | ConvEff: {row['Conversion Efficiency']:.3f}"
            if 'Energy/Image (J)' in df.columns and row.get('Energy/Image (J)') is not None:
                extra += f" | Power: {row['Power (W)']:.1f} W | Energy: {row['Energy/Image (J)']*1000:.1f} mJ/img"
            print(f"Batch {int(row['Batch Size']):2d}: {row['Throughput (imgs/s)']:7.1f} ± {row['Throughput Std (imgs/s)']:4.2f} imgs/s | "
                  f"Latency: {row['Latency (ms)']:6.1f} ± {row['Latency Std (ms)']:4.2f} ms | "
                  f"Memory: {row['Memory (GB)']:4.1f} GB ({row['Memory (%)']:5.1f}%) | "
                  f"Speedup: {row['Speedup']:.3f}x{extra}")

        avg_speedup = df['Speedup'].mean()
        print(f"{'-'*80}")
        print(f"Average Speedup: {avg_speedup:.3f}x")
    else:
        for _, row in df.iterrows():
            print(f"Batch {int(row['Batch Size']):2d}: {row['Throughput (imgs/s)']:7.1f} ± {row['Throughput Std (imgs/s)']:4.2f} imgs/s | "
                  f"Latency: {row['Latency (ms)']:6.1f} ± {row['Latency Std (ms)']:4.2f} ms | "
                  f"Memory: {row['Memory (GB)']:4.1f} GB ({row['Memory (%)']:5.1f}%)")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark vision model inference speed")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model")
    parser.add_argument("--batch_sizes", type=str, default=None, help="Comma-separated batch sizes")
    parser.add_argument("--image_sizes", type=str, default=None, help="Comma-separated image sizes")
    parser.add_argument("--num_iterations", type=int, default=100, help="Number of iterations per run")
    parser.add_argument("--num_runs", type=int, default=10, help="Number of runs to average for statistical significance")
    parser.add_argument("--save_file", type=str, default=None, help="Path to save results CSV")
    parser.add_argument("--baseline_file", type=str, default=None, help="Path to baseline CSV for speedup comparison")
    
    args = parser.parse_args()
    main(args)
