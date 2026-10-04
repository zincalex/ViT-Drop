import argparse
import json
import shutil
import sys
import time
import gc
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from accelerate import Accelerator

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "model-selection"))

from model_loader import (
    config_exists, setup_model_dir, swap_config,
    load_model_and_processor, cache_base_weights,
)
from evaluator import prepare_model, train_head, evaluate

from loader import create_dataloader, set_seed


class _FlattenTo2D(nn.Module):

    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def forward(self, x):
        if x.dim() <= 3:
            return self.inner(x)
        lead = x.shape[:-1]
        y = self.inner(x.reshape(-1, x.shape[-1]))
        return y.reshape(*lead, y.shape[-1])


def quantize_linears(model, mode, compute_dtype, skip=("classifier",)):
    import bitsandbytes as bnb

    replaced = 0
    for full_name, module in list(model.named_modules()):
        if isinstance(module, _FlattenTo2D):
            continue
        for child_name, child in list(module.named_children()):
            qualified = f"{full_name}.{child_name}" if full_name else child_name
            if not isinstance(child, nn.Linear) or any(s in qualified for s in skip):
                continue
            if mode == "int8":
                new = bnb.nn.Linear8bitLt(
                    child.in_features, child.out_features,
                    bias=child.bias is not None,
                    has_fp16_weights=False, threshold=6.0,
                )
                new.weight = bnb.nn.Int8Params(
                    child.weight.data.contiguous().to(compute_dtype),
                    requires_grad=False, has_fp16_weights=False,
                )
            elif mode == "int4":
                new = bnb.nn.Linear4bit(
                    child.in_features, child.out_features,
                    bias=child.bias is not None,
                    compute_dtype=compute_dtype, quant_type="nf4",
                )
                new.weight = bnb.nn.Params4bit(
                    child.weight.data.contiguous().to(compute_dtype),
                    requires_grad=False, quant_type="nf4",
                )
            else:
                raise ValueError(f"Unknown quantization mode: {mode}")
            if child.bias is not None:
                new.bias = nn.Parameter(child.bias.data.to(compute_dtype), requires_grad=False)
            setattr(module, child_name, _FlattenTo2D(new))
            replaced += 1
    return replaced


def weight_megabytes(model):
    total = sum(p.numel() * p.element_size() for p in model.parameters())
    total += sum(b.numel() * b.element_size() for b in model.buffers())
    return total / (1024 ** 2)


class CastInputs(nn.Module):

    def __init__(self, model, dtype):
        super().__init__()
        self.model = model
        self.cast_dtype = dtype

    def forward(self, pixel_values, **kwargs):
        return self.model(pixel_values.to(self.cast_dtype), **kwargs)


def benchmark_arm(model, image_size, batch_sizes, dtype, device,
                  num_iterations=50, num_runs=3):
    results = []
    model.eval()
    for bs in batch_sizes:
        try:
            dummy = torch.randn(bs, 3, image_size, image_size, device=device, dtype=dtype)
            with torch.no_grad():
                for _ in range(5):
                    _ = model(dummy)
            torch.cuda.synchronize()
            throughputs, latencies, memories = [], [], []
            for _ in range(num_runs):
                torch.cuda.reset_peak_memory_stats()
                times = []
                with torch.no_grad():
                    for _ in range(num_iterations):
                        torch.cuda.synchronize()
                        t0 = time.time()
                        _ = model(dummy)
                        torch.cuda.synchronize()
                        times.append(time.time() - t0)
                mean_t = float(np.mean(times))
                throughputs.append(bs / mean_t)
                latencies.append(mean_t * 1000)
                memories.append(torch.cuda.max_memory_allocated() / (1024 ** 3))
            results.append({
                "batch_size": bs,
                "throughput_imgs_s": float(np.mean(throughputs)),
                "throughput_std": float(np.std(throughputs)),
                "latency_ms": float(np.mean(latencies)),
                "peak_mem_gb": float(np.mean(memories)),
            })
            del dummy
        except RuntimeError as ex:
            if "out of memory" in str(ex).lower():
                torch.cuda.empty_cache()
                results.append({"batch_size": bs, "oom": True})
            else:
                raise
        torch.cuda.empty_cache()
    return results


def get_model_dir(arch):
    return f"./{arch}_model_quant"


def ensure_private_model_dir(arch):
    src = Path(f"./{arch}_model")
    dst = Path(get_model_dir(arch))
    if not dst.exists():
        print(f"[INFO] Creating private working copy {dst} from {src}")
        shutil.copytree(src, dst)
    return dst


def train_head_fp32(cfg, arch, drop_n, accelerator, cached_weights, dataset_dir):
    model, processor = load_model_and_processor(get_model_dir(arch), cached_state_dict=cached_weights)
    model = prepare_model(model, accelerator, dataset_dir)
    model = accelerator.prepare(model)

    train_loader = create_dataloader(
        dataset_name=cfg.dataset, base_dir=cfg.dataset_base_dir, processor=processor,
        batch_size=cfg.batch_size, num_workers=cfg.num_workers, split=cfg.train_split,
    )
    train_loader = accelerator.prepare(train_loader)
    model = train_head(model, train_loader, accelerator, cfg.epochs, cfg.lr, cfg.weight_decay)

    unwrapped = accelerator.unwrap_model(model)
    head_state = {k: v.detach().cpu().clone() for k, v in unwrapped.state_dict().items()
                  if "classifier" in k}
    accelerator.free_memory()
    del model, train_loader
    torch.cuda.empty_cache()
    return head_state, processor


def build_arm(precision, arch, accelerator, cached_weights, head_state,
              dataset_dir, compute_dtype):
    model, processor = load_model_and_processor(get_model_dir(arch), cached_state_dict=cached_weights)
    model = prepare_model(model, accelerator, dataset_dir)
    model.load_state_dict(head_state, strict=False)

    quant_seconds = 0.0
    n_quantized = 0
    if precision == "fp32":
        run_dtype = torch.float32
    elif precision == "bf16":
        model = model.to(torch.bfloat16)
        run_dtype = torch.bfloat16
    elif precision in ("int8", "int4"):
        model = model.to(compute_dtype)
        t0 = time.time()
        n_quantized = quantize_linears(model, precision, compute_dtype)
        model = model.to(accelerator.device)
        torch.cuda.synchronize()
        quant_seconds = time.time() - t0
        run_dtype = compute_dtype
    else:
        raise ValueError(f"Unknown precision: {precision}")

    model = model.to(accelerator.device)
    model.eval()
    return model, processor, run_dtype, quant_seconds, n_quantized


def run_config(cfg, arch, method, drop_n, accelerator, cached_weights, out):
    label = f"{arch}/{method}/drop{drop_n}"
    if not config_exists(cfg.results_prune_dir, arch, method, drop_n):
        accelerator.print(f"[SKIP] No config for {label}")
        return

    dataset_dir = Path(cfg.dataset_base_dir) / cfg.dataset
    if accelerator.is_main_process:
        swap_config(get_model_dir(arch), cfg.results_prune_dir, arch, method, drop_n)

    accelerator.print(f"\n=== {label}: training head ({cfg.epochs} epochs, FP32) ===")
    set_seed(42)
    head_state, _ = train_head_fp32(cfg, arch, drop_n, accelerator, cached_weights, dataset_dir)

    compute_dtype = torch.float16 if cfg.quant_compute_dtype == "fp16" else torch.bfloat16
    image_size = None
    rows = []
    for precision in cfg.precisions:
        accelerator.print(f"--- {label} | precision={precision} ---")
        set_seed(42)
        model, processor, run_dtype, quant_s, n_q = build_arm(
            precision, arch, accelerator, cached_weights, head_state,
            dataset_dir, compute_dtype,
        )
        if image_size is None:
            image_size = getattr(model.config, "image_size", 224)
            if not isinstance(image_size, int):
                image_size = int(image_size[0]) if hasattr(image_size, "__getitem__") else 224

        wrapper = CastInputs(model, run_dtype)

        eval_loader = create_dataloader(
            dataset_name=cfg.dataset, base_dir=cfg.dataset_base_dir, processor=processor,
            batch_size=cfg.batch_size_eval, num_workers=cfg.num_workers, split=cfg.eval_split,
        )
        eval_loader = accelerator.prepare(eval_loader)
        metrics = evaluate(wrapper, eval_loader, accelerator)

        bench = benchmark_arm(
            wrapper, image_size, cfg.bench_batch_sizes, run_dtype, accelerator.device,
            num_iterations=cfg.bench_iters, num_runs=cfg.bench_runs,
        )

        row = {
            "arch": arch, "method": method, "drop_n": drop_n,
            "precision": precision,
            "weight_mb": weight_megabytes(model),
            "quantized_layers": n_q,
            "quantization_seconds": round(quant_s, 2),
            "benchmark": bench,
        }
        if accelerator.is_main_process and metrics:
            row.update({
                "accuracy": metrics["accuracy"], "macro_precision": metrics["precision"],
                "macro_recall": metrics["recall"], "macro_f1": metrics["f1"],
            })
            accelerator.print(
                f"  acc={metrics['accuracy']:.4f} f1={metrics['f1']:.4f} "
                f"weights={row['weight_mb']:.0f}MB"
            )
            for b in bench:
                if b.get("oom"):
                    accelerator.print(f"  bs={b['batch_size']}: OOM")
                else:
                    accelerator.print(
                        f"  bs={b['batch_size']}: {b['throughput_imgs_s']:.1f} imgs/s | "
                        f"{b['latency_ms']:.1f} ms | peak {b['peak_mem_gb']:.2f} GB"
                    )
        rows.append(row)

        del model, wrapper, eval_loader
        gc.collect()
        torch.cuda.empty_cache()

    if accelerator.is_main_process:
        out["results"].extend(rows)
        save_output(cfg, out)


def save_output(cfg, out):
    out_dir = Path(cfg.output_dir) / cfg.dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"quant_stack_{cfg.arch}_{cfg.method}.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[log] Results written to {path}")


def parse_args():
    p = argparse.ArgumentParser(description="Pruning + weight-only quantization stacking")
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--dataset_base_dir", type=str, default="data")
    p.add_argument("--results_prune_dir", type=str, default="results_prune")
    p.add_argument("--output_dir", type=str, default="results/quantization")
    p.add_argument("--arch", type=str, required=True)
    p.add_argument("--method", type=str, required=True)
    p.add_argument("--drop_ns", type=str, default="0",
                   help="Comma-separated drop counts, e.g. '0,16'")
    p.add_argument("--precisions", type=str, default="fp32,bf16,int8,int4")
    p.add_argument("--quant_compute_dtype", type=str, default="fp16", choices=["fp16", "bf16"])
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--weight_decay", type=float, default=0.03)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--batch_size_eval", type=int, default=32)
    p.add_argument("--train_split", type=str, default="train")
    p.add_argument("--eval_split", type=str, default="test")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--bench_batch_sizes", type=str, default="1,8,128")
    p.add_argument("--bench_iters", type=int, default=50)
    p.add_argument("--bench_runs", type=int, default=3)
    cfg = p.parse_args()
    cfg.drop_ns = [int(x) for x in cfg.drop_ns.split(",")]
    cfg.precisions = [x.strip() for x in cfg.precisions.split(",")]
    cfg.bench_batch_sizes = [int(x) for x in cfg.bench_batch_sizes.split(",")]
    return cfg


def main():
    cfg = parse_args()
    set_seed(42)
    accelerator = Accelerator()
    if accelerator.num_processes > 1:
        raise RuntimeError("quant_stack.py is single-GPU; launch with plain `python`.")

    accelerator.print("=" * 80)
    accelerator.print("PRUNING + QUANTIZATION STACKING EXPERIMENT")
    accelerator.print(f"Dataset: {cfg.dataset} | {cfg.arch}/{cfg.method} | drops={cfg.drop_ns}")
    accelerator.print(f"Precision arms: {cfg.precisions}")
    accelerator.print("=" * 80)

    if not Path(f"./{cfg.arch}_model").exists():
        raise FileNotFoundError(f"./{cfg.arch}_model not found")
    model_dir = str(ensure_private_model_dir(cfg.arch))

    setup_model_dir(model_dir, cfg.results_prune_dir, cfg.arch, cfg.method, 0)
    swap_config(model_dir, cfg.results_prune_dir, cfg.arch, cfg.method, 0)
    cached_weights = cache_base_weights(model_dir)

    out = {
        "dataset": cfg.dataset, "arch": cfg.arch, "method": cfg.method,
        "epochs": cfg.epochs, "train_split": cfg.train_split, "eval_split": cfg.eval_split,
        "quant_compute_dtype": cfg.quant_compute_dtype,
        "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "results": [],
    }

    for drop_n in cfg.drop_ns:
        run_config(cfg, cfg.arch, cfg.method, drop_n, accelerator, cached_weights, out)

    accelerator.print("\n[DONE] All configurations evaluated.")


if __name__ == "__main__":
    main()
