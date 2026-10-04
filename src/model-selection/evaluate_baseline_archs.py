import gc
import csv
import sys
import torch
from pathlib import Path

from accelerate import Accelerator

sys.path.insert(0, str(Path(__file__).resolve().parent))

from model_loader import (
    config_exists, setup_model_dir, swap_config,
    load_model_and_processor,
)
from evaluator import deep_finetune_and_evaluate
from loader import set_seed


ARCH_MAX_LAYERS = {
    "dinov2": 40,
    "dinov3_vit": 24,
    "swinv2": 24,
    "vit": 12,
}

# All supported pruning methods. At drop=0 every method yields the same
# (unpruned) model, so we just need one method whose drop=0 config exists.
ALL_PRUNE_METHODS = ["block_drop", "layer_drop_attn", "layer_drop_mlp", "layer_drop_all"]

# Default dataset list (override with --datasets).
DATASETS = [
    "cifar10", "lar", "zoolake", "CrossD", "LCZ42", "InfLarynge",
    "Bark", "Pest", "ColorBG", "Kaggle38", "WHOI22", "ZooScan20",
]

# Default architectures to evaluate (override with --archs).
ARCHS = ["dinov2", "dinov3_vit", "swinv2", "vit"]


def get_model_dir(arch):
    return f"./{arch}_model"


def find_baseline_method(results_prune_dir, arch):
    """Return the first pruning method that has a drop=0 config for this arch."""
    for method in ALL_PRUNE_METHODS:
        if config_exists(results_prune_dir, arch, method, 0):
            return method
    return None


def dataset_ready(dataset, base_dir):
    """True only if train/val/test H5 splits all exist for the dataset."""
    d = Path(base_dir) / dataset
    return all((d / f"{s}.h5").exists() for s in ("train", "val", "test"))


def evaluate_baseline(arch, dataset, cfg, accelerator):
    """Fine-tune the drop=0 head and evaluate on test.

    Trains the head on train+val for 20 epochs (winner deep fine-tune protocol)
    from a startup state aligned with benchmark_vm_eval.sh.
    """
    model_dir = get_model_dir(arch)

    if not Path(model_dir).exists():
        accelerator.print(f"  [SKIP] {model_dir} not found")
        return None

    method = find_baseline_method(cfg["results_prune_dir"], arch)
    if method is None:
        accelerator.print(f"  [SKIP] No drop=0 config for {arch}")
        return None

    # Copy the custom .py + config files for the drop=0 (unpruned) variant.
    if accelerator.is_main_process:
        setup_model_dir(model_dir, cfg["results_prune_dir"], arch, method, 0)
    accelerator.wait_for_everyone()

    # The drop=0 config IS the baseline, so loading under it gives the full,
    # unpruned model.
    if accelerator.is_main_process:
        swap_config(model_dir, cfg["results_prune_dir"], arch, method, 0)
    accelerator.wait_for_everyone()

    # Load straight from the checkpoint (from_pretrained) rather than via
    # cache_base_weights + from_config. The cached path random-initializes every
    # parameter and then overwrites only those whose name and shape match, so an
    # unmatched tensor would silently keep random values. benchmark_vm_eval.sh
    # uses from_pretrained, so this keeps the two startup states identical.
    # Re-seed first so head construction is unaffected by however much RNG the
    # weight loading consumed.
    set_seed(42)
    model, processor = load_model_and_processor(model_dir)
    torch.cuda.empty_cache()

    # Per-arch output subfolder so the per-dataset best_head/logits files
    # written by deep_finetune_and_evaluate don't clobber each other.
    output_dir = str(Path(cfg["output_dir"]) / arch)
    test_metrics = deep_finetune_and_evaluate(
        model=model, processor=processor, accelerator=accelerator,
        dataset_name=dataset, dataset_base_dir=cfg["dataset_base_dir"],
        epochs=cfg["epochs"], lr=cfg["lr"],
        weight_decay=cfg["weight_decay"], batch_size=cfg["batch_size"],
        batch_size_eval=cfg["batch_size_eval"],
        num_workers=cfg["num_workers"],
        warmup_ratio=cfg["warmup_ratio"],
        output_dir=output_dir,
    )

    del model
    gc.collect()
    torch.cuda.empty_cache()
    accelerator.wait_for_everyone()

    return test_metrics


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Fine-tune + evaluate the unpruned baseline per architecture")
    parser.add_argument("--dataset_base_dir", type=str, default="data")
    parser.add_argument("--results_prune_dir", type=str, default="results_prune")
    parser.add_argument("--output_csv", type=str, default="results/selection/all_archs_baseline_test.csv")
    parser.add_argument("--output_dir", type=str, default="results/selection/all_archs_baseline_eval")
    # Deep fine-tune protocol: train+val, 20 epochs, cosine schedule with warmup.
    # lr / weight_decay / batch_size match benchmark_vm_eval.sh.
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.03)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--batch_size_eval", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--datasets", type=str, nargs="+", default=None,
                        help="Specific datasets to evaluate (default: all with H5 splits present)")
    parser.add_argument("--archs", type=str, nargs="+", default=None,
                        choices=list(ARCH_MAX_LAYERS.keys()),
                        help="Specific architectures to evaluate (default: all four)")
    args = parser.parse_args()

    archs = args.archs if args.archs else ARCHS

    # Same first action as benchmark_vision.py: seed everything before any
    # model or dataloader is constructed.
    set_seed(42)
    accelerator = Accelerator()

    cfg = {
        "dataset_base_dir": args.dataset_base_dir,
        "results_prune_dir": args.results_prune_dir,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "batch_size_eval": args.batch_size_eval,
        "num_workers": args.num_workers,
        "warmup_ratio": args.warmup_ratio,
        "output_dir": args.output_dir,
    }

    # The baseline is independent of the selection logs, so we only require the
    # dataset's H5 splits to be present.
    candidates = args.datasets if args.datasets else DATASETS
    datasets = [d for d in candidates if dataset_ready(d, args.dataset_base_dir)]

    accelerator.print("=" * 80)
    accelerator.print("EVALUATE UNPRUNED BASELINE PER ARCHITECTURE")
    accelerator.print("=" * 80)
    accelerator.print(f"Datasets: {datasets}")
    accelerator.print(f"Architectures: {archs}")
    accelerator.print("Protocol: deep fine-tune (train+val, cosine schedule with warmup)")
    accelerator.print(f"  epochs={cfg['epochs']}, lr={cfg['lr']}, wd={cfg['weight_decay']}, "
                      f"bs={cfg['batch_size']}, bs_eval={cfg['batch_size_eval']}, "
                      f"warmup={cfg['warmup_ratio']}")
    accelerator.print("  startup aligned with benchmark_vm_eval.sh: seed=42, "
                      "from_pretrained(float32, low_cpu_mem_usage=True), "
                      "head seeded with torch.manual_seed(42)")
    accelerator.print("=" * 80)

    all_results = []

    for dataset in datasets:
        accelerator.print(f"\n{'=' * 60}")
        accelerator.print(f"Dataset: {dataset}")
        accelerator.print(f"{'=' * 60}")

        row = {"dataset": dataset}

        for arch in archs:
            accelerator.print(f"\n--- {dataset} / {arch} / baseline (drop0) ---")

            metrics = evaluate_baseline(arch, dataset, cfg, accelerator)

            if metrics and accelerator.is_main_process:
                acc = metrics["accuracy"]
                accelerator.print(f"  Test acc: {acc:.4f}")
                row[f"{arch}_acc"] = f"{acc * 100:.2f}"
                row[f"{arch}_method"] = "baseline"
                row[f"{arch}_drop"] = "0"
            else:
                row[f"{arch}_acc"] = ""
                row[f"{arch}_method"] = ""
                row[f"{arch}_drop"] = ""

        all_results.append(row)

        # Save CSV incrementally
        if accelerator.is_main_process:
            output_path = Path(args.output_csv)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fieldnames = ["dataset"]
            for arch in archs:
                fieldnames.extend([f"{arch}_acc", f"{arch}_method", f"{arch}_drop"])
            with open(output_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(all_results)
            accelerator.print(f"\n  CSV saved to {output_path}")

    accelerator.print("\n" + "=" * 80)
    accelerator.print("DONE")
    accelerator.print("=" * 80)


if __name__ == "__main__":
    main()
