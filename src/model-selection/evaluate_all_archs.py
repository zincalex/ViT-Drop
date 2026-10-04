import gc
import json
import csv
import sys
import torch
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, List

from accelerate import Accelerator

sys.path.insert(0, str(Path(__file__).resolve().parent))

from model_loader import (
    config_exists, setup_model_dir, swap_config,
    load_model_and_processor, cache_base_weights,
)
from evaluator import deep_finetune_and_evaluate


ARCH_MAX_LAYERS = {
    "dinov2": 40,
    "dinov3_vit": 24,
    "swinv2": 24,
    "vit": 12,
}

DATASETS = [
    "ColorBG",
]


def get_model_dir(arch):
    return f"./{arch}_model"


def get_best_per_arch(selection_log: dict) -> Dict[str, Optional[dict]]:
    """From a selection log, find the best (method, drop_n) per architecture.

    Returns dict: arch -> {"method": ..., "drop_n": ..., "val_accuracy": ...} or None.
    """
    best = {}
    for key, entry in selection_log.get("search_results", {}).items():
        if entry.get("early_stopped"):
            continue
        parts = key.split("/")
        arch, method = parts[0], parts[1]
        drop_n = int(parts[2].replace("drop", ""))
        acc = entry["accuracy"]

        if arch not in best or acc > best[arch]["val_accuracy"] or \
           (acc == best[arch]["val_accuracy"] and drop_n > best[arch]["drop_n"]):
            best[arch] = {"method": method, "drop_n": drop_n, "val_accuracy": acc}

    # Also check baselines (drop=0) — they might be the best for some arch
    for key, entry in selection_log.get("baselines", {}).items():
        parts = key.split("/")
        arch, method = parts[0], parts[1]
        acc = entry["accuracy"]
        if arch not in best or acc > best[arch]["val_accuracy"]:
            best[arch] = {"method": method, "drop_n": 0, "val_accuracy": acc}

    return best


def evaluate_config(arch, method, drop_n, dataset, cfg, accelerator):
    """Fine-tune head on train+val, evaluate on test."""
    model_dir = get_model_dir(arch)

    if not Path(model_dir).exists():
        accelerator.print(f"  [SKIP] {model_dir} not found")
        return None

    # Find a valid config to copy .py files from
    if config_exists(cfg["results_prune_dir"], arch, method, drop_n):
        if accelerator.is_main_process:
            setup_model_dir(model_dir, cfg["results_prune_dir"], arch, method, drop_n)
        accelerator.wait_for_everyone()
    elif config_exists(cfg["results_prune_dir"], arch, method, 0):
        if accelerator.is_main_process:
            setup_model_dir(model_dir, cfg["results_prune_dir"], arch, method, 0)
        accelerator.wait_for_everyone()
    else:
        accelerator.print(f"  [SKIP] No config for {arch}/{method}/drop{drop_n}")
        return None

    # Cache base weights
    if accelerator.is_main_process:
        swap_config(model_dir, cfg["results_prune_dir"], arch, method, 0)
    accelerator.wait_for_everyone()
    cached_weights = cache_base_weights(model_dir)

    # Swap to the target config
    if accelerator.is_main_process:
        swap_config(model_dir, cfg["results_prune_dir"], arch, method, drop_n)
    accelerator.wait_for_everyone()
    model, processor = load_model_and_processor(model_dir, cached_state_dict=cached_weights)

    del cached_weights
    torch.cuda.empty_cache()

    # Use deep_finetune_and_evaluate with 5 epochs (trains on train+val, evals on test)
    test_metrics = deep_finetune_and_evaluate(
        model=model, processor=processor, accelerator=accelerator,
        dataset_name=dataset, dataset_base_dir=cfg["dataset_base_dir"],
        epochs=cfg["epochs"], lr=cfg["lr"],
        weight_decay=cfg["weight_decay"], batch_size=cfg["batch_size"],
        batch_size_eval=cfg["batch_size_eval"],
        num_workers=cfg["num_workers"],
        warmup_ratio=cfg["warmup_ratio"],
        output_dir=cfg["output_dir"],
    )

    del model
    gc.collect()
    torch.cuda.empty_cache()
    accelerator.wait_for_everyone()

    return test_metrics


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate best config per architecture on test set")
    parser.add_argument("--dataset_base_dir", type=str, default="data")
    parser.add_argument("--results_prune_dir", type=str, default="results_prune")
    parser.add_argument("--selection_dir", type=str, default="results/selection")
    parser.add_argument("--output_csv", type=str, default="results/selection/all_archs_test.csv")
    parser.add_argument("--output_dir", type=str, default="results/selection/all_archs_eval")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.03)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--batch_size_eval", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--datasets", type=str, nargs="+", default=None,
                        help="Specific datasets to evaluate (default: all with logs)")
    args = parser.parse_args()

    from loader import set_seed
    set_seed(42)
    accelerator = Accelerator()
    selection_dir = Path(args.selection_dir)

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

    # Discover datasets with completed selection logs
    if args.datasets:
        datasets = args.datasets
    else:
        datasets = []
        for d in DATASETS:
            log_path = selection_dir / d / f"selection_log_{d}.json"
            if log_path.exists():
                datasets.append(d)

    accelerator.print("=" * 80)
    accelerator.print("EVALUATE BEST CONFIG PER ARCHITECTURE")
    accelerator.print("=" * 80)
    accelerator.print(f"Datasets: {datasets}")
    accelerator.print(f"Epochs: {cfg['epochs']}, LR: {cfg['lr']}, Warmup: {cfg['warmup_ratio']}")
    accelerator.print("=" * 80)

    # Collect results: list of dicts for CSV
    all_results = []

    for dataset in datasets:
        log_path = selection_dir / dataset / f"selection_log_{dataset}.json"
        if not log_path.exists():
            accelerator.print(f"\n[SKIP] No selection log for {dataset}")
            continue

        with open(log_path, "r") as f:
            selection_log = json.load(f)

        best_per_arch = get_best_per_arch(selection_log)

        accelerator.print(f"\n{'=' * 60}")
        accelerator.print(f"Dataset: {dataset}")
        accelerator.print(f"{'=' * 60}")
        for arch in ARCH_MAX_LAYERS.keys():
            if arch in best_per_arch:
                b = best_per_arch[arch]
                accelerator.print(f"  {arch}: {b['method']}/drop{b['drop_n']} (val={b['val_accuracy']:.4f})")
            else:
                accelerator.print(f"  {arch}: [NO VALID CONFIG]")

        row = {"dataset": dataset}

        for arch in ARCH_MAX_LAYERS.keys():
            if arch not in best_per_arch:
                row[f"{arch}_acc"] = ""
                row[f"{arch}_method"] = ""
                row[f"{arch}_drop"] = ""
                continue

            best = best_per_arch[arch]
            method = best["method"]
            drop_n = best["drop_n"]

            accelerator.print(f"\n--- {dataset} / {arch} / {method} / drop{drop_n} ---")

            metrics = evaluate_config(arch, method, drop_n, dataset, cfg, accelerator)

            if metrics and accelerator.is_main_process:
                acc = metrics["accuracy"]
                accelerator.print(f"  Test acc: {acc:.4f}")
                row[f"{arch}_acc"] = f"{acc * 100:.2f}"
                row[f"{arch}_method"] = method
                row[f"{arch}_drop"] = str(drop_n)
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
            for arch in ARCH_MAX_LAYERS.keys():
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
