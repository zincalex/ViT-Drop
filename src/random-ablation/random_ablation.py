import argparse
import json
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
from accelerate import Accelerator

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "model-selection"))

from config import ARCH_MAX_LAYERS
from model_loader import (
    get_checkpoint_dir, config_exists, setup_model_dir, swap_config,
    load_model_and_processor, cache_base_weights,
)
from evaluator import finetune_and_evaluate
from loader import set_seed


def random_drop_lists(method, n, L, rng):
    if method == "block_drop":
        blocks = sorted(rng.sample(range(L), n))
        return blocks, list(blocks)
    if method == "layer_drop_attn":
        return sorted(rng.sample(range(L), n)), []
    if method == "layer_drop_mlp":
        return [], sorted(rng.sample(range(L), n))
    if method == "layer_drop_all":
        picks = rng.sample(range(2 * L), n)
        attn = sorted(i for i in picks if i < L)
        mlp = sorted(i - L for i in picks if i >= L)
        return attn, mlp
    raise ValueError(method)


def write_random_config(model_dir, prune_dir, arch, method, drop_n, attn, mlp):
    src = get_checkpoint_dir(prune_dir, arch, method, 0) / "config.json"
    cfg = json.load(open(src))
    L = ARCH_MAX_LAYERS[arch]
    cfg["drop_attn_list"] = [i in attn for i in range(L)]
    cfg["drop_mlp_list"] = [i in mlp for i in range(L)]
    with open(Path(model_dir) / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)


def get_model_dir(arch):
    return f"./{arch}_model_ablation"


def ensure_private_model_dir(arch):
    src, dst = Path(f"./{arch}_model"), Path(get_model_dir(arch))
    if not dst.exists():
        shutil.copytree(src, dst)
    return str(dst)


def broadcast_acc(metrics, accelerator):
    t = torch.tensor([0.0], device=accelerator.device)
    if accelerator.is_main_process and metrics:
        t[0] = metrics["accuracy"]
    if accelerator.num_processes > 1:
        torch.distributed.broadcast(t, src=0)
    return t.item()


def main():
    cfg = parse_args()
    set_seed(42)
    accelerator = Accelerator()

    ref_path = Path(cfg.reference_selection_dir) / cfg.dataset / f"selection_log_{cfg.dataset}.json"
    if not ref_path.exists():
        raise FileNotFoundError(f"Reference selection log not found: {ref_path}")
    ref = json.load(open(ref_path))

    out_dir = Path(cfg.output_dir) / cfg.dataset
    log_path = out_dir / f"random_ablation_{cfg.dataset}.json"
    out = {"hyperparameters": vars(cfg).copy(), "configs": {},
           "started_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    if log_path.exists():
        out = json.load(open(log_path))
        accelerator.print(f"[RESUME] {len(out['configs'])} configurations already present")

    def save():
        if accelerator.is_main_process:
            out_dir.mkdir(parents=True, exist_ok=True)
            with open(log_path, "w") as f:
                json.dump(out, f, indent=2)

    accelerator.print("=" * 80)
    accelerator.print(f"RANDOM-DROP ABLATION | dataset={cfg.dataset} | seeds={cfg.seeds}")
    accelerator.print("=" * 80)

    specs = []
    for s in cfg.configs:
        a, m, n = s.split(":")
        specs.append((a, m, int(n)))

    for arch in sorted({a for a, _, _ in specs}):
        arch_specs = [(m, n) for a, m, n in specs if a == arch]
        if not Path(f"./{arch}_model").exists():
            accelerator.print(f"[SKIP] ./{arch}_model not found")
            continue
        first_method = arch_specs[0][0]
        if not config_exists(cfg.results_prune_dir, arch, first_method, 0):
            accelerator.print(f"[SKIP] No drop0 config for {arch}")
            continue

        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            model_dir = ensure_private_model_dir(arch)
            setup_model_dir(model_dir, cfg.results_prune_dir, arch, first_method, 0)
            swap_config(model_dir, cfg.results_prune_dir, arch, first_method, 0)
        accelerator.wait_for_everyone()
        model_dir = get_model_dir(arch)
        cached_weights = cache_base_weights(model_dir)
        L = ARCH_MAX_LAYERS[arch]

        for method, drop_n in arch_specs:
            key = f"{arch}/{method}/drop{drop_n}"
            entry = out["configs"].get(key, {})
            sim = ref["search_results"].get(key)
            base = ref["baselines"].get(f"{arch}/{method}") or next(
                (v for k, v in ref["baselines"].items() if k.startswith(arch + "/")), None)
            if sim is None or base is None:
                accelerator.print(f"[SKIP] {key}: no similarity-guided reference result")
                continue
            entry["similarity_accuracy"] = sim["accuracy"]
            entry["baseline_accuracy"] = base["accuracy"]
            entry.setdefault("random", {})

            for seed in cfg.seeds:
                if str(seed) in entry["random"]:
                    continue
                rng = random.Random(1000 * seed + drop_n)
                attn, mlp = random_drop_lists(method, drop_n, L, rng)

                accelerator.print(f"\n--- {key} | random seed {seed} | attn={attn} mlp={mlp} ---")
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    write_random_config(model_dir, cfg.results_prune_dir, arch, method, drop_n, attn, mlp)
                accelerator.wait_for_everyone()
                model, processor = load_model_and_processor(model_dir, cached_state_dict=cached_weights)

                metrics = finetune_and_evaluate(
                    model=model, processor=processor, accelerator=accelerator,
                    dataset_name=cfg.dataset, dataset_base_dir=cfg.dataset_base_dir,
                    epochs=cfg.epochs, lr=cfg.lr, weight_decay=cfg.weight_decay,
                    batch_size=cfg.batch_size, batch_size_eval=cfg.batch_size_eval,
                    num_workers=cfg.num_workers, train_split="train", eval_split="val",
                )
                acc = broadcast_acc(metrics, accelerator)
                del model
                torch.cuda.empty_cache()

                if accelerator.is_main_process and metrics:
                    entry["random"][str(seed)] = {**metrics, "drop_attn": attn, "drop_mlp": mlp}
                    accs = [v["accuracy"] for v in entry["random"].values()]
                    entry["random_mean"] = float(np.mean(accs))
                    entry["random_std"] = float(np.std(accs))
                    entry["gap_similarity_minus_random"] = entry["similarity_accuracy"] - entry["random_mean"]
                    out["configs"][key] = entry
                    accelerator.print(
                        f"  random acc={acc:.4f} | similarity={sim['accuracy']:.4f} "
                        f"| baseline={base['accuracy']:.4f}")
                    save()
                accelerator.wait_for_everyone()

        del cached_weights
        torch.cuda.empty_cache()

    if accelerator.is_main_process:
        out["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        save()
        print("\n" + "=" * 80)
        print(f"RANDOM ABLATION SUMMARY: {cfg.dataset}")
        print(f"{'config':<34} {'baseline':>8} {'similar.':>8} {'random':>8} {'±std':>6} {'gap':>7}")
        for k, e in out["configs"].items():
            if "random_mean" not in e:
                continue
            print(f"{k:<34} {e['baseline_accuracy']:8.4f} {e['similarity_accuracy']:8.4f} "
                  f"{e['random_mean']:8.4f} {e['random_std']:6.4f} {e['gap_similarity_minus_random']:+7.4f}")
        print("=" * 80)
    accelerator.wait_for_everyone()


def parse_args():
    p = argparse.ArgumentParser(description="Random-drop ablation")
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--dataset_base_dir", type=str, default="data")
    p.add_argument("--results_prune_dir", type=str, default="results_prune")
    p.add_argument("--output_dir", type=str, default="results/random_ablation")
    p.add_argument("--reference_selection_dir", type=str, default="results/selection")
    p.add_argument("--configs", type=str, nargs="+", required=True,
                   help="arch:method:drop_n specs, e.g. dinov2:block_drop:8")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--weight_decay", type=float, default=0.03)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--batch_size_eval", type=int, default=10)
    p.add_argument("--num_workers", type=int, default=4)
    return p.parse_args()


if __name__ == "__main__":
    main()
