import argparse
import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# Domain-ordered spokes: natural -> remote sensing -> medical -> agriculture -> plankton
DEFAULT_DATASETS = [
    "cifar10", "LCZ42", "lar", "InfLarynge", "ColorBG", "Bark", "Pest",
    "CrossD", "zoolake", "Kaggle38", "WHOI22", "ZooScan20",
]
DATASET_LABELS = {
    "cifar10": "CIFAR-10", "LCZ42": "LCZ42", "lar": "Laryngeal",
    "InfLarynge": "InfLarynge", "ColorBG": "ColorBG", "Bark": "Bark",
    "Pest": "Pest", "CrossD": "CrossD", "zoolake": "ZooLake",
    "Kaggle38": "Kaggle38", "WHOI22": "WHOI22", "ZooScan20": "ZooScan20",
}
METRIC_PATTERNS = {
    "accuracy": r"Accuracy:\s*([0-9.]+)",
    "precision": r"Precision:\s*([0-9.]+)",
    "recall": r"Recall:\s*([0-9.]+)",
    "f1": r"F1 Score:\s*([0-9.]+)",
}
ARCH_LABELS = {"dinov2": "DINOv2", "dinov3_vit": "DINOv3", "swinv2": "SwinV2",
               "vit": "ViT", "deit": "DeiT"}
SERIES_COLORS = ["#0072B2", "#E69F00", "#CC3311", "#9467bd", "#56B4E9", "#8c564b"]


def read_metric(results_dir, dataset, arch, method, drop, metric):
    d = Path(results_dir) / dataset
    candidates = [d / f"output_vision_{arch}_drop{drop}_{method}_{dataset}.out"]
    if drop == 0:
        # drop0 runs once per model (under a single method); accept any drop0 file
        candidates += sorted(d.glob(f"output_vision_{arch}_drop0_*_{dataset}.out"))
    for p in candidates:
        if p.exists():
            m = re.search(METRIC_PATTERNS[metric], p.read_text())
            if m:
                return float(m.group(1))
    return None


def draw_wheel(ax, results_dir, arch, method, drops, datasets, metric, rmin):
    labels = [DATASET_LABELS.get(ds, ds) for ds in datasets]
    n = len(datasets)
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    for drop, color in zip(drops, SERIES_COLORS):
        vals = [read_metric(results_dir, ds, arch, method, drop, metric) for ds in datasets]
        # Values below the axis floor are clamped to the inner ring, so weak
        # spokes read as "at or below the minimum" instead of diving to the hub.
        v = [max(x, rmin) for x in vals]
        v = v + [v[0]]
        a = ang + [ang[0]]
        ax.plot(a, v, color=color, lw=2, label=f"drop {drop}")
        ax.fill(a, v, color=color, alpha=0.07)

    ax.set_xticks(ang)
    ax.set_xticklabels(labels, fontsize=9)
    # Push the dataset labels clearly outside the outer circle.
    ax.tick_params(axis="x", pad=14)
    ax.set_ylim(rmin, 1.0)
    # Radial gridlines every 0.1; no numeric labels (documented in the caption).
    ticks = [round(0.1 * k, 1) for k in range(1, 10) if 0.1 * k > rmin]
    ax.set_yticks(ticks)
    ax.set_yticklabels([])
    ax.grid(alpha=0.45)
    ax.set_title(ARCH_LABELS.get(arch, arch), fontsize=14, pad=20)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--archs", nargs="+", required=True, help="Architectures, one wheel each")
    p.add_argument("--drops", nargs="+", type=int, required=True, help="Drop counts overlaid on every wheel")
    p.add_argument("--override", nargs="*", default=[],
                   help="Per-arch drop override, e.g. vit=0,4,8")
    p.add_argument("--metric", choices=list(METRIC_PATTERNS) + ["all"], default="f1",
                   help='"all" renders precision, recall and f1 in one call')
    p.add_argument("--method", default="layer_drop_all",
                   choices=["block_drop", "layer_drop_attn", "layer_drop_mlp", "layer_drop_all"])
    p.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    p.add_argument("--results_dir", default="results/accuracy")
    p.add_argument("--output", required=True)
    p.add_argument("--rmin", type=float, default=0.0, help="Inner radius value (zoom)")
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument("--ncols", type=int, default=None,
                   help="Wheels per row (default: 1 for a single arch, else 2)")
    return p.parse_args()


def render(args, overrides, metric, output):
    # A spoke is kept only if every requested (arch, drop) series has a value
    # for it, so all wheels share an identical, gap-free spoke layout. Datasets
    # with pending campaign results are excluded and return automatically once
    # their files exist.
    datasets = []
    for ds in args.datasets:
        complete = all(
            read_metric(args.results_dir, ds, arch, args.method, drop, metric) is not None
            for arch in args.archs
            for drop in overrides.get(arch, args.drops)
        )
        if complete:
            datasets.append(ds)
        else:
            print(f"[skip] {ds}: incomplete across requested series, spoke omitted")

    n = len(args.archs)
    ncols = args.ncols if args.ncols else (1 if n == 1 else 2)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.0 * ncols, 6.3 * nrows),
                             subplot_kw=dict(polar=True), squeeze=False)

    for i, arch in enumerate(args.archs):
        ax = axes[i // ncols][i % ncols]
        drops = overrides.get(arch, args.drops)
        draw_wheel(ax, args.results_dir, arch, args.method, drops,
                   datasets, metric, args.rmin)

    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)

    # One legend for the whole figure, taken from the first wheel's series.
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels),
               fontsize=11, frameon=True, bbox_to_anchor=(0.5, -0.01))

    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(rect=(0, 0.03, 1, 1))
    plt.savefig(out, dpi=args.dpi, bbox_inches="tight")
    plt.close()
    print(f"\u2713 Saved: {out}")


def main():
    args = parse_args()
    overrides = {}
    for ov in args.override:
        arch, ds = ov.split("=")
        overrides[arch] = [int(x) for x in ds.split(",")]

    out = Path(args.output)
    if args.metric == "all":
        for metric in ("precision", "recall", "f1"):
            render(args, overrides, metric,
                   out.with_name(f"{out.stem}_{metric}_{args.method}{out.suffix}"))
    else:
        render(args, overrides, args.metric,
               out.with_name(f"{out.stem}_{args.method}{out.suffix}"))


if __name__ == "__main__":
    main()
