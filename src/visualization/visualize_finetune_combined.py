import re
from pathlib import Path

import matplotlib.pyplot as plt

from visualize_benchmark_results import (
    parse_accuracy_from_file, prettify_dataset_name,
    calculate_y_axis_range, LAST_DROPPED_MAPPING,
)

RESULTS_DIR = Path("results/accuracy")
OUT_DIR = Path("results/plots")

# Paper order: rows of four
DATASETS = ["cifar10", "zoolake", "lar", "CrossD",
            "LCZ42", "Pest", "InfLarynge", "Bark",
            "WHOI22", "Kaggle38", "ZooScan20", "ColorBG"]

MODELS = ["dinov2", "dinov3_vit", "swinv2", "vit"]
COLORS = {"dinov2": "blue", "dinov3_vit": "purple", "swinv2": "green", "vit": "red"}
LABELS = {"dinov2": "DINOv2", "dinov3_vit": "DINOv3", "swinv2": "SwinV2", "vit": "ViT"}

METHODS = {
    "layer_drop_attn": "Dropped Attention Layers",
    "layer_drop_mlp": "Dropped MLP Layers",
    "block_drop": "Dropped Blocks",
    "layer_drop_all": "The Number of Dropped Modules",
}


def collect(dataset, model, method):
    """All (drop_n, accuracy) points available on disk."""
    file_re = re.compile(
        rf"output_vision_{re.escape(model)}_drop(?P<n>\d+)_{re.escape(method)}_{re.escape(dataset)}\.out")
    points = []
    ds_dir = RESULTS_DIR / dataset
    for f in ds_dir.glob(f"output_vision_{model}_drop*_{method}_{dataset}.out"):
        m = file_re.match(f.name)
        if m is None:
            continue
        acc = parse_accuracy_from_file(f)
        if acc is not None:
            points.append((int(m.group("n")), acc))
    if not any(n == 0 for n, _ in points):
        for f in sorted(ds_dir.glob(f"output_vision_{model}_drop0_*_{dataset}.out")):
            acc = parse_accuracy_from_file(f)
            if acc is not None:
                points.append((0, acc))
                break
    return sorted(points)


def plot_panel(ax, dataset, method):
    all_acc = []
    for model in MODELS:
        points = collect(dataset, model, method)
        if not points:
            continue
        color = COLORS[model]
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        all_acc.extend(ys)
        baseline = next((a for n, a in points if n == 0), None)

        if method == "layer_drop_all":
            ax.plot(xs, ys, color=color, linestyle="-", linewidth=1.2,
                    label=LABELS[model])
            mapping = LAST_DROPPED_MAPPING[model]
            attn = [(n, a) for n, a in points if mapping.get(n) == "attn"]
            mlp = [(n, a) for n, a in points if mapping.get(n) == "mlp"]
            if attn:
                ax.scatter([p[0] for p in attn], [p[1] for p in attn],
                           color=color, marker="o", s=10, zorder=3)
            if mlp:
                ax.scatter([p[0] for p in mlp], [p[1] for p in mlp],
                           color=color, marker="*", s=32, zorder=3)
        else:
            ax.plot(xs, ys, color=color, marker="o", linestyle="-",
                    linewidth=1.2, markersize=2.6, label=LABELS[model])

        if baseline is not None:
            ax.scatter([0], [baseline], facecolors="none", edgecolors=color,
                       marker="o", s=22, linewidths=1.1, zorder=3)
            ax.axhline(y=baseline, color=color, linestyle="--", linewidth=0.8,
                       alpha=0.7, dashes=(5, 3))

    y_min, y_max = calculate_y_axis_range(all_acc)
    ax.set_ylim(y_min, y_max)
    ax.set_title(prettify_dataset_name(dataset), fontsize=10)
    ax.grid(True, color="#d4af37", alpha=0.4, linewidth=1, linestyle=":")
    ax.tick_params(labelsize=7)


def make_figure(method, xlabel):
    fig, axes = plt.subplots(3, 4, figsize=(15, 9.6))

    for ax, dataset in zip(axes.flat, DATASETS):
        plot_panel(ax, dataset, method)

    for row in range(3):
        axes[row][0].set_ylabel("Accuracy (%)", fontsize=9)
    for col in range(4):
        axes[2][col].set_xlabel(xlabel, fontsize=9)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels),
               fontsize=10, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.985))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = OUT_DIR / f"finetune_{method}_combined"
    fig.savefig(f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(f"{stem}.png", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"✓ Saved: {stem}.pdf and .png")


def main():
    for method, xlabel in METHODS.items():
        make_figure(method, xlabel)


if __name__ == "__main__":
    main()
