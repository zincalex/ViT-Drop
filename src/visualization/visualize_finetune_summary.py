import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from visualize_finetune_combined import collect, DATASETS, MODELS, COLORS, LABELS

OUT_DIR = Path("results/plots")

PANEL_TITLES = {
    "layer_drop_attn": "(a) Attn Drop",
    "layer_drop_mlp": "(b) MLP Drop",
    "block_drop": "(c) Block Drop",
    "layer_drop_all": "(d) Joint Layer Drop",
}
ORDER = ["layer_drop_attn", "layer_drop_mlp", "block_drop", "layer_drop_all"]


def retention_series(model, method):
    """drop_n -> list of per-dataset retained accuracies (% of baseline)."""
    series = {}
    for ds in DATASETS:
        points = dict(collect(ds, model, method))
        base = points.get(0)
        if not base:
            continue
        for n, acc in points.items():
            if n == 0:
                continue
            series.setdefault(n, []).append(100.0 * acc / base)
    return series


def main():
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.2), sharey=True)

    for ax, method in zip(axes, ORDER):
        for model in MODELS:
            series = retention_series(model, method)
            if not series:
                continue
            drops = sorted(series)
            med = [np.median(series[n]) for n in drops]
            q1 = [np.percentile(series[n], 25) for n in drops]
            q3 = [np.percentile(series[n], 75) for n in drops]
            color = COLORS[model]
            ax.plot(drops, med, color=color, marker="o", linewidth=1.4,
                    markersize=3, label=LABELS[model])
            ax.fill_between(drops, q1, q3, color=color, alpha=0.15, linewidth=0)

        ax.axhline(100, color="gray", linestyle="--", linewidth=0.9, alpha=0.8)
        ax.set_title(PANEL_TITLES[method], fontsize=11)
        ax.set_xlabel("Drop count", fontsize=9)
        ax.set_ylim(-5, 115)
        ax.grid(True, color="#d4af37", alpha=0.4, linewidth=1, linestyle=":")
        ax.tick_params(labelsize=8)

    axes[0].set_ylabel("Retained accuracy (% of baseline)", fontsize=10)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels),
               fontsize=10, frameon=False, bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / "finetune_retention_summary.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / "finetune_retention_summary.png", bbox_inches="tight", dpi=300)
    print(f"✓ Saved: {OUT_DIR}/finetune_retention_summary.pdf and .png")


if __name__ == "__main__":
    main()
