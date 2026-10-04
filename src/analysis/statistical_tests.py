import argparse
import re
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[2] / "results" / "accuracy"

ARCHS = {"dinov2": "DINOv2", "dinov3_vit": "DINOv3", "swinv2": "SwinV2", "vit": "ViT"}
METHODS = {"block_drop": "Block", "layer_drop_attn": "Attn",
           "layer_drop_mlp": "MLP", "layer_drop_all": "Joint"}
DROPS = [4, 8, 12]

FNAME_RE = re.compile(
    r"output_vision_(?P<arch>dinov2|dinov3_vit|swinv2|vit)_drop(?P<drop>\d+)_"
    r"(?P<method>block_drop|layer_drop_all|layer_drop_attn|layer_drop_mlp)_"
)
ACC_RE = re.compile(r"Accuracy:\s*([0-9.]+)\s*\(")


def parse_results():
    """Return acc[dataset][arch][method][drop] = accuracy in percent."""
    acc = {}
    for ds_dir in sorted(ROOT.iterdir()):
        if not ds_dir.is_dir() or ds_dir.name == "imagenet-1k":
            continue
        for f in ds_dir.glob("*.out"):
            m = FNAME_RE.match(f.name)
            a = ACC_RE.search(f.read_text()) if m else None
            if not a:
                continue
            acc.setdefault(ds_dir.name, {}).setdefault(m["arch"], {}).setdefault(
                m["method"], {})[int(m["drop"])] = float(a.group(1)) * 100.0
    return acc


def paired_vectors(acc, arch, method, drop):
    """Baseline and pruned accuracies over the datasets where both exist.
    The unpruned baseline is stored once, under block_drop drop 0."""
    base, pruned = [], []
    for ds in sorted(acc):
        b = acc[ds].get(arch, {}).get("block_drop", {}).get(0)
        p = acc[ds].get(arch, {}).get(method, {}).get(drop)
        if b is not None and p is not None:
            base.append(b)
            pruned.append(p)
    return np.array(base), np.array(pruned)


def wilcoxon_two_sided(diff):
    if np.allclose(diff, 0):
        return 1.0
    return stats.wilcoxon(diff, alternative="two-sided", zero_method="wilcox").pvalue


def tost_wilcoxon(diff, delta):
    """max of the two one-sided Wilcoxon p-values against -delta and +delta."""
    p_lower = stats.wilcoxon(diff + delta, alternative="greater", zero_method="wilcox").pvalue
    p_upper = stats.wilcoxon(diff - delta, alternative="less", zero_method="wilcox").pvalue
    return max(p_lower, p_upper)


def hodges_lehmann_ci(diff, confidence=0.90):
    """Hodges-Lehmann estimate and distribution-free CI from the Walsh averages,
    using the exact null distribution of the signed-rank statistic."""
    n = len(diff)
    walsh = np.sort([(diff[i] + diff[j]) / 2.0 for i in range(n) for j in range(i, n)])
    counts = np.zeros(n * (n + 1) // 2 + 1)
    counts[0] = 1
    for rank in range(1, n + 1):
        counts[rank:] = counts[rank:] + counts[:-rank]
    cdf = np.cumsum(counts) / counts.sum()
    k = max(1, int(np.searchsorted(cdf, (1.0 - confidence) / 2.0, side="right")))
    return float(np.median(walsh)), float(walsh[k - 1]), float(walsh[len(walsh) - k])


def holm(pvals):
    """Holm-Bonferroni adjustment for a family of p-values."""
    order = np.argsort(pvals)
    adj = np.empty(len(pvals))
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (len(pvals) - rank) * pvals[idx])
        adj[idx] = min(running, 1.0)
    return adj.tolist()


def outcome(p_holm, p_tost, lo, hi, alpha):
    if p_tost < alpha:
        return "equivalent"
    if p_holm < alpha and lo > 0:
        return "superior"
    if p_holm < alpha and hi < 0:
        return "inferior"
    return "inconclusive"


def main():
    ap = argparse.ArgumentParser(description="Difference and equivalence tests between pruned and unpruned models.")
    ap.add_argument("--delta", type=float, default=2.0,
                    help="equivalence margin in percentage points (default 2)")
    ap.add_argument("--alpha", type=float, default=0.05, help="significance level")
    args = ap.parse_args()

    acc = parse_results()
    print(f"Datasets ({len(acc)}): {', '.join(sorted(acc))}")
    print(f"Equivalence margin: +/-{args.delta:g} pp, alpha = {args.alpha}\n")
    header = (f"{'Arch':7s} {'Method':6s} {'Drop':>4s} {'n':>2s} {'mean diff':>9s} "
              f"{'Wilc p':>7s} {'Holm p':>7s} {'HL':>7s} {'90% CI':>18s} {'TOST p':>7s}  Outcome")
    print(header)
    print("-" * len(header))
    for arch, arch_lbl in ARCHS.items():
        for method, m_lbl in METHODS.items():
            rows = []
            for drop in DROPS:
                base, pruned = paired_vectors(acc, arch, method, drop)
                if len(base) == 0:
                    continue
                diff = pruned - base
                rows.append((drop, len(diff), diff.mean(), wilcoxon_two_sided(diff),
                             tost_wilcoxon(diff, args.delta), *hodges_lehmann_ci(diff)))
            for (drop, n, md, p_w, p_t, hl, lo, hi), p_h in zip(rows, holm([r[3] for r in rows])):
                print(f"{arch_lbl:7s} {m_lbl:6s} {drop:4d} {n:2d} {md:+9.2f} {p_w:7.3f} {p_h:7.3f} "
                      f"{hl:+7.2f} [{lo:+7.2f},{hi:+7.2f}] {p_t:7.3f}  "
                      f"{outcome(p_h, p_t, lo, hi, args.alpha)}")
        print()


if __name__ == "__main__":
    main()
