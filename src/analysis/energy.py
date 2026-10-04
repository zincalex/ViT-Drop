import argparse
import csv
from pathlib import Path

PER_MODEL = Path(__file__).resolve().parents[2] / "results" / "speed" / "per_model"
ARCHS = {"dinov2": "DINOv2", "dinov3_vit": "DINOv3", "swinv2": "SwinV2", "vit": "ViT"}
METHODS = {"block_drop": "Block", "layer_drop_attn": "Attn",
           "layer_drop_mlp": "MLP", "layer_drop_all": "Joint"}
DROPS = [4, 8, 12, 16, 20, 24, 28, 32, 40]


def read_row(path, batch_size=128):
    """(board power in W, throughput in images/s) of the batch-size row, or None."""
    if not path.exists():
        return None
    for r in csv.DictReader(open(path)):
        if int(r["Batch Size"]) == batch_size and r.get("Power (W)"):
            return float(r["Power (W)"]), float(r["Throughput (imgs/s)"])
    return None


def main():
    ap = argparse.ArgumentParser(description="Energy per image of the unpruned and pruned models.")
    ap.add_argument("--archs", nargs="+", default=list(ARCHS), choices=list(ARCHS),
                    help="architectures to report")
    args = ap.parse_args()

    for arch in args.archs:
        base = read_row(PER_MODEL / f"{arch}_original_speed.csv")
        if base is None:
            print(f"[SKIP] no baseline speed file for {arch}")
            continue
        p0, t0 = base
        e0 = p0 / t0
        print(f"{ARCHS[arch]} unpruned: {e0:.3f} J/img ({p0:.0f} W, {t0:.1f} img/s)")
        print(f"  {'Config':10s} {'Speedup':>8s} {'Power (W)':>9s} {'Energy (J/img)':>14s} {'Saving (%)':>10s}")
        for method, m_lbl in METHODS.items():
            for n in DROPS:
                r = read_row(PER_MODEL / f"{arch}_drop{n}_{method}_speed.csv")
                if r is None:
                    continue
                p, t = r
                e = p / t
                print(f"  {m_lbl + '-' + str(n):10s} {t / t0:7.2f}x {p:9.0f} {e:14.3f} {100 * (1 - e / e0):10.1f}")
        print()


if __name__ == "__main__":
    main()
