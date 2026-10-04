import argparse
import csv
import json
import os
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_csv_row(filepath: Path, batch_size: int) -> dict | None:
    """Return the CSV row matching batch_size, or None if not found / missing."""
    if not filepath.exists():
        return None
    try:
        with open(filepath, newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                try:
                    if int(float(row["Batch Size"])) == batch_size:
                        return row
                except (KeyError, ValueError):
                    continue
    except Exception as exc:
        print(f"[WARN] Could not read {filepath}: {exc}", file=sys.stderr)
    return None


def build_row(model: str, prune_method: str, drop_num: int,
              csv_row: dict, baseline_throughput: float | None) -> dict:
    """Build a normalised output row from a raw CSV row dict."""
    throughput = float(csv_row.get("Throughput (imgs/s)", "nan"))
    latency    = float(csv_row.get("Latency (ms)",        "nan"))
    memory     = float(csv_row.get("Memory (GB)",         "nan"))
    flops      = float(csv_row.get("FLOPs (G)",           "nan"))

    if baseline_throughput and baseline_throughput > 0:
        speedup = round(throughput / baseline_throughput, 4)
    else:
        speedup = None

    return {
        "model":         model,
        "prune_method":  prune_method,
        "drop_num":      drop_num,
        "throughput":    round(throughput, 4),
        "latency_ms":    round(latency, 4),
        "memory_gb":     round(memory, 4),
        "flops_g":       round(flops, 4),
        "speedup":       speedup,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    speed_dir   = Path(args.speed_dir)
    output_path = Path(args.output)
    batch_size  = args.batch_size

    rows: list[dict] = []

    for model in args.models:
        # ------------------------------------------------------------------
        # Baseline (original model, no pruning)
        # ------------------------------------------------------------------
        baseline_csv = speed_dir / f"{model}_original_speed.csv"
        baseline_row = parse_csv_row(baseline_csv, batch_size)

        if baseline_row is None:
            print(f"[WARN] Baseline CSV not found or empty for {model}: {baseline_csv}",
                  file=sys.stderr)
            baseline_throughput = None
        else:
            baseline_throughput = float(baseline_row.get("Throughput (imgs/s)", 0) or 0)
            # Baseline row itself — drop_num=0, prune_method="baseline"
            rows.append(build_row(model, "baseline", 0, baseline_row, baseline_throughput))

        # ------------------------------------------------------------------
        # Pruned configs
        # ------------------------------------------------------------------
        for prune_method in args.methods:
            for drop_num in args.drop_nums:
                if drop_num == 0:
                    continue  # already handled as baseline above

                csv_path = speed_dir / f"{model}_drop{drop_num}_{prune_method}_speed.csv"
                row_data = parse_csv_row(csv_path, batch_size)

                if row_data is None:
                    print(f"[SKIP] Not found: {csv_path}", file=sys.stderr)
                    continue

                rows.append(build_row(model, prune_method, drop_num,
                                      row_data, baseline_throughput))

    if not rows:
        print("[ERROR] No data found — check --speed_dir and file naming conventions.",
              file=sys.stderr)
        sys.exit(1)

    # -----------------------------------------------------------------------
    # Write CSV
    # -----------------------------------------------------------------------
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["model", "prune_method", "drop_num",
                  "throughput", "latency_ms", "memory_gb", "flops_g", "speedup"]

    with open(output_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[OK] Master speed CSV: {output_path}  ({len(rows)} rows)")

    # -----------------------------------------------------------------------
    # Write JSON (same path with .json extension)
    # -----------------------------------------------------------------------
    json_path = output_path.with_suffix(".json")
    with open(json_path, "w") as fh:
        json.dump(rows, fh, indent=2)
    print(f"[OK] Master speed JSON: {json_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Consolidate individual speed CSVs into a single master file."
    )
    parser.add_argument("--speed_dir",  type=str, default="results/speed/per_model",
                        help="Directory containing per-model speed CSVs")
    parser.add_argument("--output",     type=str, default="results/speed/speed_master.csv",
                        help="Path for the output master CSV")
    parser.add_argument("--models",     nargs="+", default=["dinov2", "dinov3_vit", "swinv2", "vit"],
                        help="Model names to include")
    parser.add_argument("--methods",    nargs="+",
                        default=["layer_drop_attn", "layer_drop_mlp", "layer_drop_all", "block_drop"],
                        help="Pruning methods to include")
    parser.add_argument("--drop_nums",  nargs="+", type=int,
                        default=[0, 2, 4, 6, 8, 12, 15, 18, 21, 28, 32, 40],
                        help="Drop counts to include")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Batch size to extract from each CSV")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
