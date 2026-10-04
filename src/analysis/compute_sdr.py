import argparse
import csv
import re
from pathlib import Path

BASELINE_METHOD = "block_drop"


def parse_accuracy_from_file(filepath: Path):
    match = re.search(r"Accuracy:\s+([\d.]+)", filepath.read_text())
    return float(match.group(1)) * 100 if match else None


def parse_throughput_from_csv(filepath: Path, batch_size=128):
    if not filepath.exists():
        return None
    for row in csv.DictReader(open(filepath)):
        if int(row["Batch Size"]) == batch_size:
            return float(row["Throughput (imgs/s)"])
    return None


def collect_accuracy_data(model, prune_method, drop_num, datasets, results_dir):
    accuracies = {}
    for dataset in datasets:
        filepath = Path(results_dir) / dataset / f"output_vision_{model}_drop{drop_num}_{prune_method}_{dataset}.out"
        if filepath.exists():
            accuracy = parse_accuracy_from_file(filepath)
            if accuracy is not None:
                accuracies[dataset] = accuracy
    return accuracies


def calculate_sdr(baseline_acc, dropped_acc, baseline_throughput, dropped_throughput):
    common = sorted(set(baseline_acc) & set(dropped_acc))
    baseline_avg = sum(baseline_acc[d] for d in common) / len(common)
    dropped_avg = sum(dropped_acc[d] for d in common) / len(common)
    delta_acc = (baseline_avg - dropped_avg) / baseline_avg * 100
    delta_speed = (dropped_throughput - baseline_throughput) / baseline_throughput * 100
    sdr = delta_acc / delta_speed if delta_speed != 0 else None
    return common, baseline_avg, dropped_avg, delta_acc, delta_speed, sdr


def write_sdr_report(output_path, drop_num, common, baseline_acc, dropped_acc, baseline_throughput,
                     dropped_throughput, baseline_avg, dropped_avg, delta_acc, delta_speed, sdr):
    line = "-" * 80 + "\n"
    with open(output_path, "w") as f:
        f.write(line + "BASELINE (drop_num=0)\n" + line)
        f.write(f"Throughput: {baseline_throughput:.2f} imgs/s\n")
        f.write(f"Average Accuracy: {baseline_avg:.2f}% over {len(common)} datasets\n\n")
        f.write("Per-Dataset Accuracy:\n")
        for d in common:
            f.write(f"  - {d}: {baseline_acc[d]:.2f}%\n")
        f.write("\n" + line + f"DROPPED MODEL (drop_num={drop_num})\n" + line)
        f.write(f"Throughput: {dropped_throughput:.2f} imgs/s\n")
        f.write(f"Average Accuracy: {dropped_avg:.2f}% over {len(common)} datasets\n\n")
        f.write("Per-Dataset Accuracy:\n")
        for d in common:
            f.write(f"  - {d}: {dropped_acc[d]:.2f}%\n")
        f.write("\n" + line + "METRICS\n" + line)
        f.write(f"Relative accuracy loss:     {delta_acc:+.2f}%\n")
        f.write(f"Relative throughput gain:   {delta_speed:+.2f}%\n")
        f.write(f"Speedup:                    {dropped_throughput / baseline_throughput:.4f}x\n")
        if sdr is not None:
            f.write(f"SDR (gamma):                {sdr:.4f}\n")
        else:
            f.write("SDR (gamma):                N/A (zero speedup)\n")


def compute_sdr_for_configuration(model, prune_method, drop_num, datasets, results_dir, speed_dir, output_dir):
    baseline_acc = collect_accuracy_data(model, BASELINE_METHOD, 0, datasets, results_dir)
    dropped_acc = collect_accuracy_data(model, prune_method, drop_num, datasets, results_dir)
    baseline_throughput = parse_throughput_from_csv(Path(speed_dir) / f"{model}_original_speed.csv")
    dropped_throughput = parse_throughput_from_csv(Path(speed_dir) / f"{model}_drop{drop_num}_{prune_method}_speed.csv")
    if not (set(baseline_acc) & set(dropped_acc)) or baseline_throughput is None or dropped_throughput is None:
        print(f"[SKIP] {model} {prune_method} drop{drop_num}: missing accuracy or speed results")
        return False
    results = calculate_sdr(baseline_acc, dropped_acc, baseline_throughput, dropped_throughput)
    output_path = Path(output_dir) / f"SDR_{model}_{prune_method}_drop{drop_num}.txt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_sdr_report(output_path, drop_num, results[0], baseline_acc, dropped_acc,
                     baseline_throughput, dropped_throughput, *results[1:])
    print(f"[OK] {output_path} (gamma = {results[-1]:.4f})" if results[-1] is not None else f"[OK] {output_path}")
    return True


def parse_args():
    parser = argparse.ArgumentParser(description="Speedup Degradation Ratio of one pruned configuration.")
    parser.add_argument("--model", type=str, required=True, help="architecture key, e.g. dinov2")
    parser.add_argument("--prune_method", type=str, required=True,
                        help="block_drop, layer_drop_attn, layer_drop_mlp or layer_drop_all")
    parser.add_argument("--drop_num", type=int, required=True, help="number of removed modules")
    parser.add_argument("--datasets", nargs="+", required=True, help="datasets averaged in the accuracy term")
    parser.add_argument("--results_dir", type=str, default="results/accuracy")
    parser.add_argument("--speed_dir", type=str, default="results/speed/per_model")
    parser.add_argument("--output_dir", type=str, default="results/sdr")
    return parser.parse_args()


def main():
    args = parse_args()
    compute_sdr_for_configuration(args.model, args.prune_method, args.drop_num, args.datasets,
                                  args.results_dir, args.speed_dir, args.output_dir)


if __name__ == "__main__":
    main()
