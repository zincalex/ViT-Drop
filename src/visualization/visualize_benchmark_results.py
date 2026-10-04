import re
import argparse
import matplotlib.pyplot as plt

from pathlib import Path


DINOV2_LAST_DROPPED = {
    1: 'attn', 2: 'attn', 3: 'attn', 4: 'attn', 5: 'attn',
    6: 'attn', 7: 'attn', 8: 'attn', 9: 'attn', 10: 'attn',
    11: 'mlp', 12: 'attn', 13: 'mlp', 14: 'attn', 15: 'mlp',
    16: 'attn', 17: 'mlp', 18: 'mlp', 19: 'mlp', 20: 'attn',
    21: 'mlp', 22: 'mlp', 23: 'mlp', 24: 'mlp', 25: 'attn',
    26: 'mlp', 27: 'mlp', 28: 'attn', 29: 'mlp', 30: 'mlp',
    31: 'mlp', 32: 'attn', 33: 'mlp', 34: 'mlp', 35: 'attn',
    36: 'mlp', 37: 'mlp', 38: 'mlp', 39: 'attn', 40: 'mlp',
}

VIT_LAST_DROPPED = {
    1: 'mlp', 2: 'mlp', 3: 'mlp', 4: 'mlp', 5: 'mlp',
    6: 'mlp', 7: 'mlp', 8: 'mlp', 9: 'mlp', 10: 'mlp',
    11: 'attn', 12: 'attn',
}

SWINV2_LAST_DROPPED = {
    1: 'attn', 2: 'attn', 3: 'attn', 4: 'attn', 5: 'attn',
    6: 'attn', 7: 'attn', 8: 'attn', 9: 'mlp', 10: 'mlp',
    11: 'attn', 12: 'mlp', 13: 'attn', 14: 'mlp', 15: 'mlp',
    16: 'mlp', 17: 'mlp', 18: 'mlp', 19: 'attn', 20: 'attn',
    21: 'mlp', 22: 'attn', 23: 'mlp', 24: 'mlp',
}

DINOV3_LAST_DROPPED = {
    1: 'mlp', 2: 'mlp', 3: 'mlp', 4: 'mlp', 5: 'attn',
    6: 'mlp', 7: 'attn', 8: 'attn', 9: 'attn', 10: 'mlp',
    11: 'attn', 12: 'mlp', 13: 'attn', 14: 'attn', 15: 'mlp',
    16: 'attn', 17: 'mlp', 18: 'mlp', 19: 'attn', 20: 'mlp',
    21: 'attn', 22: 'attn', 23: 'mlp', 24: 'attn', 25: 'mlp',
    26: 'attn', 27: 'mlp', 28: 'mlp', 29: 'attn', 30: 'attn',
    31: 'mlp', 32: 'attn', 33: 'attn', 34: 'attn', 35: 'attn',
    36: 'attn', 37: 'attn', 38: 'mlp', 39: 'mlp', 40: 'mlp',
    41: 'attn', 42: 'mlp', 43: 'mlp', 44: 'mlp', 45: 'attn',
    46: 'mlp', 47: 'mlp', 48: 'attn',
}


DEIT_LAST_DROPPED = {
    1: 'mlp', 2: 'mlp', 3: 'mlp', 4: 'mlp', 5: 'mlp',
    6: 'mlp', 7: 'mlp', 8: 'mlp', 9: 'mlp', 10: 'mlp',
    11: 'mlp', 12: 'attn', 13: 'attn', 14: 'attn', 15: 'attn',
    16: 'attn', 17: 'attn', 18: 'attn', 19: 'attn', 20: 'attn',
}

LAST_DROPPED_MAPPING = {'dinov2': DINOV2_LAST_DROPPED, 'vit': VIT_LAST_DROPPED, 'swinv2': SWINV2_LAST_DROPPED, 'dinov3_vit': DINOV3_LAST_DROPPED, 'deit': DEIT_LAST_DROPPED}


def prettify_dataset_name(dataset: str) -> str:
    mapping = {
        'imagenet-1k': 'ImageNet-1K',
        'cifar10': 'CIFAR-10',
        'zoolake': 'ZooLake',
        'crossd': 'CrossD',
        'lcz42': 'LCZ42',
        'lar': 'Lar',
        'pest': 'Pest',
        'inflarynge': 'InfLarynge',
        'whoi22': 'WHOI22',
        'kaggle38': 'Kaggle38',
        'zooscan20': 'ZooScan20',
        'colorbg': 'ColorBG',
    }
    return mapping.get(dataset.lower(), dataset.title())


def get_xlabel_from_prune_method(prune_method: str) -> str:
    if prune_method == 'layer_drop_all':
        return "The Number of Dropped Modules"
    elif 'attn' in prune_method:
        return "Dropped Attention Layers"
    elif 'mlp' in prune_method:
        return "Dropped MLP Layers"
    else:
        return "Dropped Blocks"


def parse_accuracy_from_file(filepath: Path) -> float:
    try:
        with open(filepath, 'r') as f:
            for line in f:
                if 'Accuracy:' in line:
                    match = re.search(r'Accuracy:\s+([\d.]+)', line)
                    if match:
                        return float(match.group(1)) * 100  # Convert to percentage
    except Exception as e:
        print(f"Warning: Could not parse {filepath}: {e}")

    return None


def calculate_y_axis_range(all_accuracies: list) -> tuple:
    if not all_accuracies:
        return 0, 100
    
    min_acc = min(all_accuracies)
    max_acc = max(all_accuracies)
    
    # Round to nearest 10 with padding
    y_min = (int(min_acc / 10) - 1) * 10
    y_max = (int(max_acc / 10) + 2) * 10
    
    # Ensure reasonable bounds
    y_min = max(0, y_min)
    y_max = min(100, y_max)
    
    return y_min, y_max


def collect_data(dataset, prune_method, model_names, drop_nums, results_dir):
    data = {model: [] for model in model_names}

    for model in model_names:
        for drop_num in drop_nums:
            filename = f"output_vision_{model}_drop{drop_num}_{prune_method}_{dataset}.out"
            filepath = Path(results_dir) / dataset / filename

            if not filepath.exists() and drop_num == 0:
                candidates = sorted((Path(results_dir) / dataset).glob(
                    f"output_vision_{model}_drop0_*_{dataset}.out"))
                if candidates:
                    filepath = candidates[0]

            if filepath.exists():
                accuracy = parse_accuracy_from_file(filepath)
                if accuracy is not None:
                    data[model].append((drop_num, accuracy))
                    print(f"  Found: {model} drop{drop_num} = {accuracy:.2f}%")
            else:
                print(f"  Missing: {filename}")

        data[model].sort(key=lambda x: x[0])

    return data


def create_legend_image(output_dir, model_names=None, filename="legend.png"):
    colors = {
        'dinov2': 'blue',
        'swinv2': 'green',
        'vit': 'red',
        'dinov3_vit': 'purple',
        'deit': 'orange'
    }
    
    labels_display = {
        'dinov2': 'DINOv2',
        'swinv2': 'SwinV2',
        'vit': 'ViT',
        'dinov3_vit': 'DINOv3',
        'deit': 'DeiT'
    }

    if model_names is None:
        model_names = ['dinov2', 'dinov3_vit', 'swinv2', 'vit']

    fig, ax = plt.subplots(figsize=(8, 1))
    ax.axis('off')

    lines = []
    labels = []
    for model in model_names:
        line, = ax.plot([], [], 
                       color=colors[model], 
                       linewidth=2, 
                       linestyle='-',
                       marker='o',
                       markersize=8)
        lines.append(line)
        labels.append(labels_display.get(model, model))

    legend = ax.legend(lines, labels, 
                      loc='center',
                      ncol=len(lines),
                      frameon=True,
                      fancybox=False,
                      shadow=False,
                      fontsize=14,
                      handlelength=3,
                      columnspacing=2)

    frame = legend.get_frame()
    frame.set_facecolor('white')
    frame.set_edgecolor('gray')
    frame.set_linewidth(1.5)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path / filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✓ Saved: {output_path / filename}")


def create_plot(data, dataset, prune_method, output_dir):
    fig, ax = plt.subplots(figsize=(8, 8))
    
    colors = {
        'dinov2': 'blue',
        'vit': 'red',
        'swinv2': 'green',
        'dinov3_vit': 'purple',
        'deit': 'orange'
    }
    
    all_accuracies = []
    baselines = {}

    use_layer_markers = (prune_method == 'layer_drop_all')

    for model, points in data.items():
        if not points:
            continue
        
        drop_nums = [p[0] for p in points]
        accuracies = [p[1] for p in points]
        all_accuracies.extend(accuracies)

        baseline_acc = None
        for drop_num, acc in points:
            if drop_num == 0:
                baseline_acc = acc
                baselines[model] = acc
                break
        color = colors.get(model, 'black')
        
        if use_layer_markers and model in LAST_DROPPED_MAPPING:
            last_dropped_map = LAST_DROPPED_MAPPING[model]

            attn_points = []
            mlp_points = []
            for drop_num, acc in points:
                if drop_num in last_dropped_map:
                    layer_type = last_dropped_map[drop_num]
                    if layer_type == 'attn':
                        attn_points.append((drop_num, acc))
                    elif layer_type == 'mlp':
                        mlp_points.append((drop_num, acc))

            ax.plot(drop_nums, accuracies, 
                    color=color, 
                    linestyle='-', 
                    linewidth=2, 
                    label=model)

            if baseline_acc is not None:
                ax.scatter([0], [baseline_acc],
                          facecolors='none',
                          edgecolors=color,
                          marker='o',
                          s=80,
                          linewidths=2,
                          zorder=3)

            if attn_points:
                attn_x = [p[0] for p in attn_points]
                attn_y = [p[1] for p in attn_points]
                ax.scatter(attn_x, attn_y, 
                          color=color, 
                          marker='o', 
                          s=80, 
                          zorder=3)

            if mlp_points:
                mlp_x = [p[0] for p in mlp_points]
                mlp_y = [p[1] for p in mlp_points]
                ax.scatter(mlp_x, mlp_y, 
                          color=color, 
                          marker='*', 
                          s=200, 
                          zorder=3)
        else:
            ax.plot(drop_nums, accuracies, 
                    color=color, 
                    marker='o', 
                    linestyle='-', 
                    linewidth=2, 
                    markersize=8,
                    label=model)

        if baseline_acc is not None:
            ax.axhline(y=baseline_acc, 
                      color=color, 
                      linestyle='--', 
                      linewidth=1.5, 
                      alpha=0.7,
                      dashes=(5, 3))

    ax.set_xlabel(get_xlabel_from_prune_method(prune_method), fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title(prettify_dataset_name(dataset), fontsize=14)

    ax.set_xlim(-1, 41)
    ax.set_xticks(range(0, 45, 5))

    ax.set_ylim(-5, 105)
    ax.set_yticks(range(0, 101, 10))

    ax.grid(True, color='#d4af37', alpha=0.4, linewidth=1, linestyle=':')

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    filename = f"{dataset}_{prune_method}.png"
    plt.savefig(output_path / filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {output_path / filename}")


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize benchmark results")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    parser.add_argument("--prune_method", type=str, required=True, help="Pruning method")
    parser.add_argument("--model_names", nargs='+', required=True, help="List of model names")
    parser.add_argument("--drop_nums", nargs='+', type=int, required=True, help="List of drop numbers")
    parser.add_argument("--results_dir", type=str, default="visualization/benchmark_vm_results", help="Directory with .out files")
    parser.add_argument("--output_dir", type=str, default="visualization/results", help="Output directory for plots")
    parser.add_argument("--legend_name", type=str, default="legend.png", help="Filename for the legend image")
    return parser.parse_args()


def main():
    args = parse_args()
    
    print(f"\nGenerating plot: {args.dataset} - {args.prune_method}")
    print("=" * 60)
    print("\nGenerating legend image...")
    legend_models = args.model_names if args.legend_name != "legend.png" else None
    create_legend_image(args.output_dir, model_names=legend_models, filename=args.legend_name)

    data = collect_data(args.dataset, args.prune_method, args.model_names, args.drop_nums, args.results_dir)
    create_plot(data, args.dataset, args.prune_method, args.output_dir)
    print()


if __name__ == "__main__":
    main()
