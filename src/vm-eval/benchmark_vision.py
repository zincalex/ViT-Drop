import os
import sys
import json
import h5py
import torch
import random
import argparse
import numpy as np

from pathlib import Path

SEED = 42


def set_seed(seed: int = SEED):
    """Seed all RNGs so head initialization and data shuffling are reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tqdm import tqdm
from accelerate import Accelerator
from transformers import AutoConfig, AutoImageProcessor, AutoModel, AutoModelForImageClassification
from loader import create_dataloader, get_num_classes, detect_format
from metrics import calculate_metrics, gather_predictions_across_gpus
from utils import check_dataset_formatting
from head_finetuner import finetune_head, DINOv3ForImageClassification


def get_dataset_num_classes(dataset_name: str, dataset_base_dir: str) -> int:
    dataset_dir = Path(dataset_base_dir) / dataset_name
    return get_num_classes(dataset_dir)


def parse_args():
    parser = argparse.ArgumentParser(description="Vision Model Benchmark")
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--batch_size_eval", type=int, default=32, help="Batch size for evaluation")
    parser.add_argument("--prune_method", type=str, default=None, help="Pruning method used (e.g., layer_drop_attn)")
    parser.add_argument("--drop_num", type=str, default=None, help="Number of layers dropped during pruning")
    parser.add_argument("--finetune_head", action="store_true")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs for head fine-tuning")
    parser.add_argument("--batch_size", type=int, default=30, help="Batch size for head fine-tuning training")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate for head fine-tuning")
    parser.add_argument("--weight_decay", type=float, default=0.03, help="Weight decay for head fine-tuning")
    parser.add_argument("--output_file", type=str, default=None, help="Where the benchmark results are stored")
    parser.add_argument("--dataset_base_dir", type=str, default="data")

    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(SEED)
    accelerator = Accelerator()

    dataset_dir = Path(args.dataset_base_dir) / args.dataset
    dataset_fmt = detect_format(dataset_dir)

    # STEP 1: Check dataset formatting
    if dataset_fmt == "json":
        if not check_dataset_formatting(args.dataset, args.dataset_base_dir, accelerator):
            if accelerator.is_main_process:
                print("✗ Dataset check failed. Exiting.")
                print("=" * 80)
            sys.exit(1)
    
    accelerator.print(f"Accelerator state: {accelerator.state}")
    accelerator.print("=" * 80)
    accelerator.print("Vision Model Benchmark")
    accelerator.print("=" * 80)
    accelerator.print(f"Model: {args.model_name_or_path}")
    accelerator.print(f"Dataset: {args.dataset}")
    accelerator.print(f"Dataset format: {dataset_fmt}")
    accelerator.print(f"Evaluation batch size: {args.batch_size_eval}")
    accelerator.print(f"Num processes: {accelerator.num_processes}")
    accelerator.print("=" * 80)

    config_kwargs = {"trust_remote_code": True, "cache_dir": None, "revision": "main", "token": None, "attn_implementation": "eager"}

    processor = AutoImageProcessor.from_pretrained(args.model_name_or_path, **config_kwargs)
    config = AutoConfig.from_pretrained(args.model_name_or_path, **config_kwargs)
    config.use_cache = False
    model_type = getattr(config, "model_type", None)
    auto_cls = AutoModel if model_type == "dinov3_vit" else AutoModelForImageClassification

    model = auto_cls.from_pretrained(args.model_name_or_path, config=config, torch_dtype=torch.float32,
                                     low_cpu_mem_usage=True, ignore_mismatched_sizes=True, **config_kwargs)
    # Wrap DINOv3 backbone with classification head
    if model_type == "dinov3_vit":
        dataset_num_classes_init = get_dataset_num_classes(args.dataset, args.dataset_base_dir)
        model = DINOv3ForImageClassification(model, num_classes=dataset_num_classes_init)

    num_params = sum(p.numel() for p in model.parameters())

    accelerator.print("\n" + "=" * 80)
    accelerator.print("Model Information:")
    accelerator.print("=" * 80)
    accelerator.print(f"Model type: {type(model).__name__}")
    accelerator.print(f"Total parameters: {num_params:,}")
    if hasattr(model.config, 'num_labels'): accelerator.print(f"Number of classes: {model.config.num_labels}")
    if hasattr(model.config, 'image_size'): accelerator.print(f"Image size: {model.config.image_size}")
    if hasattr(model.config, 'num_hidden_layers'): accelerator.print(f"Number of layers: {model.config.num_hidden_layers}")
    if hasattr(model.config, 'hidden_size'): accelerator.print(f"Hidden size: {model.config.hidden_size}")
    accelerator.print("=" * 80)


    # STEP 2: Head Fine-tuning
    dataset_num_classes = get_dataset_num_classes(args.dataset, args.dataset_base_dir)
    model_num_classes = model.config.num_labels if hasattr(model.config, 'num_labels') else None
    is_imagenet_dataset = "imagenet" in args.dataset.lower()
    
    # Decision logic
    classes_match = (model_num_classes == dataset_num_classes) if model_num_classes is not None else False
    skip_finetuning = is_imagenet_dataset and classes_match
    if skip_finetuning:
        model = accelerator.prepare(model)
    else :
        model = finetune_head(model=model, accelerator=accelerator, num_workers=1,
                              dataset_name=args.dataset, dataset_base_dir=args.dataset_base_dir, processor=processor,
                              batch_size=args.batch_size, num_epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay)


    # STEP 3: Evaluation
    accelerator.print("\n" + "=" * 80)
    accelerator.print("Evaluation")
    accelerator.print("=" * 80)

    layer_execution_count = {}
    attention_execution_count = {}
    mlp_execution_count = {}
    def layer_execution_hook(name):
        def hook(module, input, output):
            layer_execution_count[name] = layer_execution_count.get(name, 0) + 1
        return hook
    def attention_execution_hook(name):
        def hook(module, input, output):
            attention_execution_count[name] = attention_execution_count.get(name, 0) + 1
        return hook
    def mlp_execution_hook(name):
        def hook(module, input, output):
            mlp_execution_count[name] = mlp_execution_count.get(name, 0) + 1
        return hook

    hooks_registered = 0
    unwrapped_model = accelerator.unwrap_model(model)
    base_prefix = getattr(unwrapped_model, 'base_model_prefix', None)
    base_model = getattr(unwrapped_model, base_prefix, None) if base_prefix else None

    if base_model is not None:
        # Get flat list of layers for any vision model
        if base_prefix == "swinv2":
            layers = [block for stage in base_model.encoder.layers for block in stage.blocks]
        elif base_prefix == "dinov3_vit":
            layers = base_model.layer
        else:
            layers = base_model.encoder.layer

        for i, layer in enumerate(layers):
            layer.register_forward_hook(layer_execution_hook(f"layer_{i}"))

            attn = getattr(layer, 'attention', None)
            if attn is not None:
                attn.register_forward_hook(attention_execution_hook(f"layer_{i}_attn"))

            mlp = getattr(layer, 'mlp', None) or getattr(layer, 'intermediate', None) or getattr(layer, 'output', None)
            if mlp is not None:
                mlp.register_forward_hook(mlp_execution_hook(f"layer_{i}_mlp"))

            hooks_registered += 1
    else:
        accelerator.print(f"WARNING: Could not find base model attribute (base_model_prefix={base_prefix})")
        accelerator.print(f"Model type: {type(unwrapped_model)}")
        if hasattr(unwrapped_model, 'encoder') and hasattr(unwrapped_model.encoder, 'layer'):
            for i, layer in enumerate(unwrapped_model.encoder.layer):
                layer.register_forward_hook(layer_execution_hook(f"layer_{i}"))
                hooks_registered += 1
    
    accelerator.print(f"Registered execution hooks on {hooks_registered} layers")


    dataloader = create_dataloader(dataset_name=args.dataset, base_dir=args.dataset_base_dir, processor=processor,
                           batch_size=args.batch_size_eval, num_workers=4 * accelerator.num_processes, split="test")

    dataloader = accelerator.prepare(dataloader)
    model.eval()

    all_predictions = []
    all_labels = []
    all_logits = []
    all_indices = []
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Evaluating") if accelerator.is_main_process else dataloader
        for batch_idx, batch in enumerate(pbar):
            pixel_values = batch["pixel_values"]
            labels = batch["labels"]

            batch_size = pixel_values.shape[0]
            batch_indices = torch.arange(batch_idx * args.batch_size_eval, batch_idx * args.batch_size_eval + batch_size,
                                        device=pixel_values.device)

            outputs = model(pixel_values)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            predictions, labels, logits, batch_indices = accelerator.gather_for_metrics(
                (predictions, labels, logits, batch_indices))
            all_predictions.append(predictions)
            all_labels.append(labels)
            all_logits.append(logits)
            all_indices.append(batch_indices)

    all_predictions = torch.cat(all_predictions).cpu().tolist()
    all_labels = torch.cat(all_labels).cpu().tolist()
    all_logits = torch.cat(all_logits)
    all_indices = torch.cat(all_indices)
    
    if accelerator.is_main_process:
        metrics = calculate_metrics(all_predictions, all_labels)

        logits_dir = Path("results/logits") / args.dataset
        logits_dir.mkdir(parents=True, exist_ok=True)
        model_name = Path(args.model_name_or_path).name
        prune_str = f"{args.prune_method}_drop{args.drop_num}" if args.prune_method and args.drop_num else "original"
        logits_filename = f"{model_name}_{args.dataset}_{prune_str}.h5"
        logits_path = logits_dir / logits_filename

        predictions_np = np.array(all_predictions)
        labels_np = np.array(all_labels)
        logits_np = all_logits.cpu().numpy()
        indices_np = all_indices.cpu().numpy()
        
        with h5py.File(logits_path, 'w') as f:
            f.create_dataset('logits', data=logits_np, compression='gzip', compression_opts=4)
            f.create_dataset('predictions', data=predictions_np, compression='gzip', compression_opts=4)
            f.create_dataset('labels', data=labels_np, compression='gzip', compression_opts=4)
            f.create_dataset('image_indices', data=indices_np, compression='gzip', compression_opts=4)
        
        accelerator.print(f"Logits saved: {logits_path}")
        accelerator.print("\n" + "=" * 80)
        accelerator.print("LAYER EXECUTION VERIFICATION")
        accelerator.print("=" * 80)
        
        if not layer_execution_count:
            accelerator.print("WARNING: No layer execution data collected - hooks may not have been registered properly")

        layer_keys = sorted([k for k in layer_execution_count.keys() if k.startswith('layer_')], key=lambda x: int(x.split('_')[1]))
        for layer_key in layer_keys:
            layer_idx = int(layer_key.split('_')[1])
            attn_key = f"layer_{layer_idx}_attn"
            mlp_key = f"layer_{layer_idx}_mlp"
            layer_count = layer_execution_count.get(layer_key, 0)
            attn_count = attention_execution_count.get(attn_key, 0)
            mlp_count = mlp_execution_count.get(mlp_key, 0)
            attn_status = "✓" if attn_count > 0 else "✗"
            mlp_status = "✓" if mlp_count > 0 else "✗"
            accelerator.print(f"Layer {layer_idx:2d}: {layer_count:4d} passes | Attention: {attn_status} ({attn_count:4d}) | MLP: {mlp_status} ({mlp_count:4d})")

        if args.output_file:
            finetuning_status = "SKIPPED" if skip_finetuning else "YES"
            summary_text = f"""
                RESULTS
                Model: {args.model_name_or_path}
                Dataset: {args.dataset}
                Dataset classes: {dataset_num_classes}
                Model classes: {model_num_classes}
                Head fine-tuning: {finetuning_status}
                Pruning method: {args.prune_method}
                Layers dropped: {args.drop_num}
                Total samples: {metrics['total_samples']}
                Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy'] * 100:.2f}%)
                Precision: {metrics['precision']:.4f}
                Recall: {metrics['recall']:.4f}
                F1 Score: {metrics['f1']:.4f}
                {"=" * 80}
                """
            with open(args.output_file, "w") as f:
                f.write(summary_text)
            accelerator.print(f"\nResults saved to: {args.output_file}")

    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
