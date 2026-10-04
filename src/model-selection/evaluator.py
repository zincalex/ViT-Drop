import sys
import os
import h5py
import torch
import torch.nn as nn
import numpy as np

from pathlib import Path
from tqdm import tqdm
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from torch.utils.data import DataLoader, ConcatDataset

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vm-eval"))

from loader import create_dataloader, get_num_classes, detect_format, VisionDatasetH5, VisionDatasetJSON, set_seed
from metrics import calculate_metrics
from head_finetuner import replace_classification_head, freeze_backbone, DINOv3ForImageClassification


def cosine_with_warmup(optimizer, warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))
    return LambdaLR(optimizer, lr_lambda)


def prepare_model(model, accelerator, dataset_dir):
    num_classes = get_num_classes(dataset_dir)
    model_type = getattr(model.config, "model_type", None)

    if model_type == "dinov3_vit" and not hasattr(model, "classifier"):
        torch.manual_seed(42)
        model = DINOv3ForImageClassification(model, num_classes)
    else:
        current = model.config.num_labels if hasattr(model.config, "num_labels") else None
        if current != num_classes:
            model_dtype = next(model.parameters()).dtype
            torch.manual_seed(42)
            model = replace_classification_head(model, accelerator, num_classes, dtype=model_dtype)

    orig_print = accelerator.print
    accelerator.print = lambda *a, **k: None
    freeze_backbone(model, accelerator)
    accelerator.print = orig_print
    return model


def train_head(model, dataloader, accelerator, epochs, lr, weight_decay, warmup_ratio=0.0):
    model.train()
    total_steps = epochs * len(dataloader)
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)
    if warmup_ratio > 0:
        scheduler = cosine_with_warmup(optimizer, int(total_steps * warmup_ratio), total_steps)
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=0)
    optimizer, scheduler = accelerator.prepare(optimizer, scheduler)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}") if accelerator.is_main_process else dataloader
        for batch in pbar:
            outputs = model(batch["pixel_values"])
            loss = loss_fn(outputs.logits, batch["labels"])
            accelerator.backward(loss)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            if accelerator.is_main_process and isinstance(pbar, tqdm):
                pbar.set_postfix(loss=f"{loss.item():.4f}")
    return model


def evaluate(model, dataloader, accelerator, return_logits=False):
    model.eval()
    all_preds, all_labels, all_logits = [], [], []
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Evaluating", disable=not accelerator.is_main_process)
        for batch in pbar:
            outputs = model(batch["pixel_values"])
            logits = outputs.logits
            preds, labels = accelerator.gather_for_metrics(
                (torch.argmax(logits, dim=-1), batch["labels"]))
            all_preds.append(preds)
            all_labels.append(labels)
            if return_logits:
                all_logits.append(accelerator.gather_for_metrics(logits))
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    metrics = {}
    if accelerator.is_main_process:
        metrics = calculate_metrics(all_preds.cpu(), all_labels.cpu())
    if return_logits:
        all_logits = torch.cat(all_logits)
    accelerator.wait_for_everyone()
    return (metrics, all_logits if return_logits else None) if return_logits else metrics


def finetune_and_evaluate(model, processor, accelerator, dataset_name, dataset_base_dir,
                          epochs, lr, weight_decay, batch_size, batch_size_eval,
                          num_workers, train_split="train", eval_split="val", warmup_ratio=0.0):
    set_seed(42)
    dataset_dir = Path(dataset_base_dir) / dataset_name
    model = prepare_model(model, accelerator, dataset_dir)
    model = accelerator.prepare(model)

    train_loader = create_dataloader(
        dataset_name=dataset_name, base_dir=dataset_base_dir, processor=processor,
        batch_size=batch_size, num_workers=num_workers, split=train_split
    )
    train_loader = accelerator.prepare(train_loader)
    model = train_head(model, train_loader, accelerator, epochs, lr, weight_decay, warmup_ratio)

    eval_loader = create_dataloader(
        dataset_name=dataset_name, base_dir=dataset_base_dir, processor=processor,
        batch_size=batch_size_eval, num_workers=num_workers, split=eval_split
    )
    eval_loader = accelerator.prepare(eval_loader)
    metrics = evaluate(model, eval_loader, accelerator)
    # Explicitly shutdown dataloader workers to avoid file descriptor leaks
    if hasattr(train_loader, '_iterator') and train_loader._iterator is not None:
        train_loader._iterator._shutdown_workers()
    if hasattr(eval_loader, '_iterator') and eval_loader._iterator is not None:
        eval_loader._iterator._shutdown_workers()
    accelerator.free_memory()
    del model, train_loader, eval_loader
    torch.cuda.empty_cache()
    return metrics


def deep_finetune_and_evaluate(model, processor, accelerator, dataset_name, dataset_base_dir,
                               epochs, lr, weight_decay, batch_size, batch_size_eval,
                               num_workers, warmup_ratio=0.1, output_dir="results_selection"):
    set_seed(42)
    dataset_dir = Path(dataset_base_dir) / dataset_name
    model = prepare_model(model, accelerator, dataset_dir)
    model = accelerator.prepare(model)

    fmt = detect_format(dataset_dir)
    if fmt == "h5":
        train_ds = VisionDatasetH5(h5_path=str(dataset_dir / "train.h5"), processor=processor, split="train")
        val_ds = VisionDatasetH5(h5_path=str(dataset_dir / "val.h5"), processor=processor, split="val")
    else:
        raise ValueError(f"Deep finetune requires H5 format, got {fmt}")
    g = torch.Generator()
    g.manual_seed(42)
    combined_loader = DataLoader(
        ConcatDataset([train_ds, val_ds]), batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=False, generator=g
    )
    combined_loader = accelerator.prepare(combined_loader)
    model = train_head(model, combined_loader, accelerator, epochs, lr, weight_decay, warmup_ratio)

    if accelerator.is_main_process:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        unwrapped = accelerator.unwrap_model(model)
        head_state = {k: v.cpu().clone() for k, v in unwrapped.state_dict().items() if "classifier" in k}
        torch.save(head_state, output_path / f"best_head_{dataset_name}.pt")
        accelerator.print(f"  Head saved to {output_path}/best_head_{dataset_name}.pt")

    accelerator.wait_for_everyone()

    test_loader = create_dataloader(
        dataset_name=dataset_name, base_dir=dataset_base_dir, processor=processor,
        batch_size=batch_size_eval, num_workers=num_workers, split="test"
    )
    test_loader = accelerator.prepare(test_loader)
    test_metrics, test_logits = evaluate(model, test_loader, accelerator, return_logits=True)

    if accelerator.is_main_process and test_logits is not None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        logits_path = output_path / f"logits_{dataset_name}.h5"
        preds = torch.argmax(test_logits, dim=-1).cpu().numpy()
        with h5py.File(logits_path, "w") as f:
            f.create_dataset("logits", data=test_logits.cpu().numpy(), compression="gzip", compression_opts=4)
            f.create_dataset("predictions", data=preds, compression="gzip", compression_opts=4)
        accelerator.print(f"  Logits saved to {logits_path}")

    accelerator.free_memory()
    del model, combined_loader, test_loader
    torch.cuda.empty_cache()
    return test_metrics
