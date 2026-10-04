import json
import torch
import torch.nn as nn
from pathlib import Path

from tqdm import tqdm
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers.modeling_outputs import ImageClassifierOutput

from loader import create_dataloader, get_num_classes


class DINOv3ForImageClassification(nn.Module):
    def __init__(self, backbone, num_classes: int):
        super().__init__()
        self.dinov3_vit = backbone
        self.config = backbone.config
        self.config.num_labels = num_classes
        self.base_model_prefix = "dinov3_vit"
        hidden_size = backbone.config.hidden_size
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, pixel_values, labels=None, **kwargs):
        outputs = self.dinov3_vit(pixel_values, **kwargs)
        pooled = outputs.pooler_output
        logits = self.classifier(pooled)

        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(logits, labels)

        return ImageClassifierOutput(loss=loss, logits=logits)


def count_unique_labels(json_path: Path) -> int:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    labels = set()
    for item in data:
        labels.add(item["label"])

    return len(labels)


def get_model_hidden_size(model) -> int:
    config = model.config

    if hasattr(config, 'hidden_size'):
        return config.hidden_size
    elif hasattr(config, 'embed_dim'):
        return config.embed_dim
    elif hasattr(config, 'hidden_sizes'):
        return config.hidden_sizes[-1]
    else:
        raise ValueError(f"Could not determine hidden size for model type: {type(model).__name__}")


def replace_classification_head(model, accelerator, num_classes: int, dtype=torch.bfloat16):
    hidden_size = get_model_hidden_size(model)

    if hasattr(model, 'classifier'):
        if hasattr(model, 'dinov2') or 'dinov2' in model.__class__.__name__.lower():
            classifier_input_size = hidden_size * 2
        else:
            classifier_input_size = hidden_size
            
        model.classifier = nn.Linear(classifier_input_size, num_classes, dtype=dtype)
    else:
        raise ValueError(f"Model type {type(model).__name__} does not have a 'classifier' attribute")

    model.config.num_labels = num_classes
    return model


def freeze_backbone(model, accelerator) -> None:
    frozen_params = 0
    trainable_params = 0
    for name, param in model.named_parameters():
        if 'classifier' in name:
            param.requires_grad = True
            trainable_params += param.numel()
        else:
            param.requires_grad = False
            frozen_params += param.numel()
    
    accelerator.print(f"Backbone frozen")


def train_epoch(model, dataloader, optimizer, scheduler, accelerator, epoch: int, loss_fn) -> float:
    total_loss = 0.0
    num_batches = 0
    pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}") if accelerator.is_main_process else dataloader
    for batch in pbar:
        pixel_values = batch["pixel_values"]
        labels = batch["labels"]

        outputs = model(pixel_values)
        logits = outputs.logits

        loss = loss_fn(logits, labels)

        accelerator.backward(loss)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        total_loss += loss.item()
        num_batches += 1
        
        if accelerator.is_main_process and isinstance(pbar, tqdm):
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{scheduler.get_last_lr()[0]:.6f}'
            })
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


def finetune_head(model, accelerator, num_workers: int, dataset_name: str, dataset_base_dir: str, processor, batch_size: int = 30,
                  num_epochs: int = 20, lr: float = 0.001, weight_decay: float = 0.03):
    
    accelerator.print("\n" + "=" * 80)
    accelerator.print("Head Fine-Tuning")
    accelerator.print("=" * 80)

    dataset_dir = Path(dataset_base_dir) / dataset_name

    num_classes = get_num_classes(dataset_dir)
    
    # Replace classification head
    current_num_labels = model.config.num_labels if hasattr(model.config, 'num_labels') else None
    accelerator.print(f"Model has {current_num_labels} classes, dataset has {num_classes} classes")
    if current_num_labels != num_classes:
        accelerator.print("Replacing classification head...")
        model_dtype = next(model.parameters()).dtype
        model = replace_classification_head(model, accelerator, num_classes, dtype=model_dtype)

    freeze_backbone(model, accelerator)
    model.train()
    model = accelerator.prepare(model)

    dataloader = create_dataloader(dataset_name=dataset_name, base_dir=dataset_base_dir, processor=processor,
                            batch_size=batch_size, num_workers=num_workers, split="train")
    train_dataloader = accelerator.prepare(dataloader)
    steps_per_epoch = len(train_dataloader)
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs * steps_per_epoch, eta_min=0)
    optimizer, scheduler = accelerator.prepare(optimizer, scheduler)
    loss_fn = nn.CrossEntropyLoss()
    
    accelerator.print(f"Training configuration:")
    accelerator.print(f"  Epochs: {num_epochs}")
    accelerator.print(f"  Batch size: {batch_size}")
    accelerator.print(f"  Steps per epoch: {steps_per_epoch}")
    accelerator.print(f"  Optimizer: {type(optimizer).__name__} (lr={lr}, weight_decay={weight_decay})")
    accelerator.print(f"  Scheduler: {type(scheduler).__name__}")
    accelerator.print(f"  Loss function: {type(loss_fn).__name__}")


    # Train loop
    for epoch in range(num_epochs):
        avg_loss = train_epoch(
            model=model,
            dataloader=train_dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            accelerator=accelerator,
            epoch=epoch,
            loss_fn=loss_fn
        )
        
        if accelerator.is_main_process:
            accelerator.print(f"Epoch {epoch + 1}/{num_epochs} - Average Loss: {avg_loss:.4f}")

    accelerator.print("=" * 80)
    accelerator.wait_for_everyone()
    return model
