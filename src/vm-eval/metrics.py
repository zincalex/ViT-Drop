import torch
import numpy as np

from typing import Dict, List
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def calculate_metrics(predictions: List[int], labels: List[int]) -> Dict[str, float]:
    predictions = np.array(predictions)
    labels = np.array(labels)
    accuracy = accuracy_score(labels, predictions)

    precision = precision_score(labels, predictions, average='macro', zero_division=0)
    recall = recall_score(labels, predictions, average='macro', zero_division=0)
    f1 = f1_score(labels, predictions, average='macro', zero_division=0)

    total = len(labels)
    correct = int((predictions == labels).sum())

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "total_samples": int(total),
        "correct_predictions": int(correct),
    }


def gather_predictions_across_gpus(predictions: torch.Tensor, labels: torch.Tensor, accelerator) -> tuple:
    all_predictions = accelerator.gather(predictions)
    all_labels = accelerator.gather(labels)

    return all_predictions.cpu().tolist(), all_labels.cpu().tolist()
