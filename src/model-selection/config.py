import argparse
from dataclasses import dataclass, field
from typing import List


# Architecture name -> max layers
ARCH_MAX_LAYERS = {
    "dinov2": 40,
    "dinov3_vit": 24,
    "swinv2": 24,
    "vit": 12,
}

# All supported pruning methods
ALL_PRUNE_METHODS = ["block_drop", "layer_drop_attn", "layer_drop_mlp", "layer_drop_all"]


@dataclass
class BaselineConfig:
    """Hyperparameters for baseline (drop=0) fine-tuning."""
    epochs: int = 5
    lr: float = 0.001
    weight_decay: float = 0.03
    batch_size: int = 32


@dataclass
class SearchConfig:
    """Hyperparameters for quick search-phase fine-tuning."""
    epochs: int = 5
    lr: float = 0.001
    weight_decay: float = 0.03
    batch_size: int = 32


@dataclass
class DeepFinetuneConfig:
    """Hyperparameters for deep fine-tuning of the winner."""
    epochs: int = 20
    lr: float = 0.001
    weight_decay: float = 0.03
    batch_size: int = 32
    warmup_ratio: float = 0.1


@dataclass
class PipelineConfig:
    """Top-level pipeline configuration."""
    dataset: str = ""
    dataset_base_dir: str = "data"
    results_prune_dir: str = "results_prune"
    output_dir: str = "results_selection"
    architectures: List[str] = field(default_factory=lambda: list(ARCH_MAX_LAYERS.keys()))
    prune_methods: List[str] = field(default_factory=lambda: list(ALL_PRUNE_METHODS))
    early_stop_threshold: float = 0.05  # 5% absolute accuracy drop
    drop_step: int = 4
    batch_size_eval: int = 32
    num_workers: int = 4
    gpus: str = "0,1,2,3"
    port: str = "29500"

    baseline: BaselineConfig = field(default_factory=BaselineConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    deep_finetune: DeepFinetuneConfig = field(default_factory=DeepFinetuneConfig)


def get_drop_counts(arch: str, step: int = 4) -> List[int]:
    """Return drop counts (multiples of step) for an architecture."""
    max_layers = ARCH_MAX_LAYERS.get(arch)
    if max_layers is None:
        raise ValueError(f"Unknown architecture: {arch}")
    return list(range(step, max_layers + 1, step))


def parse_args() -> PipelineConfig:
    parser = argparse.ArgumentParser(description="Automated Model Selection Pipeline")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--dataset_base_dir", type=str, default="data")
    parser.add_argument("--results_prune_dir", type=str, default="results_prune")
    parser.add_argument("--output_dir", type=str, default="results_selection")
    parser.add_argument("--architectures", type=str, nargs="+", default=list(ARCH_MAX_LAYERS.keys()))
    parser.add_argument("--prune_methods", type=str, nargs="+", default=list(ALL_PRUNE_METHODS))
    parser.add_argument("--early_stop_threshold", type=float, default=0.05)
    parser.add_argument("--drop_step", type=int, default=4)
    parser.add_argument("--batch_size_eval", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--gpus", type=str, default="0,1,2,3")
    parser.add_argument("--port", type=str, default="29500")

    parser.add_argument("--baseline_epochs", type=int, default=5)
    parser.add_argument("--baseline_lr", type=float, default=0.001)
    parser.add_argument("--baseline_batch_size", type=int, default=32)

    parser.add_argument("--search_epochs", type=int, default=5)
    parser.add_argument("--search_lr", type=float, default=0.001)
    parser.add_argument("--search_batch_size", type=int, default=32)

    parser.add_argument("--deep_epochs", type=int, default=20)
    parser.add_argument("--deep_lr", type=float, default=0.001)
    parser.add_argument("--deep_batch_size", type=int, default=32)
    parser.add_argument("--deep_warmup_ratio", type=float, default=0.1)

    args = parser.parse_args()

    cfg = PipelineConfig(
        dataset=args.dataset,
        dataset_base_dir=args.dataset_base_dir,
        results_prune_dir=args.results_prune_dir,
        output_dir=args.output_dir,
        architectures=args.architectures,
        prune_methods=args.prune_methods,
        early_stop_threshold=args.early_stop_threshold,
        drop_step=args.drop_step,
        batch_size_eval=args.batch_size_eval,
        num_workers=args.num_workers,
        gpus=args.gpus,
        port=args.port,
        baseline=BaselineConfig(epochs=args.baseline_epochs, lr=args.baseline_lr, batch_size=args.baseline_batch_size),
        search=SearchConfig(epochs=args.search_epochs, lr=args.search_lr, batch_size=args.search_batch_size),
        deep_finetune=DeepFinetuneConfig(epochs=args.deep_epochs, lr=args.deep_lr, batch_size=args.deep_batch_size, warmup_ratio=args.deep_warmup_ratio),
    )
    return cfg
