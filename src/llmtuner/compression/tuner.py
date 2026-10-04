from typing import Any, Dict, List, Optional, TYPE_CHECKING

from llmtuner.compression.prune import run_prune
from ..extras.callbacks import LogCallback
from ..extras.logging import get_logger
from ..hparams import get_train_sparse_args

if TYPE_CHECKING:
    from transformers import TrainerCallback

logger = get_logger(__name__)


def run_exp(args: Optional[Dict[str, Any]] = None, callbacks: Optional[List["TrainerCallback"]] = None):
    model_args, data_args, training_args, finetuning_args, pruning_args = get_train_sparse_args(args)
    callbacks = [LogCallback()] if callbacks is None else callbacks
    if finetuning_args.stage == "prune":
        run_prune(model_args, data_args, training_args, finetuning_args, pruning_args, callbacks)


if __name__ == "__main__":
    run_exp()
