import shutil
import torch
from pathlib import Path
from transformers import (
    AutoConfig, AutoImageProcessor, AutoModel, AutoModelForImageClassification,
)

MODEL_HF_PATHS = {
    "dinov2": "facebook/dinov2-giant-imagenet1k-1-layer",
    "dinov3_vit": "facebook/dinov3-vitl16-pretrain-lvd1689m",
    "swinv2": "microsoft/swinv2-base-patch4-window16-256",
    "vit": "google/vit-base-patch16-224",
}
CONFIG_KWARGS = {
    "trust_remote_code": True, "cache_dir": None,
    "revision": "main", "token": None, "attn_implementation": "eager",
}


def get_config_path(prune_dir, arch, method, drop_n):
    return Path(prune_dir) / f"{arch}-{method}-discrete-drop{drop_n}" / "checkpoint" / "config.json"


def get_checkpoint_dir(prune_dir, arch, method, drop_n):
    return Path(prune_dir) / f"{arch}-{method}-discrete-drop{drop_n}" / "checkpoint"


def config_exists(prune_dir, arch, method, drop_n):
    return get_config_path(prune_dir, arch, method, drop_n).exists()


def setup_model_dir(model_dir, prune_dir, arch, method, drop_n):
    for py in get_checkpoint_dir(prune_dir, arch, method, drop_n).glob("*.py"):
        shutil.copy2(str(py), str(Path(model_dir) / py.name))


def swap_config(model_dir, prune_dir, arch, method, drop_n):
    cfg = get_config_path(prune_dir, arch, method, drop_n)
    if not cfg.exists():
        raise FileNotFoundError(f"Config not found: {cfg}")
    shutil.copy2(str(cfg), str(Path(model_dir) / "config.json"))


def cache_base_weights(model_dir):
    config = AutoConfig.from_pretrained(model_dir, **CONFIG_KWARGS)
    model_type = getattr(config, "model_type", None)
    auto_cls = AutoModel if model_type == "dinov3_vit" else AutoModelForImageClassification
    model = auto_cls.from_pretrained(
        model_dir, dtype=torch.float32, low_cpu_mem_usage=True,
        ignore_mismatched_sizes=True, **CONFIG_KWARGS
    )
    state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    del model
    torch.cuda.empty_cache()
    return state_dict

def load_model_and_processor(model_dir, cached_state_dict=None):
    processor = AutoImageProcessor.from_pretrained(model_dir, **CONFIG_KWARGS)
    config = AutoConfig.from_pretrained(model_dir, **CONFIG_KWARGS)
    config.use_cache = False
    model_type = getattr(config, "model_type", None)
    auto_cls = AutoModel if model_type == "dinov3_vit" else AutoModelForImageClassification
    if cached_state_dict is not None:
        model = auto_cls.from_config(
            config, dtype=torch.float32,
            trust_remote_code=True, attn_implementation="eager",
        )
        model_state = model.state_dict()
        filtered = {
            k: v for k, v in cached_state_dict.items()
            if k in model_state and v.shape == model_state[k].shape
        }
        model.load_state_dict(filtered, strict=False)
    else:
        model = auto_cls.from_pretrained(
            model_dir, config=config, dtype=torch.float32,
            low_cpu_mem_usage=True, ignore_mismatched_sizes=True, **CONFIG_KWARGS
        )
    return model, processor
