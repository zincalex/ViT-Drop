import os
import json
import hashlib
import h5py
import torch
import random
import numpy as np

from tqdm import tqdm
from PIL import Image
from typing import Dict
from pathlib import Path
from torch.utils.data import Dataset, DataLoader


SEED = 42


def processor_fingerprint(processor) -> str:
    cfg = {k: v for k, v in processor.to_dict().items() if k not in ("processor_class", "_processor_class")}
    cfg["__class__"] = type(processor).__name__
    return hashlib.sha1(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:10]


def cache_enabled() -> bool:
    return os.environ.get("VISION_DROP_CACHE", "1") != "0"


def set_seed(seed: int = SEED):
    """Seed all RNGs so head init and data shuffling are reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



# Multi-process helpers
def is_main_process():
    import torch.distributed as dist
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank() == 0
    return True


def wait_for_cache(cache_dir, poll_interval=2.0):
    import time
    done_flag = Path(cache_dir) / ".cache_done"
    while not done_flag.exists():
        time.sleep(poll_interval)


def detect_format(dataset_dir: Path) -> str:
    if (dataset_dir / "train.h5").exists() or (dataset_dir / "test.h5").exists():
        return "h5"
    for _ in dataset_dir.glob("*_demo.json"):
        return "json"
    raise FileNotFoundError(f"No supported dataset in {dataset_dir}")


def get_num_classes(dataset_dir: Path) -> int:
    fmt = detect_format(dataset_dir)
    if fmt == "h5":
        all_labels = set()
        for name in ["train.h5", "val.h5", "test.h5"]:
            p = dataset_dir / name
            if p.exists():
                with h5py.File(p, "r") as f:
                    all_labels.update(np.unique(f["labels"][:]).tolist())
        if all_labels:
            return len(all_labels)
    elif fmt == "json":
        for p in dataset_dir.glob("*_demo.json"):
            with open(p, "r") as f:
                data = json.load(f)
            return len(set(item["label"] for item in data))
    raise FileNotFoundError(f"Could not determine num_classes in {dataset_dir}")


# Shared caching logic for all dataset formats
class CachedDatasetMixin:
    def setup_cache(self, cache_dir: Path, num_samples: int, use_cache: bool):
        use_cache = use_cache and cache_enabled()
        self.cache_dir = cache_dir
        self.num_samples_cache = num_samples
        self.use_cache = use_cache
        if use_cache:
            if is_main_process():
                self.build_cache_if_needed()
            else:
                wait_for_cache(self.cache_dir)

    def build_cache_if_needed(self):
        done_flag = self.cache_dir / ".cache_done"
        if done_flag.exists():
            return
        needed = False
        if not self.cache_dir.exists():
            needed = True
        else:
            existing = list(self.cache_dir.glob("*.pt"))
            if len(existing) != self.num_samples_cache:
                needed = True
        if needed:
            self.build_cache()
        done_flag.touch()

    def load_cached(self, idx: int):
        if not self.use_cache:
            return None
        cache_path = self.cache_dir / f"{idx}.pt"
        if cache_path.exists():
            try:
                return torch.load(cache_path, weights_only=True)
            except Exception as e:
                print(f"Warning: cache load failed idx {idx}: {e}")
                cache_path.unlink()
        return None

    def save_to_cache(self, idx: int, result: Dict[str, torch.Tensor]):
        if not self.use_cache:
            return
        cache_path = self.cache_dir / f"{idx}.pt"
        try:
            torch.save(result, cache_path)
        except Exception as e:
            print(f"Warning: cache save failed idx {idx}: {e}")



# H5 dataset
class VisionDatasetH5(CachedDatasetMixin, Dataset):
    def __init__(self, h5_path: str, processor, split: str = "train", use_cache: bool = True):
        self.h5_path = Path(h5_path)
        self.processor = processor
        self.split = split

        # Determine actual processor output size for cache directory
        dummy = Image.new("RGB", (64, 64))
        dummy_out = processor(images=dummy, return_tensors="pt")["pixel_values"]
        cache_suffix = f"_{dummy_out.shape[2]}_{processor_fingerprint(processor)}"
        self._h5 = None

        with h5py.File(self.h5_path, "r") as f:
            self.variable_size = isinstance(f["images"], h5py.Group)
            if self.variable_size:
                self.num_samples = len(f["images"])
                self.image_shape = None
            else:
                self.num_samples = len(f["images"])
                self.image_shape = f["images"].shape[1:]

        self.setup_cache(self.h5_path.parent / f"tensor_{split}_imgs{cache_suffix}", self.num_samples, use_cache)
        self._images = None
        self._labels = None
        if not self.use_cache:
            with h5py.File(self.h5_path, "r") as f:
                self._labels = f["labels"][:]
                if not self.variable_size:
                    self._images = f["images"][:]

    def build_cache(self):
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        with h5py.File(self.h5_path, "r") as f:
            labels = f["labels"]

            if self.variable_size:
                img_grp = f["images"]
                for idx in tqdm(range(self.num_samples), desc=f"Caching {self.split}"):
                    if (self.cache_dir / f"{idx}.pt").exists():
                        continue
                    img = img_grp[str(idx)][:]
                    inputs = self.processor(images=Image.fromarray(img), return_tensors="pt")
                    result = {"pixel_values": inputs["pixel_values"].squeeze(0),
                              "labels": torch.tensor(int(labels[idx]), dtype=torch.long)}
                    torch.save(result, self.cache_dir / f"{idx}.pt")
            else:
                images = f["images"]
                image_pixels = np.prod(self.image_shape[:2]) if len(self.image_shape) >= 2 else 1024
                batch_size = 1000 if image_pixels <= 64 * 64 else (200 if image_pixels <= 128 * 128 else 50)
                print(f"Using batch size: {batch_size} (image size: {self.image_shape})")
                num_batches = (self.num_samples + batch_size - 1) // batch_size
                for bi in tqdm(range(num_batches), desc=f"Caching {self.split} (batches)"):
                    start, end = bi * batch_size, min((bi + 1) * batch_size, self.num_samples)
                    imgs_batch, lbls_batch = images[start:end], labels[start:end]
                    for i, (img, lbl) in enumerate(zip(imgs_batch, lbls_batch)):
                        idx = start + i
                        if (self.cache_dir / f"{idx}.pt").exists():
                            continue
                        inputs = self.processor(images=Image.fromarray(img), return_tensors="pt")
                        result = {"pixel_values": inputs["pixel_values"].squeeze(0),
                                  "labels": torch.tensor(int(lbl), dtype=torch.long)}
                        torch.save(result, self.cache_dir / f"{idx}.pt")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        cached = self.load_cached(idx)
        if cached is not None:
            return cached
        if self._images is not None:
            img, lbl = self._images[idx], self._labels[idx]
        else:
            if self._h5 is None:
                self._h5 = h5py.File(self.h5_path, "r")
            f = self._h5
            if self.variable_size:
                img = f["images"][str(idx)][:]
            else:
                img = f["images"][idx]
            lbl = self._labels[idx] if self._labels is not None else f["labels"][idx]
        inputs = self.processor(images=Image.fromarray(img), return_tensors="pt")
        result = {"pixel_values": inputs["pixel_values"].squeeze(0),
                  "labels": torch.tensor(int(lbl), dtype=torch.long)}
        self.save_to_cache(idx, result)
        return result



# JSON + images dataset
class VisionDatasetJSON(CachedDatasetMixin, Dataset):
    def __init__(self, dataset_name: str, base_dir: str, processor, split: str = "train", use_cache: bool = True):
        self.dataset_dir = Path(base_dir) / dataset_name
        self.processor = processor
        self.split = split

        json_path = self.dataset_dir / f"{dataset_name}_demo.json"
        with open(json_path, "r", encoding="utf-8") as f:
            all_data = json.load(f)

        if split == "train":
            self.data = [item for item in all_data if "/test/" not in item["image"]]
        elif split == "test":
            self.data = [item for item in all_data if "/test/" in item["image"]]
        else:
            self.data = all_data

        self.num_samples = len(self.data)
        self.setup_cache(self.dataset_dir / f"tensor_{split}_imgs_{processor_fingerprint(processor)}", self.num_samples, use_cache)

    def build_cache(self):
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        for idx, item in enumerate(tqdm(self.data, desc=f"Caching {self.split}")):
            if (self.cache_dir / f"{idx}.pt").exists():
                continue
            try:
                img = Image.open(self.dataset_dir / item["image"]).convert("RGB")
                inputs = self.processor(images=img, return_tensors="pt")
                result = {"pixel_values": inputs["pixel_values"].squeeze(0),
                          "labels": torch.tensor(int(item["label"]), dtype=torch.long)}
                torch.save(result, self.cache_dir / f"{idx}.pt")
            except Exception as e:
                print(f"Error processing {item['image']}: {e}")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        cached = self.load_cached(idx)
        if cached is not None:
            return cached
        item = self.data[idx]
        img = Image.open(self.dataset_dir / item["image"]).convert("RGB")
        inputs = self.processor(images=img, return_tensors="pt")
        result = {"pixel_values": inputs["pixel_values"].squeeze(0),
                  "labels": torch.tensor(int(item["label"]), dtype=torch.long)}
        self.save_to_cache(idx, result)
        return result



# Factory
def resolve_path(dataset_dir: Path, split: str, fmt: str) -> Path:
    if fmt == "h5":
        return dataset_dir / f"{split}.h5"
    return None  # JSON doesn't use per-split files


def create_dataloader(dataset_name: str, base_dir: str, processor, batch_size: int,
                      num_workers: int = 4, split: str = "train", use_cache: bool = True,
                      seed: int = SEED) -> DataLoader:
    dataset_dir = Path(base_dir) / dataset_name
    fmt = detect_format(dataset_dir)
    use_cache = use_cache and cache_enabled()

    if fmt == "h5":
        path = resolve_path(dataset_dir, split, fmt)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        dataset = VisionDatasetH5(h5_path=str(path), processor=processor, split=split, use_cache=use_cache)
    else:
        dataset = VisionDatasetJSON(dataset_name=dataset_name, base_dir=base_dir, processor=processor,
                                    split=split, use_cache=use_cache)

    shuffle = (split == "train")
    generator = None
    if shuffle and seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      pin_memory=True, persistent_workers=False, generator=generator)
