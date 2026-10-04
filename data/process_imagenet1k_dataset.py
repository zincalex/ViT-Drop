import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", type=str, default="data/imagenet-1k")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "test.h5"
    if out_path.exists():
        raise FileExistsError(f"{out_path} already exists; remove it first to rebuild.")

    from datasets import load_dataset
    print("Streaming ImageNet-1K validation split from HuggingFace...", flush=True)
    ds = load_dataset("ILSVRC/imagenet-1k", split="validation", streaming=True)

    labels = []
    sizes = defaultdict(int)
    with h5py.File(out_path, "w") as f:
        grp = f.create_group("images")
        for idx, sample in enumerate(ds):
            img = sample["image"]
            if img.mode != "RGB":
                img = img.convert("RGB")
            sizes[img.size] += 1
            grp.create_dataset(
                str(idx), data=np.asarray(img, dtype=np.uint8),
                compression="gzip", compression_opts=4,
            )
            labels.append(sample["label"])
            if idx % 2500 == 0 and idx > 0:
                print(f"  {idx} images written...", flush=True)
        labels = np.array(labels, dtype=np.int32)
        f.create_dataset("labels", data=labels, compression="gzip", compression_opts=4)
        f.attrs["num_samples"] = len(labels)
        f.attrs["num_classes"] = len(np.unique(labels))

    metadata = {
        "dataset_name": "imagenet-1k",
        "num_classes": 1000,
        "total_samples": int(len(labels)),
        "split_policy": "full ImageNet-1K validation set, test-only, zero-shot",
        "labels_0_indexed": True,
        "variable_size_images": True,
        "created_at": datetime.now().isoformat(),
        "splits": {"test": {"num_samples": int(len(labels))}},
    }
    with open(out_dir / "metadata.json", "w") as fmeta:
        json.dump(metadata, fmeta, indent=2)

    counts = np.bincount(labels, minlength=1000)
    print(f"\nWritten: {out_path}")
    print(f"Samples: {len(labels)} (expected 50000)")
    print(f"Classes: {len(np.unique(labels))} (expected 1000)")
    print(f"Per-class min/max: {counts.min()}/{counts.max()} (expected 50/50)")
    print(f"Distinct native sizes: {len(sizes)}")
    ok = len(labels) == 50000 and len(np.unique(labels)) == 1000 and len(sizes) > 1
    print("VERIFICATION PASSED" if ok else "WARNING: verification failed, check output above")


if __name__ == "__main__":
    main()
