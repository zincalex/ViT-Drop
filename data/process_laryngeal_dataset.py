import argparse
import h5py
import json
import random
import numpy as np
from pathlib import Path
from collections import defaultdict
from datetime import datetime
from PIL import Image

random.seed(42)
np.random.seed(42)

CLASS_LABELS = {"Hbv": 0, "He": 1, "IPCL": 2, "Le": 3}
FOLDS = ["FOLD 1", "FOLD 2", "FOLD 3"]


def collect_images(source_dir):
    all_images = []
    source = Path(source_dir)
    for fold in FOLDS:
        fold_path = source / fold
        if not fold_path.exists():
            print(f"  Warning: {fold_path} not found")
            continue
        for class_name, label in CLASS_LABELS.items():
            class_path = fold_path / class_name
            if not class_path.exists():
                continue
            imgs = sorted(
                list(class_path.glob("*.png")) + list(class_path.glob("*.jpg")) +
                list(class_path.glob("*.jpeg")) + list(class_path.glob("*.PNG")) +
                list(class_path.glob("*.JPG")) + list(class_path.glob("*.JPEG"))
            )
            for img_path in imgs:
                all_images.append({"path": img_path, "label": label, "class_name": class_name})
            print(f"  {fold}/{class_name}: {len(imgs)} images")
    return all_images


def stratified_split(data, ratio=0.2):
    by_label = defaultdict(list)
    for item in data:
        by_label[item["label"]].append(item)
    split_a, split_b = [], []
    for lbl in sorted(by_label.keys()):
        items = by_label[lbl]
        random.shuffle(items)
        n_b = max(1, int(len(items) * ratio))
        split_b.extend(items[:n_b])
        split_a.extend(items[n_b:])
    random.shuffle(split_a)
    random.shuffle(split_b)
    return split_a, split_b


def save_h5_variable(path, data_items):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    images, labels = [], []
    for item in data_items:
        img = np.array(Image.open(item["path"]).convert("RGB"), dtype=np.uint8)
        images.append(img)
        labels.append(item["label"])
    with h5py.File(path, "w") as f:
        grp = f.create_group("images")
        for i, img in enumerate(images):
            grp.create_dataset(str(i), data=img, compression="gzip", compression_opts=4)
        f.create_dataset("labels", data=np.array(labels, dtype=np.int32),
                         compression="gzip", compression_opts=4)
        f.attrs["num_samples"] = len(images)
        f.attrs["num_classes"] = len(np.unique(labels))
        f.attrs["variable_size"] = True
    size_mb = Path(path).stat().st_size / 1024**2
    print(f"  Saved {len(images)} samples -> {path} ({size_mb:.1f} MB)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_dir", type=str, default="laryngeal dataset")
    parser.add_argument("--output_dir", type=str, default="data/lar")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    print(f"{'='*60}")
    print(f"Processing Laryngeal dataset")
    print(f"{'='*60}")

    all_images = collect_images(args.source_dir)
    print(f"\nTotal: {len(all_images)} images, {len(CLASS_LABELS)} classes")

    train_pool, test_data = stratified_split(all_images, ratio=0.2)
    train_data, val_data = stratified_split(train_pool, ratio=0.2)
    print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

    for name, items in [("train", train_data), ("val", val_data), ("test", test_data)]:
        counts = defaultdict(int)
        for item in items:
            counts[item["label"]] += 1
        print(f"\n  {name} distribution:")
        for lbl in sorted(counts.keys()):
            cname = [k for k, v in CLASS_LABELS.items() if v == lbl][0]
            print(f"    {cname} (label {lbl}): {counts[lbl]}")

    print(f"\nSaving H5 files...")
    for name, items in [("train", train_data), ("val", val_data), ("test", test_data)]:
        save_h5_variable(output_dir / f"{name}.h5", items)

    metadata = {
        "dataset_name": "lar",
        "num_classes": len(CLASS_LABELS),
        "class_labels": CLASS_LABELS,
        "total_samples": len(all_images),
        "split_ratio": {"train_test": 0.8, "train_val": 0.8},
        "seed": 42,
        "labels_0_indexed": True,
        "variable_size_images": True,
        "created_at": datetime.now().isoformat(),
        "splits": {},
    }
    for name, items in [("train", train_data), ("val", val_data), ("test", test_data)]:
        counts = defaultdict(int)
        for item in items:
            counts[str(item["label"])] += 1
        metadata["splits"][name] = {
            "num_samples": len(items),
            "per_class_counts": dict(sorted(counts.items(), key=lambda x: int(x[0]))),
        }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nMetadata saved to {output_dir / 'metadata.json'}")

    print(f"\n{'='*60}")
    print(f"DONE: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
