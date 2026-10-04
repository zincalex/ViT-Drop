import argparse
import h5py
import numpy as np
from pathlib import Path
from collections import defaultdict
from PIL import Image

np.random.seed(42)


def load_h5(path):
    with h5py.File(path, "r") as f:
        images = f["images"][:]
        labels = f["labels"][:]
    return images, labels


def save_h5(path, images, labels):
    with h5py.File(path, "w") as f:
        f.create_dataset("images", data=images, compression="gzip", compression_opts=4)
        f.create_dataset("labels", data=labels, compression="gzip", compression_opts=4)
        f.attrs["num_samples"] = len(images)
        f.attrs["num_classes"] = len(np.unique(labels))
        f.attrs["image_shape"] = images.shape[1:]
    size_mb = path.stat().st_size / 1024**2
    print(f"  Saved {len(images)} samples → {path} ({size_mb:.1f} MB)")


def stratified_split(images, labels, val_ratio=0.2):
    by_label = defaultdict(list)
    for i, label in enumerate(labels):
        by_label[label].append(i)

    train_idx, val_idx = [], []
    for label in sorted(by_label.keys()):
        indices = by_label[label]
        np.random.shuffle(indices)
        n_val = max(1, int(len(indices) * val_ratio))
        val_idx.extend(indices[:n_val])
        train_idx.extend(indices[n_val:])

    np.random.shuffle(train_idx)
    np.random.shuffle(val_idx)
    return images[train_idx], labels[train_idx], images[val_idx], labels[val_idx]


def process_dataset(dataset_dir):
    name = dataset_dir.name
    train_path = dataset_dir / "train.h5"
    val_path = dataset_dir / "val.h5"

    if not train_path.exists():
        print(f"  SKIP: no train.h5 found")
        return

    if name == "LCZ42":
        # LCZ42: split train.h5 only, delete old val.h5
        print(f"  LCZ42 mode: splitting train.h5 only")
        images, labels = load_h5(train_path)
        if val_path.exists():
            val_path.unlink()
            print(f"  Deleted old val.h5")
    else:
        # All others: merge train.h5 + val.h5
        if not val_path.exists():
            print(f"  SKIP: no val.h5 found")
            return
        print(f"  Merging train.h5 + val.h5")
        train_imgs, train_labels = load_h5(train_path)
        val_imgs, val_labels = load_h5(val_path)

        # Resize if image dimensions don't match (use train shape as target)
        if train_imgs.shape[1:] != val_imgs.shape[1:]:
            target_h, target_w = train_imgs.shape[1], train_imgs.shape[2]
            print(f"  Resizing val images from {val_imgs.shape[1:]} to ({target_h}, {target_w}, ...)")
            resized = []
            for img in val_imgs:
                pil = Image.fromarray(img).resize((target_w, target_h), Image.LANCZOS)
                resized.append(np.array(pil, dtype=np.uint8))
            val_imgs = np.array(resized)

        images = np.concatenate([train_imgs, val_imgs], axis=0)
        labels = np.concatenate([train_labels, val_labels], axis=0)

    print(f"  Total: {len(images)} samples, {len(np.unique(labels))} classes")
    train_imgs, train_labels, val_imgs, val_labels = stratified_split(images, labels, val_ratio=0.2)
    print(f"  Split: train={len(train_imgs)}, val={len(val_imgs)}")

    save_h5(train_path, train_imgs, train_labels)
    save_h5(val_path, val_imgs, val_labels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--dataset", type=str, default=None, help="Process only this dataset folder")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    if args.dataset:
        datasets = [data_dir / args.dataset]
        if not datasets[0].is_dir():
            print(f"ERROR: {datasets[0]} not found")
            return
    else:
        datasets = sorted([d for d in data_dir.iterdir() if d.is_dir()])

    for dataset_dir in datasets:
        print(f"\n{'='*60}")
        print(f"Processing: {dataset_dir.name}")
        print(f"{'='*60}")
        process_dataset(dataset_dir)

    print(f"\nDone! Processed {len(datasets)} datasets.")


if __name__ == "__main__":
    main()
