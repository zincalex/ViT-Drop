import argparse
import h5py
import json
import numpy as np
import random
from PIL import Image
from pathlib import Path
from collections import defaultdict

random.seed(42)
np.random.seed(42)


def collect_images_and_labels(base_path, domains, subsets):
    image_paths = []
    labels = []

    for subset in subsets:
        subset_path = Path(base_path) / subset
        if not subset_path.exists():
            print(f"  Warning: {subset_path} does not exist, skipping...")
            continue

        for domain in domains:
            domain_path = subset_path / domain
            if not domain_path.exists():
                print(f"  Warning: {domain_path} does not exist, skipping...")
                continue

            print(f"  Scanning: {domain_path}")
            for class_folder in sorted(domain_path.iterdir()):
                if not class_folder.is_dir():
                    continue
                imgs = sorted(list(class_folder.glob("*.jpg")) + list(class_folder.glob("*.png")))
                for img_file in imgs:
                    image_paths.append(str(img_file))
                    labels.append(class_folder.name)
                if imgs:
                    print(f"    {class_folder.name}: {len(imgs)} images")

    return image_paths, labels


def create_label_mapping(all_labels):
    unique = sorted(set(all_labels))
    label_to_id = {label: idx for idx, label in enumerate(unique)}
    id_to_label = {idx: label for label, idx in label_to_id.items()}
    return label_to_id, id_to_label


def load_images(image_paths, labels, label_to_id):
    images = []
    label_ids = []
    sizes = defaultdict(int)

    print(f"  Loading {len(image_paths)} images...")
    for idx, (img_path, label) in enumerate(zip(image_paths, labels)):
        if idx % 1000 == 0 and idx > 0:
            print(f"    Processed {idx}/{len(image_paths)}...")
        try:
            img = Image.open(img_path).convert("RGB")
            sizes[img.size] += 1
            images.append(img)
            label_ids.append(label_to_id[label])
        except Exception as e:
            print(f"    Error loading {img_path}: {e}")

    print(f"  Loaded {len(images)} images")
    return images, label_ids, sizes


def images_to_array(images, sizes):
    target_size = max(sizes, key=sizes.get)
    if len(sizes) > 1:
        print(f"  Found {len(sizes)} different sizes, resizing outliers to {target_size}")
        for sz, cnt in sorted(sizes.items(), key=lambda x: -x[1])[:5]:
            print(f"    {sz}: {cnt} images")

    arrays = []
    for img in images:
        if img.size != target_size:
            img = img.resize(target_size, Image.LANCZOS)
        arrays.append(np.array(img, dtype=np.uint8))

    return np.array(arrays, dtype=np.uint8)


def stratified_split(labels, val_ratio=0.2):
    by_label = defaultdict(list)
    for idx, label in enumerate(labels):
        by_label[label].append(idx)

    train_idx, val_idx = [], []
    for label in sorted(by_label.keys()):
        indices = by_label[label]
        random.shuffle(indices)
        n_val = max(1, int(len(indices) * val_ratio))
        val_idx.extend(indices[:n_val])
        train_idx.extend(indices[n_val:])

    random.shuffle(train_idx)
    random.shuffle(val_idx)
    return train_idx, val_idx


def save_h5(path, images_arr, labels_arr, desc=""):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.create_dataset("images", data=images_arr, compression="gzip", compression_opts=4)
        f.create_dataset("labels", data=labels_arr, compression="gzip", compression_opts=4)
        f.attrs["num_samples"] = len(images_arr)
        f.attrs["num_classes"] = len(np.unique(labels_arr))
        f.attrs["image_shape"] = images_arr.shape[1:]
    size_mb = Path(path).stat().st_size / 1024**2
    print(f"  {desc}: {len(images_arr)} samples, shape={images_arr.shape} ({size_mb:.1f} MB)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_path", type=str, default="DAPlankton")
    parser.add_argument("--output_dir", type=str, default="data/CrossD")
    parser.add_argument("--val_ratio", type=float, default=0.2)
    args = parser.parse_args()

    subsets = ["DAPlankton_lab", "DAPlankton_sea"]
    output_dir = Path(args.output_dir)

    print("=" * 60)
    print("DAPlankton → CrossD (train/val/test)")
    print("=" * 60)

    # Collect all images for unified label mapping
    print("\nCollecting CS images (train+val)...")
    cs_paths, cs_labels = collect_images_and_labels(args.base_path, ["CS"], subsets)
    print(f"\nCollecting IFCB images (test)...")
    ifcb_paths, ifcb_labels = collect_images_and_labels(args.base_path, ["IFCB"], subsets)

    label_to_id, id_to_label = create_label_mapping(cs_labels + ifcb_labels)
    print(f"\nUnified label mapping: {len(label_to_id)} classes")

    # Load CS images → train + val
    print("\nLoading CS images...")
    cs_images, cs_label_ids, cs_sizes = load_images(cs_paths, cs_labels, label_to_id)
    cs_arr = images_to_array(cs_images, cs_sizes)
    cs_labels_arr = np.array(cs_label_ids, dtype=np.int32)

    train_idx, val_idx = stratified_split(cs_label_ids, args.val_ratio)
    print(f"  CS split: train={len(train_idx)}, val={len(val_idx)}")

    save_h5(output_dir / "train.h5", cs_arr[train_idx], cs_labels_arr[train_idx], desc="Train")
    save_h5(output_dir / "val.h5", cs_arr[val_idx], cs_labels_arr[val_idx], desc="Val")

    # Load IFCB images → test
    print("\nLoading IFCB images...")
    ifcb_images, ifcb_label_ids, ifcb_sizes = load_images(ifcb_paths, ifcb_labels, label_to_id)
    ifcb_arr = images_to_array(ifcb_images, ifcb_sizes)
    ifcb_labels_arr = np.array(ifcb_label_ids, dtype=np.int32)

    save_h5(output_dir / "test.h5", ifcb_arr, ifcb_labels_arr, desc="Test")

    # Save metadata
    metadata = {
        "dataset_name": "CrossD",
        "num_classes": len(label_to_id),
        "label_mapping": {"label_to_id": label_to_id, "id_to_label": {int(k): v for k, v in id_to_label.items()}},
        "train_domains": ["CS"], "test_domains": ["IFCB"], "excluded_domains": ["FC"],
        "splits": {
            "train": {"num_samples": len(train_idx), "per_class_counts": {str(k): int(v) for k, v in zip(*np.unique(cs_labels_arr[train_idx], return_counts=True))}},
            "val": {"num_samples": len(val_idx), "per_class_counts": {str(k): int(v) for k, v in zip(*np.unique(cs_labels_arr[val_idx], return_counts=True))}},
            "test": {"num_samples": len(ifcb_labels_arr), "per_class_counts": {str(k): int(v) for k, v in zip(*np.unique(ifcb_labels_arr, return_counts=True))}},
        },
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nMetadata saved to {output_dir / 'metadata.json'}")

    print(f"\n{'='*60}")
    print(f"DONE: train={len(train_idx)}, val={len(val_idx)}, test={len(ifcb_labels_arr)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
