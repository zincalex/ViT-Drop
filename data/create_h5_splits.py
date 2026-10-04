import sys
print("Script started", flush=True)

import argparse, json, h5py, numpy as np, random
from pathlib import Path
from PIL import Image
from collections import defaultdict
from tqdm import tqdm

random.seed(42)
np.random.seed(42)

def load_json(dataset_dir, dataset_name):
    with open(dataset_dir / f"{dataset_name}_demo.json", "r") as f:
        return json.load(f)

def split_train_test(data):
    train = [d for d in data if "/test/" not in d["image"]]
    test = [d for d in data if "/test/" in d["image"]]
    return train, test

def stratified_split(data, val_ratio=0.2):
    by_label = defaultdict(list)
    for item in data:
        by_label[item["label"]].append(item)
    train_out, val_out = [], []
    for label, items in by_label.items():
        random.shuffle(items)
        n_val = max(1, int(len(items) * val_ratio))
        val_out.extend(items[:n_val])
        train_out.extend(items[n_val:])
    random.shuffle(train_out)
    random.shuffle(val_out)
    return train_out, val_out



def load_and_save_h5(data, dataset_dir, output_path, desc=""):
    images, labels = [], []
    skipped = 0
    sizes = defaultdict(int)
    for item in tqdm(data, desc=desc):
        img_path = dataset_dir / item["image"]
        try:
            img = Image.open(img_path).convert("RGB")
            sizes[img.size] += 1
            images.append(img)
            labels.append(item["label"])
        except Exception as e:
            skipped += 1
            if skipped <= 5:
                print(f"  Warning: {img_path}: {e}", flush=True)
    if skipped > 0:
        print(f"  Skipped {skipped} images", flush=True)
    # Find most common size, resize outliers
    target_size = max(sizes, key=sizes.get)
    if len(sizes) > 1:
        print(f"  Found {len(sizes)} sizes, resizing to {target_size}", flush=True)
        for sz, cnt in sorted(sizes.items(), key=lambda x: -x[1])[:5]:
            print(f"    {sz}: {cnt} images", flush=True)
    images_np = []
    for img in images:
        if img.size != target_size:
            img = img.resize(target_size, Image.LANCZOS)
        images_np.append(np.array(img, dtype=np.uint8))
    images_arr = np.array(images_np, dtype=np.uint8)
    labels_arr = np.array(labels, dtype=np.int32)
    print(f"  Shape: {images_arr.shape}, dtype: {images_arr.dtype}", flush=True)
    print(f"  Labels: [{labels_arr.min()}, {labels_arr.max()}]", flush=True)
    print(f"  First 5 labels: {labels_arr[:5].tolist()}", flush=True)
    print(f"  Pixel range: [{images_arr[0].min()}, {images_arr[0].max()}]", flush=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as f:
        f.create_dataset("images", data=images_arr, compression="gzip", compression_opts=4)
        f.create_dataset("labels", data=labels_arr, compression="gzip", compression_opts=4)
        f.attrs["num_samples"] = len(images_np)
        f.attrs["num_classes"] = len(np.unique(labels_arr))
        f.attrs["image_shape"] = images_arr.shape[1:]
    size_mb = output_path.stat().st_size / 1024**2
    print(f"  Saved {len(images_np)} -> {output_path} ({size_mb:.1f} MB)", flush=True)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--val_ratio", type=float, default=0.2)
    args = parser.parse_args()
    dataset_dir = Path(args.data_dir) / args.dataset
    print(f"Dataset: {args.dataset}", flush=True)
    print(f"Directory: {dataset_dir}", flush=True)
    data = load_json(dataset_dir, args.dataset)
    print(f"Total entries: {len(data)}", flush=True)
    train_data, test_data = split_train_test(data)
    print(f"Train: {len(train_data)}, Test: {len(test_data)}", flush=True)
    train_split, val_split = stratified_split(train_data, args.val_ratio)
    print(f"After split: train={len(train_split)}, val={len(val_split)}", flush=True)
    load_and_save_h5(train_split, dataset_dir, dataset_dir / "train.h5", desc="Train")
    load_and_save_h5(val_split, dataset_dir, dataset_dir / "val.h5", desc="Val")
    load_and_save_h5(test_data, dataset_dir, dataset_dir / "test.h5", desc="Test")
    print(f"Done! Files saved to {dataset_dir}/", flush=True)

if __name__ == "__main__":
    main()
