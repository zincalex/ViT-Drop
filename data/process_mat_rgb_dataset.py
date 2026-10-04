import argparse
import h5py
import json
import random
import numpy as np
from pathlib import Path
from collections import defaultdict
from datetime import datetime

random.seed(42)
np.random.seed(42)


def load_mat(path):
    path = str(path)
    try:
        import scipy.io as sio
        data = sio.loadmat(path)
        cell = data["DATA"]
        images = list(cell[0, 0].flatten())
        labels = cell[0, 1].flatten().astype(np.int32)
        perm = data["PERM"]
        print(f"Loaded as MATLAB v5")
        return images, labels, perm
    except NotImplementedError:
        pass

    f = h5py.File(path, "r")
    data_refs = f["DATA"][:].flatten()
    img_obj = f[data_refs[0]]
    images = []
    if isinstance(img_obj, h5py.Dataset) and img_obj.dtype == object:
        img_refs = img_obj[:].flatten()
        for r in img_refs:
            ds = f[r]
            if isinstance(ds, h5py.Dataset):
                images.append(np.transpose(ds[:], (2, 1, 0)))
    elif isinstance(img_obj, h5py.Group):
        for k in list(img_obj.keys()):
            ds = img_obj[k]
            if isinstance(ds, h5py.Dataset):
                images.append(np.transpose(ds[:], (2, 1, 0)))
    else:
        raise ValueError(f"Unexpected image cell type: {type(img_obj)}")
    print(f"  Loaded {len(images)} images")
    labels = f[data_refs[1]][:].flatten().astype(np.int32)
    perm = f[data_refs[2]][:]
    if perm.shape[0] > perm.shape[1]:
        perm = perm.T
    f.close()
    print(f"Loaded as MATLAB v7.3")
    return images, labels, perm


def stratified_split(indices, labels, val_ratio=0.2):
    by_label = defaultdict(list)
    for idx in indices:
        by_label[labels[idx]].append(idx)
    train_idx, val_idx = [], []
    for lbl in sorted(by_label.keys()):
        idxs = by_label[lbl]
        random.shuffle(idxs)
        n_val = max(1, int(len(idxs) * val_ratio))
        val_idx.extend(idxs[:n_val])
        train_idx.extend(idxs[n_val:])
    random.shuffle(train_idx)
    random.shuffle(val_idx)
    return train_idx, val_idx


def save_h5_variable(path, images, labels):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
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


def get_train_size(mat_path, n_perms, n_samples, fold):
    train_size = None
    try:
        import scipy.io as sio
        data = sio.loadmat(str(mat_path))
        hint = data["DATA"][0, 3].flatten()
        if len(hint) == 1:
            train_size = int(hint[0])
        elif len(hint) >= fold + 1:
            train_size = int(hint[fold])
    except NotImplementedError:
        f = h5py.File(str(mat_path), "r")
        data_refs = f["DATA"][:].flatten()
        for ref in data_refs:
            obj = f[ref]
            if isinstance(obj, h5py.Dataset):
                arr = obj[:]
                if arr.ndim <= 2 and np.issubdtype(arr.dtype, np.number):
                    vals = arr.flatten().astype(int)
                    if len(vals) == n_perms and all(0 < v < n_samples for v in vals):
                        train_size = int(vals[fold])
                        break
                    elif len(vals) == 1 and 0 < vals[0] < n_samples:
                        train_size = int(vals[0])
                        break
        f.close()
    if train_size is None:
        raise ValueError("Could not determine train size from DATA[0,3]")
    return train_size


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, choices=["Bark", "InfLarynge", "Pest"])
    parser.add_argument("--mat_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--fold", type=int, default=0)
    args = parser.parse_args()

    output_dir = Path(args.output_dir or f"data/{args.dataset}")
    print(f"{'='*60}")
    print(f"Processing {args.dataset} dataset")
    print(f"{'='*60}")
    print(f"Source: {args.mat_path}")
    print(f"Output: {output_dir}")
    print(f"Permutation: {args.fold}")

    images, labels, perm = load_mat(args.mat_path)
    n_samples = len(images)
    n_perms = perm.shape[0]
    print(f"\nTotal: {n_samples} images, {len(np.unique(labels))} classes, {n_perms} permutations")

    labels = labels - labels.min()
    train_size = get_train_size(args.mat_path, n_perms, n_samples, args.fold)
    print(f"Train size from DATA[0,3]: {train_size}")

    perm_row = perm[args.fold].flatten().astype(np.int32) - 1
    train_pool = list(perm_row[:train_size])
    test_indices = list(perm_row[train_size:])
    print(f"Train pool: {len(train_pool)}, Test: {len(test_indices)}")

    train_indices, val_indices = stratified_split(train_pool, labels, val_ratio=0.2)
    print(f"Train: {len(train_indices)}, Val: {len(val_indices)}")

    for name, idxs in [("train", train_indices), ("val", val_indices), ("test", test_indices)]:
        counts = defaultdict(int)
        for i in idxs:
            counts[int(labels[i])] += 1
        print(f"\n  {name} distribution:")
        for lbl in sorted(counts.keys()):
            print(f"    Label {lbl}: {counts[lbl]}")

    print(f"\nSaving H5 files...")
    for name, idxs in [("train", train_indices), ("val", val_indices), ("test", test_indices)]:
        split_images = [images[i] for i in idxs]
        split_labels = [int(labels[i]) for i in idxs]
        save_h5_variable(output_dir / f"{name}.h5", split_images, split_labels)

    metadata = {
        "dataset_name": args.dataset,
        "source_file": str(args.mat_path),
        "num_classes": int(len(np.unique(labels))),
        "total_samples": n_samples,
        "num_perms": n_perms,
        "permutation_used": args.fold,
        "train_size_from_perm": train_size,
        "split_ratio": {"train": 0.8, "val": 0.2},
        "seed": 42,
        "labels_0_indexed": True,
        "variable_size_images": True,
        "created_at": datetime.now().isoformat(),
        "splits": {},
    }
    for name, idxs in [("train", train_indices), ("val", val_indices), ("test", test_indices)]:
        counts = defaultdict(int)
        for i in idxs:
            counts[str(int(labels[i]))] += 1
        metadata["splits"][name] = {
            "num_samples": len(idxs),
            "per_class_counts": dict(sorted(counts.items(), key=lambda x: int(x[0]))),
        }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nMetadata saved to {output_dir / 'metadata.json'}")

    print(f"\n{'='*60}")
    print(f"DONE: train={len(train_indices)}, val={len(val_indices)}, test={len(test_indices)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
