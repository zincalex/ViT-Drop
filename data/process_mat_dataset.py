import argparse
import json
import random
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import scipy.io

random.seed(42)
np.random.seed(42)

DEFAULT_MAT_PATHS = {
    "ColorBG": "DatasColor_BG.mat",
    "Kaggle38": "Datas_Kaggle38.mat",
    "WHOI22": "Datas_WHOI22.mat",
    "ZooScan20": "Datas_ZooScan20.mat",
}


def normalize_layout(img):
    img = np.asarray(img)

    if img.ndim == 2:
        return np.stack([img, img, img], axis=-1)

    if img.ndim == 3:
        if img.shape[2] == 3:
            return img
        if img.shape[0] == 3:
            return np.transpose(img, (1, 2, 0))

    raise ValueError(
        f"Unsupported image layout: shape {img.shape} (ndim {img.ndim}). "
        "Expected 2D grayscale (H, W) or a 3D array with a dimension equal to 3 "
        "(channels-first (3, H, W) or channels-last (H, W, 3))."
    )


def _source_hw(img):
    img = np.asarray(img)

    if img.ndim == 2:
        return int(img.shape[0]), int(img.shape[1])

    if img.ndim == 3:
        if img.shape[2] == 3:
            return int(img.shape[0]), int(img.shape[1])
        if img.shape[0] == 3:
            return int(img.shape[1]), int(img.shape[2])

    raise ValueError(
        f"Unsupported image layout: shape {img.shape} (ndim {img.ndim}). "
        "Expected 2D grayscale (H, W) or a 3D array with a dimension equal to 3 "
        "(channels-first (3, H, W) or channels-last (H, W, 3))."
    )


def validate_counts_and_integrity(images, normalized, total):
    produced = len(normalized)
    if produced != total:
        raise ValueError(
            f"Image count mismatch: expected {total} (from DATA[0,4]) but "
            f"produced {produced} normalized image(s)."
        )

    for index, (src, norm) in enumerate(zip(images, normalized)):
        src_h, src_w = _source_hw(src)
        norm = np.asarray(norm)
        norm_h, norm_w = int(norm.shape[0]), int(norm.shape[1])
        if (norm_h, norm_w) != (src_h, src_w):
            raise ValueError(
                f"Image {index}: normalized H/W ({norm_h}, {norm_w}) differs "
                f"from source H/W ({src_h}, {src_w}); normalize_layout must "
                f"preserve height and width."
            )


def to_zero_indexed(labels):
    labels = np.asarray(labels)

    if not np.issubdtype(labels.dtype, np.integer):
        if not np.issubdtype(labels.dtype, np.number):
            raise ValueError(
                f"Labels must be integer-valued; got non-numeric dtype "
                f"{labels.dtype}."
            )
        finite = np.isfinite(labels)
        is_whole = finite & (labels == np.floor(labels))
        if not np.all(is_whole):
            bad = labels[~is_whole]
            raise ValueError(
                f"Labels must be integer-valued; found non-integer label "
                f"{bad.flat[0]!r}."
            )

    return (labels - labels.min()).astype(np.int32)


def perm_split(perm_row, train_size):
    row = np.asarray(perm_row).reshape(-1)
    n = row.shape[0]

    if train_size is None or isinstance(train_size, bool):
        raise ValueError(
            f"Invalid train_size: {train_size!r}. Expected a non-negative "
            f"integer <= row length ({n})."
        )

    ts_arr = np.asarray(train_size)
    if ts_arr.size != 1 or not np.issubdtype(ts_arr.dtype, np.number):
        raise ValueError(
            f"Invalid train_size: {train_size!r}. Expected a non-negative "
            f"integer <= row length ({n})."
        )

    ts_val = ts_arr.reshape(-1)[0]
    if not np.isfinite(ts_val) or ts_val != np.floor(ts_val):
        raise ValueError(
            f"Invalid train_size: {ts_val!r} is not an integer. Expected a "
            f"non-negative integer <= row length ({n})."
        )

    ts = int(ts_val)
    if ts < 0:
        raise ValueError(
            f"Invalid train_size: {ts} is negative. Expected a non-negative "
            f"integer <= row length ({n})."
        )
    if ts > n:
        raise ValueError(
            f"Invalid train_size: {ts} exceeds the PERM row length ({n})."
        )

    zero_indexed = (row - 1).astype(int)
    train_pool = zero_indexed[:ts].tolist()
    test_set = zero_indexed[ts:].tolist()
    return train_pool, test_set


def stratified_split(indices, labels, val_ratio=0.2):
    by_label = defaultdict(list)
    for idx in indices:
        by_label[labels[idx]].append(idx)

    for lbl in sorted(by_label.keys()):
        if len(by_label[lbl]) < 2:
            raise ValueError(
                f"Class {lbl} has only {len(by_label[lbl])} sample(s) in the "
                f"train pool; at least 2 are required to place >=1 in both "
                f"train and val."
            )

    train_idx, val_idx = [], []
    for lbl in sorted(by_label.keys()):
        idxs = by_label[lbl]
        random.shuffle(idxs)
        n_val = max(1, round(val_ratio * len(idxs)))
        val_idx.extend(idxs[:n_val])
        train_idx.extend(idxs[n_val:])
    return train_idx, val_idx


def load_mat(path):
    path = str(path)

    import scipy.io as sio
    try:
        data = sio.loadmat(path)
    except NotImplementedError:
        return _load_mat_v73(path)
    except Exception as v5_error:
        try:
            return _load_mat_v73(path)
        except Exception as v73_error:
            raise ValueError(
                f"Failed loading MAT file {path}: loading failed with both the "
                f"v5 (scipy) and v7.3 (h5py) loaders "
                f"(v5 error: {v5_error}; v7.3 error: {v73_error})."
            )

    return _load_mat_v5(data, path)


def _load_mat_v5(data, path):
    cell = data["DATA"]

    images = [np.asarray(img) for img in cell[0, 0].flatten()]

    labels = np.asarray(cell[0, 1]).flatten()

    perm = np.asarray(cell[0, 2])

    train_size = np.asarray(cell[0, 3]).flatten()

    total = int(np.asarray(cell[0, 4]).flatten()[0])

    print("Loaded as MATLAB v5")
    return images, labels, perm, train_size, total


def _load_mat_v73(path):
    import h5py

    with h5py.File(path, "r") as f:
        if "DATA" not in f:
            raise ValueError(
                f"Failed loading MAT file {path}: required 'DATA' variable is "
                f"missing (cell index 0 unavailable)."
            )

        data_refs = np.asarray(f["DATA"][:]).flatten()

        def resolve_cell(index):
            if index >= data_refs.size:
                raise ValueError(
                    f"Failed loading MAT file {path}: DATA cell index {index} "
                    f"is missing (only {data_refs.size} cell(s) present)."
                )
            ref = data_refs[index]
            if not ref:
                raise ValueError(
                    f"Failed loading MAT file {path}: DATA cell index {index} "
                    f"is missing (null reference)."
                )
            return f[ref]

        img_obj = resolve_cell(0)
        images = []
        if isinstance(img_obj, h5py.Group):
            for key in img_obj.keys():
                ds = img_obj[key]
                if isinstance(ds, h5py.Dataset):
                    images.append(np.asarray(ds[:]))
        elif isinstance(img_obj, h5py.Dataset):
            if img_obj.dtype == object:
                for ref in img_obj[:].flatten():
                    ds = f[ref]
                    if isinstance(ds, h5py.Dataset):
                        images.append(np.asarray(ds[:]))
            else:
                images.append(np.asarray(img_obj[:]))
        else:
            raise ValueError(
                f"Failed loading MAT file {path}: DATA cell index 0 (images) "
                f"has an unexpected type {type(img_obj).__name__}."
            )

        labels = np.asarray(resolve_cell(1)[:]).flatten()

        perm = np.asarray(resolve_cell(2)[:])
        if perm.ndim == 2 and perm.shape[0] > perm.shape[1]:
            perm = perm.T

        train_size = np.asarray(resolve_cell(3)[:]).flatten()

        total = int(np.asarray(resolve_cell(4)[:]).flatten()[0])

    print("Loaded as MATLAB v7.3 (HDF5)")
    return images, labels, perm, train_size, total


def save_h5_variable(path, images, labels):
    import os
    import tempfile

    path = Path(path)
    labels_arr = np.asarray(labels, dtype=np.int32)

    path.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    os.close(fd)
    tmp_path = Path(tmp_name)

    try:
        with h5py.File(tmp_path, "w") as f:
            grp = f.create_group("images")
            for i, img in enumerate(images):
                grp.create_dataset(
                    str(i), data=np.asarray(img),
                    compression="gzip", compression_opts=4,
                )
            f.create_dataset(
                "labels", data=labels_arr,
                compression="gzip", compression_opts=4,
            )
            f.attrs["num_samples"] = len(labels_arr)
            f.attrs["num_classes"] = int(np.unique(labels_arr).size)
            f.attrs["variable_size"] = True
        os.replace(tmp_path, path)
    except BaseException:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except OSError:
            pass
        raise

    size_mb = path.stat().st_size / 1024**2
    print(f"  Saved {len(labels_arr)} samples -> {path} ({size_mb:.1f} MB)")


def write_metadata(
    output_dir,
    *,
    dataset_name,
    source_file,
    num_classes,
    total_samples,
    num_perms,
    permutation_used,
    train_size_from_perm,
    val_ratio,
    split_labels,
    seed=42,
):
    output_dir = Path(output_dir)
    metadata_path = output_dir / "metadata.json"

    metadata = {
        "dataset_name": dataset_name,
        "source_file": str(source_file),
        "num_classes": int(num_classes),
        "total_samples": int(total_samples),
        "num_perms": int(num_perms),
        "permutation_used": int(permutation_used),
        "train_size_from_perm": int(train_size_from_perm),
        "split_ratio": {"train": 1 - val_ratio, "val": val_ratio},
        "seed": int(seed),
        "labels_0_indexed": True,
        "variable_size_images": True,
        "created_at": datetime.now().isoformat(),
        "splits": {},
    }

    for name in ("train", "val", "test"):
        labels = split_labels.get(name, [])
        counts = defaultdict(int)
        for lbl in labels:
            counts[str(int(lbl))] += 1
        metadata["splits"][name] = {
            "num_samples": len(labels),
            "per_class_counts": dict(
                sorted(counts.items(), key=lambda kv: int(kv[0]))
            ),
        }

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
    except OSError as exc:
        raise ValueError(
            f"Failed to write metadata file {metadata_path}: {exc}. "
            f"The split files already written were left in place (not rolled back)."
        ) from exc

    print(f"\nMetadata saved to {metadata_path}")
    return metadata


def main():
    parser = argparse.ArgumentParser(
        description="Process a .mat dataset into train/val/test H5 splits."
    )
    parser.add_argument("--dataset", type=str, required=True, choices=list(DEFAULT_MAT_PATHS))
    parser.add_argument("--mat_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    args = parser.parse_args()

    dataset_name = args.dataset

    if args.mat_path is None:
        mat_path = DEFAULT_MAT_PATHS[dataset_name]
    else:
        mat_path = args.mat_path
        if not Path(mat_path).is_file():
            raise ValueError(
                f"Source MAT file not found: {mat_path}. Supply a path to an "
                f"existing .mat file."
            )

    output_dir = Path(args.output_dir or f"data/{dataset_name}")
    fold = args.fold
    val_ratio = args.val_ratio

    print("=" * 60)
    print(f"Processing {dataset_name} dataset")
    print("=" * 60)
    print(f"Source: {mat_path}")
    print(f"Output: {output_dir}")
    print(f"Permutation (fold): {fold}")
    print(f"Val ratio: {val_ratio}")

    images, labels, perm, train_size, total = load_mat(mat_path)

    n_perm_rows = int(np.asarray(perm).shape[0])
    print(
        f"\nTotal: {total} images, {len(np.unique(labels))} class(es) "
        f"(raw), {n_perm_rows} permutation row(s)"
    )

    normalized = [normalize_layout(img) for img in images]

    validate_counts_and_integrity(images, normalized, total)

    labels0 = to_zero_indexed(labels)
    num_classes = int(np.unique(labels0).size)

    if fold < 0 or fold >= n_perm_rows:
        raise ValueError(
            f"Invalid fold {fold}: must be in [0, {n_perm_rows}) "
            f"({n_perm_rows} PERM row(s) available)."
        )

    ts_arr = np.asarray(train_size).reshape(-1)
    train_size_val = ts_arr[0] if ts_arr.size == 1 else ts_arr[fold]

    perm_row = np.asarray(perm)[fold]
    train_pool, test_set = perm_split(perm_row, train_size_val)
    train_size_used = len(train_pool)
    print(f"Train size from DATA[0,3]: {train_size_used}")
    print(f"Train pool: {len(train_pool)}, Test: {len(test_set)}")

    train_idx, val_idx = stratified_split(train_pool, labels0, val_ratio=val_ratio)
    print(f"Train: {len(train_idx)}, Val: {len(val_idx)}")

    splits = {
        "train": train_idx,
        "val": val_idx,
        "test": test_set,
    }
    split_images = {
        name: [normalized[i] for i in idxs] for name, idxs in splits.items()
    }
    split_labels = {
        name: [int(labels0[i]) for i in idxs] for name, idxs in splits.items()
    }

    for name in ("train", "val", "test"):
        labels_for_split = split_labels[name]
        counts = defaultdict(int)
        for lbl in labels_for_split:
            counts[lbl] += 1
        print(f"\n  {name} distribution (total {len(labels_for_split)}):")
        for lbl in sorted(counts.keys()):
            print(f"    Label {lbl}: {counts[lbl]}")

    print("\nSaving H5 files...")
    for name in ("train", "val", "test"):
        save_h5_variable(
            output_dir / f"{name}.h5", split_images[name], split_labels[name]
        )

    write_metadata(
        output_dir,
        dataset_name=dataset_name,
        source_file=mat_path,
        num_classes=num_classes,
        total_samples=total,
        num_perms=n_perm_rows,
        permutation_used=fold,
        train_size_from_perm=train_size_used,
        val_ratio=val_ratio,
        split_labels=split_labels,
    )

    print("\n" + "=" * 60)
    print(
        f"DONE: train={len(train_idx)}, val={len(val_idx)}, test={len(test_set)}"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()
