"""Checksummed downloads and explicitly versioned BRF preprocessing."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import urllib.request

import h5py
import numpy as np
from scipy.io import loadmat

UPSTREAM = "1a42b8c8aceedb13cae3b2327774c2fcc04fd696"
HEIDELBERG = "https://zenkelab.org/datasets"
MNIST = {
    "train-images-idx3-ubyte.gz": "f68b3c2dcbeaaa9fbdd348bbdeb94873",
    "train-labels-idx1-ubyte.gz": "d53e105ee54ea40749a09fcbcd1e9432",
    "t10k-images-idx3-ubyte.gz": "9fb629c4189551a2d022fa330f9573f3",
    "t10k-labels-idx1-ubyte.gz": "ec29112dd5afa0611ce80d1b7f02629c",
}


def digest(path, algorithm="sha256"):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, algorithm).hexdigest()


def download(url, path, md5=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        temporary = path.with_suffix(path.suffix + ".part")
        print(f"Downloading {url}", flush=True)
        with urllib.request.urlopen(url, timeout=120) as source, temporary.open("wb") as out:
            shutil.copyfileobj(source, out)
        if md5 and digest(temporary, "md5") != md5:
            raise ValueError(f"Checksum mismatch: {url}")
        temporary.replace(path)
    if md5 and digest(path, "md5") != md5:
        raise ValueError(f"Cached checksum mismatch: {path}")
    return {"url": url, "file": path.name, "sha256": digest(path), "md5": md5}


def bin_shd(times, units, steps=250, dt=0.004):
    """Exact upstream bins: first upper edge 0, final .996; discard unit zero.

    searchsorted uses the same floating point upper edges as the original loop,
    avoiding an off-by-one for events exactly on a boundary.
    """
    times, units = np.asarray(times), np.asarray(units)
    bins = np.searchsorted(np.arange(steps) * dt, times, side="left")
    valid = (bins < steps) & (units > 0) & (units < 700)
    image = np.zeros((steps, 700), np.uint8)
    image[bins[valid], 700 - units[valid]] = 1
    return image


def prepare(root, task):
    root = Path(root) / ("mnist" if task in {"smnist", "psmnist"} else task)
    root.mkdir(parents=True, exist_ok=True)
    record = root / "provenance.json"
    if record.exists():
        return root
    sources = []
    if task in {"smnist", "psmnist"}:
        for name, md5 in MNIST.items():
            sources.append(download(f"https://ossci-datasets.s3.amazonaws.com/mnist/{name}", root / name, md5))
        for split, prefix in (("train", "train"), ("test", "t10k")):
            with gzip.open(root / f"{prefix}-images-idx3-ubyte.gz", "rb") as f:
                x = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 784, 1)
            with gzip.open(root / f"{prefix}-labels-idx1-ubyte.gz", "rb") as f:
                y = np.frombuffer(f.read(), np.uint8, offset=8)
            np.save(root / f"{split}_x.npy", x)
            np.save(root / f"{split}_y.npy", y)
        protocol = "Official IDX pixels, divide by 255 at batch loading; no centering."
    elif task == "shd":
        with urllib.request.urlopen(f"{HEIDELBERG}/md5sums.txt", timeout=60) as response:
            hashes = {line.split()[1]: line.split()[0] for line in response.read().decode().splitlines() if len(line.split()) == 2}
        for split in ("train", "test"):
            name = f"shd_{split}.h5.gz"
            sources.append(download(f"{HEIDELBERG}/{name}", root / name, hashes[name]))
            uncompressed = root / name[:-3]
            if not uncompressed.exists():
                with gzip.open(root / name, "rb") as source, uncompressed.open("wb") as out:
                    shutil.copyfileobj(source, out)
            with h5py.File(uncompressed) as f:
                count = len(f["labels"])
                x = np.lib.format.open_memmap(root / f"{split}_x.npy", mode="w+", dtype=np.uint8, shape=(count, 250, 700))
                for i in range(count):
                    x[i] = bin_shd(f["spikes/times"][i], f["spikes/units"][i])
                x.flush()
                np.save(root / f"{split}_y.npy", f["labels"][:].astype(np.int64))
                if "extra/speaker" in f:
                    np.save(root / f"{split}_speaker.npy", f["extra/speaker"][:])
        protocol = "BRF upstream: 250 bins with upper edges 0,.004,...,.996 seconds; channel 700-unit; discard unit zero."
    elif task == "ecg":
        for split in ("train", "test"):
            name = f"QTDB_{split}.mat"
            url = f"https://raw.githubusercontent.com/AdaptiveAILab/brf-neurons/{UPSTREAM}/experiments/ecg/data/{name}"
            sources.append(download(url, root / name))
            mat = loadmat(root / name)
            np.save(root / f"{split}_x.npy", mat["x"][:, :-1].astype(np.float32))
            np.save(root / f"{split}_y.npy", mat["y"][:, :-1].argmax(-1).astype(np.int64))
        protocol = "Yin et al. preprocessed QTDB distributed by BRF; remove final sentinel row; retain supplied split."
    else:
        raise ValueError(task)
    arrays = {p.name: {"sha256": digest(p), "shape": list(np.load(p, mmap_mode="r").shape)} for p in root.glob("*.npy")}
    record.write_text(json.dumps({"protocol": protocol, "sources": sources, "arrays": arrays}, indent=2) + "\n")
    return root


def datasets(root, task, split_seed=20260906):
    root = prepare(root, task)
    train_x, train_y = [np.load(root / f"train_{s}.npy", mmap_mode="r") for s in ("x", "y")]
    order = np.random.default_rng(split_seed).permutation(len(train_y))
    n_val = int(len(order) * .1)
    permutation = np.random.default_rng(2024).permutation(784) if task == "psmnist" else None
    split = {"train": (train_x, train_y, order[n_val:]), "validation": (train_x, train_y, order[:n_val])}
    test_x, test_y = [np.load(root / f"test_{s}.npy", mmap_mode="r") for s in ("x", "y")]
    split["test"] = test_x, test_y, np.arange(len(test_y))
    metadata = json.loads((root / "provenance.json").read_text())
    metadata["split_seed"] = split_seed
    metadata["validation_indices_sha256"] = hashlib.sha256(order[:n_val].tobytes()).hexdigest()
    metadata["permutation_sha256"] = hashlib.sha256(permutation.tobytes()).hexdigest() if permutation is not None else None
    return split, permutation, metadata


def batches(dataset, batch_size, task, permutation=None, shuffle_seed=None, limit=None):
    x, y, indices = dataset
    indices = indices[:limit] if limit else indices
    if shuffle_seed is not None:
        indices = np.random.default_rng(shuffle_seed).permutation(indices)
    for start in range(0, len(indices), batch_size):
        real = indices[start:start + batch_size]
        padded = np.pad(real, (0, batch_size - len(real)), mode="edge")
        xb = np.asarray(x[padded], np.float32)
        yb = np.asarray(y[padded], np.int32)
        if task in {"smnist", "psmnist"}:
            xb = xb / 255
            if permutation is not None:
                xb = xb[:, permutation]
        if task == "ecg":
            yb = yb.T
        yield xb.transpose(1, 0, 2), yb, (np.arange(batch_size) < len(real)).astype(np.float32)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("task", choices=["smnist", "psmnist", "ecg", "shd", "all"])
    parser.add_argument("--root", default="data")
    args = parser.parse_args()
    for task in (["smnist", "ecg", "shd"] if args.task == "all" else [args.task]):
        print(prepare(args.root, task), flush=True)
