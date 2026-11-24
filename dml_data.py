"""
Minimal MNIST loader for the LeNet-5 project.

This module provides the interface expected by scaffold.py:

    ds = dml_data.load("mnist")
    ds.x_train, ds.y_train, ds.x_test, ds.y_test

The four original MNIST IDX files are downloaded on first use and cached
locally. Images are returned as uint8 arrays with shape (N, 28, 28);
labels are returned as int64 arrays with shape (N,).
"""

from __future__ import annotations

import gzip
import struct
import urllib.error
import urllib.request
from pathlib import Path
from types import SimpleNamespace

_BASE_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/"
_FILES = {
    "x_train": "train-images-idx3-ubyte.gz",
    "y_train": "train-labels-idx1-ubyte.gz",
    "x_test": "t10k-images-idx3-ubyte.gz",
    "y_test": "t10k-labels-idx1-ubyte.gz",
}

# Cache outside the project directory so the dataset is not mixed with source.
_CACHE_DIR = Path.home() / ".cache" / "lenet5-mnist"


def _download(filename: str) -> Path:
    """Download one compressed MNIST file if it is not already cached."""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    destination = _CACHE_DIR / filename

    if destination.exists() and destination.stat().st_size > 0:
        return destination

    url = _BASE_URL + filename
    temporary = destination.with_suffix(destination.suffix + ".part")

    try:
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "LeNet5-MNIST-loader/1.0"},
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            data = response.read()

        # Check the gzip stream before treating the download as complete.
        import io
        with gzip.GzipFile(fileobj=io.BytesIO(data)) as stream:
            stream.read(1)

        temporary.write_bytes(data)
        temporary.replace(destination)
    except Exception as exc:
        if temporary.exists():
            temporary.unlink()
        raise RuntimeError(
            f"Could not download {filename} from {url}. "
            "Check your internet connection, proxy/firewall settings, "
            "and try again."
        ) from exc

    return destination


def _read_images(path: Path):
    """Read an IDX image file and return uint8 images (N, rows, cols)."""
    with gzip.open(path, "rb") as stream:
        header = stream.read(16)
        if len(header) != 16:
            raise ValueError(f"Invalid MNIST image file: {path}")

        magic, count, rows, cols = struct.unpack(">IIII", header)
        if magic != 2051:
            raise ValueError(
                f"Invalid image-file magic number {magic} in {path}"
            )

        raw = stream.read()
        expected = count * rows * cols
        if len(raw) != expected:
            raise ValueError(
                f"Incomplete MNIST image data in {path}: "
                f"expected {expected} bytes, got {len(raw)}"
            )

    import numpy as np
    return np.frombuffer(raw, dtype=np.uint8).reshape(count, rows, cols).copy()


def _read_labels(path: Path):
    """Read an IDX label file and return int64 labels (N,)."""
    with gzip.open(path, "rb") as stream:
        header = stream.read(8)
        if len(header) != 8:
            raise ValueError(f"Invalid MNIST label file: {path}")

        magic, count = struct.unpack(">II", header)
        if magic != 2049:
            raise ValueError(
                f"Invalid label-file magic number {magic} in {path}"
            )

        raw = stream.read()
        if len(raw) != count:
            raise ValueError(
                f"Incomplete MNIST label data in {path}: "
                f"expected {count} bytes, got {len(raw)}"
            )

    import numpy as np
    return np.frombuffer(raw, dtype=np.uint8).astype(np.int64)


def load(name="mnist"):
    """
    Load MNIST and return an object with x_train, y_train, x_test, y_test.

    Parameters
    ----------
    name : str
        Only "mnist" is supported.

    Returns
    -------
    SimpleNamespace
        x_train: uint8, shape (60000, 28, 28)
        y_train: int64, shape (60000,)
        x_test:  uint8, shape (10000, 28, 28)
        y_test:  int64, shape (10000,)
    """
    if name.lower() != "mnist":
        raise ValueError(
            f"Unsupported dataset {name!r}. This module supports only 'mnist'."
        )

    paths = {key: _download(filename) for key, filename in _FILES.items()}

    x_train = _read_images(paths["x_train"])
    y_train = _read_labels(paths["y_train"])
    x_test = _read_images(paths["x_test"])
    y_test = _read_labels(paths["y_test"])

    if len(x_train) != len(y_train) or len(x_test) != len(y_test):
        raise ValueError("MNIST image and label counts do not match.")

    return SimpleNamespace(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
    )
