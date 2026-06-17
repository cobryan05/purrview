"""Dataset loader and image fetcher abstraction for the training server.

ImageFetcher handles the physical retrieval of image bytes; ImageLabelDataset
wraps that into a torch Dataset suitable for DataLoader.

Two fetcher implementations:
- FilesystemImageFetcher  — reads files directly from a shared volume (preferred)
- HttpImageFetcher        — fetches via db_server HTTP API and caches locally

The split is deterministic: images are sorted by UUID, then the first
(1 - val_fraction) fraction is train and the rest is val.  Reproducible across
runs regardless of DB insertion order.
"""

from __future__ import annotations

import hashlib
import logging
import os
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional

import httpx

logger = logging.getLogger(__name__)

# ─── Image Fetcher abstraction ────────────────────────────────────────────────


class ImageFetcher(ABC):
    """Retrieve raw image bytes given a relative image path and UUID."""

    @abstractmethod
    def fetch(self, image_uuid: str, filename: str) -> bytes:
        """Return JPEG/PNG bytes for the image.

        Args:
            image_uuid: DB UUID of the image.
            filename:   Relative filename as stored in ImageRecord.filename.
        """
        ...


class FilesystemImageFetcher(ImageFetcher):
    """Read images directly from a shared filesystem volume (zero HTTP overhead)."""

    def __init__(self, fs_root: str, path_prefix_map: Optional[dict[str, str]] = None):
        """
        Args:
            fs_root:          Root directory where images live (e.g. /app/image_datasets).
            path_prefix_map:  Optional {source_prefix: target_prefix} remapping applied
                              to filename before joining with fs_root.  Handles the case
                              where the training container mounts the volume at a different
                              path than the main container.
        """
        self._fs_root = Path(fs_root)
        self._prefix_map = path_prefix_map or {}

    def _resolve(self, filename: str) -> Path:
        for src, dst in self._prefix_map.items():
            if filename.startswith(src):
                filename = dst + filename[len(src):]
                break
        return self._fs_root / filename

    def fetch(self, image_uuid: str, filename: str) -> bytes:
        path = self._resolve(filename)
        return path.read_bytes()


class HttpImageFetcher(ImageFetcher):
    """Fetch images from db_server over HTTP and cache to a local temp directory."""

    def __init__(self, db_base_url: str, cache_dir: Optional[str] = None):
        """
        Args:
            db_base_url:  Base URL of the db_server (e.g. http://localhost:8003).
            cache_dir:    Directory for caching downloaded images.  A temporary
                          directory is created and owned by this instance if None.
        """
        self._db_base_url = db_base_url.rstrip("/")
        if cache_dir is None:
            self._tmp = tempfile.mkdtemp(prefix="training_img_cache_")
            self._cache_dir = Path(self._tmp)
        else:
            self._cache_dir = Path(cache_dir)
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            self._tmp = None

    def _cache_path(self, image_uuid: str) -> Path:
        return self._cache_dir / f"{image_uuid}.jpg"

    def fetch(self, image_uuid: str, filename: str) -> bytes:
        cached = self._cache_path(image_uuid)
        if cached.exists():
            return cached.read_bytes()

        # The db_server file endpoint returns JSON with a base64-encoded image_base64 field.
        url = f"{self._db_base_url}/api/images/{filename}/file"
        resp = httpx.get(url, params={"include_binary": True}, timeout=30.0)
        resp.raise_for_status()
        body = resp.json()
        if body.get("status") != "success" or not body.get("image_base64"):
            raise ValueError(f"Failed to fetch image {filename!r}: {body.get('message')}")

        import base64 as _b64
        data = _b64.b64decode(body["image_base64"])
        cached.write_bytes(data)
        return data

    def cleanup(self) -> None:
        """Remove the temporary cache directory (only if we created it)."""
        if self._tmp:
            import shutil

            shutil.rmtree(self._tmp, ignore_errors=True)


# ─── Dataset ─────────────────────────────────────────────────────────────────


def _build_dataset(
    label_uuid: str,
    db_base_url: str,
    fetcher: ImageFetcher,
    split: str = "train",
    val_fraction: float = 0.2,
) -> "ImageLabelDataset":
    """Factory that loads image-label rows from db_server and returns an ImageLabelDataset.

    Args:
        label_uuid:   UUID of the label to load present/absent examples for.
        db_base_url:  Base URL of the db_server.
        fetcher:      ImageFetcher to use for loading image bytes.
        split:        "train" or "val".
        val_fraction: Fraction of images to hold out for validation (0–1).
    """
    # Fetch all present/absent rows for this label.
    url = f"{db_base_url.rstrip('/')}/api/image-labels"
    resp = httpx.get(
        url,
        params={"label_uuid": label_uuid, "status": ["present", "absent"]},
        timeout=60.0,
    )
    resp.raise_for_status()
    body = resp.json()
    rows = body.get("images", [])
    if not rows:
        raise ValueError(f"No labelled images found for label_uuid={label_uuid!r}")

    # Deterministic train/val split: sort by UUID (stable hash), then slice.
    rows_sorted = sorted(rows, key=lambda r: r["image_uuid"])
    n_val = max(1, int(len(rows_sorted) * val_fraction))
    n_train = len(rows_sorted) - n_val
    if split == "train":
        split_rows = rows_sorted[:n_train]
    else:
        split_rows = rows_sorted[n_train:]

    return ImageLabelDataset(split_rows=split_rows, fetcher=fetcher, split=split)


class ImageLabelDataset:
    """Torch Dataset returning (image_tensor [3,224,224], label_int) pairs.

    ``label_int`` mapping: 0 = absent, 1 = present.

    Augmentation is applied on the train split (horizontal flip, colour jitter,
    random crop) and skipped on the val split (centre crop only).

    Instantiate via ``_build_dataset()`` rather than directly.
    """

    INPUT_SIZE = (224, 224)

    def __init__(self, split_rows: list[dict], fetcher: ImageFetcher, split: str):
        # Defer torch/albumentations imports so the module can be imported in
        # environments where those packages are absent (e.g. test runner).
        import albumentations as A
        import numpy as np
        import torch
        from albumentations.pytorch import ToTensorV2

        self._torch = torch
        self._np = np
        self._rows = split_rows
        self._fetcher = fetcher
        self._split = split

        h, w = self.INPUT_SIZE
        if split == "train":
            self._transform = A.Compose([
                A.RandomResizedCrop(height=h, width=w, scale=(0.8, 1.0)),
                A.HorizontalFlip(p=0.5),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.5),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])
        else:
            self._transform = A.Compose([
                A.Resize(height=int(h * 1.14), width=int(w * 1.14)),
                A.CenterCrop(height=h, width=w),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, idx: int):
        import cv2

        row = self._rows[idx]
        image_bytes = self._fetcher.fetch(row["image_uuid"], row["filename"])

        # Decode to numpy RGB array.
        buf = self._np.frombuffer(image_bytes, dtype=self._np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Failed to decode image uuid={row['image_uuid']}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        augmented = self._transform(image=img)
        tensor = augmented["image"]  # float32 [3,H,W]

        label = 1 if row["status"] == "present" else 0
        return tensor, label

    def class_weights(self):
        """Return a 2-element tensor [w_absent, w_present] for weighted loss."""
        import torch

        n_present = sum(1 for r in self._rows if r["status"] == "present")
        n_absent = len(self._rows) - n_present
        if n_present == 0 or n_absent == 0:
            return torch.tensor([1.0, 1.0])
        total = len(self._rows)
        w_absent = total / (2.0 * n_absent)
        w_present = total / (2.0 * n_present)
        return torch.tensor([w_absent, w_present], dtype=torch.float32)
