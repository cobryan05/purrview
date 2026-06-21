"""Manifest-driven dataset for the training server.

Images arrive via the push ingest API and are stored in the uuid-keyed
``ImageCache``.  This module builds a torch Dataset from a pushed manifest
(list of ``{image_uuid, filename, status}`` dicts) + the local cache.

The split is deterministic: images are sorted by UUID then sliced.
This is reproducible across runs regardless of submission order.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def build_dataset_from_manifest(
    manifest: list[dict],
    cache_dir: str,
    split: str = "train",
    val_fraction: float = 0.2,
) -> "ImageLabelDataset":
    """Build an ``ImageLabelDataset`` from a pushed manifest + local cache.

    Args:
        manifest:     List of ``{image_uuid, filename, status}`` dicts.
                      ``status`` must be ``"present"`` or ``"absent"``.
        cache_dir:    Path to the UUID-keyed image cache directory.
        split:        ``"train"`` or ``"val"``.
        val_fraction: Fraction of images to hold out for validation.

    Raises:
        ValueError: When no manifest rows are available or required images are
                    missing from the cache.
    """
    if not manifest:
        raise ValueError("Empty manifest — no images to train on")

    cache = Path(cache_dir)

    # Validate cache completeness.
    missing = [r["image_uuid"] for r in manifest if not (cache / r["image_uuid"]).exists()]
    if missing:
        raise ValueError(
            f"{len(missing)} images required by the manifest are missing from the cache: "
            f"{missing[:5]}{'...' if len(missing) > 5 else ''}"
        )

    # Deterministic train/val split: sort by UUID (stable), then slice.
    rows_sorted = sorted(manifest, key=lambda r: r["image_uuid"])
    n_val = max(1, int(len(rows_sorted) * val_fraction))
    n_train = len(rows_sorted) - n_val
    if split == "train":
        split_rows = rows_sorted[:n_train]
    else:
        split_rows = rows_sorted[n_train:]

    return ImageLabelDataset(split_rows=split_rows, cache_dir=cache_dir, split=split)


class ImageLabelDataset:
    """Torch Dataset returning (image_tensor [3,224,224], label_int) pairs.

    ``label_int`` mapping: 0 = absent, 1 = present.

    Augmentation: train split gets random crop + flip + colour jitter;
    val split gets centre crop only.
    """

    INPUT_SIZE = (224, 224)

    def __init__(self, split_rows: list[dict], cache_dir: str, split: str):
        # Defer torch/albumentations imports — only available in training container.
        import albumentations as A
        import numpy as np
        import torch
        from albumentations.pytorch import ToTensorV2

        self._torch = torch
        self._np = np
        self._rows = split_rows
        self._cache_dir = Path(cache_dir)
        self._split = split

        h, w = self.INPUT_SIZE
        if split == "train":
            self._transform = A.Compose(
                [
                    A.RandomResizedCrop(height=h, width=w, scale=(0.8, 1.0)),
                    A.HorizontalFlip(p=0.5),
                    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.5),
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    ToTensorV2(),
                ]
            )
        else:
            self._transform = A.Compose(
                [
                    A.Resize(height=int(h * 1.14), width=int(w * 1.14)),
                    A.CenterCrop(height=h, width=w),
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    ToTensorV2(),
                ]
            )

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, idx: int):
        import cv2

        row = self._rows[idx]
        image_path = self._cache_dir / row["image_uuid"]
        image_bytes = image_path.read_bytes()

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
        """Return a 2-element tensor [w_absent, w_present] for weighted CE loss."""
        import torch

        n_present = sum(1 for r in self._rows if r["status"] == "present")
        n_absent = len(self._rows) - n_present
        if n_present == 0 or n_absent == 0:
            return torch.tensor([1.0, 1.0])
        total = len(self._rows)
        w_absent = total / (2.0 * n_absent)
        w_present = total / (2.0 * n_present)
        return torch.tensor([w_absent, w_present], dtype=torch.float32)


def build_multilabel_dataset_from_manifest(
    manifest: list[dict],
    label_uuids: list[str],
    label_targets: dict[str, dict[str, int]],
    cache_dir: str,
    split: str = "train",
    val_fraction: float = 0.2,
) -> "MultiLabelDataset":
    """Build a MultiLabelDataset from a pushed manifest + per-label status targets.

    Args:
        manifest:      List of ``{image_uuid, filename, status}`` dicts (unique images).
                       ``status`` is ignored — per-label status is in ``label_targets``.
        label_uuids:   Ordered list of label UUIDs defining the output vector order.
        label_targets: ``{image_uuid: {label_uuid: 0_or_1}}`` mapping.
                       Missing entries are treated as unknown (-1) and masked from loss.
        cache_dir:     Path to UUID-keyed image cache.
        split:         ``"train"`` or ``"val"``.
        val_fraction:  Fraction of images held out for validation.

    Raises:
        ValueError: When no manifest rows are provided or cache is incomplete.
    """
    if not manifest:
        raise ValueError("Empty manifest — no images to train on")

    cache = Path(cache_dir)
    missing = [r["image_uuid"] for r in manifest if not (cache / r["image_uuid"]).exists()]
    if missing:
        raise ValueError(
            f"{len(missing)} images required by the manifest are missing from the cache: "
            f"{missing[:5]}{'...' if len(missing) > 5 else ''}"
        )

    uuids_sorted = sorted(r["image_uuid"] for r in manifest)
    n_val = max(1, int(len(uuids_sorted) * val_fraction))
    n_train = len(uuids_sorted) - n_val
    split_uuids = uuids_sorted[:n_train] if split == "train" else uuids_sorted[n_train:]

    return MultiLabelDataset(
        image_uuids=split_uuids,
        label_uuids=label_uuids,
        label_targets=label_targets,
        cache_dir=cache_dir,
        split=split,
    )


class MultiLabelDataset:
    """Torch Dataset returning ``(image_tensor [3,224,224], target [N])`` pairs.

    ``target[i]`` is:
        1.0  — label i is present for this image
        0.0  — label i is absent for this image
       -1.0  — label i has no annotation for this image (masked from loss)

    Augmentation mirrors ``ImageLabelDataset``.
    """

    INPUT_SIZE = (224, 224)

    def __init__(
        self,
        image_uuids: list[str],
        label_uuids: list[str],
        label_targets: dict[str, dict[str, int]],
        cache_dir: str,
        split: str,
    ):
        import albumentations as A
        import numpy as np
        import torch
        from albumentations.pytorch import ToTensorV2

        self._torch = torch
        self._np = np
        self._image_uuids = image_uuids
        self._label_uuids = label_uuids
        self._label_targets = label_targets
        self._cache_dir = Path(cache_dir)

        h, w = self.INPUT_SIZE
        if split == "train":
            self._transform = A.Compose(
                [
                    A.RandomResizedCrop(height=h, width=w, scale=(0.8, 1.0)),
                    A.HorizontalFlip(p=0.5),
                    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.5),
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    ToTensorV2(),
                ]
            )
        else:
            self._transform = A.Compose(
                [
                    A.Resize(height=int(h * 1.14), width=int(w * 1.14)),
                    A.CenterCrop(height=h, width=w),
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    ToTensorV2(),
                ]
            )

    def __len__(self) -> int:
        return len(self._image_uuids)

    def __getitem__(self, idx: int):
        import cv2

        uuid = self._image_uuids[idx]
        image_path = self._cache_dir / uuid
        image_bytes = image_path.read_bytes()

        buf = self._np.frombuffer(image_bytes, dtype=self._np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Failed to decode image uuid={uuid}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        augmented = self._transform(image=img)
        tensor = augmented["image"]

        per_label = self._label_targets.get(uuid, {})
        target = self._torch.tensor(
            [float(per_label.get(luuid, -1)) for luuid in self._label_uuids],
            dtype=self._torch.float32,
        )
        return tensor, target

    def pos_weights(self):
        """Per-label positive-class weight for ``BCEWithLogitsLoss(pos_weight=...)``.

        Returns a tensor of shape ``[N]`` where each element is
        ``n_absent / n_present`` for that label (clamped to avoid division by zero).
        """
        import torch

        N = len(self._label_uuids)
        n_present = torch.zeros(N)
        n_absent = torch.zeros(N)
        for uuid in self._image_uuids:
            per_label = self._label_targets.get(uuid, {})
            for i, luuid in enumerate(self._label_uuids):
                val = per_label.get(luuid, -1)
                if val == 1:
                    n_present[i] += 1
                elif val == 0:
                    n_absent[i] += 1
        return n_absent / (n_present + 1e-8)
