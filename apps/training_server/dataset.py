"""Manifest-driven dataset for the training server.

Images arrive via the push ingest API and are stored in the uuid-keyed
``ImageCache``.  This module builds a torch Dataset from a pushed manifest
(list of ``{image_uuid, filename, status}`` dicts) + the local cache.

The split is deterministic: images are sorted by UUID then sliced.
This is reproducible across runs regardless of submission order.

Images are always read from the local cache. The training server never calls
back into db_server — every connection in the training path is outbound from
tasks_server, which is what lets the training server run on a host that cannot
reach the rest of the stack.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from apps.helpers.classifier_utils import DEFAULT_CROP_PADDING, crop_region
from apps.training_server.augmentation import (
    DEFAULT_AUGMENTATION_PRESET,
    build_classifier_train_transform,
    resolve_augmentation_preset,
)

logger = logging.getLogger(__name__)


class _CacheFetcher:
    """Internal fetcher that reads from a UUID-keyed cache directory."""

    def __init__(self, cache_dir: str):
        self._dir = Path(cache_dir)

    def fetch(self, uuid: str, filename: str) -> bytes:
        return (self._dir / uuid).read_bytes()


# ---------------------------------------------------------------------------
# Public dataset builders
# ---------------------------------------------------------------------------


def build_dataset_from_manifest(
    manifest: list[dict],
    cache_dir: str,
    split: str = "train",
    val_fraction: float = 0.2,
    augmentation: str = DEFAULT_AUGMENTATION_PRESET,
) -> "ImageLabelDataset":
    """Build an ``ImageLabelDataset`` from a pushed manifest + local UUID-keyed cache.

    Args:
        manifest:     List of ``{image_uuid, filename, status}`` dicts.
                      ``status`` must be ``"present"`` or ``"absent"``.
        cache_dir:    Path to the UUID-keyed image cache directory.
        split:        ``"train"`` or ``"val"``.
        val_fraction: Fraction of images to hold out for validation.
        augmentation: Preset name from ``augmentation.AUGMENTATION_PRESETS``,
                      applied to the train split only (val stays deterministic).

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

    return ImageLabelDataset(split_rows=split_rows, cache_dir=cache_dir, split=split, augmentation=augmentation)


# ---------------------------------------------------------------------------
# Dataset classes
# ---------------------------------------------------------------------------


class ImageLabelDataset:
    """Torch Dataset returning (image_tensor [3,224,224], label_int) pairs.

    ``label_int`` mapping: 0 = absent, 1 = present.

    Augmentation: train split gets random crop + flip + colour jitter;
    val split gets centre crop only.

    Reads from the UUID-keyed cache, like MultiLabelDataset.
    """

    INPUT_SIZE = (224, 224)

    def __init__(
        self,
        split_rows: list[dict],
        split: str,
        cache_dir: str,
        augmentation: str = DEFAULT_AUGMENTATION_PRESET,
    ):
        fetcher = _CacheFetcher(cache_dir)

        # Defer torch/albumentations imports — only available in training container.
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
            self._transform = build_classifier_train_transform((h, w), resolve_augmentation_preset(augmentation))
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
        image_bytes = self._fetcher.fetch(row["image_uuid"], row["filename"])

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
    augmentation: str = DEFAULT_AUGMENTATION_PRESET,
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
        augmentation=augmentation,
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
        augmentation: str = DEFAULT_AUGMENTATION_PRESET,
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
            self._transform = build_classifier_train_transform((h, w), resolve_augmentation_preset(augmentation))
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


# ---------------------------------------------------------------------------
# Crop classification dataset
# ---------------------------------------------------------------------------


def build_crop_dataset_from_manifest(
    manifest: list[dict],
    crop_targets: dict[str, list[dict]],
    num_classes: int,
    cache_dir: str,
    split: str = "train",
    val_fraction: float = 0.2,
    crop_padding: float = DEFAULT_CROP_PADDING,
    augmentation: str = DEFAULT_AUGMENTATION_PRESET,
) -> "CropDataset":
    """Build a ``CropDataset`` from a pushed manifest + per-image crop boxes.

    Args:
        manifest:     List of ``{image_uuid, filename, status}`` dicts (unique images —
                      the same shape every other builder takes, so image push is
                      unchanged). ``status`` is ignored; a crop's class is its own.
        crop_targets: ``{image_uuid: [{class_idx, x, y, width, height}]}``, relative
                      coords. Images absent from this map contribute no samples.
        num_classes:  Number of output classes (defines the label space).
        cache_dir:    Path to UUID-keyed image cache.
        split:        ``"train"`` or ``"val"``.
        val_fraction: Fraction of *images* held out for validation.
        crop_padding: Fraction of box size added as context on each side.

    The split is by **image**, not by crop. Splitting crops directly would put two
    boxes of the same animal from the same frame — and, with a stream collector
    feeding the dataset, near-identical boxes from adjacent frames of the same
    image — on both sides of the split, which inflates val accuracy without
    improving the model.

    Raises:
        ValueError: When no manifest rows are provided, the cache is incomplete, or
                    the split contains no crops at all.
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

    # Flatten to one row per crop. Degenerate boxes (zero/negative extent) are
    # dropped here rather than at __getitem__ time so len() is honest and a
    # DataLoader worker can't die mid-epoch on a box that was never usable.
    rows: list[dict] = []
    for image_uuid in split_uuids:
        for crop in crop_targets.get(image_uuid, []):
            if crop["width"] <= 0 or crop["height"] <= 0:
                continue
            if not (0 <= int(crop["class_idx"]) < num_classes):
                continue
            rows.append({"image_uuid": image_uuid, **crop})

    if not rows:
        raise ValueError(f"No usable crops in the {split!r} split — check that the in-scope bboxes are verified")

    return CropDataset(
        rows=rows,
        num_classes=num_classes,
        cache_dir=cache_dir,
        split=split,
        crop_padding=crop_padding,
        augmentation=augmentation,
    )


class CropDataset:
    """Torch Dataset returning ``(image_tensor [3,224,224], class_idx)`` pairs, one
    sample per bounding box rather than per image.

    Each sample is the box's region cut out of its source image, padded by
    ``crop_padding`` on each side for context (a tight box on a curled-up cat is
    mostly cat, but the ears/tail just outside it carry a lot of the signal), then
    put through the same augmentation pipeline as the whole-image datasets.

    Unlike ``MultiLabelDataset`` this is single-label: a crop is exactly one class,
    so the head is softmax/cross-entropy, not per-label sigmoid.
    """

    INPUT_SIZE = (224, 224)

    def __init__(
        self,
        rows: list[dict],
        num_classes: int,
        cache_dir: str,
        split: str,
        crop_padding: float = DEFAULT_CROP_PADDING,
        augmentation: str = DEFAULT_AUGMENTATION_PRESET,
    ):
        import albumentations as A
        import numpy as np
        import torch
        from albumentations.pytorch import ToTensorV2

        self._torch = torch
        self._np = np
        self._rows = rows
        self._num_classes = num_classes
        self._cache_dir = Path(cache_dir)
        self._crop_padding = crop_padding

        h, w = self.INPUT_SIZE
        if split == "train":
            self._transform = build_classifier_train_transform((h, w), resolve_augmentation_preset(augmentation))
        else:
            # No 1.14x resize + centre crop here (unlike the whole-image datasets):
            # the region is already the subject, and centre-cropping it would shave
            # off the context padding this dataset deliberately added.
            self._transform = A.Compose(
                [
                    A.Resize(height=h, width=w),
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    ToTensorV2(),
                ]
            )

    @property
    def image_uuids(self) -> list[str]:
        """Unique source images backing this split, in first-seen order."""
        seen: dict[str, None] = {}
        for row in self._rows:
            seen.setdefault(row["image_uuid"], None)
        return list(seen)

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, idx: int):
        import cv2

        row = self._rows[idx]
        image_bytes = (self._cache_dir / row["image_uuid"]).read_bytes()

        buf = self._np.frombuffer(image_bytes, dtype=self._np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Failed to decode image uuid={row['image_uuid']}")

        crop = crop_region(img, row["x"], row["y"], row["width"], row["height"], self._crop_padding)
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

        augmented = self._transform(image=crop)
        return augmented["image"], int(row["class_idx"])

    def class_weights(self):
        """Per-class weight for ``CrossEntropyLoss(weight=...)``, inverse-frequency
        normalised so the mean weight is 1. Classes with no samples get weight 1
        (they contribute no loss anyway)."""
        import torch

        counts = torch.zeros(self._num_classes)
        for row in self._rows:
            counts[int(row["class_idx"])] += 1
        weights = torch.where(counts > 0, len(self._rows) / (self._num_classes * counts.clamp(min=1)), torch.ones(1))
        return weights.float()

    def class_counts(self) -> list[int]:
        counts = [0] * self._num_classes
        for row in self._rows:
            counts[int(row["class_idx"])] += 1
        return counts


# ---------------------------------------------------------------------------
# YOLO detection dataset builder
# ---------------------------------------------------------------------------


def build_yolo_detection_dataset(
    manifest: list[dict],
    annotations: dict[str, list],
    class_names: list[str],
    cache_dir: str,
    out_dir: str,
    val_fraction: float = 0.2,
) -> dict:
    """Materialise a YOLO-format dataset directory from the UUID-keyed cache.

    Args:
        manifest:     List of ``{image_uuid, filename, status}`` dicts (all images).
        annotations:  ``{image_uuid: [[cls_idx, cx, cy, w, h], ...]}`` — may be empty
                      list for negative-sample images.
        class_names:  Ordered list of class name strings.
        cache_dir:    Path to the UUID-keyed image cache (flat, one file per UUID).
        out_dir:      Directory to write the YOLO dataset structure into.
        val_fraction: Fraction of images to use for validation (default 0.2).

    Returns:
        ``{data_yaml: str, train_uuids: list[str], val_uuids: list[str]}``
    """
    import yaml

    cache = Path(cache_dir)
    out = Path(out_dir)

    # Deterministic split: sort by UUID then slice.
    sorted_entries = sorted(manifest, key=lambda e: e["image_uuid"])
    n_val = max(1, int(len(sorted_entries) * val_fraction)) if len(sorted_entries) > 1 else 0
    n_train = len(sorted_entries) - n_val
    train_entries = sorted_entries[:n_train]
    val_entries = sorted_entries[n_train:]

    for split, entries in (("train", train_entries), ("val", val_entries)):
        img_dir = out / "images" / split
        lbl_dir = out / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for entry in entries:
            uuid = entry["image_uuid"]
            src = cache / uuid
            if not src.exists():
                logger.warning("Cache miss for %s — skipping", uuid)
                continue
            # Copy image as <uuid>.jpg (extension is cosmetic; YOLO reads by filename stem)
            dst_img = img_dir / f"{uuid}.jpg"
            shutil.copy2(src, dst_img)

            # Write label file (empty = negative sample).
            ann = annotations.get(uuid, [])
            dst_lbl = lbl_dir / f"{uuid}.txt"
            lines = [f"{int(a[0])} {a[1]:.6f} {a[2]:.6f} {a[3]:.6f} {a[4]:.6f}" for a in ann]
            dst_lbl.write_text("\n".join(lines))

    data_yaml_path = out / "data.yaml"
    data_yaml_content = {
        "train": str(out / "images" / "train"),
        "val": str(out / "images" / "val"),
        "nc": len(class_names),
        "names": class_names,
    }
    data_yaml_path.write_text(yaml.dump(data_yaml_content, default_flow_style=False))

    return {
        "data_yaml": str(data_yaml_path),
        "train_uuids": [e["image_uuid"] for e in train_entries],
        "val_uuids": [e["image_uuid"] for e in val_entries],
    }
