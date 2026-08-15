"""Tests for the training dataset loader.

Requires: torch, albumentations, opencv-python.
Run inside the training container.

Images are addressed the way production addresses them: a flat UUID-keyed cache
directory, populated by the push ingest API. The dataset has no other input —
the training server never fetches images over HTTP.
"""

import tempfile
from pathlib import Path

import numpy as np

from apps.training_server.dataset import ImageLabelDataset

# ─── ImageLabelDataset ───────────────────────────────────────────────────────


def _make_jpeg_bytes(h=64, w=64) -> bytes:
    """Create a minimal valid JPEG in memory."""
    import cv2

    img = np.zeros((h, w, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return buf.tobytes()


def _make_cache_for_rows(rows: list[dict]) -> str:
    """Populate a UUID-keyed cache dir, one JPEG per row, and return its path.

    Keyed by image_uuid rather than filename — that is what ImageCache writes and
    what _CacheFetcher reads.
    """
    tmp = tempfile.mkdtemp()
    for row in rows:
        (Path(tmp) / row["image_uuid"]).write_bytes(_make_jpeg_bytes())
    return tmp


def test_dataset_len():
    rows = [{"image_uuid": f"uuid{i}", "filename": f"{i}.jpg", "status": "present"} for i in range(5)]
    ds = ImageLabelDataset(split_rows=rows, cache_dir=_make_cache_for_rows(rows), split="val")
    assert len(ds) == 5


def test_dataset_returns_tensor_and_label():
    import torch

    rows = [
        {"image_uuid": "uuid-p", "filename": "p.jpg", "status": "present"},
        {"image_uuid": "uuid-a", "filename": "a.jpg", "status": "absent"},
    ]
    ds = ImageLabelDataset(split_rows=rows, cache_dir=_make_cache_for_rows(rows), split="val")
    tensor, label = ds[0]
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (3, 224, 224)
    assert label == 1  # present


def test_dataset_label_encoding():
    rows = [
        {"image_uuid": "uuid-p", "filename": "p.jpg", "status": "present"},
        {"image_uuid": "uuid-a", "filename": "a.jpg", "status": "absent"},
    ]
    ds = ImageLabelDataset(split_rows=rows, cache_dir=_make_cache_for_rows(rows), split="val")
    _, label_p = ds[0]
    _, label_a = ds[1]
    assert label_p == 1
    assert label_a == 0


def test_dataset_class_weights_balanced():
    import torch

    rows = [
        {"image_uuid": "uuid-p", "filename": "p.jpg", "status": "present"},
        {"image_uuid": "uuid-a", "filename": "a.jpg", "status": "absent"},
    ]
    ds = ImageLabelDataset(split_rows=rows, cache_dir=_make_cache_for_rows(rows), split="train")
    weights = ds.class_weights()
    assert isinstance(weights, torch.Tensor)
    assert weights.shape == (2,)
    assert torch.allclose(weights, torch.tensor([1.0, 1.0]))


def test_build_dataset_train_val_split():
    """Deterministic 80/20 split: sorted by UUID, last 20% is val."""
    from apps.training_server.dataset import build_dataset_from_manifest

    n = 10
    manifest = [{"image_uuid": f"uuid-{i:03d}", "filename": f"{i}.jpg", "status": "present"} for i in range(n)]
    cache_dir = _make_cache_for_rows(manifest)

    ds_train = build_dataset_from_manifest(manifest, cache_dir, split="train")
    ds_val = build_dataset_from_manifest(manifest, cache_dir, split="val")

    assert len(ds_train) == 8
    assert len(ds_val) == 2
