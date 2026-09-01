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


# ─── CropDataset ─────────────────────────────────────────────────────────────


def _crop_manifest(n_images: int, crops_per_image: int, num_classes: int = 2):
    """A manifest + crop_targets pair with `crops_per_image` boxes on every image."""
    manifest = [{"image_uuid": f"img-{i:03d}", "filename": f"{i}.jpg", "status": "present"} for i in range(n_images)]
    crop_targets = {
        m["image_uuid"]: [
            {"class_idx": j % num_classes, "x": 0.1 * j, "y": 0.1, "width": 0.2, "height": 0.2}
            for j in range(crops_per_image)
        ]
        for m in manifest
    }
    return manifest, crop_targets, _make_cache_for_rows(manifest)


def test_crop_dataset_yields_one_sample_per_box():
    """Unlike the image-level datasets, length is counted in boxes, not images."""
    from apps.training_server.dataset import build_crop_dataset_from_manifest

    manifest, crop_targets, cache_dir = _crop_manifest(n_images=10, crops_per_image=3)
    ds = build_crop_dataset_from_manifest(manifest, crop_targets, 2, cache_dir, split="train")

    assert len(ds) == 8 * 3  # 8 train images x 3 boxes
    tensor, label = ds[0]
    assert tensor.shape == (3, 224, 224)
    assert label in (0, 1)


def test_crop_dataset_splits_by_image_not_by_box():
    """No source image may appear in both splits.

    Boxes from one frame are near-duplicates of each other — and, with a stream
    collector feeding the dataset, of the neighbouring frames' boxes too. Splitting
    the flattened box list would put those on both sides and report a validation
    accuracy the model has not earned.
    """
    from apps.training_server.dataset import build_crop_dataset_from_manifest

    manifest, crop_targets, cache_dir = _crop_manifest(n_images=10, crops_per_image=4)
    ds_train = build_crop_dataset_from_manifest(manifest, crop_targets, 2, cache_dir, split="train")
    ds_val = build_crop_dataset_from_manifest(manifest, crop_targets, 2, cache_dir, split="val")

    assert set(ds_train.image_uuids).isdisjoint(ds_val.image_uuids)
    assert len(ds_train) + len(ds_val) == 10 * 4


def test_crop_dataset_drops_degenerate_and_out_of_range_boxes():
    """A zero-extent box or an out-of-range class index is dropped at build time,
    so len() is honest and no DataLoader worker dies partway through an epoch."""
    from apps.training_server.dataset import build_crop_dataset_from_manifest

    manifest = [{"image_uuid": f"img-{i:03d}", "filename": f"{i}.jpg", "status": "present"} for i in range(5)]
    cache_dir = _make_cache_for_rows(manifest)
    crop_targets = {
        m["image_uuid"]: [
            {"class_idx": 0, "x": 0.1, "y": 0.1, "width": 0.2, "height": 0.2},  # keep
            {"class_idx": 0, "x": 0.1, "y": 0.1, "width": 0.0, "height": 0.2},  # zero width
            {"class_idx": 7, "x": 0.1, "y": 0.1, "width": 0.2, "height": 0.2},  # class out of range
        ]
        for m in manifest
    }

    ds = build_crop_dataset_from_manifest(manifest, crop_targets, 2, cache_dir, split="train")
    assert len(ds) == 4  # 4 train images x 1 usable box
    for i in range(len(ds)):
        tensor, _ = ds[i]
        assert tensor.shape == (3, 224, 224)


def test_crop_dataset_class_weights_favour_the_rarer_class():
    """Inverse-frequency weighting: the class with fewer crops gets the larger
    weight, so a lopsided cat gets real gradient instead of being predicted away."""
    from apps.training_server.dataset import build_crop_dataset_from_manifest

    manifest = [{"image_uuid": f"img-{i:03d}", "filename": f"{i}.jpg", "status": "present"} for i in range(10)]
    cache_dir = _make_cache_for_rows(manifest)
    # Class 0 on every image, class 1 on only the first two.
    crop_targets = {
        m["image_uuid"]: (
            [{"class_idx": 0, "x": 0.1, "y": 0.1, "width": 0.2, "height": 0.2}]
            + (
                [{"class_idx": 1, "x": 0.5, "y": 0.5, "width": 0.2, "height": 0.2}]
                if int(m["image_uuid"].split("-")[1]) < 2
                else []
            )
        )
        for m in manifest
    }

    ds = build_crop_dataset_from_manifest(manifest, crop_targets, 2, cache_dir, split="train")
    counts = ds.class_counts()
    weights = ds.class_weights()

    assert counts[1] < counts[0]
    assert weights[1] > weights[0]
