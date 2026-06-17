"""Tests for the training dataset loader.

Requires: torch, albumentations, opencv-python.
Run inside the training container.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from apps.training_server.dataset import (
    FilesystemImageFetcher,
    HttpImageFetcher,
    ImageLabelDataset,
    _build_dataset,
)


# ─── FilesystemImageFetcher ──────────────────────────────────────────────────


def test_filesystem_fetcher_reads_file(tmp_path):
    data = b"\xff\xd8\xff\xe0test"
    (tmp_path / "img.jpg").write_bytes(data)
    fetcher = FilesystemImageFetcher(fs_root=str(tmp_path))
    assert fetcher.fetch("uuid1", "img.jpg") == data


def test_filesystem_fetcher_prefix_remap(tmp_path):
    sub = tmp_path / "remapped"
    sub.mkdir()
    data = b"imgdata"
    (sub / "a.jpg").write_bytes(data)

    fetcher = FilesystemImageFetcher(
        fs_root=str(tmp_path),
        path_prefix_map={"/original": "/remapped"},
    )
    assert fetcher.fetch("uuid1", "/original/a.jpg") == data


def test_filesystem_fetcher_missing_file(tmp_path):
    fetcher = FilesystemImageFetcher(fs_root=str(tmp_path))
    with pytest.raises(FileNotFoundError):
        fetcher.fetch("uuid1", "nonexistent.jpg")


# ─── HttpImageFetcher ────────────────────────────────────────────────────────


def test_http_fetcher_caches_result(tmp_path):
    import base64

    raw = b"\xff\xd8\xff" + b"\x00" * 20
    b64 = base64.b64encode(raw).decode()

    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {"status": "success", "image_base64": b64}

    fetcher = HttpImageFetcher(db_base_url="http://fake", cache_dir=str(tmp_path))

    with patch("httpx.get", return_value=mock_resp) as mock_get:
        result1 = fetcher.fetch("uuid1", "img.jpg")
        result2 = fetcher.fetch("uuid1", "img.jpg")  # should use cache

    assert result1 == raw
    assert result2 == raw
    assert mock_get.call_count == 1  # second call used cache


def test_http_fetcher_raises_on_failure(tmp_path):
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {"status": "failure", "message": "not found"}

    fetcher = HttpImageFetcher(db_base_url="http://fake", cache_dir=str(tmp_path))
    with patch("httpx.get", return_value=mock_resp):
        with pytest.raises(ValueError, match="not found"):
            fetcher.fetch("uuid1", "img.jpg")


def test_http_fetcher_cleanup_removes_tmp():
    fetcher = HttpImageFetcher(db_base_url="http://fake")
    tmp_dir = Path(fetcher._tmp)
    assert tmp_dir.exists()
    fetcher.cleanup()
    assert not tmp_dir.exists()


# ─── ImageLabelDataset ───────────────────────────────────────────────────────


def _make_jpeg_bytes(h=64, w=64) -> bytes:
    """Create a minimal valid JPEG in memory."""
    import cv2

    img = np.zeros((h, w, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return buf.tobytes()


def _make_fetcher_for_rows(rows: list[dict]) -> FilesystemImageFetcher:
    """Build a temp dir with one JPEG per row, return a FilesystemImageFetcher."""
    tmp = tempfile.mkdtemp()
    for row in rows:
        fname = Path(row["filename"]).name
        (Path(tmp) / fname).write_bytes(_make_jpeg_bytes())
    return FilesystemImageFetcher(fs_root=tmp)


def test_dataset_len():
    rows = [{"image_uuid": f"uuid{i}", "filename": f"{i}.jpg", "status": "present"} for i in range(5)]
    fetcher = _make_fetcher_for_rows(rows)
    ds = ImageLabelDataset(split_rows=rows, fetcher=fetcher, split="val")
    assert len(ds) == 5


def test_dataset_returns_tensor_and_label():
    import torch

    rows = [
        {"image_uuid": "uuid-p", "filename": "p.jpg", "status": "present"},
        {"image_uuid": "uuid-a", "filename": "a.jpg", "status": "absent"},
    ]
    fetcher = _make_fetcher_for_rows(rows)
    ds = ImageLabelDataset(split_rows=rows, fetcher=fetcher, split="val")
    tensor, label = ds[0]
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (3, 224, 224)
    assert label == 1  # present


def test_dataset_label_encoding():
    rows = [
        {"image_uuid": "uuid-p", "filename": "p.jpg", "status": "present"},
        {"image_uuid": "uuid-a", "filename": "a.jpg", "status": "absent"},
    ]
    fetcher = _make_fetcher_for_rows(rows)
    ds = ImageLabelDataset(split_rows=rows, fetcher=fetcher, split="val")
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
    fetcher = _make_fetcher_for_rows(rows)
    ds = ImageLabelDataset(split_rows=rows, fetcher=fetcher, split="train")
    weights = ds.class_weights()
    assert isinstance(weights, torch.Tensor)
    assert weights.shape == (2,)
    assert torch.allclose(weights, torch.tensor([1.0, 1.0]))


def test_build_dataset_train_val_split():
    """Deterministic 80/20 split: sorted by UUID, last 20% is val."""
    n = 10
    rows_payload = [
        {"image_uuid": f"uuid-{i:03d}", "filename": f"{i}.jpg", "status": "present", "label_uuid": "l"}
        for i in range(n)
    ]

    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {"images": rows_payload}

    tmp = tempfile.mkdtemp()
    for i in range(n):
        (Path(tmp) / f"{i}.jpg").write_bytes(_make_jpeg_bytes())
    fetcher = FilesystemImageFetcher(fs_root=tmp)

    with patch("httpx.get", return_value=mock_resp):
        ds_train = _build_dataset("label-uuid", "http://fake", fetcher, split="train")
        ds_val = _build_dataset("label-uuid", "http://fake", fetcher, split="val")

    assert len(ds_train) == 8
    assert len(ds_val) == 2
