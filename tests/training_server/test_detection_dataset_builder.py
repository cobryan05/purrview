"""Tests for build_yolo_detection_dataset.

Verifies the YOLO dataset layout, train/val split, data.yaml contents,
and annotation format. No real training is performed.
"""

import os
from pathlib import Path

import pytest
import yaml

from apps.training_server.dataset import build_yolo_detection_dataset

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _make_cache(tmp_path: Path, uuids: list[str]) -> Path:
    """Create a flat UUID-keyed cache directory with 1-byte placeholder files."""
    cache = tmp_path / "cache"
    cache.mkdir()
    for u in uuids:
        (cache / u).write_bytes(b"\xff\xd8\xff\xe0")  # minimal JPEG-ish header
    return cache


def _make_manifest(uuids: list[str]) -> list[dict]:
    return [{"image_uuid": u, "filename": f"{u}.jpg", "status": "present"} for u in uuids]


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestBuildYoloDetectionDataset:
    def test_directory_layout(self, tmp_path):
        """Images and labels directories are created for both splits."""
        uuids = [f"uuid-{i:02d}" for i in range(5)]
        cache = _make_cache(tmp_path, uuids)
        out = tmp_path / "dataset"
        build_yolo_detection_dataset(
            manifest=_make_manifest(uuids),
            annotations={u: [] for u in uuids},
            class_names=["cat"],
            cache_dir=str(cache),
            out_dir=str(out),
        )
        for split in ("train", "val"):
            assert (out / "images" / split).is_dir()
            assert (out / "labels" / split).is_dir()

    def test_data_yaml_contents(self, tmp_path):
        """data.yaml has correct train/val paths, nc, and names."""
        uuids = [f"uuid-{i:02d}" for i in range(5)]
        cache = _make_cache(tmp_path, uuids)
        out = tmp_path / "dataset"
        build_yolo_detection_dataset(
            manifest=_make_manifest(uuids),
            annotations={u: [] for u in uuids},
            class_names=["cat", "dog"],
            cache_dir=str(cache),
            out_dir=str(out),
        )
        data = yaml.safe_load((out / "data.yaml").read_text())
        assert data["nc"] == 2
        assert data["names"] == ["cat", "dog"]
        assert str(out / "images" / "train") in data["train"]
        assert str(out / "images" / "val") in data["val"]

    def test_deterministic_split(self, tmp_path):
        """Split is deterministic: sorted by UUID, last 20% is val."""
        uuids = [f"uuid-{i:02d}" for i in range(10)]
        cache = _make_cache(tmp_path, uuids)
        out = tmp_path / "dataset"
        result = build_yolo_detection_dataset(
            manifest=_make_manifest(uuids),
            annotations={u: [] for u in uuids},
            class_names=["cat"],
            cache_dir=str(cache),
            out_dir=str(out),
            val_fraction=0.2,
        )
        # 10 images × 0.2 = 2 val, 8 train
        assert len(result["train_uuids"]) == 8
        assert len(result["val_uuids"]) == 2
        # Sorted by UUID: last 2 go to val.
        sorted_uuids = sorted(uuids)
        assert result["train_uuids"] == sorted_uuids[:8]
        assert result["val_uuids"] == sorted_uuids[8:]

    def test_annotation_file_contents(self, tmp_path):
        """Label files have one YOLO line per annotation: cls cx cy w h."""
        uuids = ["img-a", "img-b", "img-c", "img-d", "img-e"]
        cache = _make_cache(tmp_path, uuids)
        out = tmp_path / "dataset"
        annotations = {
            "img-a": [[0, 0.5, 0.5, 0.3, 0.4]],
            "img-b": [[1, 0.2, 0.3, 0.1, 0.1], [0, 0.8, 0.8, 0.2, 0.2]],
        }
        # remaining images have no bboxes
        for u in ["img-c", "img-d", "img-e"]:
            annotations[u] = []

        build_yolo_detection_dataset(
            manifest=_make_manifest(uuids),
            annotations=annotations,
            class_names=["cat", "dog"],
            cache_dir=str(cache),
            out_dir=str(out),
        )
        # Find img-a label file (could be in train or val split).
        lbl_a = None
        for split in ("train", "val"):
            candidate = out / "labels" / split / "img-a.txt"
            if candidate.exists():
                lbl_a = candidate
                break
        assert lbl_a is not None, "label file for img-a not found"
        lines = lbl_a.read_text().strip().split("\n")
        assert len(lines) == 1
        parts = lines[0].split()
        assert parts[0] == "0"
        assert abs(float(parts[1]) - 0.5) < 1e-5
        assert abs(float(parts[2]) - 0.5) < 1e-5
        assert abs(float(parts[3]) - 0.3) < 1e-5
        assert abs(float(parts[4]) - 0.4) < 1e-5

    def test_empty_annotation_gives_empty_label_file(self, tmp_path):
        """Images with no bboxes get an empty label file (valid negative sample)."""
        uuids = ["neg-a", "neg-b", "neg-c", "neg-d", "neg-e"]
        cache = _make_cache(tmp_path, uuids)
        out = tmp_path / "dataset"
        build_yolo_detection_dataset(
            manifest=_make_manifest(uuids),
            annotations={u: [] for u in uuids},
            class_names=["cat"],
            cache_dir=str(cache),
            out_dir=str(out),
        )
        for split in ("train", "val"):
            for lbl_file in (out / "labels" / split).glob("*.txt"):
                assert lbl_file.read_text().strip() == ""

    def test_image_files_copied_to_split_dirs(self, tmp_path):
        """Image files from the cache are copied into the correct split directories."""
        uuids = [f"img-{i:02d}" for i in range(5)]
        cache = _make_cache(tmp_path, uuids)
        out = tmp_path / "dataset"
        result = build_yolo_detection_dataset(
            manifest=_make_manifest(uuids),
            annotations={u: [] for u in uuids},
            class_names=["cat"],
            cache_dir=str(cache),
            out_dir=str(out),
        )
        for u in result["train_uuids"]:
            assert (out / "images" / "train" / f"{u}.jpg").exists()
        for u in result["val_uuids"]:
            assert (out / "images" / "val" / f"{u}.jpg").exists()

    def test_single_image_all_train(self, tmp_path):
        """A single image goes to train (no val when only 1 image)."""
        uuids = ["solo"]
        cache = _make_cache(tmp_path, uuids)
        out = tmp_path / "dataset"
        result = build_yolo_detection_dataset(
            manifest=_make_manifest(uuids),
            annotations={"solo": []},
            class_names=["cat"],
            cache_dir=str(cache),
            out_dir=str(out),
        )
        assert result["train_uuids"] == ["solo"]
        assert result["val_uuids"] == []
