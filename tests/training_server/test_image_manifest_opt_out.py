"""The per-training-run "remember exact image list" opt-out.

Every trainer writes its train/val image uuid lists into the sidecar JSON beside
the ONNX. This covers the one helper all four of them route that through
(`trainers/base.py:image_manifest_fields`) plus a guard that none of them has
drifted back to writing the keys directly — the failure mode this design exists
to prevent is three trainers honouring the opt-out and the fourth ignoring it.
"""

from pathlib import Path

import pytest

from apps.training_server.trainers.base import RECORD_IMAGE_MANIFEST_KEY, image_manifest_fields

TRAINERS_DIR = Path(__file__).resolve().parents[2] / "apps" / "training_server" / "trainers"

#: The trainers that write a sidecar containing an image manifest.
SIDECAR_TRAINERS = [
    "binary_image_classifier.py",
    "multi_label_classifier.py",
    "object_detection_trainer.py",
    "ultralytics_detection_trainer.py",
]


class TestImageManifestFields:
    def test_records_by_default_when_key_absent(self):
        """Absent key means ON — this is the whole reason the opt-out is an
        opt-OUT: no already-saved training config carries the key."""
        assert image_manifest_fields({}, ["a", "b"], ["c"]) == {
            "train_image_uuids": ["a", "b"],
            "val_image_uuids": ["c"],
        }

    def test_records_when_explicitly_enabled(self):
        fields = image_manifest_fields({RECORD_IMAGE_MANIFEST_KEY: True}, ["a"], ["b"])
        assert fields == {"train_image_uuids": ["a"], "val_image_uuids": ["b"]}

    def test_omits_keys_entirely_when_disabled(self):
        """Omitted, not emptied — a reader must be able to tell "not recorded"
        from "recorded, and the split happened to be empty"."""
        assert image_manifest_fields({RECORD_IMAGE_MANIFEST_KEY: False}, ["a"], ["b"]) == {}

    def test_copies_the_lists(self):
        train = ["a"]
        fields = image_manifest_fields({}, train, [])
        train.append("b")
        assert fields["train_image_uuids"] == ["a"]


class TestAllTrainersRouteThroughTheHelper:
    @pytest.mark.parametrize("filename", SIDECAR_TRAINERS)
    def test_trainer_calls_image_manifest_fields(self, filename):
        source = (TRAINERS_DIR / filename).read_text()
        assert (
            "image_manifest_fields(" in source
        ), f"{filename} no longer routes its sidecar manifest through the helper"

    @pytest.mark.parametrize("filename", SIDECAR_TRAINERS)
    def test_trainer_does_not_write_the_sidecar_keys_directly(self, filename):
        """A direct `"train_image_uuids": ...` entry would bypass the opt-out."""
        source = (TRAINERS_DIR / filename).read_text()
        assert (
            '"train_image_uuids"' not in source
        ), f"{filename} writes train_image_uuids directly, bypassing the opt-out"
        assert '"val_image_uuids"' not in source, f"{filename} writes val_image_uuids directly, bypassing the opt-out"
