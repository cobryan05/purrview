"""Unit tests for apps.helpers.model_type_map.resolve_trainer_type.

Covers the arch-aware object_detection dispatch (LibreYOLO9's yolo_v9_* prefix
vs. the newer Ultralytics architectures) plus the arch-independent lookup for
other model types, and the unknown/unmapped error paths (must raise, not
silently pass through — see MORE_YOLO_PLAN.md Section 1).
"""

import pytest

from apps.helpers.model_type_map import MODEL_TYPE_TO_TRAINER_TYPE, resolve_trainer_type


class TestResolveTrainerTypeObjectDetection:
    @pytest.mark.parametrize("arch", ["yolo_v9_t", "yolo_v9_s", "yolo_v9_m", "yolo_v9_c", "yolo_v9_e"])
    def test_libreyolo9_architectures_route_to_object_detection(self, arch):
        assert resolve_trainer_type("object_detection", arch) == "object_detection"

    @pytest.mark.parametrize(
        "arch",
        [
            "yolov8n",
            "yolov8s",
            "yolov8m",
            "yolov8l",
            "yolov8x",
            "yolov9t",
            "yolov9s",
            "yolov9m",
            "yolov9c",
            "yolov9e",
            "yolo11n",
            "yolo11s",
            "yolo11m",
            "yolo11l",
            "yolo11x",
            "yolo12n",
            "yolo12s",
            "yolo12m",
            "yolo12l",
            "yolo12x",
        ],
    )
    def test_ultralytics_architectures_route_to_ultralytics_object_detection(self, arch):
        assert resolve_trainer_type("object_detection", arch) == "ultralytics_object_detection"

    def test_ultralytics_own_v9_is_distinct_from_libreyolo_v9(self):
        """Ultralytics' own GELAN-backbone v9 (arch name 'yolov9c', no underscore
        prefix) must NOT be confused with LibreYOLO9's 'yolo_v9_c' — they are
        different trainers/model families despite similar names."""
        assert resolve_trainer_type("object_detection", "yolov9c") == "ultralytics_object_detection"
        assert resolve_trainer_type("object_detection", "yolo_v9_c") == "object_detection"

    def test_unknown_architecture_raises(self):
        with pytest.raises(ValueError, match="Unrecognized object_detection architecture"):
            resolve_trainer_type("object_detection", "some_future_arch")

    def test_empty_architecture_raises(self):
        with pytest.raises(ValueError):
            resolve_trainer_type("object_detection", "")


class TestResolveTrainerTypeOtherModelTypes:
    """Non-object_detection model types are a simple arch-independent lookup —
    architecture is accepted but ignored."""

    def test_image_classifier(self):
        assert resolve_trainer_type("image_classifier", "mobilenet_v3_small") == "binary_image_classifier"

    def test_image_classifier_ignores_architecture_value(self):
        assert resolve_trainer_type("image_classifier", "resnet_50") == "binary_image_classifier"
        assert resolve_trainer_type("image_classifier", "") == "binary_image_classifier"

    def test_crop_classifier(self):
        assert resolve_trainer_type("crop_classifier", "efficientnet_b0") == "crop_classifier"

    def test_unknown_model_type_raises(self):
        with pytest.raises(ValueError, match="Unrecognized model_type"):
            resolve_trainer_type("some_future_model_type", "whatever")

    def test_covers_every_entry_in_the_shared_map(self):
        """Every model_type in MODEL_TYPE_TO_TRAINER_TYPE (except object_detection,
        tested separately above) must resolve identically through resolve_trainer_type."""
        for model_type, trainer_type in MODEL_TYPE_TO_TRAINER_TYPE.items():
            if model_type == "object_detection":
                continue
            assert resolve_trainer_type(model_type, "irrelevant") == trainer_type
