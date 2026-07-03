"""Tests for the detection-dataset endpoint and DBClient method.

Covers:
- class ordering matches _derive_labels_for_node
- rollup children map to parent class index
- only VERIFIED bboxes are included (unverified filtered out)
- absent-verified labels → empty annotations (image still in result)
- images not in dataset are excluded
"""

import uuid

import pytest

from apps.helpers.db.types import (
    UNVERIFIED_TAG_UUID,
    VERIFIED_TAG_UUID,
    WHOLE_DATABASE_DATASET_UUID,
)
from tests.conftest import make_png

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

_WHOLE_DB_UUID = WHOLE_DATABASE_DATASET_UUID


async def _create_label(http_client, name: str, parent_uuid: str | None = None) -> str:
    payload: dict = {"name": name, "color": "#ff0000"}
    if parent_uuid is not None:
        payload["parentUuid"] = parent_uuid
    r = await http_client.post("/api/labels", json=payload)
    assert r.status_code == 200, r.text
    return r.json()["label"]["uuid"]


async def _create_dataset(http_client, name: str, label_uuids: list | None = None) -> dict:
    r = await http_client.post(
        "/api/datasets",
        json={"name": name, "filter_json": {}, "label_uuids": label_uuids or []},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "success", body
    return body["dataset"]


async def _create_model_config(http_client, name: str, nodes: list | None = None) -> dict:
    r = await http_client.post("/api/model-configs", json={"name": name, "nodes": nodes or []})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "success", body
    return body["config"]


async def _create_training_config(
    http_client, name: str, dataset_uuids: list[str], *, model_config_node_uuid: str | None = None
) -> dict:
    payload: dict = {"name": name, "datasetUuids": dataset_uuids}
    if model_config_node_uuid is not None:
        payload["modelConfigNodeUuid"] = model_config_node_uuid
    r = await http_client.post("/api/training-configs", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "success", body
    return body["config"]


async def _register_image(http_client, files_root, name: str) -> str:
    """Register a PNG image and return its path."""
    (files_root / name).write_bytes(make_png())
    r = await http_client.post("/api/images/metadata/update", json={"imagePath": name, "boxes": []})
    assert r.status_code == 200, r.text
    return name


async def _add_bbox(
    http_client,
    image_path: str,
    label_uuid: str,
    tag_uuids: list[str],
    *,
    x: float = 0.1,
    y: float = 0.1,
    width: float = 0.2,
    height: float = 0.2,
) -> str:
    box_uuid = str(uuid.uuid4())
    r = await http_client.post(
        "/api/images/metadata/update",
        json={
            "imagePath": image_path,
            "boxes": [
                {
                    "uuid": box_uuid,
                    "labelUuid": label_uuid,
                    "x": x,
                    "y": y,
                    "width": width,
                    "height": height,
                    "tagUuids": tag_uuids,
                }
            ],
        },
    )
    assert r.status_code == 200, r.text
    return box_uuid


async def _set_image_label(http_client, image_path: str, label_uuid: str, status: str) -> None:
    r = await http_client.post(
        "/api/images/labels/set",
        json={"imagePath": image_path, "labelUuid": label_uuid, "status": status},
    )
    assert r.status_code == 200, r.text


async def _get_detection_dataset(http_client, tc_uuid: str) -> dict:
    r = await http_client.get(f"/api/training-configs/{tc_uuid}/detection-dataset")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "success", body
    return body


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# Each test uses the whole-database dataset (uuid=WHOLE_DATABASE_DATASET_UUID)
# which includes all registered images, so no explicit dataset membership needed.
# ─────────────────────────────────────────────────────────────────────────────


class TestGetTrainingConfigDetectionDataset:
    async def test_returns_not_found_for_missing_config(self, http_client):
        r = await http_client.get("/api/training-configs/nonexistent/detection-dataset")
        assert r.status_code == 200
        assert r.json()["status"] == "failure"

    async def test_verified_bbox_appears_in_annotations(self, http_client, files_root):
        """A VERIFIED bbox in scope appears in annotations with correct class index and coords."""
        cat_uuid = await _create_label(http_client, "cat")
        mc = await _create_model_config(
            http_client,
            "mc-verified",
            nodes=[{"labelUuid": cat_uuid, "modelType": "object_detection", "architecture": "yolo_v9_c"}],
        )
        node_uuid = mc["nodes"][0]["uuid"]
        tc = await _create_training_config(
            http_client, "tc-verified", [_WHOLE_DB_UUID], model_config_node_uuid=node_uuid
        )
        img = await _register_image(http_client, files_root, "cat_img.png")

        # Verified bbox for 'cat'.
        await _add_bbox(http_client, img, cat_uuid, [VERIFIED_TAG_UUID], x=0.1, y=0.1, width=0.3, height=0.4)
        # Mark image label present (eligibility requires this).
        await _set_image_label(http_client, img, cat_uuid, "present")

        result = await _get_detection_dataset(http_client, tc["uuid"])
        assert result["classNames"] == ["cat"]
        images = result["images"]
        assert len(images) == 1
        ann = images[0]["annotations"]
        assert len(ann) == 1
        cls_idx, cx, cy, w, h = ann[0]
        assert int(cls_idx) == 0
        # cx = x + width/2 = 0.1 + 0.15 = 0.25, cy = 0.1 + 0.2 = 0.30
        assert abs(cx - 0.25) < 1e-5
        assert abs(cy - 0.30) < 1e-5
        assert abs(w - 0.3) < 1e-5
        assert abs(h - 0.4) < 1e-5

    async def test_unverified_bbox_excluded(self, http_client, files_root):
        """Bboxes with UNVERIFIED tag are excluded from annotations."""
        cat_uuid = await _create_label(http_client, "cat2")
        mc = await _create_model_config(
            http_client,
            "mc-unverified",
            nodes=[{"labelUuid": cat_uuid, "modelType": "object_detection", "architecture": "yolo_v9_c"}],
        )
        node_uuid = mc["nodes"][0]["uuid"]
        tc = await _create_training_config(
            http_client, "tc-unverified", [_WHOLE_DB_UUID], model_config_node_uuid=node_uuid
        )
        img = await _register_image(http_client, files_root, "unverified_img.png")

        # Only unverified bbox — should not appear in annotations.
        await _add_bbox(http_client, img, cat_uuid, [UNVERIFIED_TAG_UUID])
        # Explicitly mark label absent so image is eligible.
        await _set_image_label(http_client, img, cat_uuid, "absent")

        result = await _get_detection_dataset(http_client, tc["uuid"])
        images = result["images"]
        assert len(images) == 1
        assert images[0]["annotations"] == []

    async def test_absent_label_gives_empty_annotations(self, http_client, files_root):
        """Image verified absent for a label is included with empty annotations (negative sample)."""
        cat_uuid = await _create_label(http_client, "cat3")
        mc = await _create_model_config(
            http_client,
            "mc-absent",
            nodes=[{"labelUuid": cat_uuid, "modelType": "object_detection", "architecture": "yolo_v9_c"}],
        )
        node_uuid = mc["nodes"][0]["uuid"]
        tc = await _create_training_config(
            http_client, "tc-absent", [_WHOLE_DB_UUID], model_config_node_uuid=node_uuid
        )
        img = await _register_image(http_client, files_root, "absent_img.png")
        await _set_image_label(http_client, img, cat_uuid, "absent")

        result = await _get_detection_dataset(http_client, tc["uuid"])
        images = result["images"]
        assert len(images) == 1
        assert images[0]["annotations"] == []

    async def test_ineligible_image_excluded(self, http_client, files_root):
        """Image without any verified image_label row for scope labels is excluded."""
        cat_uuid = await _create_label(http_client, "cat4")
        mc = await _create_model_config(
            http_client,
            "mc-inelig",
            nodes=[{"labelUuid": cat_uuid, "modelType": "object_detection", "architecture": "yolo_v9_c"}],
        )
        node_uuid = mc["nodes"][0]["uuid"]
        tc = await _create_training_config(
            http_client, "tc-inelig", [_WHOLE_DB_UUID], model_config_node_uuid=node_uuid
        )
        # Image registered but NOT verified for cat_uuid → not eligible.
        img = await _register_image(http_client, files_root, "inelig_img.png")
        # No image_label row set for cat_uuid.

        result = await _get_detection_dataset(http_client, tc["uuid"])
        images = result["images"]
        assert len(images) == 0

    async def test_multiple_classes_class_index(self, http_client, files_root):
        """With multiple labels, each gets a distinct class index in order."""
        cat_uuid = await _create_label(http_client, "cat5")
        dog_uuid = await _create_label(http_client, "dog5")
        mc = await _create_model_config(
            http_client,
            "mc-multi",
            nodes=[
                # Root node (labelUuid absent / None) covers all labeled children.
                {"modelType": "object_detection", "architecture": "yolo_v9_c"},
                {"labelUuid": cat_uuid, "modelType": "object_detection", "architecture": "yolo_v9_c"},
                {"labelUuid": dog_uuid, "modelType": "object_detection", "architecture": "yolo_v9_c"},
            ],
        )
        # Root node has labelUuid == None.
        root_nodes = [n for n in mc["nodes"] if n.get("labelUuid") is None]
        assert root_nodes, f"Expected a root node in {mc['nodes']}"
        root_node = root_nodes[0]
        tc = await _create_training_config(
            http_client, "tc-multi", [_WHOLE_DB_UUID], model_config_node_uuid=root_node["uuid"]
        )
        img = await _register_image(http_client, files_root, "multi_class.png")

        # Both bboxes must be sent in a single update call; each call replaces all boxes.
        r = await http_client.post(
            "/api/images/metadata/update",
            json={
                "imagePath": img,
                "boxes": [
                    {"uuid": str(uuid.uuid4()), "labelUuid": cat_uuid, "x": 0.1, "y": 0.1, "width": 0.2, "height": 0.2, "tagUuids": [VERIFIED_TAG_UUID]},
                    {"uuid": str(uuid.uuid4()), "labelUuid": dog_uuid, "x": 0.5, "y": 0.5, "width": 0.2, "height": 0.2, "tagUuids": [VERIFIED_TAG_UUID]},
                ],
            },
        )
        assert r.status_code == 200, r.text
        await _set_image_label(http_client, img, cat_uuid, "present")
        await _set_image_label(http_client, img, dog_uuid, "present")

        result = await _get_detection_dataset(http_client, tc["uuid"])
        class_names = result["classNames"]
        assert "cat5" in class_names
        assert "dog5" in class_names
        # Class indices should be distinct.
        assert class_names.index("cat5") != class_names.index("dog5")
        # Both bboxes should appear.
        ann = result["images"][0]["annotations"]
        assert len(ann) == 2
        # Each annotation's class index should correspond to the right label.
        for a in ann:
            assert int(a[0]) in range(len(class_names))


class TestGetBundleTargetDetectionDataset:
    """POST /api/model-configs/{uuid}/training-targets/{node_uuid}/detection-dataset —

    Same assembly core as the TrainingConfig-based endpoint, but scoped via
    derive_bundle_training_targets (role/label_uuids/class_remap) so a bundle run's
    object-detection sub-models get the same classes shown in the model tree preview.
    """

    async def _get_bundle_target_dataset(
        self, http_client, config_uuid: str, node_uuid: str, dataset_uuids: list[str]
    ) -> dict:
        r = await http_client.post(
            f"/api/model-configs/{config_uuid}/training-targets/{node_uuid}/detection-dataset",
            json={"datasetUuids": dataset_uuids},
        )
        assert r.status_code == 200, r.text
        return r.json()

    async def test_returns_not_found_for_missing_config(self, http_client):
        body = await self._get_bundle_target_dataset(http_client, "nonexistent", "nonexistent", [])
        assert body["status"] == "failure"

    async def test_returns_not_found_when_node_is_not_a_bundle_target(self, http_client):
        """A config with only a root node has no boundary targets — any node_uuid lookup misses."""
        mc = await _create_model_config(
            http_client,
            "mc-no-bundle-target",
            nodes=[{"modelType": "object_detection", "architecture": "yolo_v9_c"}],
        )
        body = await self._get_bundle_target_dataset(http_client, mc["uuid"], "does-not-exist", [_WHOLE_DB_UUID])
        assert body["status"] == "failure"

    async def test_submodel_boundary_matches_target_classes_and_verified_bbox(self, http_client, files_root):
        """A sub-model boundary's target discriminates its CHILDREN, not itself (a childless
        boundary has an empty class list — see derive_bundle_training_targets' docstring), so
        this needs a boundary node (animal) with a child label (cat) beneath it."""
        animal_uuid = await _create_label(http_client, "animal-bundle")
        cat_uuid = await _create_label(http_client, "cat-bundle", parent_uuid=animal_uuid)
        mc = await _create_model_config(
            http_client,
            "mc-bundle-target",
            nodes=[
                # Root: a different spec than the sub-model below, so `animal` is a genuine
                # sub-model boundary (spec differs from inherited).
                {"modelType": "image_classifier", "architecture": "mobilenet_v3_small"},
                {"labelUuid": animal_uuid, "modelType": "object_detection", "architecture": "yolo_v9_c"},
                # `cat` inherits animal's spec (not its own boundary) — becomes a class
                # of animal's sub-model target.
                {"labelUuid": cat_uuid, "modelType": "object_detection", "architecture": "yolo_v9_c"},
            ],
        )
        sub_node = next(n for n in mc["nodes"] if n.get("labelUuid") == animal_uuid)

        img = await _register_image(http_client, files_root, "bundle_cat.png")
        await _add_bbox(http_client, img, cat_uuid, [VERIFIED_TAG_UUID], x=0.1, y=0.1, width=0.3, height=0.4)
        await _set_image_label(http_client, img, cat_uuid, "present")

        result = await self._get_bundle_target_dataset(http_client, mc["uuid"], sub_node["uuid"], [_WHOLE_DB_UUID])
        assert result["status"] == "success"
        assert result["classNames"] == ["cat-bundle"]
        images = result["images"]
        assert len(images) == 1
        assert len(images[0]["annotations"]) == 1
