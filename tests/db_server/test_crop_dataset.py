"""Tests for the crop-dataset endpoint (crop_classifier sub-model training data).

POST /api/model-configs/{uuid}/training-targets/{node_uuid}/crop-dataset

The crop assembler shares a scope contract with the detection assembler
(derive_training_targets decides the classes and the remap) but deliberately
diverges on what makes an image usable — see test_no_image_label_readiness_gate,
which is the reason the two are separate methods rather than one with a flag.
"""

import uuid

from apps.helpers.db.types import (
    UNVERIFIED_TAG_UUID,
    VERIFIED_TAG_UUID,
    WHOLE_DATABASE_DATASET_UUID,
)
from tests.conftest import make_png

_WHOLE_DB_UUID = WHOLE_DATABASE_DATASET_UUID


async def _create_label(http_client, name: str, parent_uuid: str | None = None) -> str:
    payload: dict = {"name": name, "color": "#ff0000"}
    if parent_uuid is not None:
        payload["parentUuid"] = parent_uuid
    r = await http_client.post("/api/labels", json=payload)
    assert r.status_code == 200, r.text
    return r.json()["label"]["uuid"]


async def _create_model_config(http_client, name: str, nodes: list | None = None) -> dict:
    r = await http_client.post("/api/model-configs", json={"name": name, "nodes": nodes or []})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "success", body
    return body["config"]


async def _register_image(http_client, files_root, name: str) -> str:
    (files_root / name).write_bytes(make_png())
    r = await http_client.post("/api/images/metadata/update", json={"imagePath": name, "boxes": []})
    assert r.status_code == 200, r.text
    return name


async def _add_bboxes(http_client, image_path: str, boxes: list[dict]) -> None:
    """Replace an image's boxes. Each entry: {labelUuid, tagUuids, x, y, width, height}."""
    r = await http_client.post(
        "/api/images/metadata/update",
        json={
            "imagePath": image_path,
            "boxes": [
                {
                    "uuid": str(uuid.uuid4()),
                    "labelUuid": b["labelUuid"],
                    "x": b.get("x", 0.1),
                    "y": b.get("y", 0.1),
                    "width": b.get("width", 0.2),
                    "height": b.get("height", 0.2),
                    "tagUuids": b["tagUuids"],
                }
                for b in boxes
            ],
        },
    )
    assert r.status_code == 200, r.text


async def _cat_config(http_client, prefix: str) -> tuple[dict, dict, str, list[str]]:
    """A Cat boundary with three child labels — the shape this feature exists for.

    Root is an object detector; Cat overrides to crop_classifier, which makes it a
    genuine sub-model boundary (its spec differs from the inherited one). Returns
    (config, cat_node, cat_uuid, [child uuids]).
    """
    cat_uuid = await _create_label(http_client, f"{prefix}-cat")
    children = [await _create_label(http_client, f"{prefix}-{n}", parent_uuid=cat_uuid) for n in ("willow", "maple")]
    mc = await _create_model_config(
        http_client,
        f"mc-{prefix}",
        nodes=[
            {"modelType": "object_detection", "architecture": "yolo_v9_s"},
            {"labelUuid": cat_uuid, "modelType": "crop_classifier", "architecture": "resnet_18"},
            *[{"labelUuid": c, "modelType": "crop_classifier", "architecture": "resnet_18"} for c in children],
        ],
    )
    cat_node = next(n for n in mc["nodes"] if n.get("labelUuid") == cat_uuid)
    return mc, cat_node, cat_uuid, children


async def _get_crop_dataset(
    http_client,
    config_uuid: str,
    node_uuid: str,
    dataset_uuids: list[str],
    label_uuids: list[str] | None = None,
    class_remap: dict[str, str] | None = None,
) -> dict:
    body: dict = {"datasetUuids": dataset_uuids}
    if label_uuids is not None:
        body["labelUuids"] = label_uuids
    if class_remap is not None:
        body["classRemap"] = class_remap
    r = await http_client.post(
        f"/api/model-configs/{config_uuid}/training-targets/{node_uuid}/crop-dataset",
        json=body,
    )
    assert r.status_code == 200, r.text
    return r.json()


async def _get_detection_dataset(http_client, config_uuid: str, node_uuid: str, dataset_uuids: list[str]) -> dict:
    r = await http_client.post(
        f"/api/model-configs/{config_uuid}/training-targets/{node_uuid}/detection-dataset",
        json={"datasetUuids": dataset_uuids},
    )
    assert r.status_code == 200, r.text
    return r.json()


class TestCropDataset:
    async def test_returns_failure_for_missing_config(self, http_client):
        body = await _get_crop_dataset(http_client, "nonexistent", "nonexistent", [])
        assert body["status"] == "failure"

    async def test_returns_failure_when_node_is_not_a_target(self, http_client):
        mc = await _create_model_config(
            http_client,
            "mc-crop-no-target",
            nodes=[{"modelType": "object_detection", "architecture": "yolo_v9_s"}],
        )
        body = await _get_crop_dataset(http_client, mc["uuid"], "does-not-exist", [_WHOLE_DB_UUID])
        assert body["status"] == "failure"

    async def test_sibling_labels_become_the_classes_with_stable_indices(self, http_client, files_root):
        """The boundary's children are the classes; each crop carries the index of its
        own label, and that index is the label's position in classLabelUuids."""
        mc, cat_node, _cat_uuid, (willow, maple) = await _cat_config(http_client, "idx")

        img = await _register_image(http_client, files_root, "crop_idx.png")
        await _add_bboxes(
            http_client,
            img,
            [
                {"labelUuid": willow, "tagUuids": [VERIFIED_TAG_UUID], "x": 0.1, "y": 0.1},
                {"labelUuid": maple, "tagUuids": [VERIFIED_TAG_UUID], "x": 0.5, "y": 0.5},
            ],
        )

        body = await _get_crop_dataset(http_client, mc["uuid"], cat_node["uuid"], [_WHOLE_DB_UUID])
        assert body["status"] == "success"
        assert len(body["images"]) == 1

        idx_of = {lu: i for i, lu in enumerate(body["classLabelUuids"])}
        assert set(idx_of) == {willow, maple}
        crops = body["images"][0]["crops"]
        assert len(crops) == 2
        # Locate each crop by its coordinates, then check it carries its label's index.
        by_x = {round(c["x"], 3): c for c in crops}
        assert by_x[0.1]["classIdx"] == idx_of[willow]
        assert by_x[0.5]["classIdx"] == idx_of[maple]

    async def test_unverified_bboxes_are_excluded(self, http_client, files_root):
        """Only verified boxes train the classifier — an unverified box is a guess,
        often the very mislabel this second pass exists to fix."""
        mc, cat_node, _cat_uuid, (willow, maple) = await _cat_config(http_client, "unver")

        img = await _register_image(http_client, files_root, "crop_unver.png")
        await _add_bboxes(
            http_client,
            img,
            [
                {"labelUuid": willow, "tagUuids": [VERIFIED_TAG_UUID], "x": 0.1, "y": 0.1},
                {"labelUuid": maple, "tagUuids": [UNVERIFIED_TAG_UUID], "x": 0.5, "y": 0.5},
            ],
        )

        body = await _get_crop_dataset(http_client, mc["uuid"], cat_node["uuid"], [_WHOLE_DB_UUID])
        crops = body["images"][0]["crops"]
        assert len(crops) == 1
        assert round(crops[0]["x"], 3) == 0.1

    async def test_images_without_in_scope_crops_are_omitted(self, http_client, files_root):
        """No negatives: an image with nothing to crop contributes no samples, so it
        is left out entirely rather than emitted with an empty list."""
        mc, cat_node, cat_uuid, (willow, _maple) = await _cat_config(http_client, "empty")

        with_cat = await _register_image(http_client, files_root, "crop_with_cat.png")
        await _add_bboxes(http_client, with_cat, [{"labelUuid": willow, "tagUuids": [VERIFIED_TAG_UUID]}])
        # The boundary label itself is not a class of its own sub-model, so a verified
        # "Cat" box is out of scope here and its image contributes nothing.
        without = await _register_image(http_client, files_root, "crop_without_cat.png")
        await _add_bboxes(http_client, without, [{"labelUuid": cat_uuid, "tagUuids": [VERIFIED_TAG_UUID]}])

        body = await _get_crop_dataset(http_client, mc["uuid"], cat_node["uuid"], [_WHOLE_DB_UUID])
        assert [i["imageUuid"] for i in body["images"]] != []
        assert len(body["images"]) == 1
        assert all(len(i["crops"]) > 0 for i in body["images"])

    async def test_no_image_label_readiness_gate(self, http_client, files_root):
        """The divergence from the detection assembler, and the reason both exist.

        An image holding one cat has a PRESENT row for that cat and *nothing* for
        its siblings — nobody ever marks the other two explicitly absent. The
        detection assembler needs every in-scope label decided before it can treat
        the unboxed remainder as background, so it drops the image. The crop
        assembler doesn't care: the sample *is* the box, and that box is verified.

        This is the whole live-data case. On the real DB the same gate takes ~1500
        verified per-cat boxes down to a few dozen.
        """
        mc, cat_node, _cat_uuid, (willow, _maple) = await _cat_config(http_client, "gate")

        img = await _register_image(http_client, files_root, "crop_gate.png")
        # One class boxed, the sibling never decided either way.
        await _add_bboxes(http_client, img, [{"labelUuid": willow, "tagUuids": [VERIFIED_TAG_UUID], "x": 0.1}])

        detection = await _get_detection_dataset(http_client, mc["uuid"], cat_node["uuid"], [_WHOLE_DB_UUID])
        crop = await _get_crop_dataset(http_client, mc["uuid"], cat_node["uuid"], [_WHOLE_DB_UUID])

        assert detection["status"] == "success"
        assert detection["images"] == [], "partially-decided image should not reach a detection trainer"
        assert crop["status"] == "success"
        assert len(crop["images"]) == 1
        assert len(crop["images"][0]["crops"]) == 1

    async def test_remap_outside_the_class_set_does_not_pull_labels_in(self, http_client, files_root):
        """A caller may hand over a snapshot remap wider than this target's classes
        (the top model's collapse map names every assigned label). Those labels must
        stay out of scope, not be silently claimed by class 0."""
        mc, cat_node, cat_uuid, (willow, maple) = await _cat_config(http_client, "remap")
        outsider = await _create_label(http_client, "remap-dog")

        img = await _register_image(http_client, files_root, "crop_remap.png")
        await _add_bboxes(
            http_client,
            img,
            [
                {"labelUuid": willow, "tagUuids": [VERIFIED_TAG_UUID], "x": 0.1, "y": 0.1},
                {"labelUuid": outsider, "tagUuids": [VERIFIED_TAG_UUID], "x": 0.5, "y": 0.5},
            ],
        )

        body = await _get_crop_dataset(
            http_client,
            mc["uuid"],
            cat_node["uuid"],
            [_WHOLE_DB_UUID],
            label_uuids=[willow, maple],
            # `outsider` remaps to a label that is not a class of this target.
            class_remap={outsider: cat_uuid},
        )
        assert body["status"] == "success"
        crops = body["images"][0]["crops"]
        assert len(crops) == 1
        assert round(crops[0]["x"], 3) == 0.1

    async def test_remap_inside_the_class_set_folds_boxes_into_that_class(self, http_client, files_root):
        """The remap that *does* apply: a label folded into one of this target's
        classes trains as that class rather than being dropped."""
        mc, cat_node, _cat_uuid, (willow, maple) = await _cat_config(http_client, "fold")
        kitten = await _create_label(http_client, "fold-kitten", parent_uuid=willow)

        img = await _register_image(http_client, files_root, "crop_fold.png")
        await _add_bboxes(http_client, img, [{"labelUuid": kitten, "tagUuids": [VERIFIED_TAG_UUID], "x": 0.3}])

        body = await _get_crop_dataset(
            http_client,
            mc["uuid"],
            cat_node["uuid"],
            [_WHOLE_DB_UUID],
            label_uuids=[willow, maple],
            class_remap={kitten: willow},
        )
        assert body["status"] == "success"
        crops = body["images"][0]["crops"]
        assert len(crops) == 1
        assert crops[0]["classIdx"] == body["classLabelUuids"].index(willow)
