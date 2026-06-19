"""Tests for the training configurations feature — CRUD, readiness endpoint, label stats."""

import uuid

from apps.helpers.db.types import UNVERIFIED_TAG_UUID, VERIFIED_TAG_UUID
from tests.conftest import make_png

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


async def _create_label(http_client, name: str) -> str:
    r = await http_client.post("/api/labels", json={"name": name, "color": "#ff0000"})
    assert r.status_code == 200, r.text
    return r.json()["label"]["uuid"]


async def _create_dataset(http_client, name: str) -> dict:
    r = await http_client.post("/api/datasets", json={"name": name, "filter_json": {}, "label_uuids": []})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "success", body
    return body["dataset"]


async def _create_training_config(
    http_client,
    name: str,
    dataset_uuids: list[str],
    *,
    trainer_type: str = "image_classifier",
    label_uuids: list | None = None,
    description: str | None = None,
    model_config_uuid: str | None = None,
    hyperparams_json: dict | None = None,
) -> dict:
    payload: dict = {
        "name": name,
        "trainerType": trainer_type,
        "datasetUuids": dataset_uuids,
        "labelUuids": label_uuids or [],
    }
    if description is not None:
        payload["description"] = description
    if model_config_uuid is not None:
        payload["modelConfigUuid"] = model_config_uuid
    if hyperparams_json is not None:
        payload["hyperparamsJson"] = hyperparams_json
    r = await http_client.post("/api/training-configs", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "success", body
    return body["config"]


async def _register_image(http_client, files_root, name: str) -> str:
    (files_root / name).write_bytes(make_png())
    r = await http_client.post(
        "/api/images/metadata/update",
        json={"imagePath": name, "boxes": []},
    )
    assert r.status_code == 200, r.text
    return name


# ─────────────────────────────────────────────────────────────────────────────
# CRUD
# ─────────────────────────────────────────────────────────────────────────────


class TestTrainingConfigCrud:
    async def test_create_and_get(self, http_client):
        ds = await _create_dataset(http_client, "ds-crud")
        cfg = await _create_training_config(http_client, "tc-basic", [ds["uuid"]])
        assert cfg["name"] == "tc-basic"
        assert cfg["trainerType"] == "image_classifier"
        assert cfg["datasetUuids"] == [ds["uuid"]]
        assert cfg["labelUuids"] == []
        assert cfg["uuid"]

        r = await http_client.get(f"/api/training-configs/{cfg['uuid']}")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "success"
        assert body["config"]["uuid"] == cfg["uuid"]

    async def test_create_with_multiple_datasets(self, http_client):
        ds_a = await _create_dataset(http_client, "ds-multi-a")
        ds_b = await _create_dataset(http_client, "ds-multi-b")
        cfg = await _create_training_config(http_client, "tc-multi-ds", [ds_a["uuid"], ds_b["uuid"]])
        assert set(cfg["datasetUuids"]) == {ds_a["uuid"], ds_b["uuid"]}

    async def test_create_with_no_datasets(self, http_client):
        """Empty datasetUuids is allowed at create time (validation is in the UI)."""
        cfg = await _create_training_config(http_client, "tc-no-ds", [])
        assert cfg["datasetUuids"] == []

    async def test_create_with_labels(self, http_client):
        ds = await _create_dataset(http_client, "ds-labels")
        label_a = await _create_label(http_client, "cat")
        label_b = await _create_label(http_client, "dog")
        cfg = await _create_training_config(
            http_client, "tc-labels", [ds["uuid"]],
            label_uuids=[label_a, label_b],
        )
        assert set(cfg["labelUuids"]) == {label_a, label_b}

    async def test_create_with_hyperparams(self, http_client):
        ds = await _create_dataset(http_client, "ds-hyper")
        cfg = await _create_training_config(
            http_client, "tc-hyper", [ds["uuid"]],
            hyperparams_json={"epochs": 50, "lr": 0.001},
        )
        assert cfg["hyperparamsJson"]["epochs"] == 50
        assert cfg["hyperparamsJson"]["lr"] == 0.001

    async def test_list(self, http_client):
        ds = await _create_dataset(http_client, "ds-list")
        await _create_training_config(http_client, "tc-list-a", [ds["uuid"]])
        await _create_training_config(http_client, "tc-list-b", [ds["uuid"]])

        r = await http_client.get("/api/training-configs")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "success"
        names = [c["name"] for c in body["configs"]]
        assert "tc-list-a" in names
        assert "tc-list-b" in names

    async def test_update_datasets(self, http_client):
        ds_a = await _create_dataset(http_client, "ds-upd-a")
        ds_b = await _create_dataset(http_client, "ds-upd-b")
        cfg = await _create_training_config(http_client, "tc-update-ds", [ds_a["uuid"]])
        assert cfg["datasetUuids"] == [ds_a["uuid"]]

        r = await http_client.put(
            f"/api/training-configs/{cfg['uuid']}",
            json={"datasetUuids": [ds_a["uuid"], ds_b["uuid"]]},
        )
        assert r.status_code == 200
        updated = r.json()["config"]
        assert set(updated["datasetUuids"]) == {ds_a["uuid"], ds_b["uuid"]}

    async def test_update_clears_datasets_with_empty_list(self, http_client):
        ds = await _create_dataset(http_client, "ds-clear-ds")
        cfg = await _create_training_config(http_client, "tc-clear-ds", [ds["uuid"]])
        assert cfg["datasetUuids"] == [ds["uuid"]]

        r = await http_client.put(
            f"/api/training-configs/{cfg['uuid']}",
            json={"datasetUuids": []},
        )
        assert r.json()["config"]["datasetUuids"] == []

    async def test_update(self, http_client):
        ds = await _create_dataset(http_client, "ds-update")
        label_uuid = await _create_label(http_client, "bird")
        cfg = await _create_training_config(http_client, "tc-update", [ds["uuid"]])

        r = await http_client.put(
            f"/api/training-configs/{cfg['uuid']}",
            json={"name": "tc-updated", "labelUuids": [label_uuid]},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "success"
        updated = body["config"]
        assert updated["name"] == "tc-updated"
        assert updated["labelUuids"] == [label_uuid]

    async def test_update_clears_labels_with_empty_list(self, http_client):
        ds = await _create_dataset(http_client, "ds-clear-labels")
        label_uuid = await _create_label(http_client, "fish")
        cfg = await _create_training_config(
            http_client, "tc-clear", [ds["uuid"]], label_uuids=[label_uuid]
        )
        assert cfg["labelUuids"] == [label_uuid]

        r = await http_client.put(
            f"/api/training-configs/{cfg['uuid']}",
            json={"labelUuids": []},
        )
        assert r.json()["config"]["labelUuids"] == []

    async def test_delete(self, http_client):
        ds = await _create_dataset(http_client, "ds-delete")
        cfg = await _create_training_config(http_client, "tc-delete", [ds["uuid"]])

        r = await http_client.delete(f"/api/training-configs/{cfg['uuid']}")
        assert r.status_code == 200
        assert r.json()["status"] == "success"

        r2 = await http_client.get(f"/api/training-configs/{cfg['uuid']}")
        assert r2.json()["status"] == "failure"

    async def test_delete_dataset_rejected_when_linked(self, http_client):
        """Deleting a dataset that a training config references is rejected (409)."""
        ds = await _create_dataset(http_client, "ds-linked")
        await _create_training_config(http_client, "tc-linked", [ds["uuid"]])

        r = await http_client.delete(f"/api/datasets/{ds['uuid']}")
        assert r.status_code == 409

    async def test_delete_dataset_allowed_after_config_deleted(self, http_client):
        """Once the training config is removed, the dataset can be deleted."""
        ds = await _create_dataset(http_client, "ds-unlinked")
        cfg = await _create_training_config(http_client, "tc-unlinked", [ds["uuid"]])

        await http_client.delete(f"/api/training-configs/{cfg['uuid']}")

        r = await http_client.delete(f"/api/datasets/{ds['uuid']}")
        assert r.status_code == 200
        assert r.json()["status"] == "success"

    async def test_duplicate_name_rejected(self, http_client):
        ds = await _create_dataset(http_client, "ds-dup")
        await _create_training_config(http_client, "tc-dup", [ds["uuid"]])
        r = await http_client.post(
            "/api/training-configs",
            json={"name": "tc-dup", "trainerType": "yolo_v8", "datasetUuids": [ds["uuid"]]},
        )
        assert r.json()["status"] == "failure"

    async def test_get_nonexistent_returns_failure(self, http_client):
        r = await http_client.get(f"/api/training-configs/{uuid.uuid4()}")
        assert r.json()["status"] == "failure"


# ─────────────────────────────────────────────────────────────────────────────
# Readiness endpoint
# ─────────────────────────────────────────────────────────────────────────────


class TestTrainingConfigReadiness:
    async def test_empty_dataset_returns_zero_counts(self, http_client):
        ds = await _create_dataset(http_client, "ds-ready-empty")
        label_uuid = await _create_label(http_client, "empty-label")
        cfg = await _create_training_config(
            http_client, "tc-ready-empty", [ds["uuid"]], label_uuids=[label_uuid]
        )
        r = await http_client.get(f"/api/training-configs/{cfg['uuid']}/readiness")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "success"
        assert body["readyCount"] == 0
        assert body["needsReviewCount"] == 0

    async def test_no_labels_returns_zero_counts(self, http_client, files_root):
        ds = await _create_dataset(http_client, "ds-ready-nolabels")
        await _register_image(http_client, files_root, "ready_nolabels.png")
        cfg = await _create_training_config(
            http_client, "tc-ready-nolabels", [ds["uuid"]], label_uuids=[]
        )
        r = await http_client.get(f"/api/training-configs/{cfg['uuid']}/readiness")
        body = r.json()
        assert body["readyCount"] == 0
        assert body["needsReviewCount"] == 0

    async def test_unreviewed_image_counted_as_needs_review(self, http_client, files_root):
        ds = await _create_dataset(http_client, "ds-ready-unreviewed")
        label_uuid = await _create_label(http_client, "unreviewed-label")
        await _register_image(http_client, files_root, "ready_unreviewed.png")
        cfg = await _create_training_config(
            http_client, "tc-ready-unreviewed", [ds["uuid"]], label_uuids=[label_uuid]
        )
        r = await http_client.get(f"/api/training-configs/{cfg['uuid']}/readiness")
        body = r.json()
        assert body["readyCount"] == 0
        assert body["needsReviewCount"] == 1

    async def test_present_label_counted_as_ready(self, http_client, files_root):
        ds = await _create_dataset(http_client, "ds-ready-present")
        label_uuid = await _create_label(http_client, "present-label")
        await _register_image(http_client, files_root, "ready_present.png")

        # Set label to PRESENT manually
        r = await http_client.post(
            "/api/images/labels/set",
            json={"imagePath": "ready_present.png", "labelUuid": label_uuid, "status": "present"},
        )
        assert r.json()["status"] == "success"

        cfg = await _create_training_config(
            http_client, "tc-ready-present", [ds["uuid"]], label_uuids=[label_uuid]
        )
        r = await http_client.get(f"/api/training-configs/{cfg['uuid']}/readiness")
        body = r.json()
        assert body["readyCount"] == 1
        assert body["needsReviewCount"] == 0

    async def test_absent_label_counted_as_ready(self, http_client, files_root):
        ds = await _create_dataset(http_client, "ds-ready-absent")
        label_uuid = await _create_label(http_client, "absent-label")
        await _register_image(http_client, files_root, "ready_absent.png")

        r = await http_client.post(
            "/api/images/labels/set",
            json={"imagePath": "ready_absent.png", "labelUuid": label_uuid, "status": "absent"},
        )
        assert r.json()["status"] == "success"

        cfg = await _create_training_config(
            http_client, "tc-ready-absent", [ds["uuid"]], label_uuids=[label_uuid]
        )
        r = await http_client.get(f"/api/training-configs/{cfg['uuid']}/readiness")
        body = r.json()
        assert body["readyCount"] == 1
        assert body["needsReviewCount"] == 0

    async def test_partial_label_coverage_needs_review(self, http_client, files_root):
        """An image with 1 of 2 target labels verified is 'needs review'."""
        ds = await _create_dataset(http_client, "ds-ready-partial")
        label_a = await _create_label(http_client, "partial-a")
        label_b = await _create_label(http_client, "partial-b")
        await _register_image(http_client, files_root, "ready_partial.png")

        # Only verify label_a; label_b left unreviewed
        await http_client.post(
            "/api/images/labels/set",
            json={"imagePath": "ready_partial.png", "labelUuid": label_a, "status": "present"},
        )

        cfg = await _create_training_config(
            http_client, "tc-ready-partial", [ds["uuid"]], label_uuids=[label_a, label_b]
        )
        r = await http_client.get(f"/api/training-configs/{cfg['uuid']}/readiness")
        body = r.json()
        assert body["readyCount"] == 0
        assert body["needsReviewCount"] == 1

    async def test_mixed_ready_and_needs_review(self, http_client, files_root):
        """Multiple images — some ready, some not."""
        ds = await _create_dataset(http_client, "ds-ready-mixed")
        label_uuid = await _create_label(http_client, "mixed-label")
        await _register_image(http_client, files_root, "ready_mixed_a.png")
        await _register_image(http_client, files_root, "ready_mixed_b.png")

        # Verify label on image A only; image B stays unreviewed
        await http_client.post(
            "/api/images/labels/set",
            json={"imagePath": "ready_mixed_a.png", "labelUuid": label_uuid, "status": "absent"},
        )

        cfg = await _create_training_config(
            http_client, "tc-ready-mixed", [ds["uuid"]], label_uuids=[label_uuid]
        )
        r = await http_client.get(f"/api/training-configs/{cfg['uuid']}/readiness")
        body = r.json()
        assert body["readyCount"] == 1
        assert body["needsReviewCount"] == 1

    async def test_readiness_nonexistent_config(self, http_client):
        r = await http_client.get(f"/api/training-configs/{uuid.uuid4()}/readiness")
        assert r.json()["status"] == "failure"

    async def test_readiness_metadata(self, http_client):
        """Response includes dataset_uuids and label_uuids."""
        ds = await _create_dataset(http_client, "ds-ready-meta")
        label_uuid = await _create_label(http_client, "meta-label")
        cfg = await _create_training_config(
            http_client, "tc-ready-meta", [ds["uuid"]], label_uuids=[label_uuid]
        )
        r = await http_client.get(f"/api/training-configs/{cfg['uuid']}/readiness")
        body = r.json()
        assert ds["uuid"] in body["datasetUuids"]
        assert label_uuid in body["labelUuids"]

    async def test_readiness_union_of_two_datasets(self, http_client, files_root):
        """Readiness counts images from both linked datasets (union, de-duped)."""
        ds_a = await _create_dataset(http_client, "ds-union-a")
        ds_b = await _create_dataset(http_client, "ds-union-b")
        label_uuid = await _create_label(http_client, "union-label")

        # img_a lands in dataset A (all-images filter)
        await _register_image(http_client, files_root, "union_img_a.png")
        # img_b lands in dataset B
        await _register_image(http_client, files_root, "union_img_b.png")

        # Verify label on img_a only
        await http_client.post(
            "/api/images/labels/set",
            json={"imagePath": "union_img_a.png", "labelUuid": label_uuid, "status": "present"},
        )

        cfg = await _create_training_config(
            http_client, "tc-union", [ds_a["uuid"], ds_b["uuid"]], label_uuids=[label_uuid]
        )
        r = await http_client.get(f"/api/training-configs/{cfg['uuid']}/readiness")
        body = r.json()
        # Both images in the union: 1 ready (img_a), 1 needs review (img_b)
        assert body["readyCount"] == 1
        assert body["needsReviewCount"] == 1
        assert set(body["datasetUuids"]) == {ds_a["uuid"], ds_b["uuid"]}


# ─────────────────────────────────────────────────────────────────────────────
# Label stats endpoint
# ─────────────────────────────────────────────────────────────────────────────


async def _create_bbox(http_client, image_path: str, label_uuid: str, tag_uuids: list[str]) -> str:
    bid = str(uuid.uuid4())
    r = await http_client.post(
        "/api/images/metadata/update",
        json={
            "imagePath": image_path,
            "boxes": [
                {
                    "uuid": bid,
                    "labelUuid": label_uuid,
                    "x": 0.1,
                    "y": 0.1,
                    "width": 0.3,
                    "height": 0.3,
                    "tagUuids": tag_uuids,
                }
            ],
        },
    )
    assert r.status_code == 200, r.text
    return bid


class TestTrainingConfigLabelStats:
    ENDPOINT = "/api/training-configs/{uuid}/label-stats"

    async def test_nonexistent_config_returns_failure(self, http_client):
        r = await http_client.get(self.ENDPOINT.format(uuid=uuid.uuid4()))
        assert r.json()["status"] == "failure"

    async def test_no_datasets_returns_empty(self, http_client):
        cfg = await _create_training_config(http_client, "ls-no-ds", [])
        r = await http_client.get(self.ENDPOINT.format(uuid=cfg["uuid"]))
        body = r.json()
        assert body["status"] == "success"
        assert body["labelStats"] == []
        assert body["totalImages"] == 0

    async def test_object_scope_bbox_counts(self, http_client, files_root):
        """Object-scope label: verified/unverified bbox counts and per-image rollups.

        Dataset uses label_uuids to filter images that have bboxes for that label.
        """
        label_uuid = (await http_client.post(
            "/api/labels", json={"name": "ls-obj-lbl", "color": "#00ff00"}
        )).json()["label"]["uuid"]

        # Dataset carries label_uuids so the stats function knows which labels to measure
        # and so _list_images_for_dataset finds the annotated images via the bbox filter.
        ds = (await http_client.post(
            "/api/datasets", json={"name": "ls-obj-scope", "filter_json": {}, "label_uuids": [label_uuid]}
        )).json()["dataset"]

        # img_a: one verified bbox + one unverified bbox — sent in a single update to avoid overwrite
        img_a = await _register_image(http_client, files_root, "ls_obj_a.png")
        r = await http_client.post(
            "/api/images/metadata/update",
            json={
                "imagePath": img_a,
                "boxes": [
                    {"uuid": str(uuid.uuid4()), "labelUuid": label_uuid,
                     "x": 0.1, "y": 0.1, "width": 0.2, "height": 0.2, "tagUuids": [VERIFIED_TAG_UUID]},
                    {"uuid": str(uuid.uuid4()), "labelUuid": label_uuid,
                     "x": 0.5, "y": 0.5, "width": 0.2, "height": 0.2, "tagUuids": [UNVERIFIED_TAG_UUID]},
                ],
            },
        )
        assert r.status_code == 200, r.text

        # img_b: one verified bbox (fully clear)
        img_b = await _register_image(http_client, files_root, "ls_obj_b.png")
        await _create_bbox(http_client, img_b, label_uuid, [VERIFIED_TAG_UUID])

        cfg = await _create_training_config(http_client, "ls-obj-cfg", [ds["uuid"]])
        r = await http_client.get(self.ENDPOINT.format(uuid=cfg["uuid"]))
        body = r.json()
        assert body["status"] == "success"
        assert body["totalImages"] == 2

        stat = next(s for s in body["labelStats"] if s["label_uuid"] == label_uuid)
        assert stat["label_scope"] == "object"
        assert stat["verified_bbox_count"] == 2    # one on img_a, one on img_b
        assert stat["unverified_bbox_count"] == 1  # one on img_a
        assert stat["images_missing_verification_count"] == 1  # img_a has unverified
        assert stat["fully_verified_image_count"] == 1         # img_b is all-clear

    async def test_union_of_two_datasets_deduped(self, http_client, files_root):
        """Images shared between two datasets are counted only once in totalImages."""
        label_uuid = (await http_client.post(
            "/api/labels", json={"name": "ls-union-lbl", "color": "#0000ff"}
        )).json()["label"]["uuid"]

        # img with a bbox so the dataset bbox-filter finds it in both datasets
        img = await _register_image(http_client, files_root, "ls_union_img.png")
        await _create_bbox(http_client, img, label_uuid, [VERIFIED_TAG_UUID])

        # Two datasets both carrying the same label — both will match the same image
        ds_a = (await http_client.post(
            "/api/datasets", json={"name": "ls-union-a", "filter_json": {}, "label_uuids": [label_uuid]}
        )).json()["dataset"]
        ds_b = (await http_client.post(
            "/api/datasets", json={"name": "ls-union-b", "filter_json": {}, "label_uuids": [label_uuid]}
        )).json()["dataset"]

        cfg = await _create_training_config(http_client, "ls-union-cfg", [ds_a["uuid"], ds_b["uuid"]])
        r = await http_client.get(self.ENDPOINT.format(uuid=cfg["uuid"]))
        body = r.json()
        assert body["status"] == "success"
        # Image appears in both datasets but must be counted only once
        assert body["totalImages"] == 1
