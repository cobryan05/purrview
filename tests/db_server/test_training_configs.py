"""Tests for the training configurations feature — CRUD, readiness endpoint, label stats."""

import uuid

from apps.helpers.db.types import UNVERIFIED_TAG_UUID, VERIFIED_TAG_UUID
from tests.conftest import make_png

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


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
    http_client,
    name: str,
    dataset_uuids: list[str],
    *,
    model_config_node_uuid: str | None = None,
    description: str | None = None,
    hyperparams_json: dict | None = None,
) -> dict:
    payload: dict = {
        "name": name,
        "datasetUuids": dataset_uuids,
    }
    if description is not None:
        payload["description"] = description
    if model_config_node_uuid is not None:
        payload["modelConfigNodeUuid"] = model_config_node_uuid
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


async def _set_image_label(http_client, image_path: str, label_uuid: str, status: str) -> None:
    r = await http_client.post(
        "/api/images/labels/set",
        json={"imagePath": image_path, "labelUuid": label_uuid, "status": status},
    )
    assert r.status_code == 200, r.text


# ─────────────────────────────────────────────────────────────────────────────
# CRUD
# ─────────────────────────────────────────────────────────────────────────────


class TestTrainingConfigCrud:
    async def test_create_and_get(self, http_client):
        ds = await _create_dataset(http_client, "ds-crud")
        cfg = await _create_training_config(http_client, "tc-basic", [ds["uuid"]])
        assert cfg["name"] == "tc-basic"
        assert cfg["datasetUuids"] == [ds["uuid"]]
        assert cfg["modelConfigNodeUuid"] is None
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

    async def test_create_with_model_config_node(self, http_client):
        label_uuid = await _create_label(http_client, "tc-node-label")
        mc = await _create_model_config(
            http_client,
            "tc-node-mc",
            nodes=[
                {
                    "label_uuid": label_uuid,
                    "model_type": "image_classifier",
                    "architecture": "resnet_18",
                }
            ],
        )
        node_uuid = mc["nodes"][0]["uuid"]
        cfg = await _create_training_config(http_client, "tc-with-node", [], model_config_node_uuid=node_uuid)
        assert cfg["modelConfigNodeUuid"] == node_uuid

    async def test_create_with_hyperparams(self, http_client):
        ds = await _create_dataset(http_client, "ds-hyper")
        cfg = await _create_training_config(
            http_client,
            "tc-hyper",
            [ds["uuid"]],
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

    async def test_update_name(self, http_client):
        ds = await _create_dataset(http_client, "ds-update")
        cfg = await _create_training_config(http_client, "tc-update", [ds["uuid"]])

        r = await http_client.put(
            f"/api/training-configs/{cfg['uuid']}",
            json={"name": "tc-updated"},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "success"
        assert body["config"]["name"] == "tc-updated"

    async def test_update_model_config_node_uuid(self, http_client):
        """Updating modelConfigNodeUuid replaces the node link."""
        label_a = await _create_label(http_client, "upd-node-lbl-a")
        label_b = await _create_label(http_client, "upd-node-lbl-b")
        mc = await _create_model_config(
            http_client,
            "mc-upd-node",
            nodes=[
                {
                    "label_uuid": label_a,
                    "model_type": "image_classifier",
                    "architecture": "resnet_18",
                },
                {
                    "label_uuid": label_b,
                    "model_type": "image_classifier",
                    "architecture": "resnet_50",
                },
            ],
        )
        node_a = next(n for n in mc["nodes"] if n["labelUuid"] == label_a)
        node_b = next(n for n in mc["nodes"] if n["labelUuid"] == label_b)

        cfg = await _create_training_config(http_client, "tc-upd-node", [], model_config_node_uuid=node_a["uuid"])
        assert cfg["modelConfigNodeUuid"] == node_a["uuid"]

        r = await http_client.put(
            f"/api/training-configs/{cfg['uuid']}",
            json={"modelConfigNodeUuid": node_b["uuid"]},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "success"
        assert body["config"]["modelConfigNodeUuid"] == node_b["uuid"]

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
            json={"name": "tc-dup", "datasetUuids": [ds["uuid"]]},
        )
        assert r.json()["status"] == "failure"

    async def test_get_nonexistent_returns_failure(self, http_client):
        r = await http_client.get(f"/api/training-configs/{uuid.uuid4()}")
        assert r.json()["status"] == "failure"


# ─────────────────────────────────────────────────────────────────────────────
# Readiness endpoint
# ─────────────────────────────────────────────────────────────────────────────


class TestTrainingConfigReadiness:
    async def _make_tc_with_node(self, http_client, tc_name: str, ds_uuid: str, label_uuid: str) -> tuple[dict, str]:
        """Create a model config with an image_classifier node for label_uuid, then a training config."""
        mc = await _create_model_config(
            http_client,
            f"mc-{tc_name}",
            nodes=[
                {
                    "label_uuid": label_uuid,
                    "model_type": "image_classifier",
                    "architecture": "resnet_18",
                }
            ],
        )
        node_uuid = mc["nodes"][0]["uuid"]
        cfg = await _create_training_config(http_client, tc_name, [ds_uuid], model_config_node_uuid=node_uuid)
        return cfg, node_uuid

    async def test_empty_dataset_returns_zero_counts(self, http_client):
        ds = await _create_dataset(http_client, "ds-ready-empty")
        label_uuid = await _create_label(http_client, "empty-label")
        cfg, _ = await self._make_tc_with_node(http_client, "tc-ready-empty", ds["uuid"], label_uuid)
        r = await http_client.get(f"/api/training-configs/{cfg['uuid']}/readiness")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "success"
        assert body["readyCount"] == 0
        assert body["needsReviewCount"] == 0

    async def test_no_node_returns_zero_counts(self, http_client, files_root):
        """Training config with no model_config_node_uuid has no labels to check."""
        ds = await _create_dataset(http_client, "ds-ready-nonode")
        await _register_image(http_client, files_root, "ready_nonode.png")
        cfg = await _create_training_config(http_client, "tc-ready-nonode", [ds["uuid"]])
        r = await http_client.get(f"/api/training-configs/{cfg['uuid']}/readiness")
        body = r.json()
        assert body["readyCount"] == 0
        assert body["needsReviewCount"] == 0

    async def test_unreviewed_image_counted_as_needs_review(self, http_client, files_root):
        ds = await _create_dataset(http_client, "ds-ready-unreviewed")
        label_uuid = await _create_label(http_client, "unreviewed-label")
        await _register_image(http_client, files_root, "ready_unreviewed.png")
        cfg, _ = await self._make_tc_with_node(http_client, "tc-ready-unreviewed", ds["uuid"], label_uuid)
        r = await http_client.get(f"/api/training-configs/{cfg['uuid']}/readiness")
        body = r.json()
        assert body["readyCount"] == 0
        assert body["needsReviewCount"] == 1

    async def test_present_label_counted_as_ready(self, http_client, files_root):
        ds = await _create_dataset(http_client, "ds-ready-present")
        label_uuid = await _create_label(http_client, "present-label")
        await _register_image(http_client, files_root, "ready_present.png")

        r = await http_client.post(
            "/api/images/labels/set",
            json={
                "imagePath": "ready_present.png",
                "labelUuid": label_uuid,
                "status": "present",
            },
        )
        assert r.json()["status"] == "success"

        cfg, _ = await self._make_tc_with_node(http_client, "tc-ready-present", ds["uuid"], label_uuid)
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
            json={
                "imagePath": "ready_absent.png",
                "labelUuid": label_uuid,
                "status": "absent",
            },
        )
        assert r.json()["status"] == "success"

        cfg, _ = await self._make_tc_with_node(http_client, "tc-ready-absent", ds["uuid"], label_uuid)
        r = await http_client.get(f"/api/training-configs/{cfg['uuid']}/readiness")
        body = r.json()
        assert body["readyCount"] == 1
        assert body["needsReviewCount"] == 0

    async def test_mixed_ready_and_needs_review(self, http_client, files_root):
        """Multiple images — some ready, some not."""
        ds = await _create_dataset(http_client, "ds-ready-mixed")
        label_uuid = await _create_label(http_client, "mixed-label")
        await _register_image(http_client, files_root, "ready_mixed_a.png")
        await _register_image(http_client, files_root, "ready_mixed_b.png")

        # Verify label on image A only; image B stays unreviewed
        await http_client.post(
            "/api/images/labels/set",
            json={
                "imagePath": "ready_mixed_a.png",
                "labelUuid": label_uuid,
                "status": "absent",
            },
        )

        cfg, _ = await self._make_tc_with_node(http_client, "tc-ready-mixed", ds["uuid"], label_uuid)
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
        cfg, _ = await self._make_tc_with_node(http_client, "tc-ready-meta", ds["uuid"], label_uuid)
        r = await http_client.get(f"/api/training-configs/{cfg['uuid']}/readiness")
        body = r.json()
        assert ds["uuid"] in body["datasetUuids"]
        assert label_uuid in body["labelUuids"]

    async def test_readiness_union_of_two_datasets(self, http_client, files_root):
        """Readiness counts images from both linked datasets (union, de-duped)."""
        ds_a = await _create_dataset(http_client, "ds-union-a")
        ds_b = await _create_dataset(http_client, "ds-union-b")
        label_uuid = await _create_label(http_client, "union-label")

        await _register_image(http_client, files_root, "union_img_a.png")
        await _register_image(http_client, files_root, "union_img_b.png")

        # Verify label on img_a only
        await http_client.post(
            "/api/images/labels/set",
            json={
                "imagePath": "union_img_a.png",
                "labelUuid": label_uuid,
                "status": "present",
            },
        )

        mc = await _create_model_config(
            http_client,
            "mc-union",
            nodes=[
                {
                    "label_uuid": label_uuid,
                    "model_type": "image_classifier",
                    "architecture": "resnet_18",
                }
            ],
        )
        node_uuid = mc["nodes"][0]["uuid"]
        cfg = await _create_training_config(
            http_client,
            "tc-union",
            [ds_a["uuid"], ds_b["uuid"]],
            model_config_node_uuid=node_uuid,
        )
        r = await http_client.get(f"/api/training-configs/{cfg['uuid']}/readiness")
        body = r.json()
        assert body["readyCount"] == 1
        assert body["needsReviewCount"] == 1
        assert set(body["datasetUuids"]) == {ds_a["uuid"], ds_b["uuid"]}

    async def test_node_label_includes_descendants(self, http_client, files_root):
        """Readiness for a node with label_uuid=A includes A's child labels too."""
        parent_label = await _create_label(http_client, "parent-hier-lbl")
        child_label = await _create_label(http_client, "child-hier-lbl", parent_uuid=parent_label)

        ds = await _create_dataset(http_client, "ds-hier")
        await _register_image(http_client, files_root, "hier_img.png")

        # Mark parent label verified; child unverified → not fully ready (2 labels needed)
        await http_client.post(
            "/api/images/labels/set",
            json={
                "imagePath": "hier_img.png",
                "labelUuid": parent_label,
                "status": "present",
            },
        )

        mc = await _create_model_config(
            http_client,
            "mc-hier",
            nodes=[
                {
                    "label_uuid": parent_label,
                    "model_type": "image_classifier",
                    "architecture": "resnet_18",
                },
            ],
        )
        node_uuid = mc["nodes"][0]["uuid"]
        cfg = await _create_training_config(http_client, "tc-hier", [ds["uuid"]], model_config_node_uuid=node_uuid)
        r = await http_client.get(f"/api/training-configs/{cfg['uuid']}/readiness")
        body = r.json()
        # Both parent and child label required; only parent verified → needs_review
        assert body["needsReviewCount"] == 1
        assert body["readyCount"] == 0
        assert parent_label in body["labelUuids"]
        assert child_label in body["labelUuids"]

    async def test_root_node_uses_all_config_labels(self, http_client, files_root):
        """Root node (label_uuid=None) readiness uses all non-root labels in the config."""
        ds = await _create_dataset(http_client, "ds-root-node")
        label_a = await _create_label(http_client, "root-lbl-a")
        label_b = await _create_label(http_client, "root-lbl-b")
        await _register_image(http_client, files_root, "root_node_img.png")

        # Mark only label_a; label_b remains unreviewed → image not fully ready
        await http_client.post(
            "/api/images/labels/set",
            json={
                "imagePath": "root_node_img.png",
                "labelUuid": label_a,
                "status": "present",
            },
        )

        mc = await _create_model_config(
            http_client,
            "mc-root-node",
            nodes=[
                {
                    "label_uuid": None,
                    "model_type": "object_detection",
                    "architecture": "yolo_v8_n",
                },
                {
                    "label_uuid": label_a,
                    "model_type": "image_classifier",
                    "architecture": "resnet_18",
                },
                {
                    "label_uuid": label_b,
                    "model_type": "image_classifier",
                    "architecture": "resnet_18",
                },
            ],
        )
        root_node = next(n for n in mc["nodes"] if n["labelUuid"] is None)
        cfg = await _create_training_config(
            http_client,
            "tc-root-node",
            [ds["uuid"]],
            model_config_node_uuid=root_node["uuid"],
        )
        r = await http_client.get(f"/api/training-configs/{cfg['uuid']}/readiness")
        body = r.json()
        # Needs both label_a and label_b; only label_a is verified → needs_review
        assert body["needsReviewCount"] == 1
        assert body["readyCount"] == 0


class TestRollupLabelDerivation:
    async def test_nested_rollup_parents_claim_nearest_ancestor(self, http_client, db_client):
        """A descendant shared by nested rollup parents rolls up into the NEAREST one.

        With A→B→C where A and B both have roll_up_unchecked nodes and C has no node,
        C must be claimed by B (its nearest rollup ancestor) and counted exactly once,
        regardless of the order the model_config_nodes are returned by the DB.
        """
        label_a = await _create_label(http_client, "rollup-a")
        label_b = await _create_label(http_client, "rollup-b", parent_uuid=label_a)
        label_c = await _create_label(http_client, "rollup-c", parent_uuid=label_b)

        mc = await _create_model_config(
            http_client,
            "mc-nested-rollup",
            nodes=[
                {"label_uuid": None, "model_type": "object_detection", "architecture": "yolo_v8_n"},
                {
                    "label_uuid": label_a,
                    "model_type": "image_classifier",
                    "architecture": "resnet_18",
                    "roll_up_unchecked": True,
                },
                {
                    "label_uuid": label_b,
                    "model_type": "image_classifier",
                    "architecture": "resnet_18",
                    "roll_up_unchecked": True,
                },
            ],
        )
        root_node = next(n for n in mc["nodes"] if n["labelUuid"] is None)

        label_uuids, rollup_map = await db_client._derive_labels_for_node(root_node["uuid"])

        # C is claimed by its nearest rollup ancestor (B), not the farther one (A).
        assert label_c in rollup_map.get(label_b, [])
        assert label_c not in rollup_map.get(label_a, [])

        # No label is double-counted across rollup parents.
        all_claimed = [u for children in rollup_map.values() for u in children]
        assert len(all_claimed) == len(set(all_claimed))

        # All three labels are in scope; parents precede their rollup children.
        assert {label_a, label_b, label_c} <= set(label_uuids)
        assert label_uuids.index(label_b) < label_uuids.index(label_c)


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
        """Object-scope label: verified/unverified bbox counts and per-image rollups."""
        label_uuid = (await http_client.post("/api/labels", json={"name": "ls-obj-lbl", "color": "#00ff00"})).json()[
            "label"
        ]["uuid"]

        ds = (
            await http_client.post(
                "/api/datasets",
                json={
                    "name": "ls-obj-scope",
                    "filter_json": {},
                    "label_uuids": [label_uuid],
                },
            )
        ).json()["dataset"]

        img_a = await _register_image(http_client, files_root, "ls_obj_a.png")
        r = await http_client.post(
            "/api/images/metadata/update",
            json={
                "imagePath": img_a,
                "boxes": [
                    {
                        "uuid": str(uuid.uuid4()),
                        "labelUuid": label_uuid,
                        "x": 0.1,
                        "y": 0.1,
                        "width": 0.2,
                        "height": 0.2,
                        "tagUuids": [VERIFIED_TAG_UUID],
                    },
                    {
                        "uuid": str(uuid.uuid4()),
                        "labelUuid": label_uuid,
                        "x": 0.5,
                        "y": 0.5,
                        "width": 0.2,
                        "height": 0.2,
                        "tagUuids": [UNVERIFIED_TAG_UUID],
                    },
                ],
            },
        )
        assert r.status_code == 200, r.text

        img_b = await _register_image(http_client, files_root, "ls_obj_b.png")
        await _create_bbox(http_client, img_b, label_uuid, [VERIFIED_TAG_UUID])

        # Create model config with a node for this label, link via model_config_node_uuid.
        mc = (
            await http_client.post(
                "/api/model-configs",
                json={
                    "name": "mc-ls-obj",
                    "nodes": [
                        {
                            "label_uuid": label_uuid,
                            "model_type": "object_detection",
                            "architecture": "yolo_v8_n",
                        }
                    ],
                },
            )
        ).json()["config"]
        node_uuid = mc["nodes"][0]["uuid"]

        cfg = await _create_training_config(http_client, "ls-obj-cfg", [ds["uuid"]], model_config_node_uuid=node_uuid)
        r = await http_client.get(self.ENDPOINT.format(uuid=cfg["uuid"]))
        body = r.json()
        assert body["status"] == "success"
        assert body["totalImages"] == 2

        stat = next(s for s in body["labelStats"] if s["labelUuid"] == label_uuid)
        assert stat["labelScope"] == "object"
        assert stat["verifiedBboxCount"] == 2
        assert stat["unverifiedBboxCount"] == 1
        assert stat["imagesMissingVerificationCount"] == 1
        assert stat["fullyVerifiedImageCount"] == 1

    async def test_root_node_derives_labels_from_config(self, http_client, files_root):
        """Label stats with a root-node uses all non-root nodes' labels in the config."""
        label_uuid = (await http_client.post("/api/labels", json={"name": "ls-root-lbl", "color": "#ff00ff"})).json()[
            "label"
        ]["uuid"]

        ds = (
            await http_client.post(
                "/api/datasets",
                json={
                    "name": "ls-root-ds",
                    "filter_json": {},
                    "label_uuids": [label_uuid],
                },
            )
        ).json()["dataset"]

        img = await _register_image(http_client, files_root, "ls_root_img.png")
        await _create_bbox(http_client, img, label_uuid, [VERIFIED_TAG_UUID])

        mc = (
            await http_client.post(
                "/api/model-configs",
                json={
                    "name": "mc-ls-root",
                    "nodes": [
                        {
                            "label_uuid": None,
                            "model_type": "object_detection",
                            "architecture": "yolo_v8_n",
                        },
                        {
                            "label_uuid": label_uuid,
                            "model_type": "object_detection",
                            "architecture": "yolo_v8_n",
                        },
                    ],
                },
            )
        ).json()["config"]
        root_node = next(n for n in mc["nodes"] if n["labelUuid"] is None)

        cfg = await _create_training_config(
            http_client,
            "ls-root-cfg",
            [ds["uuid"]],
            model_config_node_uuid=root_node["uuid"],
        )
        r = await http_client.get(self.ENDPOINT.format(uuid=cfg["uuid"]))
        body = r.json()
        assert body["status"] == "success"
        assert body["totalImages"] == 1
        # Root node should produce stats for label_uuid (the only non-root node label)
        assert any(s["labelUuid"] == label_uuid for s in body["labelStats"])

    async def test_union_of_two_datasets_deduped(self, http_client, files_root):
        """Images shared between two datasets are counted only once in totalImages."""
        label_uuid = (await http_client.post("/api/labels", json={"name": "ls-union-lbl", "color": "#0000ff"})).json()[
            "label"
        ]["uuid"]

        img = await _register_image(http_client, files_root, "ls_union_img.png")
        await _create_bbox(http_client, img, label_uuid, [VERIFIED_TAG_UUID])

        ds_a = (
            await http_client.post(
                "/api/datasets",
                json={
                    "name": "ls-union-a",
                    "filter_json": {},
                    "label_uuids": [label_uuid],
                },
            )
        ).json()["dataset"]
        ds_b = (
            await http_client.post(
                "/api/datasets",
                json={
                    "name": "ls-union-b",
                    "filter_json": {},
                    "label_uuids": [label_uuid],
                },
            )
        ).json()["dataset"]

        mc = (
            await http_client.post(
                "/api/model-configs",
                json={
                    "name": "mc-ls-union",
                    "nodes": [
                        {
                            "label_uuid": label_uuid,
                            "model_type": "object_detection",
                            "architecture": "yolo_v8_n",
                        }
                    ],
                },
            )
        ).json()["config"]
        node_uuid = mc["nodes"][0]["uuid"]

        cfg = await _create_training_config(
            http_client,
            "ls-union-cfg",
            [ds_a["uuid"], ds_b["uuid"]],
            model_config_node_uuid=node_uuid,
        )
        r = await http_client.get(self.ENDPOINT.format(uuid=cfg["uuid"]))
        body = r.json()
        assert body["status"] == "success"
        assert body["totalImages"] == 1

    async def test_image_scope_present_absent_counts(self, http_client, files_root):
        """Image-scope label: present/absent/unverified counts are correct."""
        label_uuid = (
            await http_client.post("/api/labels", json={"name": "ls-img-lbl", "color": "#ff0000", "scope": "image"})
        ).json()["label"]["uuid"]

        # No label_uuids filter — image-scope labels have no bboxes, so filtering by
        # label_uuids (which scopes to bbox labels) would return 0 images.
        # The relevant labels are derived from the model config node below.
        ds = (
            await http_client.post(
                "/api/datasets",
                json={"name": "ls-img-scope", "filter_json": {}},
            )
        ).json()["dataset"]

        img_present_1 = await _register_image(http_client, files_root, "ls_img_present1.png")
        img_present_2 = await _register_image(http_client, files_root, "ls_img_present2.png")
        img_absent = await _register_image(http_client, files_root, "ls_img_absent.png")
        _unverified = await _register_image(http_client, files_root, "ls_img_unverified.png")

        await _set_image_label(http_client, img_present_1, label_uuid, "present")
        await _set_image_label(http_client, img_present_2, label_uuid, "present")
        await _set_image_label(http_client, img_absent, label_uuid, "absent")
        # ls_img_unverified.png has no label status set

        mc = (
            await http_client.post(
                "/api/model-configs",
                json={
                    "name": "mc-ls-img",
                    "nodes": [
                        {
                            "label_uuid": label_uuid,
                            "model_type": "image_classifier",
                            "architecture": "resnet_18",
                        }
                    ],
                },
            )
        ).json()["config"]
        node_uuid = mc["nodes"][0]["uuid"]

        cfg = await _create_training_config(http_client, "ls-img-cfg", [ds["uuid"]], model_config_node_uuid=node_uuid)
        r = await http_client.get(self.ENDPOINT.format(uuid=cfg["uuid"]))
        body = r.json()
        assert body["status"] == "success"
        assert body["totalImages"] == 4

        stat = next(s for s in body["labelStats"] if s["labelUuid"] == label_uuid)
        assert stat["labelScope"] == "image"
        assert stat["presentImageCount"] == 2
        assert stat["absentImageCount"] == 1
        assert stat["verifiedImageCount"] == 3
        assert stat["unverifiedImageCount"] == 1
