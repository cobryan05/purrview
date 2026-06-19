"""Tests for the model bundles feature — CRUD and FK constraint behavior."""

import uuid

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


async def _create_model_config(http_client, name: str) -> dict:
    r = await http_client.post("/api/model-configs", json={"name": name, "nodes": []})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "success", body
    return body["config"]


async def _create_bundle(
    http_client,
    name: str,
    model_config_uuid: str,
    *,
    description: str | None = None,
    bundle_path: str = "/models/bundle.zip",
    manifest_json: dict | None = None,
    training_config_uuids: list | None = None,
) -> dict:
    payload: dict = {
        "name": name,
        "modelConfigUuid": model_config_uuid,
        "bundlePath": bundle_path,
    }
    if description is not None:
        payload["description"] = description
    if manifest_json is not None:
        payload["manifestJson"] = manifest_json
    if training_config_uuids is not None:
        payload["trainingConfigUuids"] = training_config_uuids
    r = await http_client.post("/api/model-bundles", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "success", body
    return body["bundle"]


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestModelBundleCrud:
    async def test_create_and_list(self, http_client):
        mc = await _create_model_config(http_client, "mc-bundle-list")
        bundle = await _create_bundle(http_client, "bundle-list", mc["uuid"])
        assert bundle["name"] == "bundle-list"
        assert bundle["modelConfigUuid"] == mc["uuid"]
        assert bundle["uuid"]

        r = await http_client.get("/api/model-bundles")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "success"
        uuids = [b["uuid"] for b in body["bundles"]]
        assert bundle["uuid"] in uuids

    async def test_get_single(self, http_client):
        mc = await _create_model_config(http_client, "mc-bundle-get")
        bundle = await _create_bundle(http_client, "bundle-get", mc["uuid"], description="test desc")
        r = await http_client.get(f"/api/model-bundles/{bundle['uuid']}")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "success"
        assert body["bundle"]["uuid"] == bundle["uuid"]
        assert body["bundle"]["description"] == "test desc"

    async def test_get_nonexistent_returns_failure(self, http_client):
        r = await http_client.get(f"/api/model-bundles/{uuid.uuid4()}")
        assert r.json()["status"] == "failure"

    async def test_create_with_manifest_and_training_configs(self, http_client):
        mc = await _create_model_config(http_client, "mc-bundle-manifest")
        tc_uuid = str(uuid.uuid4())
        bundle = await _create_bundle(
            http_client,
            "bundle-manifest",
            mc["uuid"],
            manifest_json={"version": "1.0", "format": "onnx"},
            training_config_uuids=[tc_uuid],
        )
        assert bundle["manifestJson"]["version"] == "1.0"
        assert tc_uuid in bundle["trainingConfigUuids"]

    async def test_update_name(self, http_client):
        mc = await _create_model_config(http_client, "mc-bundle-upd-name")
        bundle = await _create_bundle(http_client, "bundle-upd-name", mc["uuid"])

        r = await http_client.patch(
            f"/api/model-bundles/{bundle['uuid']}",
            json={"name": "bundle-renamed"},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "success"
        assert body["bundle"]["name"] == "bundle-renamed"

    async def test_update_description(self, http_client):
        mc = await _create_model_config(http_client, "mc-bundle-upd-desc")
        bundle = await _create_bundle(http_client, "bundle-upd-desc", mc["uuid"])

        r = await http_client.patch(
            f"/api/model-bundles/{bundle['uuid']}",
            json={"description": "updated description"},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "success"
        assert body["bundle"]["description"] == "updated description"

    async def test_delete(self, http_client):
        mc = await _create_model_config(http_client, "mc-bundle-del")
        bundle = await _create_bundle(http_client, "bundle-del", mc["uuid"])

        r = await http_client.delete(f"/api/model-bundles/{bundle['uuid']}")
        assert r.status_code == 200
        assert r.json()["status"] == "success"

        r2 = await http_client.get(f"/api/model-bundles/{bundle['uuid']}")
        assert r2.json()["status"] == "failure"

    async def test_delete_nonexistent_returns_failure(self, http_client):
        r = await http_client.delete(f"/api/model-bundles/{uuid.uuid4()}")
        assert r.json()["status"] == "failure"

    async def test_duplicate_name_rejected(self, http_client):
        mc = await _create_model_config(http_client, "mc-bundle-dup")
        await _create_bundle(http_client, "bundle-dup", mc["uuid"])
        r = await http_client.post(
            "/api/model-bundles",
            json={"name": "bundle-dup", "modelConfigUuid": mc["uuid"], "bundlePath": "/models/x.zip"},
        )
        assert r.json()["status"] == "failure"

    async def test_model_config_deletion_blocked_when_bundle_references_it(self, http_client):
        """Deleting a model config that has a bundle referencing it should fail."""
        mc = await _create_model_config(http_client, "mc-bundle-fk")
        await _create_bundle(http_client, "bundle-fk", mc["uuid"])

        r = await http_client.delete(f"/api/model-configs/{mc['uuid']}")
        # The FK constraint (ondelete=RESTRICT) should prevent this.
        body = r.json()
        assert body["status"] == "failure"

    async def test_bundle_survives_after_model_config_deleted_if_bundle_removed_first(self, http_client):
        """Once the bundle is removed, the model config can be deleted."""
        mc = await _create_model_config(http_client, "mc-bundle-order")
        bundle = await _create_bundle(http_client, "bundle-order", mc["uuid"])

        await http_client.delete(f"/api/model-bundles/{bundle['uuid']}")

        r = await http_client.delete(f"/api/model-configs/{mc['uuid']}")
        assert r.json()["status"] == "success"
