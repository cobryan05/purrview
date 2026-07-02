"""Tests for the training config ready-images endpoint and DbClient method.

Coverage:
- Fully-verified images appear; images missing a target label's verification are excluded.
- 'absent' status counts as verified, same as 'present'.
- Union of two datasets is de-duped correctly.
- Training config with no model_config_node / no datasets → empty result.
- Keyset pagination (cursor) works across multiple pages.
- Readiness count == len(ready-images) for the same config.
- Rollup: an image counts as ready as soon as ANY ONE in-scope label is verified.
"""

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
    return r.json()["dataset"]


async def _create_model_config(http_client, name: str, nodes: list) -> dict:
    r = await http_client.post(
        "/api/model-configs",
        json={
            "name": name,
            "rootModelType": "image_classifier",
            "rootArchitecture": "resnet_18",
            "labelUuids": [n["label_uuid"] for n in nodes if n.get("label_uuid")],
            "nodes": nodes,
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "success", body
    return body["config"]


async def _create_training_config(
    http_client,
    name: str,
    dataset_uuids: list[str],
    model_config_node_uuid: str | None = None,
) -> dict:
    payload: dict = {"name": name, "datasetUuids": dataset_uuids}
    if model_config_node_uuid:
        payload["modelConfigNodeUuid"] = model_config_node_uuid
    r = await http_client.post("/api/training-configs", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "success", body
    return body["config"]


async def _register_image(http_client, files_root, name: str) -> str:
    (files_root / name).write_bytes(make_png())
    r = await http_client.post("/api/images/metadata/update", json={"imagePath": name, "boxes": []})
    assert r.status_code == 200, r.text
    return name


async def _verify_label(http_client, image_path: str, label_uuid: str, status: str = "present") -> None:
    r = await http_client.post(
        "/api/images/labels/set",
        json={"imagePath": image_path, "labelUuid": label_uuid, "status": status},
    )
    assert r.status_code == 200, r.text


async def _ready_images(http_client, config_uuid: str, **payload_overrides) -> dict:
    """POST to the ready-images endpoint and return the parsed body."""
    payload = {"limit": 50}
    payload.update(payload_overrides)
    r = await http_client.post(f"/api/training-configs/{config_uuid}/ready-images", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "success", body
    return body


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestReadyImages:
    async def _make_tc(self, http_client, name: str, ds_uuids: list[str], label_uuid: str):
        """Create a model config node for label_uuid and a training config."""
        mc = await _create_model_config(
            http_client,
            f"mc-{name}",
            nodes=[
                {
                    "label_uuid": label_uuid,
                    "model_type": "image_classifier",
                    "architecture": "resnet_18",
                }
            ],
        )
        node_uuid = mc["nodes"][0]["uuid"]
        cfg = await _create_training_config(http_client, name, ds_uuids, model_config_node_uuid=node_uuid)
        return cfg, node_uuid

    async def test_fully_verified_image_appears(self, http_client, files_root):
        """An image with its target label verified appears in the ready list."""
        ds = await _create_dataset(http_client, "ri-ds-appears")
        label_uuid = await _create_label(http_client, "ri-label-appears")
        await _register_image(http_client, files_root, "ri_appears.png")
        await _verify_label(http_client, "ri_appears.png", label_uuid, "present")
        cfg, _ = await self._make_tc(http_client, "ri-tc-appears", [ds["uuid"]], label_uuid)

        body = await _ready_images(http_client, cfg["uuid"])
        assert len(body["images"]) == 1
        assert "ri_appears" in body["images"][0]["filename"]

    async def test_unverified_image_excluded(self, http_client, files_root):
        """An image without a verified target label does NOT appear."""
        ds = await _create_dataset(http_client, "ri-ds-unverified")
        label_uuid = await _create_label(http_client, "ri-label-unverified")
        await _register_image(http_client, files_root, "ri_unverified.png")
        cfg, _ = await self._make_tc(http_client, "ri-tc-unverified", [ds["uuid"]], label_uuid)

        body = await _ready_images(http_client, cfg["uuid"])
        assert body["images"] == []
        assert body["hasMore"] is False

    async def test_absent_status_treated_as_verified(self, http_client, files_root):
        """'absent' verification status counts as verified — image DOES appear."""
        ds = await _create_dataset(http_client, "ri-ds-absent")
        label_uuid = await _create_label(http_client, "ri-label-absent")
        await _register_image(http_client, files_root, "ri_absent.png")
        await _verify_label(http_client, "ri_absent.png", label_uuid, "absent")
        cfg, _ = await self._make_tc(http_client, "ri-tc-absent", [ds["uuid"]], label_uuid)

        body = await _ready_images(http_client, cfg["uuid"])
        assert len(body["images"]) == 1

    async def test_no_datasets_returns_empty(self, http_client):
        """Training config with a node but no linked datasets → empty list."""
        label_uuid = await _create_label(http_client, "ri-label-nods")
        mc = await _create_model_config(
            http_client,
            "mc-ri-nods",
            nodes=[
                {
                    "label_uuid": label_uuid,
                    "model_type": "image_classifier",
                    "architecture": "resnet_18",
                }
            ],
        )
        node_uuid = mc["nodes"][0]["uuid"]
        cfg = await _create_training_config(http_client, "ri-tc-nods", [], model_config_node_uuid=node_uuid)

        body = await _ready_images(http_client, cfg["uuid"])
        assert body["images"] == []

    async def test_no_node_returns_empty(self, http_client, files_root):
        """Training config without a model_config_node_uuid → empty list."""
        ds = await _create_dataset(http_client, "ri-ds-nonode")
        await _register_image(http_client, files_root, "ri_nonode.png")
        cfg = await _create_training_config(http_client, "ri-tc-nonode", [ds["uuid"]])

        body = await _ready_images(http_client, cfg["uuid"])
        assert body["images"] == []

    async def test_nonexistent_config_returns_empty(self, http_client):
        """Unknown config UUID → success with empty images (not a 404 in this convention)."""
        r = await http_client.post(
            "/api/training-configs/00000000-0000-0000-0000-000000000000/ready-images",
            json={},
        )
        body = r.json()
        assert body["status"] == "success"
        assert body["images"] == []

    async def test_mixed_images(self, http_client, files_root):
        """Only verified images appear; unverified ones are excluded."""
        ds = await _create_dataset(http_client, "ri-ds-mixed")
        label_uuid = await _create_label(http_client, "ri-label-mixed")

        await _register_image(http_client, files_root, "ri_mix_unverified.png")
        await _register_image(http_client, files_root, "ri_mix_verified.png")
        await _verify_label(http_client, "ri_mix_verified.png", label_uuid, "present")

        cfg, _ = await self._make_tc(http_client, "ri-tc-mixed", [ds["uuid"]], label_uuid)

        body = await _ready_images(http_client, cfg["uuid"])
        assert len(body["images"]) == 1
        filenames = [img["filename"] for img in body["images"]]
        assert any("ri_mix_verified" in f for f in filenames)
        assert not any("ri_mix_unverified" in f for f in filenames)

    async def test_union_of_two_datasets(self, http_client, files_root):
        """Images from both linked datasets appear in the merged list."""
        ds_a = await _create_dataset(http_client, "ri-union-a")
        ds_b = await _create_dataset(http_client, "ri-union-b")
        label_uuid = await _create_label(http_client, "ri-union-label")

        await _register_image(http_client, files_root, "ri_union_img_a.png")
        await _register_image(http_client, files_root, "ri_union_img_b.png")

        # Verify img_a — it should appear; img_b unverified → excluded
        await _verify_label(http_client, "ri_union_img_a.png", label_uuid, "present")

        cfg, _ = await self._make_tc(http_client, "ri-tc-union", [ds_a["uuid"], ds_b["uuid"]], label_uuid)

        body = await _ready_images(http_client, cfg["uuid"])
        assert len(body["images"]) == 1
        assert "ri_union_img_a" in body["images"][0]["filename"]

    async def test_union_deduplication(self, http_client, files_root):
        """An image that belongs to both datasets appears only once."""
        ds_a = await _create_dataset(http_client, "ri-dedup-a")
        ds_b = await _create_dataset(http_client, "ri-dedup-b")
        label_uuid = await _create_label(http_client, "ri-dedup-label")

        # Single image, both datasets are open (no filter) so it belongs to both
        await _register_image(http_client, files_root, "ri_dedup_img.png")
        await _verify_label(http_client, "ri_dedup_img.png", label_uuid, "present")

        cfg, _ = await self._make_tc(http_client, "ri-tc-dedup", [ds_a["uuid"], ds_b["uuid"]], label_uuid)

        body = await _ready_images(http_client, cfg["uuid"])
        assert len(body["images"]) == 1  # de-duped, not 2

    async def test_readiness_count_matches_list_length(self, http_client, files_root):
        """readiness.readyCount == len(ready-images) for the same config."""
        ds = await _create_dataset(http_client, "ri-match-ds")
        label_uuid = await _create_label(http_client, "ri-match-label")

        await _register_image(http_client, files_root, "ri_match_img1.png")
        await _register_image(http_client, files_root, "ri_match_img2.png")
        # Verify one of them
        await _verify_label(http_client, "ri_match_img2.png", label_uuid, "absent")

        cfg, _ = await self._make_tc(http_client, "ri-tc-match", [ds["uuid"]], label_uuid)

        readiness_r = await http_client.get(f"/api/training-configs/{cfg['uuid']}/readiness")
        readiness = readiness_r.json()
        assert readiness["status"] == "success"

        ready_body = await _ready_images(http_client, cfg["uuid"])
        assert len(ready_body["images"]) == readiness["readyCount"]

    async def test_pagination(self, http_client, files_root):
        """Keyset cursor correctly pages through all ready images."""
        ds = await _create_dataset(http_client, "ri-page-ds")
        label_uuid = await _create_label(http_client, "ri-page-label")

        image_names = [f"ri_page_img_{i}.png" for i in range(5)]
        for name in image_names:
            await _register_image(http_client, files_root, name)
            await _verify_label(http_client, name, label_uuid, "present")

        cfg, _ = await self._make_tc(http_client, "ri-tc-page", [ds["uuid"]], label_uuid)

        # Collect all 5 images by paging with limit=2
        all_uuids: list[str] = []
        after_uuid = None
        after_sort_value = None

        for _ in range(10):  # safety limit
            payload: dict = {"limit": 2}
            if after_uuid:
                payload["afterUuid"] = after_uuid
                payload["afterSortValue"] = after_sort_value
            body = await _ready_images(http_client, cfg["uuid"], **payload)
            page_images = body["images"]
            if not page_images:
                break
            for img in page_images:
                all_uuids.append(img["uuid"])
            if not body["hasMore"]:
                break
            last = page_images[-1]
            after_uuid = last["uuid"]
            after_sort_value = last.get("createdAt")

        assert len(all_uuids) == 5
        # No duplicates
        assert len(set(all_uuids)) == 5

    async def test_two_labels_partial_verification_excluded(self, http_client, files_root):
        """When a node's scope spans two labels (parent + child), only ONE verified is not enough."""
        parent_label = await _create_label(http_client, "ri-hier-parent")
        child_label = await _create_label(http_client, "ri-hier-child", parent_uuid=parent_label)

        ds = await _create_dataset(http_client, "ri-ds-hier")
        await _register_image(http_client, files_root, "ri_hier_img.png")
        await _verify_label(http_client, "ri_hier_img.png", parent_label, "present")
        # child_label left unverified

        mc = await _create_model_config(
            http_client,
            "mc-ri-hier",
            nodes=[
                {
                    "label_uuid": parent_label,
                    "model_type": "image_classifier",
                    "architecture": "resnet_18",
                },
            ],
        )
        node_uuid = mc["nodes"][0]["uuid"]
        cfg = await _create_training_config(http_client, "ri-tc-hier", [ds["uuid"]], model_config_node_uuid=node_uuid)

        body = await _ready_images(http_client, cfg["uuid"])
        assert body["images"] == []

    async def test_rollup_ready_with_any_one_label_verified(self, http_client, files_root):
        """With roll_up_unchecked active, verifying ANY ONE in-scope label is enough to be ready."""
        parent_label = await _create_label(http_client, "ri-rollup-parent")
        child_label = await _create_label(http_client, "ri-rollup-child", parent_uuid=parent_label)

        ds = await _create_dataset(http_client, "ri-ds-rollup")
        await _register_image(http_client, files_root, "ri_rollup_img.png")
        # Only verify the parent — under rollup this alone should be enough.
        await _verify_label(http_client, "ri_rollup_img.png", parent_label, "present")

        mc = await _create_model_config(
            http_client,
            "mc-ri-rollup",
            nodes=[
                {
                    "label_uuid": parent_label,
                    "model_type": "image_classifier",
                    "architecture": "resnet_18",
                    "roll_up_unchecked": True,
                },
            ],
        )
        node_uuid = mc["nodes"][0]["uuid"]
        cfg = await _create_training_config(
            http_client, "ri-tc-rollup", [ds["uuid"]], model_config_node_uuid=node_uuid
        )

        body = await _ready_images(http_client, cfg["uuid"])
        assert len(body["images"]) == 1
        assert "ri_rollup_img" in body["images"][0]["filename"]

        readiness_r = await http_client.get(f"/api/training-configs/{cfg['uuid']}/readiness")
        readiness = readiness_r.json()
        assert readiness["readyCount"] == 1
        assert readiness["rollUpActive"] is True
