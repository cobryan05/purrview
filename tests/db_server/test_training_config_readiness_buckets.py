"""Tests for the readiness three-bucket split: ready / needs-review (partial) / not-started.

Coverage:
- partial-review-images: an image with SOME but not all in-scope labels verified appears
  (and is excluded from ready-images and not-started-images).
- not-started-images: an image with ZERO in-scope labels verified appears (and is excluded
  from ready-images and partial-review-images).
- readiness endpoint's three counts sum to the total number of images in scope.
- rollup: needs_review_count is always 0 (no partial state possible when threshold == 1).
- keyset pagination works for both new list endpoints.
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


async def _post_bucket(http_client, config_uuid: str, path: str, **payload_overrides) -> dict:
    payload = {"limit": 50}
    payload.update(payload_overrides)
    r = await http_client.post(f"/api/training-configs/{config_uuid}/{path}", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "success", body
    return body


async def _partial_review_images(http_client, config_uuid: str, **payload_overrides) -> dict:
    return await _post_bucket(http_client, config_uuid, "partial-review-images", **payload_overrides)


async def _not_started_images(http_client, config_uuid: str, **payload_overrides) -> dict:
    return await _post_bucket(http_client, config_uuid, "not-started-images", **payload_overrides)


async def _readiness(http_client, config_uuid: str) -> dict:
    r = await http_client.get(f"/api/training-configs/{config_uuid}/readiness")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "success", body
    return body


async def _make_two_label_tc(http_client, name: str, ds_uuid: str) -> tuple[dict, str, str]:
    """Create a parent+child label pair and a training config node scoped to the parent
    (so the node's in-scope labels are [parent, child] — a threshold of 2)."""
    parent_label = await _create_label(http_client, f"{name}-parent")
    child_label = await _create_label(http_client, f"{name}-child", parent_uuid=parent_label)
    mc = await _create_model_config(
        http_client,
        f"mc-{name}",
        nodes=[
            {
                "label_uuid": parent_label,
                "model_type": "image_classifier",
                "architecture": "resnet_18",
            },
        ],
    )
    node_uuid = mc["nodes"][0]["uuid"]
    cfg = await _create_training_config(http_client, name, [ds_uuid], model_config_node_uuid=node_uuid)
    return cfg, parent_label, child_label


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestPartialReviewImages:
    async def test_partial_image_appears_and_is_excluded_from_other_buckets(self, http_client, files_root):
        ds = await _create_dataset(http_client, "bk-ds-partial")
        cfg, parent_label, child_label = await _make_two_label_tc(http_client, "bk-tc-partial", ds["uuid"])

        await _register_image(http_client, files_root, "bk_partial.png")
        # Only the parent label verified — partial coverage (1 of 2).
        await _verify_label(http_client, "bk_partial.png", parent_label, "present")

        partial_body = await _partial_review_images(http_client, cfg["uuid"])
        assert len(partial_body["images"]) == 1
        assert "bk_partial" in partial_body["images"][0]["filename"]

        ready_r = await http_client.post(f"/api/training-configs/{cfg['uuid']}/ready-images", json={"limit": 50})
        assert ready_r.json()["images"] == []

        not_started_body = await _not_started_images(http_client, cfg["uuid"])
        assert not_started_body["images"] == []

    async def test_fully_verified_image_excluded_from_partial(self, http_client, files_root):
        ds = await _create_dataset(http_client, "bk-ds-full")
        cfg, parent_label, child_label = await _make_two_label_tc(http_client, "bk-tc-full", ds["uuid"])

        await _register_image(http_client, files_root, "bk_full.png")
        await _verify_label(http_client, "bk_full.png", parent_label, "present")
        await _verify_label(http_client, "bk_full.png", child_label, "present")

        partial_body = await _partial_review_images(http_client, cfg["uuid"])
        assert partial_body["images"] == []

    async def test_pagination(self, http_client, files_root):
        ds = await _create_dataset(http_client, "bk-ds-page")
        cfg, parent_label, _child_label = await _make_two_label_tc(http_client, "bk-tc-page", ds["uuid"])

        image_names = [f"bk_partial_page_{i}.png" for i in range(5)]
        for name in image_names:
            await _register_image(http_client, files_root, name)
            await _verify_label(http_client, name, parent_label, "present")

        all_uuids: list[str] = []
        after_uuid = None
        after_sort_value = None
        for _ in range(10):
            payload: dict = {"limit": 2}
            if after_uuid:
                payload["afterUuid"] = after_uuid
                payload["afterSortValue"] = after_sort_value
            body = await _partial_review_images(http_client, cfg["uuid"], **payload)
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
        assert len(set(all_uuids)) == 5


class TestNotStartedImages:
    async def test_untouched_image_appears(self, http_client, files_root):
        ds = await _create_dataset(http_client, "bk-ds-untouched")
        cfg, _parent_label, _child_label = await _make_two_label_tc(http_client, "bk-tc-untouched", ds["uuid"])

        await _register_image(http_client, files_root, "bk_untouched.png")

        not_started_body = await _not_started_images(http_client, cfg["uuid"])
        assert len(not_started_body["images"]) == 1
        assert "bk_untouched" in not_started_body["images"][0]["filename"]

        partial_body = await _partial_review_images(http_client, cfg["uuid"])
        assert partial_body["images"] == []

    async def test_partially_touched_image_excluded_from_not_started(self, http_client, files_root):
        ds = await _create_dataset(http_client, "bk-ds-nsexcl")
        cfg, parent_label, _child_label = await _make_two_label_tc(http_client, "bk-tc-nsexcl", ds["uuid"])

        await _register_image(http_client, files_root, "bk_nsexcl.png")
        await _verify_label(http_client, "bk_nsexcl.png", parent_label, "present")

        not_started_body = await _not_started_images(http_client, cfg["uuid"])
        assert not_started_body["images"] == []

    async def test_nonexistent_config_returns_empty(self, http_client):
        r = await http_client.post(
            "/api/training-configs/00000000-0000-0000-0000-000000000000/not-started-images",
            json={},
        )
        body = r.json()
        assert body["status"] == "success"
        assert body["images"] == []

    async def test_pagination(self, http_client, files_root):
        ds = await _create_dataset(http_client, "bk-ds-ns-page")
        cfg, _parent_label, _child_label = await _make_two_label_tc(http_client, "bk-tc-ns-page", ds["uuid"])

        image_names = [f"bk_ns_page_{i}.png" for i in range(5)]
        for name in image_names:
            await _register_image(http_client, files_root, name)

        all_uuids: list[str] = []
        after_uuid = None
        after_sort_value = None
        for _ in range(10):
            payload: dict = {"limit": 2}
            if after_uuid:
                payload["afterUuid"] = after_uuid
                payload["afterSortValue"] = after_sort_value
            body = await _not_started_images(http_client, cfg["uuid"], **payload)
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
        assert len(set(all_uuids)) == 5


class TestReadinessBucketConsistency:
    async def test_counts_sum_to_total_images_and_match_list_lengths(self, http_client, files_root):
        ds = await _create_dataset(http_client, "bk-ds-consistency")
        cfg, parent_label, child_label = await _make_two_label_tc(http_client, "bk-tc-consistency", ds["uuid"])

        await _register_image(http_client, files_root, "bk_c_ready.png")
        await _verify_label(http_client, "bk_c_ready.png", parent_label, "present")
        await _verify_label(http_client, "bk_c_ready.png", child_label, "absent")

        await _register_image(http_client, files_root, "bk_c_partial.png")
        await _verify_label(http_client, "bk_c_partial.png", parent_label, "present")

        await _register_image(http_client, files_root, "bk_c_untouched.png")

        readiness = await _readiness(http_client, cfg["uuid"])
        total = readiness["readyCount"] + readiness["needsReviewCount"] + readiness["notStartedCount"]
        assert total == 3
        assert readiness["readyCount"] == 1
        assert readiness["needsReviewCount"] == 1
        assert readiness["notStartedCount"] == 1

        ready_r = await http_client.post(f"/api/training-configs/{cfg['uuid']}/ready-images", json={"limit": 50})
        assert len(ready_r.json()["images"]) == readiness["readyCount"]

        partial_body = await _partial_review_images(http_client, cfg["uuid"])
        assert len(partial_body["images"]) == readiness["needsReviewCount"]

        not_started_body = await _not_started_images(http_client, cfg["uuid"])
        assert len(not_started_body["images"]) == readiness["notStartedCount"]

    async def test_rollup_needs_review_always_zero(self, http_client, files_root):
        """Under rollup (threshold == 1), needs_review_count is always 0."""
        parent_label = await _create_label(http_client, "bk-rollup-parent")
        _child_label = await _create_label(http_client, "bk-rollup-child", parent_uuid=parent_label)

        ds = await _create_dataset(http_client, "bk-ds-rollup")
        await _register_image(http_client, files_root, "bk_rollup_ready.png")
        await _register_image(http_client, files_root, "bk_rollup_untouched.png")
        await _verify_label(http_client, "bk_rollup_ready.png", parent_label, "present")

        mc = await _create_model_config(
            http_client,
            "mc-bk-rollup",
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
        cfg = await _create_training_config(http_client, "bk-tc-rollup", [ds["uuid"]], model_config_node_uuid=node_uuid)

        readiness = await _readiness(http_client, cfg["uuid"])
        assert readiness["rollUpActive"] is True
        assert readiness["needsReviewCount"] == 0
        assert readiness["readyCount"] == 1
        assert readiness["notStartedCount"] == 1

        partial_body = await _partial_review_images(http_client, cfg["uuid"])
        assert partial_body["images"] == []
