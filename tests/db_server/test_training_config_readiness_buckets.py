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


class TestNeedsReviewRequiresVerifiedPresence:
    """'needs review' must be scoped to images that actually contain (verified PRESENT)
    one or more of the training-set labels — an image whose only partial verification
    is ABSENT hasn't surfaced any in-scope label and belongs in 'not started' instead."""

    async def test_absent_only_partial_image_falls_into_not_started(self, http_client, files_root):
        ds = await _create_dataset(http_client, "bk-ds-absent-only")
        cfg, parent_label, child_label = await _make_two_label_tc(http_client, "bk-tc-absent-only", ds["uuid"])

        await _register_image(http_client, files_root, "bk_absent_only.png")
        # Only the parent label verified, and as ABSENT — no in-scope label actually present.
        await _verify_label(http_client, "bk_absent_only.png", parent_label, "absent")

        partial_body = await _partial_review_images(http_client, cfg["uuid"])
        assert partial_body["images"] == []

        not_started_body = await _not_started_images(http_client, cfg["uuid"])
        assert len(not_started_body["images"]) == 1
        assert "bk_absent_only" in not_started_body["images"][0]["filename"]

        readiness = await _readiness(http_client, cfg["uuid"])
        assert readiness["needsReviewCount"] == 0
        assert readiness["notStartedCount"] == 1
        assert readiness["readyCount"] == 0

    async def test_present_partial_image_still_needs_review(self, http_client, files_root):
        ds = await _create_dataset(http_client, "bk-ds-present-partial")
        cfg, parent_label, _child_label = await _make_two_label_tc(http_client, "bk-tc-present-partial", ds["uuid"])

        await _register_image(http_client, files_root, "bk_present_partial.png")
        await _verify_label(http_client, "bk_present_partial.png", parent_label, "present")

        partial_body = await _partial_review_images(http_client, cfg["uuid"])
        assert len(partial_body["images"]) == 1
        assert "bk_present_partial" in partial_body["images"][0]["filename"]

        readiness = await _readiness(http_client, cfg["uuid"])
        assert readiness["needsReviewCount"] == 1
        assert readiness["notStartedCount"] == 0


class TestReadyPositiveNegativeSplit:
    """'Ready' conflates images that carry an annotation with images verified as pure
    background. A mostly-background dataset therefore reads as far more training signal
    than it holds, so the ready count is split into positive/background sub-buckets."""

    async def test_counts_split_and_sum_to_ready(self, http_client, files_root):
        ds = await _create_dataset(http_client, "bk-ds-split")
        cfg, parent_label, child_label = await _make_two_label_tc(http_client, "bk-tc-split", ds["uuid"])

        # Positive: fully verified with one label actually present.
        await _register_image(http_client, files_root, "bk_split_positive.png")
        await _verify_label(http_client, "bk_split_positive.png", parent_label, "present")
        await _verify_label(http_client, "bk_split_positive.png", child_label, "absent")

        # Background: fully verified, every in-scope label absent.
        for i in range(3):
            name = f"bk_split_background_{i}.png"
            await _register_image(http_client, files_root, name)
            await _verify_label(http_client, name, parent_label, "absent")
            await _verify_label(http_client, name, child_label, "absent")

        readiness = await _readiness(http_client, cfg["uuid"])
        assert readiness["readyCount"] == 4
        assert readiness["readyPositiveCount"] == 1
        assert readiness["readyNegativeCount"] == 3
        assert readiness["readyPositiveCount"] + readiness["readyNegativeCount"] == readiness["readyCount"]

    async def test_buckets_are_disjoint_and_cover_ready(self, http_client, files_root):
        ds = await _create_dataset(http_client, "bk-ds-split-lists")
        cfg, parent_label, child_label = await _make_two_label_tc(http_client, "bk-tc-split-lists", ds["uuid"])

        await _register_image(http_client, files_root, "bk_lists_positive.png")
        await _verify_label(http_client, "bk_lists_positive.png", parent_label, "present")
        await _verify_label(http_client, "bk_lists_positive.png", child_label, "present")

        await _register_image(http_client, files_root, "bk_lists_background.png")
        await _verify_label(http_client, "bk_lists_background.png", parent_label, "absent")
        await _verify_label(http_client, "bk_lists_background.png", child_label, "absent")

        # Partially verified — ready in neither sub-bucket.
        await _register_image(http_client, files_root, "bk_lists_partial.png")
        await _verify_label(http_client, "bk_lists_partial.png", parent_label, "present")

        positive = await _post_bucket(http_client, cfg["uuid"], "ready-positive-images")
        negative = await _post_bucket(http_client, cfg["uuid"], "ready-negative-images")
        ready = await _post_bucket(http_client, cfg["uuid"], "ready-images")

        positive_uuids = {img["uuid"] for img in positive["images"]}
        negative_uuids = {img["uuid"] for img in negative["images"]}
        ready_uuids = {img["uuid"] for img in ready["images"]}

        assert len(positive["images"]) == 1
        assert "bk_lists_positive" in positive["images"][0]["filename"]
        assert len(negative["images"]) == 1
        assert "bk_lists_background" in negative["images"][0]["filename"]
        assert positive_uuids & negative_uuids == set()
        assert positive_uuids | negative_uuids == ready_uuids

    async def test_rollup_image_with_only_absent_label_counts_as_background(self, http_client, files_root):
        """Under rollup the threshold drops to 1, so a single ABSENT verification makes an
        image ready. It must land in background, not positive."""
        ds = await _create_dataset(http_client, "bk-ds-split-rollup")
        parent_label = await _create_label(http_client, "bk-split-rollup-parent")
        child_label = await _create_label(http_client, "bk-split-rollup-child", parent_uuid=parent_label)
        mc = await _create_model_config(
            http_client,
            "mc-bk-split-rollup",
            nodes=[
                {
                    "label_uuid": parent_label,
                    "model_type": "image_classifier",
                    "architecture": "resnet_18",
                    "roll_up_unchecked": True,
                },
            ],
        )
        cfg = await _create_training_config(
            http_client, "bk-tc-split-rollup", [ds["uuid"]], model_config_node_uuid=mc["nodes"][0]["uuid"]
        )

        await _register_image(http_client, files_root, "bk_rollup_background.png")
        await _verify_label(http_client, "bk_rollup_background.png", parent_label, "absent")

        await _register_image(http_client, files_root, "bk_rollup_positive.png")
        await _verify_label(http_client, "bk_rollup_positive.png", child_label, "present")

        readiness = await _readiness(http_client, cfg["uuid"])
        assert readiness["rollUpActive"] is True
        assert readiness["readyCount"] == 2
        assert readiness["readyPositiveCount"] == 1
        assert readiness["readyNegativeCount"] == 1

    async def test_pagination_on_background_bucket(self, http_client, files_root):
        ds = await _create_dataset(http_client, "bk-ds-split-page")
        cfg, parent_label, child_label = await _make_two_label_tc(http_client, "bk-tc-split-page", ds["uuid"])

        for i in range(5):
            name = f"bk_split_page_{i}.png"
            await _register_image(http_client, files_root, name)
            await _verify_label(http_client, name, parent_label, "absent")
            await _verify_label(http_client, name, child_label, "absent")

        all_uuids: list[str] = []
        after_uuid = None
        after_sort_value = None
        for _ in range(10):
            payload: dict = {"limit": 2}
            if after_uuid:
                payload["afterUuid"] = after_uuid
                payload["afterSortValue"] = after_sort_value
            body = await _post_bucket(http_client, cfg["uuid"], "ready-negative-images", **payload)
            page_images = body["images"]
            if not page_images:
                break
            all_uuids.extend(img["uuid"] for img in page_images)
            if not body["hasMore"]:
                break
            last = page_images[-1]
            after_uuid = last["uuid"]
            after_sort_value = last.get("createdAt")

        assert len(all_uuids) == 5
        assert len(set(all_uuids)) == 5
