"""Tests for the training config needs-review images endpoint and the D1 decoupled
verification_label_uuids feature in ImageSearchParams.

Coverage:
- Unverified images appear; fully-verified images are excluded.
- Union of two datasets is de-duped correctly.
- Training config with no model_config_node → empty result.
- Extra label_uuids in the payload narrows the pool.
- Keyset pagination (cursor) works across multiple pages.
- Readiness count == needs-review list length for the same config.
- D1: verification_label_uuids != label_uuids — verification check is applied to the
  decoupled target set, not the pool filter.
"""

import uuid as _uuid_mod

from tests.conftest import make_png

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


async def _create_label(http_client, name: str) -> str:
    r = await http_client.post("/api/labels", json={"name": name, "color": "#ff0000"})
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


async def _needs_review(http_client, config_uuid: str, **payload_overrides) -> dict:
    """POST to the needs-review-images endpoint and return the parsed body."""
    payload = {"limit": 50}
    payload.update(payload_overrides)
    r = await http_client.post(f"/api/training-configs/{config_uuid}/needs-review-images", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "success", body
    return body


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestNeedsReviewImages:
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

    async def test_unverified_image_appears(self, http_client, files_root):
        """An image without a verified target label appears in the list."""
        ds = await _create_dataset(http_client, "nr-ds-appears")
        label_uuid = await _create_label(http_client, "nr-label-appears")
        await _register_image(http_client, files_root, "nr_appears.png")
        cfg, _ = await self._make_tc(http_client, "nr-tc-appears", [ds["uuid"]], label_uuid)

        body = await _needs_review(http_client, cfg["uuid"])
        assert len(body["images"]) == 1

    async def test_verified_image_excluded(self, http_client, files_root):
        """An image with all target labels verified does NOT appear."""
        ds = await _create_dataset(http_client, "nr-ds-verified")
        label_uuid = await _create_label(http_client, "nr-label-verified")
        await _register_image(http_client, files_root, "nr_verified.png")
        await _verify_label(http_client, "nr_verified.png", label_uuid, "present")
        cfg, _ = await self._make_tc(http_client, "nr-tc-verified", [ds["uuid"]], label_uuid)

        body = await _needs_review(http_client, cfg["uuid"])
        assert body["images"] == []
        assert body["hasMore"] is False

    async def test_absent_status_treated_as_verified(self, http_client, files_root):
        """'absent' verification status counts as verified — image must NOT appear."""
        ds = await _create_dataset(http_client, "nr-ds-absent")
        label_uuid = await _create_label(http_client, "nr-label-absent")
        await _register_image(http_client, files_root, "nr_absent.png")
        await _verify_label(http_client, "nr_absent.png", label_uuid, "absent")
        cfg, _ = await self._make_tc(http_client, "nr-tc-absent", [ds["uuid"]], label_uuid)

        body = await _needs_review(http_client, cfg["uuid"])
        assert body["images"] == [], "absent-verified image should be excluded from needs-review"

    async def test_no_datasets_returns_empty(self, http_client, files_root):
        """Training config with a node but no linked datasets → empty list."""
        label_uuid = await _create_label(http_client, "nr-label-nods")
        mc = await _create_model_config(
            http_client,
            "mc-nr-nods",
            nodes=[
                {
                    "label_uuid": label_uuid,
                    "model_type": "image_classifier",
                    "architecture": "resnet_18",
                }
            ],
        )
        node_uuid = mc["nodes"][0]["uuid"]
        cfg = await _create_training_config(http_client, "nr-tc-nods", [], model_config_node_uuid=node_uuid)

        body = await _needs_review(http_client, cfg["uuid"])
        assert body["images"] == []

    async def test_mixed_images(self, http_client, files_root):
        """Only unverified images appear; verified ones are excluded."""
        ds = await _create_dataset(http_client, "nr-ds-mixed")
        label_uuid = await _create_label(http_client, "nr-label-mixed")

        await _register_image(http_client, files_root, "nr_mix_unverified.png")
        await _register_image(http_client, files_root, "nr_mix_verified.png")
        await _verify_label(http_client, "nr_mix_verified.png", label_uuid, "present")

        cfg, _ = await self._make_tc(http_client, "nr-tc-mixed", [ds["uuid"]], label_uuid)

        body = await _needs_review(http_client, cfg["uuid"])
        assert len(body["images"]) == 1
        filenames = [img["filename"] for img in body["images"]]
        assert any("nr_mix_unverified" in f for f in filenames)
        assert not any("nr_mix_verified" in f for f in filenames)

    async def test_no_node_returns_empty(self, http_client, files_root):
        """Training config without a model_config_node_uuid → empty list."""
        ds = await _create_dataset(http_client, "nr-ds-nonode")
        await _register_image(http_client, files_root, "nr_nonode.png")
        cfg = await _create_training_config(http_client, "nr-tc-nonode", [ds["uuid"]])

        body = await _needs_review(http_client, cfg["uuid"])
        assert body["images"] == []

    async def test_nonexistent_config_returns_empty(self, http_client):
        """Unknown config UUID → success with empty images (not a 404 in this convention)."""
        r = await http_client.post(
            "/api/training-configs/00000000-0000-0000-0000-000000000000/needs-review-images",
            json={},
        )
        body = r.json()
        assert body["status"] == "success"
        assert body["images"] == []

    async def test_union_of_two_datasets(self, http_client, files_root):
        """Images from both linked datasets appear in the merged list."""
        ds_a = await _create_dataset(http_client, "nr-union-a")
        ds_b = await _create_dataset(http_client, "nr-union-b")
        label_uuid = await _create_label(http_client, "nr-union-label")

        await _register_image(http_client, files_root, "nr_union_img_a.png")
        await _register_image(http_client, files_root, "nr_union_img_b.png")

        # Verify img_a — it should NOT appear; img_b unverified → should appear
        await _verify_label(http_client, "nr_union_img_a.png", label_uuid, "present")

        cfg, _ = await self._make_tc(http_client, "nr-tc-union", [ds_a["uuid"], ds_b["uuid"]], label_uuid)

        body = await _needs_review(http_client, cfg["uuid"])
        assert len(body["images"]) == 1
        assert "nr_union_img_b" in body["images"][0]["filename"]

    async def test_union_deduplication(self, http_client, files_root):
        """An image that belongs to both datasets appears only once."""
        ds_a = await _create_dataset(http_client, "nr-dedup-a")
        ds_b = await _create_dataset(http_client, "nr-dedup-b")
        label_uuid = await _create_label(http_client, "nr-dedup-label")

        # Single image, both datasets are open (no filter) so it belongs to both
        await _register_image(http_client, files_root, "nr_dedup_img.png")

        cfg, _ = await self._make_tc(http_client, "nr-tc-dedup", [ds_a["uuid"], ds_b["uuid"]], label_uuid)

        body = await _needs_review(http_client, cfg["uuid"])
        assert len(body["images"]) == 1  # de-duped, not 2

    async def test_readiness_count_matches_list_length(self, http_client, files_root):
        """The bbox-scoped needs-review-images list (has_unverified) covers exactly the images
        that are NOT ready — i.e. the union of the readiness endpoint's needsReviewCount
        (partial) and notStartedCount (untouched) buckets. needs-review-images itself was not
        changed by the readiness ready/needs-review/not-started split — it answers a different,
        bbox-verification-status question.
        """
        ds = await _create_dataset(http_client, "nr-match-ds")
        label_uuid = await _create_label(http_client, "nr-match-label")

        await _register_image(http_client, files_root, "nr_match_img1.png")
        await _register_image(http_client, files_root, "nr_match_img2.png")
        # Verify one of them
        await _verify_label(http_client, "nr_match_img2.png", label_uuid, "absent")

        cfg, _ = await self._make_tc(http_client, "nr-tc-match", [ds["uuid"]], label_uuid)

        readiness_r = await http_client.get(f"/api/training-configs/{cfg['uuid']}/readiness")
        readiness = readiness_r.json()
        assert readiness["status"] == "success"

        needs_review_body = await _needs_review(http_client, cfg["uuid"])
        assert len(needs_review_body["images"]) == readiness["needsReviewCount"] + readiness["notStartedCount"]

    async def test_pagination(self, http_client, files_root):
        """Keyset cursor correctly pages through all needs-review images."""
        ds = await _create_dataset(http_client, "nr-page-ds")
        label_uuid = await _create_label(http_client, "nr-page-label")

        image_names = [f"nr_page_img_{i}.png" for i in range(5)]
        for name in image_names:
            await _register_image(http_client, files_root, name)

        cfg, _ = await self._make_tc(http_client, "nr-tc-page", [ds["uuid"]], label_uuid)

        # Collect all 5 images by paging with limit=2
        all_uuids: list[str] = []
        after_uuid = None
        after_sort_value = None

        for _ in range(10):  # safety limit
            payload: dict = {"limit": 2}
            if after_uuid:
                payload["afterUuid"] = after_uuid
                payload["afterSortValue"] = after_sort_value
            body = await _needs_review(http_client, cfg["uuid"], **payload)
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

    async def test_extra_label_uuids_narrows_pool(self, http_client, files_root):
        """payload.labelUuids further narrows the pool to images with those label bboxes."""
        import uuid as _uuid

        from apps.helpers.db.types import UNVERIFIED_TAG_UUID

        ds = await _create_dataset(http_client, "nr-extra-ds")
        target_label_uuid = await _create_label(http_client, "nr-extra-target")
        pool_label_uuid = await _create_label(http_client, "nr-extra-pool")

        await _register_image(http_client, files_root, "nr_extra_with_pool.png")
        await _register_image(http_client, files_root, "nr_extra_without_pool.png")

        # Add a bbox with the pool label on the first image
        bbox_id = str(_uuid.uuid4())
        r = await http_client.post(
            "/api/images/metadata/update",
            json={
                "imagePath": "nr_extra_with_pool.png",
                "boxes": [
                    {
                        "uuid": bbox_id,
                        "labelUuid": pool_label_uuid,
                        "x": 0.1,
                        "y": 0.1,
                        "width": 0.3,
                        "height": 0.3,
                        "tagUuids": [UNVERIFIED_TAG_UUID],
                    }
                ],
            },
        )
        assert r.status_code == 200, r.text

        cfg, _ = await self._make_tc(http_client, "nr-tc-extra", [ds["uuid"]], target_label_uuid)

        # Without extra filter: both images appear (both have unverified target label)
        body_all = await _needs_review(http_client, cfg["uuid"])
        assert len(body_all["images"]) == 2

        # With extra filter: only the one with a pool_label bbox appears
        body_filtered = await _needs_review(http_client, cfg["uuid"], labelUuids=[pool_label_uuid])
        assert len(body_filtered["images"]) == 1
        assert "nr_extra_with_pool" in body_filtered["images"][0]["filename"]

    async def test_extra_label_uuids_merges_into_dataset_pool_term_not_unrelated_term(self, http_client, files_root):
        """extra_label_uuids must widen the dataset's own label_uuids-derived pool
        ObjectLabelTerm, not some unrelated non-negated objectLabel term the user
        already put in filter_json.terms — merging into the wrong term would let
        images through that only satisfy the unrelated term's widened set while
        failing the pool label filter it was supposed to widen."""
        target_label_uuid = await _create_label(http_client, "nr-merge-target")
        pool_label_uuid = await _create_label(http_client, "nr-merge-pool")
        decoy_label_uuid = await _create_label(http_client, "nr-merge-decoy")
        extra_label_uuid = await _create_label(http_client, "nr-merge-extra")

        r = await http_client.post(
            "/api/datasets",
            json={
                "name": "nr-merge-ds",
                "filter_json": {"terms": [{"kind": "objectLabel", "labelUuids": [decoy_label_uuid]}]},
                "label_uuids": [pool_label_uuid],
            },
        )
        assert r.status_code == 200, r.text
        ds = r.json()["dataset"]

        # Has bboxes for the pool label and the extra label, but NOT the decoy
        # label — the decoy term must stay strict (unwidened by extra_label_uuids)
        # and exclude this image even though the pool term alone is satisfied.
        await _register_image(http_client, files_root, "nr_merge_pool_only.png")
        r = await http_client.post(
            "/api/images/metadata/update",
            json={
                "imagePath": "nr_merge_pool_only.png",
                "boxes": [
                    {
                        "uuid": str(_uuid_mod.uuid4()),
                        "labelUuid": pool_label_uuid,
                        "x": 0.1,
                        "y": 0.1,
                        "width": 0.3,
                        "height": 0.3,
                        "tagUuids": [],
                    },
                    {
                        "uuid": str(_uuid_mod.uuid4()),
                        "labelUuid": extra_label_uuid,
                        "x": 0.5,
                        "y": 0.5,
                        "width": 0.2,
                        "height": 0.2,
                        "tagUuids": [],
                    },
                ],
            },
        )
        assert r.status_code == 200, r.text

        cfg, _ = await self._make_tc(http_client, "nr-tc-merge", [ds["uuid"]], target_label_uuid)

        body = await _needs_review(http_client, cfg["uuid"], labelUuids=[extra_label_uuid])
        assert len(body["images"]) == 0


class TestVerificationLabelUuidsDecoupled:
    """D1: verification_label_uuids in the review-queue endpoint operates independently of label_uuids.

    We test this via POST /api/images/review (which directly accepts ImageSearchParams fields)
    to ensure the decoupled predicate is wired end-to-end.
    """

    async def test_verification_targets_differ_from_pool(self, http_client, files_root):
        """When verificationLabelUuids is set, has_unverified checks those labels, not label_uuids."""
        verify_label_uuid = await _create_label(http_client, "vlu-verify")
        pool_label_uuid = await _create_label(http_client, "vlu-pool")

        await _register_image(http_client, files_root, "vlu_img_a.png")
        await _register_image(http_client, files_root, "vlu_img_b.png")

        # Verify verify_label on img_a only; img_b remains unverified.
        await _verify_label(http_client, "vlu_img_a.png", verify_label_uuid, "present")

        # POST /api/images/review with verificationLabelUuids decoupled from label_uuids.
        # No pool label filter (label_uuids=[]), so both images are candidates.
        # Verification check is against verify_label_uuid only.
        r = await http_client.post(
            "/api/images/review",
            json={
                "terms": [{"kind": "labelVerification", "check": "has_unverified", "labelUuids": [verify_label_uuid]}],
                "limit": 50,
            },
        )
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "success"
        filenames = [img["filename"] for img in body["images"]]

        # Only img_b (unverified for verify_label) should appear.
        assert any("vlu_img_b" in f for f in filenames), f"img_b missing: {filenames}"
        assert not any("vlu_img_a" in f for f in filenames), f"img_a should be excluded: {filenames}"

    async def test_pool_label_filter_and_verify_target_independent(self, http_client, files_root):
        """Pool filter (labelUuids) narrows candidates; verification check is against verificationLabelUuids."""
        import uuid as _uuid

        from apps.helpers.db.types import UNVERIFIED_TAG_UUID

        verify_label_uuid = await _create_label(http_client, "vlu2-verify")
        pool_label_uuid = await _create_label(http_client, "vlu2-pool")

        await _register_image(http_client, files_root, "vlu2_pool.png")
        await _register_image(http_client, files_root, "vlu2_nopool.png")

        # Add a bbox with pool_label on vlu2_pool.png so it passes the label_uuids pool filter.
        bbox_id = str(_uuid.uuid4())
        r = await http_client.post(
            "/api/images/metadata/update",
            json={
                "imagePath": "vlu2_pool.png",
                "boxes": [
                    {
                        "uuid": bbox_id,
                        "labelUuid": pool_label_uuid,
                        "x": 0.1,
                        "y": 0.1,
                        "width": 0.3,
                        "height": 0.3,
                        "tagUuids": [UNVERIFIED_TAG_UUID],
                    }
                ],
            },
        )
        assert r.status_code == 200, r.text

        # Neither image has a verified verify_label row → both are unverified for verify_label.
        # But pool filter restricts to images with pool_label bbox → only vlu2_pool.png.
        r = await http_client.post(
            "/api/images/review",
            json={
                "terms": [
                    {"kind": "objectLabel", "labelUuids": [pool_label_uuid]},
                    {"kind": "labelVerification", "check": "has_unverified", "labelUuids": [verify_label_uuid]},
                ],
                "limit": 50,
            },
        )
        assert r.status_code == 200
        body = r.json()
        filenames = [img["filename"] for img in body["images"]]

        # vlu2_pool.png passes pool filter AND is unverified for verify_label → appears.
        assert any("vlu2_pool" in f for f in filenames), f"pool img missing: {filenames}"
        # vlu2_nopool.png excluded by pool filter.
        assert not any("vlu2_nopool" in f for f in filenames), f"nopool img should be excluded: {filenames}"
