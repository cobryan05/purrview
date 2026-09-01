"""Tests for persisted "Run Model on Results" preview runs and candidates.

Covers CRUD on preview_runs, add_preview_candidates conflict resolution,
query_preview_candidates filtering/sorting/pagination, bulk candidate PATCH,
staleness reconciliation, apply_preview_run, and the TTL sweep.

See PREVIEW_REMODEL.md §4, §6, §8, §9.
"""

from datetime import datetime, timedelta, timezone

import pytest

from apps.helpers.db.types import (
    UNVERIFIED_TAG_UUID,
    VERIFIED_TAG_UUID,
    PreviewCandidateCreate,
    PreviewRunTTL,
)
from tests.conftest import make_png

# ─────────────────────────────────────────────────────────────────────────────
# Fixtures / helpers
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def image_file(files_root):
    img = files_root / "preview_test.png"
    img.write_bytes(make_png())
    return img


@pytest.fixture
async def label_uuid(http_client):
    r = await http_client.post("/api/labels", json={"name": "preview_label_a", "color": "#abcdef"})
    return r.json()["label"]["uuid"]


@pytest.fixture
async def other_label_uuid(http_client):
    r = await http_client.post("/api/labels", json={"name": "preview_label_b", "color": "#123456"})
    return r.json()["label"]["uuid"]


async def _create_bbox(http_client, label_uuid, x=0.1, y=0.1, width=0.2, height=0.2, tag_uuids=None):
    """Insert one bbox on preview_test.png without disturbing any siblings.

    Uses the single-bbox insert endpoint (POST /api/images/bbox), not
    /api/images/metadata/update — the latter *replaces* the whole bbox list
    for the image, so two sequential calls would silently drop the first box.
    """
    r = await http_client.post(
        "/api/images/bbox",
        json={
            "imagePath": "preview_test.png",
            "labelUuid": label_uuid,
            "x": x,
            "y": y,
            "width": width,
            "height": height,
            "tagUuids": tag_uuids if tag_uuids is not None else [UNVERIFIED_TAG_UUID],
        },
    )
    assert r.status_code == 200, r.text
    return r.json()["bboxUuid"]


async def _get_image_uuid(http_client, image_path="preview_test.png"):
    r = await http_client.post("/api/images/metadata/get", json={"imagePath": image_path})
    assert r.status_code == 200, r.text
    return r.json()["image"]["uuid"]


async def _create_run(http_client, name=None, model_name="yolov8", task_typename="PreviewModelRelabelTask"):
    payload = {"modelName": model_name, "taskTypename": task_typename}
    if name is not None:
        payload["name"] = name
    r = await http_client.post("/api/preview-runs", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "success", body
    return body["run"]


async def _add_candidates(http_client, run_uuid, candidates, conflict_policy=None):
    payload = {"candidates": candidates}
    if conflict_policy is not None:
        payload["conflictPolicy"] = conflict_policy
    r = await http_client.post(f"/api/preview-runs/{run_uuid}/candidates", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "success", body
    return body["updated"]


def _candidate(image_uuid, label_uuid, bbox_uuid=None, is_new=False, confidence=0.5, source_model="model_a"):
    return {
        "imageUuid": image_uuid,
        "bboxUuid": bbox_uuid,
        "isNew": is_new,
        "proposedLabelUuid": label_uuid,
        "confidence": confidence,
        "sourceModel": source_model,
        "geometry": {"x": 0.2, "y": 0.2, "width": 0.1, "height": 0.1} if is_new else None,
    }


async def _query_candidates(http_client, run_uuid, **query):
    r = await http_client.post(f"/api/preview-runs/{run_uuid}/candidates/query", json=query)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "success", body
    return body["candidates"], body["totalCount"]


# ─────────────────────────────────────────────────────────────────────────────
# 1. Run CRUD
# ─────────────────────────────────────────────────────────────────────────────


class TestPreviewRunCrud:
    async def test_create_with_explicit_name(self, http_client):
        run = await _create_run(http_client, name="MyRun", model_name="yolov8")
        assert run["name"] == "MyRun"
        assert run["modelName"] == "yolov8"
        assert run["pinned"] is False
        assert run["uuid"]

    async def test_create_without_name_derives_from_model_and_timestamp(self, http_client):
        run = await _create_run(http_client, name=None, model_name="water_valves")
        assert run["name"].startswith("water_valves")
        assert run["name"] != "water_valves"  # a timestamp must follow

    async def test_list_includes_created_run(self, http_client):
        run = await _create_run(http_client, name="ListedRun")
        r = await http_client.get("/api/preview-runs")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "success"
        uuids = [x["uuid"] for x in body["runs"]]
        assert run["uuid"] in uuids

    async def test_get_single(self, http_client):
        run = await _create_run(http_client, name="GetRun")
        r = await http_client.get(f"/api/preview-runs/{run['uuid']}")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "success"
        assert body["run"]["uuid"] == run["uuid"]
        assert body["run"]["name"] == "GetRun"

    async def test_get_missing_returns_failure(self, http_client):
        r = await http_client.get("/api/preview-runs/no-such-uuid")
        assert r.status_code == 200
        assert r.json()["status"] == "failure"

    async def test_patch_rename_and_pin(self, http_client):
        run = await _create_run(http_client, name="OldName")
        r = await http_client.patch(
            f"/api/preview-runs/{run['uuid']}",
            json={"name": "NewName", "pinned": True},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "success"
        assert body["run"]["name"] == "NewName"
        assert body["run"]["pinned"] is True

    async def test_patch_missing_returns_failure(self, http_client):
        r = await http_client.patch("/api/preview-runs/no-such-uuid", json={"name": "X"})
        assert r.status_code == 200
        assert r.json()["status"] == "failure"

    async def test_delete_removes_run(self, http_client):
        run = await _create_run(http_client, name="ToDelete")
        r = await http_client.delete(f"/api/preview-runs/{run['uuid']}")
        assert r.status_code == 200
        assert r.json()["status"] == "success"

        r2 = await http_client.get(f"/api/preview-runs/{run['uuid']}")
        assert r2.json()["status"] == "failure"

    async def test_delete_missing_returns_failure(self, http_client):
        r = await http_client.delete("/api/preview-runs/no-such-uuid")
        assert r.status_code == 200
        assert r.json()["status"] == "failure"

    async def test_delete_cascades_candidates(self, http_client, image_file, label_uuid):
        run = await _create_run(http_client)
        image_uuid = await _get_image_uuid(http_client)
        await _add_candidates(http_client, run["uuid"], [_candidate(image_uuid, label_uuid)])

        candidates, total = await _query_candidates(http_client, run["uuid"])
        assert total == 1

        r = await http_client.delete(f"/api/preview-runs/{run['uuid']}")
        assert r.json()["status"] == "success"

        # Candidates table row must be gone too — query against the (now
        # nonexistent) run returns nothing rather than erroring, so assert via
        # the db_client-level count instead of the (deleted) run's endpoint.
        r2 = await http_client.get(f"/api/preview-runs/{run['uuid']}")
        assert r2.json()["status"] == "failure"


class TestPreviewRunCrudDbClient:
    async def test_delete_cascade_removes_candidate_rows(self, db_client, image_file, label_uuid):
        from apps.helpers.db.types import PreviewCandidateQuery

        run = await db_client.create_preview_run(name="UnitRun", model_name="m", task_typename="t")
        image_uuid = (await db_client.add_image(image_file)).uuid
        await db_client.add_preview_candidates(
            run.uuid,
            [PreviewCandidateCreate(image_uuid=image_uuid, proposed_label_uuid=label_uuid, confidence=0.5)],
        )
        candidates, total = await db_client.query_preview_candidates(run.uuid, PreviewCandidateQuery())
        assert total == 1

        n = await db_client.delete_preview_runs(run.uuid)
        assert n == 1

        candidates_after, total_after = await db_client.query_preview_candidates(run.uuid, PreviewCandidateQuery())
        assert total_after == 0
        assert candidates_after == []


# ─────────────────────────────────────────────────────────────────────────────
# 2. add_preview_candidates conflict policy
# ─────────────────────────────────────────────────────────────────────────────


class TestConflictPolicy:
    async def test_last_policy_overwrites_prior_proposal(self, http_client, image_file, label_uuid, other_label_uuid):
        run = await _create_run(http_client)
        image_uuid = await _get_image_uuid(http_client)
        bbox_uuid = await _create_bbox(http_client, label_uuid)

        first = _candidate(image_uuid, label_uuid, bbox_uuid=bbox_uuid, confidence=0.4, source_model="model_a")
        second = _candidate(image_uuid, other_label_uuid, bbox_uuid=bbox_uuid, confidence=0.9, source_model="model_b")

        written = await _add_candidates(http_client, run["uuid"], [first], conflict_policy="last")
        assert written == 1
        written2 = await _add_candidates(http_client, run["uuid"], [second], conflict_policy="last")
        assert written2 == 1  # it's an overwrite, still counted as written

        candidates, total = await _query_candidates(http_client, run["uuid"])
        assert total == 1  # one bbox → one row, not two
        assert candidates[0]["proposedLabelUuid"] == other_label_uuid
        assert candidates[0]["confidence"] == pytest.approx(0.9)
        assert candidates[0]["sourceModel"] == "model_b"

    async def test_first_policy_keeps_original_proposal(self, http_client, image_file, label_uuid, other_label_uuid):
        run = await _create_run(http_client)
        image_uuid = await _get_image_uuid(http_client)
        bbox_uuid = await _create_bbox(http_client, label_uuid)

        first = _candidate(image_uuid, label_uuid, bbox_uuid=bbox_uuid, confidence=0.4, source_model="model_a")
        second = _candidate(image_uuid, other_label_uuid, bbox_uuid=bbox_uuid, confidence=0.9, source_model="model_b")

        await _add_candidates(http_client, run["uuid"], [first], conflict_policy="first")
        await _add_candidates(http_client, run["uuid"], [second], conflict_policy="first")

        candidates, total = await _query_candidates(http_client, run["uuid"])
        assert total == 1
        assert candidates[0]["proposedLabelUuid"] == label_uuid
        assert candidates[0]["confidence"] == pytest.approx(0.4)
        assert candidates[0]["sourceModel"] == "model_a"

    async def test_new_candidates_sharing_an_image_are_not_conflicts(
        self, http_client, image_file, label_uuid, other_label_uuid
    ):
        """Two is_new candidates (bbox_uuid=None) on the same image are two
        separate proposed detections, not a conflict — even under `last`."""
        run = await _create_run(http_client)
        image_uuid = await _get_image_uuid(http_client)

        a = _candidate(image_uuid, label_uuid, bbox_uuid=None, is_new=True, confidence=0.3, source_model="model_a")
        b = _candidate(
            image_uuid, other_label_uuid, bbox_uuid=None, is_new=True, confidence=0.6, source_model="model_b"
        )
        written = await _add_candidates(http_client, run["uuid"], [a, b], conflict_policy="last")
        assert written == 2

        candidates, total = await _query_candidates(http_client, run["uuid"])
        assert total == 2


# ─────────────────────────────────────────────────────────────────────────────
# 3. query_preview_candidates — filters, total_count, pagination, sorting
# ─────────────────────────────────────────────────────────────────────────────


class TestQueryPreviewCandidates:
    async def _seed_run(self, http_client, label_uuid, other_label_uuid):
        run = await _create_run(http_client)
        image_uuid = await _get_image_uuid(http_client)
        candidates = [
            _candidate(image_uuid, label_uuid, confidence=0.9, source_model="model_a", is_new=True),
            _candidate(image_uuid, other_label_uuid, confidence=0.2, source_model="model_b", is_new=True),
            _candidate(image_uuid, label_uuid, confidence=0.5, source_model="model_a", is_new=True),
        ]
        await _add_candidates(http_client, run["uuid"], candidates)
        return run

    async def test_filter_by_proposed_label(self, http_client, image_file, label_uuid, other_label_uuid):
        run = await self._seed_run(http_client, label_uuid, other_label_uuid)
        candidates, total = await _query_candidates(http_client, run["uuid"], proposedLabelUuids=[other_label_uuid])
        assert total == 1
        assert candidates[0]["proposedLabelUuid"] == other_label_uuid

    async def test_filter_by_confidence_range(self, http_client, image_file, label_uuid, other_label_uuid):
        run = await self._seed_run(http_client, label_uuid, other_label_uuid)
        candidates, total = await _query_candidates(http_client, run["uuid"], confidenceMin=0.4, confidenceMax=0.95)
        assert total == 2
        assert all(0.4 <= c["confidence"] <= 0.95 for c in candidates)

    async def test_filter_by_source_models(self, http_client, image_file, label_uuid, other_label_uuid):
        run = await self._seed_run(http_client, label_uuid, other_label_uuid)
        candidates, total = await _query_candidates(http_client, run["uuid"], sourceModels=["model_b"])
        assert total == 1
        assert candidates[0]["sourceModel"] == "model_b"

    async def test_filter_by_is_new(self, http_client, image_file, label_uuid, other_label_uuid):
        run = await _create_run(http_client)
        image_uuid = await _get_image_uuid(http_client)
        bbox_uuid = await _create_bbox(http_client, label_uuid)
        existing = _candidate(image_uuid, label_uuid, bbox_uuid=bbox_uuid, confidence=0.5)
        new = _candidate(image_uuid, label_uuid, bbox_uuid=None, is_new=True, confidence=0.6)
        await _add_candidates(http_client, run["uuid"], [existing, new])

        candidates, total = await _query_candidates(http_client, run["uuid"], isNew=True)
        assert total == 1
        assert candidates[0]["isNew"] is True

        candidates2, total2 = await _query_candidates(http_client, run["uuid"], isNew=False)
        assert total2 == 1
        assert candidates2[0]["isNew"] is False

    async def test_filter_by_decisions(self, http_client, image_file, label_uuid, other_label_uuid):
        run = await self._seed_run(http_client, label_uuid, other_label_uuid)
        candidates, _ = await _query_candidates(http_client, run["uuid"])
        target = candidates[0]["uuid"]
        r = await http_client.patch(
            f"/api/preview-runs/{run['uuid']}/candidates",
            json={"candidateUuids": [target], "decision": "accepted"},
        )
        assert r.json()["status"] == "success"

        accepted, total = await _query_candidates(http_client, run["uuid"], decisions=["accepted"])
        assert total == 1
        assert accepted[0]["uuid"] == target

        pending, total_pending = await _query_candidates(http_client, run["uuid"], decisions=["pending"])
        assert total_pending == 2

    async def test_excluding_a_label_keeps_everything_else(self, http_client, image_file, label_uuid, other_label_uuid):
        """The "≠" half of the label filter — how a twelve-thousand-candidate run is
        triaged by taking classes away rather than naming the ones you want."""
        run = await self._seed_run(http_client, label_uuid, other_label_uuid)

        candidates, total = await _query_candidates(http_client, run["uuid"], excludeProposedLabelUuids=[label_uuid])

        assert total == 1
        assert candidates[0]["proposedLabelUuid"] == other_label_uuid

    async def test_include_and_exclude_apply_together(self, http_client, image_file, label_uuid, other_label_uuid):
        """ "Every cat except Willow" is one filter, not two passes — which is why the two
        lists travel together rather than being a mode. Excluding everything that was
        included is the case that proves they are both being applied: the answer is
        empty, not "the include won"."""
        run = await self._seed_run(http_client, label_uuid, other_label_uuid)

        _, total = await _query_candidates(
            http_client,
            run["uuid"],
            proposedLabelUuids=[label_uuid, other_label_uuid],
            excludeProposedLabelUuids=[label_uuid],
        )
        assert total == 1

        _, none_left = await _query_candidates(
            http_client,
            run["uuid"],
            proposedLabelUuids=[label_uuid],
            excludeProposedLabelUuids=[label_uuid],
        )
        assert none_left == 0

    async def test_excluding_a_decision_leaves_the_undecided(
        self, http_client, image_file, label_uuid, other_label_uuid
    ):
        """The filter someone reaches for after an hour of triage: "show me everything I
        have not already dealt with"."""
        run = await self._seed_run(http_client, label_uuid, other_label_uuid)
        candidates, _ = await _query_candidates(http_client, run["uuid"])
        target = candidates[0]["uuid"]
        r = await http_client.patch(
            f"/api/preview-runs/{run['uuid']}/candidates",
            json={"candidateUuids": [target], "decision": "accepted"},
        )
        assert r.json()["status"] == "success"

        rest, total = await _query_candidates(http_client, run["uuid"], excludeDecisions=["accepted"])

        assert total == 2
        assert target not in {c["uuid"] for c in rest}

    async def test_total_count_reflects_filtered_not_run_total(
        self, http_client, image_file, label_uuid, other_label_uuid
    ):
        run = await self._seed_run(http_client, label_uuid, other_label_uuid)
        _, total_all = await _query_candidates(http_client, run["uuid"])
        assert total_all == 3
        _, total_filtered = await _query_candidates(http_client, run["uuid"], proposedLabelUuids=[label_uuid])
        assert total_filtered == 2

    async def test_pagination_limit_offset(self, http_client, image_file, label_uuid):
        run = await _create_run(http_client)
        image_uuid = await _get_image_uuid(http_client)
        candidates = [
            _candidate(image_uuid, label_uuid, confidence=c / 10, is_new=True, source_model="m") for c in range(10)
        ]
        await _add_candidates(http_client, run["uuid"], candidates)

        page1, total = await _query_candidates(
            http_client, run["uuid"], sortBy="confidence", sortOrder="asc", limit=4, offset=0
        )
        assert total == 10
        assert len(page1) == 4
        page2, _ = await _query_candidates(
            http_client, run["uuid"], sortBy="confidence", sortOrder="asc", limit=4, offset=4
        )
        assert len(page2) == 4
        page3, _ = await _query_candidates(
            http_client, run["uuid"], sortBy="confidence", sortOrder="asc", limit=4, offset=8
        )
        assert len(page3) == 2

        all_uuids = [c["uuid"] for c in page1 + page2 + page3]
        assert len(set(all_uuids)) == 10  # no overlap/dupes across pages

        confidences = [c["confidence"] for c in page1 + page2 + page3]
        assert confidences == sorted(confidences)

    async def test_sort_by_confidence(self, http_client, image_file, label_uuid):
        run = await _create_run(http_client)
        image_uuid = await _get_image_uuid(http_client)
        candidates = [
            _candidate(image_uuid, label_uuid, confidence=0.1, is_new=True),
            _candidate(image_uuid, label_uuid, confidence=0.9, is_new=True),
            _candidate(image_uuid, label_uuid, confidence=0.5, is_new=True),
        ]
        await _add_candidates(http_client, run["uuid"], candidates)

        desc, _ = await _query_candidates(http_client, run["uuid"], sortBy="confidence", sortOrder="desc")
        assert [c["confidence"] for c in desc] == [0.9, 0.5, 0.1]

        asc, _ = await _query_candidates(http_client, run["uuid"], sortBy="confidence", sortOrder="asc")
        assert [c["confidence"] for c in asc] == [0.1, 0.5, 0.9]

    async def test_sort_by_created_at(self, http_client, image_file, label_uuid):
        run = await _create_run(http_client)
        image_uuid = await _get_image_uuid(http_client)
        # Written in two separate calls, so created_at ordering is meaningful.
        first = await _add_candidates(
            http_client, run["uuid"], [_candidate(image_uuid, label_uuid, confidence=0.5, is_new=True)]
        )
        second = await _add_candidates(
            http_client, run["uuid"], [_candidate(image_uuid, label_uuid, confidence=0.5, is_new=True)]
        )
        assert first == 1 and second == 1

        asc, _ = await _query_candidates(http_client, run["uuid"], sortBy="created_at", sortOrder="asc")
        desc, _ = await _query_candidates(http_client, run["uuid"], sortBy="created_at", sortOrder="desc")
        assert [c["uuid"] for c in asc] == list(reversed([c["uuid"] for c in desc]))

    async def test_sort_image_confidence_groups_by_image_ranked_by_best(self, http_client, files_root, label_uuid):
        """image_confidence groups every candidate from one image contiguously,
        images ordered by their own best candidate."""
        (files_root / "img_a.png").write_bytes(make_png())
        (files_root / "img_b.png").write_bytes(make_png())

        run = await _create_run(http_client)
        image_a = await _get_image_uuid(http_client, "img_a.png")
        image_b = await _get_image_uuid(http_client, "img_b.png")

        # Image A's best is 0.4 (lower); image B's best is 0.95 (higher).
        candidates = [
            _candidate(image_a, label_uuid, confidence=0.4, is_new=True),
            _candidate(image_a, label_uuid, confidence=0.1, is_new=True),
            _candidate(image_b, label_uuid, confidence=0.95, is_new=True),
            _candidate(image_b, label_uuid, confidence=0.3, is_new=True),
        ]
        await _add_candidates(http_client, run["uuid"], candidates)

        results, total = await _query_candidates(http_client, run["uuid"], sortBy="image_confidence", sortOrder="desc")
        assert total == 4

        # Contiguity: every candidate belonging to one image is adjacent.
        seen_images = []
        for c in results:
            if not seen_images or seen_images[-1] != c["imageUuid"]:
                seen_images.append(c["imageUuid"])
        assert len(seen_images) == 2  # each image appears in exactly one contiguous block

        # Ranking: image B (best 0.95) comes before image A (best 0.4) under desc.
        assert seen_images[0] == image_b
        assert seen_images[1] == image_a

    async def test_sort_label_confidence_groups_by_proposed_label(
        self, http_client, files_root, label_uuid, other_label_uuid
    ):
        """label_confidence keeps every candidate of one proposed label
        contiguous, labels ordered by their own best candidate — the "triage a
        whole class at once" grouping in the preview panel."""
        (files_root / "lbl_a.png").write_bytes(make_png())
        (files_root / "lbl_b.png").write_bytes(make_png())

        run = await _create_run(http_client)
        image_a = await _get_image_uuid(http_client, "lbl_a.png")
        image_b = await _get_image_uuid(http_client, "lbl_b.png")

        # Interleaved across images, so only the label grouping can make each
        # label's candidates adjacent. Label A's best is 0.4, label B's is 0.95.
        candidates = [
            _candidate(image_a, label_uuid, confidence=0.4, is_new=True),
            _candidate(image_b, other_label_uuid, confidence=0.95, is_new=True),
            _candidate(image_b, label_uuid, confidence=0.1, is_new=True),
            _candidate(image_a, other_label_uuid, confidence=0.3, is_new=True),
        ]
        await _add_candidates(http_client, run["uuid"], candidates)

        results, total = await _query_candidates(http_client, run["uuid"], sortBy="label_confidence", sortOrder="desc")
        assert total == 4

        seen_labels = []
        for c in results:
            if not seen_labels or seen_labels[-1] != c["proposedLabelUuid"]:
                seen_labels.append(c["proposedLabelUuid"])
        assert len(seen_labels) == 2, seen_labels  # one contiguous block per label

        # Ranking: label B (best 0.95) before label A (best 0.4) under desc.
        assert seen_labels[0] == other_label_uuid
        assert seen_labels[1] == label_uuid

    async def test_sort_label_confidence_ranks_within_the_filter(
        self, http_client, files_root, label_uuid, other_label_uuid
    ):
        """A label's rank comes from its best candidate *among the filtered
        rows* — filtering away the top candidate must re-rank its label,
        otherwise a page opens on a group whose lead candidate isn't shown."""
        (files_root / "lbl_c.png").write_bytes(make_png())
        run = await _create_run(http_client)
        image_uuid = await _get_image_uuid(http_client, "lbl_c.png")

        await _add_candidates(
            http_client,
            run["uuid"],
            [
                # Label A: 0.95 (filtered out below) and 0.2.
                _candidate(image_uuid, label_uuid, confidence=0.95, is_new=True),
                _candidate(image_uuid, label_uuid, confidence=0.2, is_new=True),
                # Label B: 0.5, which wins once A's 0.95 is filtered away.
                _candidate(image_uuid, other_label_uuid, confidence=0.5, is_new=True),
            ],
        )

        results, total = await _query_candidates(
            http_client, run["uuid"], sortBy="label_confidence", sortOrder="desc", confidenceMax=0.6
        )
        assert total == 2
        assert results[0]["proposedLabelUuid"] == other_label_uuid
        assert results[1]["proposedLabelUuid"] == label_uuid


# ─────────────────────────────────────────────────────────────────────────────
# 4. Bulk PATCH of candidates
# ─────────────────────────────────────────────────────────────────────────────


class TestBulkPatchCandidates:
    async def test_bulk_patch_sets_decision_action_label_geometry(
        self, http_client, image_file, label_uuid, other_label_uuid
    ):
        run = await _create_run(http_client)
        image_uuid = await _get_image_uuid(http_client)
        bbox_uuid = await _create_bbox(http_client, label_uuid)
        await _add_candidates(http_client, run["uuid"], [_candidate(image_uuid, label_uuid, bbox_uuid=bbox_uuid)])

        candidates, _ = await _query_candidates(http_client, run["uuid"])
        target = candidates[0]["uuid"]

        r = await http_client.patch(
            f"/api/preview-runs/{run['uuid']}/candidates",
            json={
                "candidateUuids": [target],
                "decision": "accepted",
                "decisionAction": "relabel",
                "proposedLabelUuid": other_label_uuid,
                "geometry": {"x": 0.3, "y": 0.3, "width": 0.15, "height": 0.15},
            },
        )
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "success"
        assert body["updated"] == 1

        updated, _ = await _query_candidates(http_client, run["uuid"])
        row = updated[0]
        assert row["decision"] == "accepted"
        assert row["decisionAction"] == "relabel"
        assert row["proposedLabelUuid"] == other_label_uuid
        assert row["geometry"] == {"x": 0.3, "y": 0.3, "width": 0.15, "height": 0.15}

    async def test_bulk_patch_is_scoped_by_run(self, http_client, image_file, label_uuid):
        run_a = await _create_run(http_client, name="RunA")
        run_b = await _create_run(http_client, name="RunB")
        image_uuid = await _get_image_uuid(http_client)

        await _add_candidates(http_client, run_a["uuid"], [_candidate(image_uuid, label_uuid, is_new=True)])
        candidates_a, _ = await _query_candidates(http_client, run_a["uuid"])
        candidate_a_uuid = candidates_a[0]["uuid"]

        # Attempt to patch run A's candidate uuid through run B's endpoint.
        r = await http_client.patch(
            f"/api/preview-runs/{run_b['uuid']}/candidates",
            json={"candidateUuids": [candidate_a_uuid], "decision": "accepted"},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "success"
        assert body["updated"] == 0  # scoped out — nothing in run B matched

        # Candidate in run A is untouched.
        candidates_a_after, _ = await _query_candidates(http_client, run_a["uuid"])
        assert candidates_a_after[0]["decision"] == "pending"


# ─────────────────────────────────────────────────────────────────────────────
# 5. Staleness
# ─────────────────────────────────────────────────────────────────────────────


class TestStaleness:
    async def test_deleted_bbox_becomes_stale_and_is_persisted(self, http_client, image_file, label_uuid):
        run = await _create_run(http_client)
        image_uuid = await _get_image_uuid(http_client)
        bbox_uuid = await _create_bbox(http_client, label_uuid)
        await _add_candidates(http_client, run["uuid"], [_candidate(image_uuid, label_uuid, bbox_uuid=bbox_uuid)])

        del_r = await http_client.delete(f"/api/images/bbox/{bbox_uuid}")
        assert del_r.status_code == 204

        candidates, _ = await _query_candidates(http_client, run["uuid"])
        assert candidates[0]["decision"] == "stale"

        # Persisted: re-listing the run's decision counts must also show it.
        r = await http_client.get(f"/api/preview-runs/{run['uuid']}")
        assert r.json()["run"]["decisionCounts"].get("stale") == 1

    async def test_label_changed_since_run_without_going_stale(
        self, http_client, image_file, label_uuid, other_label_uuid
    ):
        run = await _create_run(http_client)
        image_uuid = await _get_image_uuid(http_client)
        bbox_uuid = await _create_bbox(http_client, label_uuid)
        candidate = _candidate(image_uuid, other_label_uuid, bbox_uuid=bbox_uuid)
        candidate["currentLabelUuid"] = label_uuid  # snapshot at run time
        await _add_candidates(http_client, run["uuid"], [candidate])

        # Relabel the live bbox away from the snapshot.
        patch_r = await http_client.patch(
            f"/api/images/bbox/{bbox_uuid}",
            json={"labelUuid": other_label_uuid, "tagUuids": [UNVERIFIED_TAG_UUID]},
        )
        assert patch_r.status_code == 204

        candidates, _ = await _query_candidates(http_client, run["uuid"])
        assert candidates[0]["decision"] != "stale"
        assert candidates[0]["labelChangedSinceRun"] is True

    async def test_is_new_candidate_validated_against_image_not_bbox(self, http_client, image_file, label_uuid):
        run = await _create_run(http_client)
        image_uuid = await _get_image_uuid(http_client)
        await _add_candidates(http_client, run["uuid"], [_candidate(image_uuid, label_uuid, is_new=True)])

        candidates, _ = await _query_candidates(http_client, run["uuid"])
        assert candidates[0]["decision"] != "stale"  # image exists, is_new candidate is fine with no bbox

    async def test_applied_candidate_with_deleted_bbox_is_not_flipped_to_stale(
        self, http_client, image_file, label_uuid, other_label_uuid
    ):
        """An applied `delete` legitimately removes its own bbox — that must not
        re-read as staleness."""
        run = await _create_run(http_client)
        image_uuid = await _get_image_uuid(http_client)
        bbox_uuid = await _create_bbox(http_client, label_uuid)
        await _add_candidates(http_client, run["uuid"], [_candidate(image_uuid, label_uuid, bbox_uuid=bbox_uuid)])

        candidates, _ = await _query_candidates(http_client, run["uuid"])
        target = candidates[0]["uuid"]
        await http_client.patch(
            f"/api/preview-runs/{run['uuid']}/candidates",
            json={"candidateUuids": [target], "decision": "accepted", "decisionAction": "delete"},
        )

        apply_r = await http_client.post(f"/api/preview-runs/{run['uuid']}/apply", json={})
        assert apply_r.status_code == 200
        result = apply_r.json()["result"]
        assert result["deleted"] == 1
        assert result["stale"] == 0

        candidates_after, _ = await _query_candidates(http_client, run["uuid"])
        assert candidates_after[0]["decision"] == "applied"  # not stale, despite the bbox now being gone


# ─────────────────────────────────────────────────────────────────────────────
# 6. apply_preview_run
# ─────────────────────────────────────────────────────────────────────────────


class TestApplyPreviewRun:
    async def _accept(self, http_client, run_uuid, candidate_uuid, action=None, label_uuid=None, geometry=None):
        payload = {"candidateUuids": [candidate_uuid], "decision": "accepted"}
        if action is not None:
            payload["decisionAction"] = action
        if label_uuid is not None:
            payload["proposedLabelUuid"] = label_uuid
        if geometry is not None:
            payload["geometry"] = geometry
        r = await http_client.patch(f"/api/preview-runs/{run_uuid}/candidates", json=payload)
        assert r.json()["status"] == "success"

    async def test_apply_only_touches_accepted_leaves_pending_and_rejected(
        self, http_client, image_file, label_uuid, other_label_uuid
    ):
        run = await _create_run(http_client)
        image_uuid = await _get_image_uuid(http_client)
        accepted_bbox = await _create_bbox(http_client, label_uuid, x=0.1, y=0.1)
        pending_bbox = await _create_bbox(http_client, label_uuid, x=0.4, y=0.4)
        rejected_bbox = await _create_bbox(http_client, label_uuid, x=0.7, y=0.7)

        await _add_candidates(
            http_client,
            run["uuid"],
            [
                _candidate(image_uuid, other_label_uuid, bbox_uuid=accepted_bbox, source_model="a"),
                _candidate(image_uuid, other_label_uuid, bbox_uuid=pending_bbox, source_model="b"),
                _candidate(image_uuid, other_label_uuid, bbox_uuid=rejected_bbox, source_model="c"),
            ],
        )
        candidates, _ = await _query_candidates(http_client, run["uuid"], sortBy="created_at", sortOrder="asc")
        by_bbox = {c["bboxUuid"]: c["uuid"] for c in candidates}

        await self._accept(
            http_client, run["uuid"], by_bbox[accepted_bbox], action="relabel", label_uuid=other_label_uuid
        )
        r_reject = await http_client.patch(
            f"/api/preview-runs/{run['uuid']}/candidates",
            json={"candidateUuids": [by_bbox[rejected_bbox]], "decision": "rejected"},
        )
        assert r_reject.json()["status"] == "success"

        apply_r = await http_client.post(f"/api/preview-runs/{run['uuid']}/apply", json={})
        result = apply_r.json()["result"]
        assert result["modified"] == 1
        assert result["added"] == 0

        meta_r = await http_client.post("/api/images/metadata/get", json={"imagePath": "preview_test.png"})
        bboxes = {b["uuid"]: b for b in meta_r.json()["image"]["bboxes"]}
        assert bboxes[accepted_bbox]["labelUuid"] == other_label_uuid  # relabeled
        assert bboxes[pending_bbox]["labelUuid"] == label_uuid  # untouched
        assert bboxes[rejected_bbox]["labelUuid"] == label_uuid  # untouched

    async def test_apply_delete_action_removes_bbox(self, http_client, image_file, label_uuid):
        run = await _create_run(http_client)
        image_uuid = await _get_image_uuid(http_client)
        bbox_uuid = await _create_bbox(http_client, label_uuid)
        await _add_candidates(http_client, run["uuid"], [_candidate(image_uuid, label_uuid, bbox_uuid=bbox_uuid)])
        candidates, _ = await _query_candidates(http_client, run["uuid"])
        target = candidates[0]["uuid"]

        await self._accept(http_client, run["uuid"], target, action="delete")
        apply_r = await http_client.post(f"/api/preview-runs/{run['uuid']}/apply", json={})
        result = apply_r.json()["result"]
        assert result["deleted"] == 1

        meta_r = await http_client.post("/api/images/metadata/get", json={"imagePath": "preview_test.png"})
        assert meta_r.json()["image"]["bboxes"] == []

    async def test_apply_new_candidate_creates_bbox_with_unverified_by_default(
        self, http_client, image_file, label_uuid
    ):
        run = await _create_run(http_client)
        image_uuid = await _get_image_uuid(http_client)
        await _add_candidates(
            http_client, run["uuid"], [_candidate(image_uuid, label_uuid, is_new=True, confidence=0.7)]
        )
        candidates, _ = await _query_candidates(http_client, run["uuid"])
        target = candidates[0]["uuid"]
        await self._accept(http_client, run["uuid"], target)

        apply_r = await http_client.post(f"/api/preview-runs/{run['uuid']}/apply", json={})
        result = apply_r.json()["result"]
        assert result["added"] == 1

        meta_r = await http_client.post("/api/images/metadata/get", json={"imagePath": "preview_test.png"})
        bboxes = meta_r.json()["image"]["bboxes"]
        assert len(bboxes) == 1
        tag_uuids = [t["uuid"] for t in bboxes[0]["tags"]]
        assert UNVERIFIED_TAG_UUID in tag_uuids
        assert VERIFIED_TAG_UUID not in tag_uuids

    async def test_apply_new_candidate_persists_confidence(self, http_client, image_file, label_uuid):
        """The canvas label and the review card render the model's confidence
        from `metadata_json.confidence` (the key CollectTrainingImagesTask
        add_bbox already uses). Without a writer the whole display path is inert,
        which is exactly how it shipped the first time.
        """
        run = await _create_run(http_client)
        image_uuid = await _get_image_uuid(http_client)
        await _add_candidates(
            http_client, run["uuid"], [_candidate(image_uuid, label_uuid, is_new=True, confidence=0.82)]
        )
        candidates, _ = await _query_candidates(http_client, run["uuid"])
        await self._accept(http_client, run["uuid"], candidates[0]["uuid"])
        await http_client.post(f"/api/preview-runs/{run['uuid']}/apply", json={})

        meta_r = await http_client.post("/api/images/metadata/get", json={"imagePath": "preview_test.png"})
        # Read side is `metadataJson`; `extra` is only the *write*-side name that
        # web.py maps onto metadata_json (see CLAUDE.md).
        metadata = meta_r.json()["image"]["bboxes"][0]["metadataJson"]
        assert metadata["confidence"] == pytest.approx(0.82)
        assert metadata["source_model"] == "model_a"

    async def test_apply_carries_harvest_provenance_onto_the_box(self, http_client, image_file, label_uuid):
        """Where a harvested frame came from has to survive the preview run that made it.

        The candidate is swept by the run's TTL; the bbox it creates is permanent
        training data. If the provenance only ever lived on the candidate, a harvest
        batch that turns out to be wrong stops being findable the moment the run is
        applied — and `source_model` is no help, because it says "some blip harvest,
        ever", not which event or run.
        """
        run = await _create_run(http_client)
        image_uuid = await _get_image_uuid(http_client)
        candidate = _candidate(image_uuid, label_uuid, is_new=True, confidence=0.5)
        candidate["geometry"] = {
            **candidate["geometry"],
            "sourceEventUuid": "evt-1234",
            "sourceClipTimeS": 4.6,
            "estimated": True,
        }
        await _add_candidates(http_client, run["uuid"], [candidate])
        candidates, _ = await _query_candidates(http_client, run["uuid"])
        await self._accept(http_client, run["uuid"], candidates[0]["uuid"])
        await http_client.post(f"/api/preview-runs/{run['uuid']}/apply", json={})

        meta_r = await http_client.post("/api/images/metadata/get", json={"imagePath": "preview_test.png"})
        bbox = meta_r.json()["image"]["bboxes"][0]
        metadata = bbox["metadataJson"]
        assert metadata["sourceEventUuid"] == "evt-1234"
        assert metadata["sourceClipTimeS"] == pytest.approx(4.6)
        assert metadata["estimated"] is True
        # The box itself stays in the bbox columns — copying it into metadata would
        # give every reader two disagreeing answers to "where is this box".
        assert "x" not in metadata and "width" not in metadata
        assert bbox["x"] == pytest.approx(0.2)

    async def test_apply_new_candidate_mark_verified_true(self, http_client, image_file, label_uuid):
        run = await _create_run(http_client)
        image_uuid = await _get_image_uuid(http_client)
        await _add_candidates(
            http_client, run["uuid"], [_candidate(image_uuid, label_uuid, is_new=True, confidence=0.7)]
        )
        candidates, _ = await _query_candidates(http_client, run["uuid"])
        target = candidates[0]["uuid"]
        await self._accept(http_client, run["uuid"], target)

        apply_r = await http_client.post(f"/api/preview-runs/{run['uuid']}/apply", json={"markVerified": True})
        result = apply_r.json()["result"]
        assert result["added"] == 1

        meta_r = await http_client.post("/api/images/metadata/get", json={"imagePath": "preview_test.png"})
        tag_uuids = [t["uuid"] for t in meta_r.json()["image"]["bboxes"][0]["tags"]]
        assert VERIFIED_TAG_UUID in tag_uuids
        assert UNVERIFIED_TAG_UUID not in tag_uuids

    async def test_applied_candidates_flip_decision_and_set_applied_at(self, http_client, image_file, label_uuid):
        run = await _create_run(http_client)
        image_uuid = await _get_image_uuid(http_client)
        bbox_uuid = await _create_bbox(http_client, label_uuid)
        await _add_candidates(http_client, run["uuid"], [_candidate(image_uuid, label_uuid, bbox_uuid=bbox_uuid)])
        candidates, _ = await _query_candidates(http_client, run["uuid"])
        target = candidates[0]["uuid"]
        await self._accept(http_client, run["uuid"], target, action="relabel", label_uuid=label_uuid)

        await http_client.post(f"/api/preview-runs/{run['uuid']}/apply", json={})

        after, _ = await _query_candidates(http_client, run["uuid"])
        assert after[0]["decision"] == "applied"
        assert after[0]["appliedAt"] is not None

    async def test_apply_counts_are_correct(self, http_client, image_file, label_uuid, other_label_uuid):
        run = await _create_run(http_client)
        image_uuid = await _get_image_uuid(http_client)
        relabel_bbox = await _create_bbox(http_client, label_uuid, x=0.05, y=0.05)
        delete_bbox = await _create_bbox(http_client, label_uuid, x=0.3, y=0.3)

        await _add_candidates(
            http_client,
            run["uuid"],
            [
                _candidate(image_uuid, other_label_uuid, bbox_uuid=relabel_bbox, source_model="a"),
                _candidate(image_uuid, other_label_uuid, bbox_uuid=delete_bbox, source_model="b"),
                _candidate(image_uuid, label_uuid, is_new=True, confidence=0.6, source_model="c"),
            ],
        )
        candidates, _ = await _query_candidates(http_client, run["uuid"], sortBy="created_at", sortOrder="asc")
        by_bbox = {c["bboxUuid"]: c for c in candidates if c["bboxUuid"]}
        new_candidate = next(c for c in candidates if c["isNew"])

        await self._accept(
            http_client, run["uuid"], by_bbox[relabel_bbox]["uuid"], action="relabel", label_uuid=other_label_uuid
        )
        await self._accept(http_client, run["uuid"], by_bbox[delete_bbox]["uuid"], action="delete")
        await self._accept(http_client, run["uuid"], new_candidate["uuid"])

        apply_r = await http_client.post(f"/api/preview-runs/{run['uuid']}/apply", json={})
        result = apply_r.json()["result"]
        assert result["modified"] == 1
        assert result["deleted"] == 1
        assert result["added"] == 1
        assert result["stale"] == 0
        assert result["failed"] == 0

    async def test_reapply_is_idempotent(self, http_client, image_file, label_uuid, other_label_uuid):
        run = await _create_run(http_client)
        image_uuid = await _get_image_uuid(http_client)
        bbox_uuid = await _create_bbox(http_client, label_uuid)
        await _add_candidates(http_client, run["uuid"], [_candidate(image_uuid, other_label_uuid, bbox_uuid=bbox_uuid)])
        candidates, _ = await _query_candidates(http_client, run["uuid"])
        target = candidates[0]["uuid"]
        await self._accept(http_client, run["uuid"], target, action="relabel", label_uuid=other_label_uuid)

        first = await http_client.post(f"/api/preview-runs/{run['uuid']}/apply", json={})
        assert first.json()["result"]["modified"] == 1

        # Second apply: the candidate is now `applied`, not `accepted`, so the
        # accepted-only fetch in apply_preview_run must not pick it up again.
        second = await http_client.post(f"/api/preview-runs/{run['uuid']}/apply", json={})
        result2 = second.json()["result"]
        assert result2["modified"] == 0
        assert result2["added"] == 0
        assert result2["deleted"] == 0


# ─────────────────────────────────────────────────────────────────────────────
# 7. TTL sweep
# ─────────────────────────────────────────────────────────────────────────────


class TestTtlSweep:
    async def test_unpinned_expired_run_is_swept_on_list(self, http_client, db_client):
        run = await _create_run(http_client, name="Expiring")
        await db_client.update_preview_run(run["uuid"], expires_at=datetime.now(timezone.utc) - timedelta(days=1))

        r = await http_client.get("/api/preview-runs")
        uuids = [x["uuid"] for x in r.json()["runs"]]
        assert run["uuid"] not in uuids

        r2 = await http_client.get(f"/api/preview-runs/{run['uuid']}")
        assert r2.json()["status"] == "failure"

    async def test_pinned_expired_run_survives(self, http_client, db_client):
        run = await _create_run(http_client, name="PinnedExpiring")
        await db_client.update_preview_run(
            run["uuid"], pinned=True, expires_at=datetime.now(timezone.utc) - timedelta(days=1)
        )

        r = await http_client.get("/api/preview-runs")
        uuids = [x["uuid"] for x in r.json()["runs"]]
        assert run["uuid"] in uuids

    async def test_fully_triaged_run_swept_early_even_if_not_yet_expired(
        self, http_client, db_client, image_file, label_uuid
    ):
        run = await _create_run(http_client, name="TriagedEarly")
        image_uuid = await _get_image_uuid(http_client)
        await _add_candidates(http_client, run["uuid"], [_candidate(image_uuid, label_uuid, is_new=True)])
        candidates, _ = await _query_candidates(http_client, run["uuid"])
        await http_client.patch(
            f"/api/preview-runs/{run['uuid']}/candidates",
            json={"candidateUuids": [candidates[0]["uuid"]], "decision": "rejected"},
        )

        # expires_at still far in the future, but updated_at is pushed back past
        # the TRIAGED_HOURS cutoff and every candidate is applied/rejected.
        past = datetime.now(timezone.utc) - timedelta(hours=PreviewRunTTL.TRIAGED_HOURS + 1)
        await db_client.update_preview_run(run["uuid"], expires_at=datetime.now(timezone.utc) + timedelta(days=5))
        # Force updated_at back directly — update_preview_run always bumps it
        # forward via ORM refresh, so reach into the session to backdate it.
        from apps.helpers.db.types import PreviewRun

        async with db_client._async_session_maker() as session:
            row = await session.get(PreviewRun, run["uuid"])
            row.updated_at = past
            session.add(row)
            await session.commit()

        r = await http_client.get("/api/preview-runs")
        uuids = [x["uuid"] for x in r.json()["runs"]]
        assert run["uuid"] not in uuids
