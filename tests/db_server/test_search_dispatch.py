"""Tests for the IN-vs-EXISTS selectivity dispatch in image search.

Two layers:
- Equivalence: _apply_image_search_filters(use_exists=True) must return exactly
  the same image set as use_exists=False for every filter shape it dispatches.
- Dispatch decision: _should_use_exists_dispatch picks EXISTS only for common
  predicates and bails out whenever an unestimatable filter is present.
"""

import uuid as uuidlib

import pytest
from sqlmodel import select

from apps.helpers.db.db_client import _apply_image_search_filters
from apps.helpers.db.types import (
    UNVERIFIED_TAG_UUID,
    BBoxUpdate,
    ImageLabelFilter,
    ImageRecord,
    ImageRecordUpdate,
    ImageSearchParams,
    TagPresenceFilter,
)

# ──────────────────────────────────────────────────────────────────────────────
# Fixtures — a small skewed dataset: 20 images, common tag on ~90% of them,
# rare tag on 1, common label on ~80%, rare label on 1, plus a child label.
# ──────────────────────────────────────────────────────────────────────────────


@pytest.fixture
async def seeded(db_client):
    common_label = (await db_client.add_label(name="common", color="#111111")).uuid
    rare_label = (await db_client.add_label(name="rare", color="#222222")).uuid
    child_label = (await db_client.add_label(name="child", color="#333333", parent_uuid=common_label)).uuid
    rare_tag = (await db_client.add_tag(name="rare-tag")).uuid

    images = []
    for i in range(20):
        img = await db_client.add_image(filename=f"/imgs/dispatch_{i:02d}.jpg")
        images.append(img.uuid)
        boxes = []
        if i < 16:  # common label, common (unverified) tag
            boxes.append((common_label, [UNVERIFIED_TAG_UUID]))
        if i == 5:  # the one rare-label bbox, also rare-tagged
            boxes.append((rare_label, [UNVERIFIED_TAG_UUID, rare_tag]))
        if i in (6, 7):  # child of common label
            boxes.append((child_label, [UNVERIFIED_TAG_UUID]))
        if i in (17, 18):  # bbox with common tag but no common label
            boxes.append((rare_label, [UNVERIFIED_TAG_UUID]))
        if boxes:
            await db_client.write_image_metadata_to_db(
                img.uuid,
                ImageRecordUpdate(
                    bboxes=[
                        BBoxUpdate(
                            uuid=str(uuidlib.uuid4()),
                            label_uuid=label_uuid,
                            x=0.1,
                            y=0.1,
                            width=0.2,
                            height=0.2,
                            tag_uuids=tag_uuids,
                        )
                        for label_uuid, tag_uuids in boxes
                    ]
                ),
            )
    return {
        "images": images,
        "common_label": common_label,
        "rare_label": rare_label,
        "child_label": child_label,
        "rare_tag": rare_tag,
    }


async def _uuids(db_client, params: ImageSearchParams, use_exists: bool) -> set:
    stmt = _apply_image_search_filters(select(ImageRecord.uuid), params, use_exists=use_exists)
    async with db_client._async_session_maker() as session:
        return set((await session.exec(stmt)).all())


# ──────────────────────────────────────────────────────────────────────────────
# Equivalence — EXISTS and IN shapes must produce identical image sets
# ──────────────────────────────────────────────────────────────────────────────


class TestExistsInEquivalence:
    @pytest.mark.parametrize(
        "make_params",
        [
            lambda s: ImageSearchParams(tag_uuids=[UNVERIFIED_TAG_UUID]),
            lambda s: ImageSearchParams(tag_uuids=[s["rare_tag"]]),
            lambda s: ImageSearchParams(tag_uuids=[UNVERIFIED_TAG_UUID, s["rare_tag"]]),
            lambda s: ImageSearchParams(tag_uuids=[UNVERIFIED_TAG_UUID, s["rare_tag"]], tag_match_mode="all"),
            lambda s: ImageSearchParams(label_uuids=[s["common_label"]]),
            lambda s: ImageSearchParams(label_uuids=[s["common_label"], s["rare_label"]]),
            lambda s: ImageSearchParams(
                label_uuids=[s["common_label"], s["rare_label"]], label_match_mode="all"
            ),
            lambda s: ImageSearchParams(label_uuids=[s["common_label"]], label_include_children=True),
            lambda s: ImageSearchParams(
                label_uuids=[s["common_label"], s["rare_label"]],
                label_match_mode="all",
                label_include_children=True,
            ),
            lambda s: ImageSearchParams(label_uuid=s["rare_label"]),
            lambda s: ImageSearchParams(tag_uuids=[UNVERIFIED_TAG_UUID], label_uuids=[s["common_label"]]),
            lambda s: ImageSearchParams(tag_uuids=[s["rare_tag"]], label_uuids=[s["rare_label"]]),
        ],
        ids=[
            "tag_common",
            "tag_rare",
            "tags_any",
            "tags_all",
            "label_common",
            "labels_any",
            "labels_all",
            "label_children",
            "labels_all_children",
            "label_backcompat",
            "tag_plus_label",
            "rare_tag_plus_rare_label",
        ],
    )
    async def test_same_image_set(self, db_client, seeded, make_params):
        params = make_params(seeded)
        in_set = await _uuids(db_client, params, use_exists=False)
        exists_set = await _uuids(db_client, params, use_exists=True)
        assert exists_set == in_set
        assert in_set  # every scenario above matches at least one image

    async def test_zero_match_same_empty_set(self, db_client, seeded):
        ghost = str(uuidlib.uuid4())
        for params in (ImageSearchParams(tag_uuids=[ghost]), ImageSearchParams(label_uuids=[ghost])):
            assert await _uuids(db_client, params, use_exists=False) == set()
            assert await _uuids(db_client, params, use_exists=True) == set()

    async def test_list_images_search_end_to_end_matches_count(self, db_client, seeded):
        """The wired-up path: common tag triggers EXISTS dispatch and must
        return the same images as count_images_search (which stays on IN)."""
        params = ImageSearchParams(tag_uuids=[UNVERIFIED_TAG_UUID], limit=100)
        assert await db_client._should_use_exists_dispatch(params) is True
        batch = await db_client.list_images_search(params)
        total = await db_client.count_images_search(params)
        assert len(batch) == total == 18  # images 0-15 plus 17/18


# ──────────────────────────────────────────────────────────────────────────────
# Dispatch decision
# ──────────────────────────────────────────────────────────────────────────────


class TestDispatchDecision:
    async def test_common_tag_dispatches_exists(self, db_client, seeded):
        assert await db_client._should_use_exists_dispatch(ImageSearchParams(tag_uuids=[UNVERIFIED_TAG_UUID]))

    async def test_rare_tag_keeps_in(self, db_client, seeded):
        assert not await db_client._should_use_exists_dispatch(ImageSearchParams(tag_uuids=[seeded["rare_tag"]]))

    async def test_zero_match_tag_keeps_in(self, db_client, seeded):
        ghost = str(uuidlib.uuid4())
        assert not await db_client._should_use_exists_dispatch(ImageSearchParams(tag_uuids=[ghost]))

    async def test_one_rare_predicate_disables_dispatch_entirely(self, db_client, seeded):
        """All-or-nothing: a rare label alongside a common tag must force IN —
        a single rare predicate in the conjunction kills early termination."""
        params = ImageSearchParams(tag_uuids=[UNVERIFIED_TAG_UUID], label_uuids=[seeded["rare_label"]])
        assert not await db_client._should_use_exists_dispatch(params)

    async def test_common_tag_and_common_label_dispatches(self, db_client, seeded):
        params = ImageSearchParams(tag_uuids=[UNVERIFIED_TAG_UUID], label_uuids=[seeded["common_label"]])
        assert await db_client._should_use_exists_dispatch(params)

    async def test_all_mode_uses_min_estimate(self, db_client, seeded):
        # "any" of (common, rare) is common → dispatch; "all" is at most the
        # rare one → keep IN.
        any_params = ImageSearchParams(tag_uuids=[UNVERIFIED_TAG_UUID, seeded["rare_tag"]])
        all_params = any_params.model_copy(update={"tag_match_mode": "all"})
        assert await db_client._should_use_exists_dispatch(any_params)
        assert not await db_client._should_use_exists_dispatch(all_params)

    @pytest.mark.parametrize(
        "extra",
        [
            {"filename_search": "frame"},
            {"filename_searches": ["frame"]},
            {"date_from": "2020-01-01T00:00:00Z"},
            {"updated_date_to": "2030-01-01T00:00:00Z"},
            {"image_label_filters": [ImageLabelFilter(label_uuid=str(uuidlib.uuid4()), status="present")]},
            {"tag_presence_filters": [TagPresenceFilter(tag_uuid=str(uuidlib.uuid4()), exclude=True)]},
            {"label_exclusions": [str(uuidlib.uuid4())]},
            {"label_verification_check": "has_unverified"},
        ],
        ids=[
            "filename_search",
            "filename_searches",
            "date_from",
            "updated_date_to",
            "image_label_filter",
            "tag_presence_filter",
            "label_exclusion",
            "verification_check",
        ],
    )
    async def test_unestimatable_filters_disable_dispatch(self, db_client, seeded, extra):
        params = ImageSearchParams(tag_uuids=[UNVERIFIED_TAG_UUID], **extra)
        assert not await db_client._should_use_exists_dispatch(params)

    async def test_no_bbox_predicates_keeps_in(self, db_client, seeded):
        assert not await db_client._should_use_exists_dispatch(ImageSearchParams())
        assert not await db_client._should_use_exists_dispatch(ImageSearchParams(require_bboxes=True))

    async def test_show_all_bboxes_ignores_tag_probe(self, db_client, seeded):
        """show_all_bboxes suppresses the tag filter, so the tag probe must not
        run — with no dispatchable predicate left, IN is kept."""
        params = ImageSearchParams(tag_uuids=[UNVERIFIED_TAG_UUID], show_all_bboxes=True)
        assert not await db_client._should_use_exists_dispatch(params)

    async def test_empty_db_keeps_in(self, db_client):
        params = ImageSearchParams(tag_uuids=[UNVERIFIED_TAG_UUID])
        assert not await db_client._should_use_exists_dispatch(params)

    async def test_probe_cache_reused(self, db_client, seeded):
        params = ImageSearchParams(tag_uuids=[UNVERIFIED_TAG_UUID])
        await db_client._should_use_exists_dispatch(params)
        key = f"tag:{UNVERIFIED_TAG_UUID}"
        assert key in db_client._selectivity_cache
        first = db_client._selectivity_cache[key]
        await db_client._should_use_exists_dispatch(params)
        assert db_client._selectivity_cache[key] is first  # cache hit, not re-probed
