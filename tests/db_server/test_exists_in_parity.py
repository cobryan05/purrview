"""EXISTS/IN parity for the term-based filter query builder.

PR 1.4b made `_should_use_exists_dispatch` unconditionally return False (the
selectivity-estimation logic it used to run operated exclusively on the flat
ImageSearchParams fields deleted in that PR; `terms` has no selectivity
estimation, deliberately out of scope). That leaves the `use_exists=True`
branches inside `_apply_filter_terms` (`_tag_predicate`/`_label_predicate`,
the correlated-EXISTS shape with `.correlate(ImageRecord)`) with **no caller
in production and no test coverage** — a real trap for whoever re-enables
EXISTS dispatch later: they'd flip it on and run code that has never
executed.

These tests call `_apply_image_search_filters` directly with
`use_exists=True` and `use_exists=False` over the same fixture data and
assert identical result sets (sorted uuid lists, not just counts), across
every term kind that actually has an EXISTS branch: objectLabel (bboxTag's
sibling, and the one with roll_up/descendant-CTE interaction) and bboxTag.
"""

import uuid as uuidlib

import pytest
from sqlmodel import select

from apps.helpers.db.db_client import _apply_image_search_filters
from apps.helpers.db.types import (
    BBoxTagTerm,
    BBoxUpdate,
    ImageRecord,
    ImageRecordUpdate,
    ImageSearchParams,
    ObjectLabelTerm,
)

# ──────────────────────────────────────────────────────────────────────────────
# Fixture — labels (with a parent/child pair for roll_up), tags, and a small
# set of images/bboxes exercising both include and exclude directions.
# ──────────────────────────────────────────────────────────────────────────────


@pytest.fixture
async def seeded(db_client):
    parent_label = (await db_client.add_label(name="parent", color="#111111")).uuid
    child_label = (await db_client.add_label(name="child", color="#222222", parent_uuid=parent_label)).uuid
    other_label = (await db_client.add_label(name="other", color="#333333")).uuid

    tag_a = (await db_client.add_tag(name="tag-a")).uuid
    tag_b = (await db_client.add_tag(name="tag-b")).uuid

    async def _make_image(filename: str, boxes: list) -> str:
        img = await db_client.add_image(filename=filename)
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
        return img.uuid

    # img_parent: one bbox, parent_label, tag_a only.
    img_parent = await _make_image("parity_parent.jpg", [(parent_label, [tag_a])])
    # img_child: one bbox, child_label, both tags (bbox with tag_a AND tag_b —
    # needed for the bboxTag match_mode="all" case).
    img_child = await _make_image("parity_child.jpg", [(child_label, [tag_a, tag_b])])
    # img_other: one bbox, other_label, tag_b only.
    img_other = await _make_image("parity_other.jpg", [(other_label, [tag_b])])
    # img_parent_tagb: one bbox, parent_label, tag_b only (no tag_a) — lets
    # "objectLabel any [parent]" and "bboxTag any [tag_a]" disagree.
    img_parent_tagb = await _make_image("parity_parent_tagb.jpg", [(parent_label, [tag_b])])
    # img_both_labels: two bboxes on one image — parent_label and other_label
    # (different bboxes) — needed for objectLabel match_mode="all".
    img_both_labels = await _make_image("parity_both_labels.jpg", [(parent_label, [tag_a]), (other_label, [tag_b])])
    # img_no_bboxes: registered image with zero bboxes — exercises the
    # require_bboxes + negated-bbox-implying-term interaction.
    img_no_bboxes = await _make_image("parity_empty.jpg", [])

    return {
        "parent_label": parent_label,
        "child_label": child_label,
        "other_label": other_label,
        "tag_a": tag_a,
        "tag_b": tag_b,
        "img_parent": img_parent,
        "img_child": img_child,
        "img_other": img_other,
        "img_parent_tagb": img_parent_tagb,
        "img_both_labels": img_both_labels,
        "img_no_bboxes": img_no_bboxes,
    }


async def _uuids(db_client, params: ImageSearchParams, use_exists: bool) -> list:
    stmt = _apply_image_search_filters(select(ImageRecord.uuid), params, use_exists=use_exists)
    async with db_client._async_session_maker() as session:
        rows = (await session.exec(stmt)).all()
    return sorted(rows)


async def _assert_parity(db_client, params: ImageSearchParams) -> list:
    """Assert IN and EXISTS shapes return identical result sets; return that set."""
    in_result = await _uuids(db_client, params, use_exists=False)
    exists_result = await _uuids(db_client, params, use_exists=True)
    assert exists_result == in_result, (
        f"EXISTS/IN parity broken for terms={params.terms!r}: " f"IN={in_result!r} EXISTS={exists_result!r}"
    )
    return in_result


# ──────────────────────────────────────────────────────────────────────────────
# objectLabel
# ──────────────────────────────────────────────────────────────────────────────


class TestObjectLabelParity:
    async def test_match_mode_any(self, db_client, seeded):
        params = ImageSearchParams(terms=[ObjectLabelTerm(label_uuids=[seeded["parent_label"]], roll_up=False)])
        result = await _assert_parity(db_client, params)
        assert set(result) == {seeded["img_parent"], seeded["img_parent_tagb"], seeded["img_both_labels"]}

    async def test_match_mode_all(self, db_client, seeded):
        """A single image must have a bbox with EACH label (bboxes may differ)."""
        params = ImageSearchParams(
            terms=[
                ObjectLabelTerm(
                    label_uuids=[seeded["parent_label"], seeded["other_label"]],
                    match_mode="all",
                    roll_up=False,
                )
            ]
        )
        result = await _assert_parity(db_client, params)
        assert set(result) == {seeded["img_both_labels"]}

    async def test_negate(self, db_client, seeded):
        params = ImageSearchParams(
            terms=[ObjectLabelTerm(label_uuids=[seeded["parent_label"]], roll_up=False, negate=True)]
        )
        result = await _assert_parity(db_client, params)
        # Every image whose bboxes never include parent_label, PLUS the
        # bbox-free image (a negated term is satisfied vacuously).
        assert set(result) == {seeded["img_child"], seeded["img_other"], seeded["img_no_bboxes"]}

    async def test_roll_up_matches_descendant(self, db_client, seeded):
        """roll_up=True: filtering on parent_label also matches bboxes on
        child_label — the descendant CTE meeting the EXISTS shape."""
        params = ImageSearchParams(terms=[ObjectLabelTerm(label_uuids=[seeded["parent_label"]], roll_up=True)])
        result = await _assert_parity(db_client, params)
        assert set(result) == {
            seeded["img_parent"],
            seeded["img_parent_tagb"],
            seeded["img_child"],
            seeded["img_both_labels"],
        }


# ──────────────────────────────────────────────────────────────────────────────
# bboxTag
# ──────────────────────────────────────────────────────────────────────────────


class TestBBoxTagParity:
    async def test_match_mode_any(self, db_client, seeded):
        params = ImageSearchParams(terms=[BBoxTagTerm(tag_uuids=[seeded["tag_a"]])])
        result = await _assert_parity(db_client, params)
        assert set(result) == {seeded["img_parent"], seeded["img_child"], seeded["img_both_labels"]}

    async def test_match_mode_all(self, db_client, seeded):
        """match_mode='all' means the image has *a* bbox carrying tag_a AND
        *a* bbox carrying tag_b — independent EXISTS per tag, bboxes may
        differ (matches the pre-1.4b flat tag_uuids/tag_match_mode='all'
        semantics). img_child satisfies both on the SAME bbox; img_both_labels
        satisfies them on two DIFFERENT bboxes — both must match."""
        params = ImageSearchParams(terms=[BBoxTagTerm(tag_uuids=[seeded["tag_a"], seeded["tag_b"]], match_mode="all")])
        result = await _assert_parity(db_client, params)
        assert set(result) == {seeded["img_child"], seeded["img_both_labels"]}

    async def test_negate(self, db_client, seeded):
        params = ImageSearchParams(terms=[BBoxTagTerm(tag_uuids=[seeded["tag_a"]], negate=True)])
        result = await _assert_parity(db_client, params)
        assert set(result) == {seeded["img_other"], seeded["img_parent_tagb"], seeded["img_no_bboxes"]}


# ──────────────────────────────────────────────────────────────────────────────
# Multi-term AND combining both kinds
# ──────────────────────────────────────────────────────────────────────────────


class TestCombinedTermParity:
    async def test_object_label_and_bbox_tag(self, db_client, seeded):
        params = ImageSearchParams(
            terms=[
                ObjectLabelTerm(label_uuids=[seeded["parent_label"]], roll_up=False),
                BBoxTagTerm(tag_uuids=[seeded["tag_a"]]),
            ]
        )
        result = await _assert_parity(db_client, params)
        # Each term is its own independent EXISTS predicate (ANDed together):
        # img_parent's one bbox is parent_label + tag_a — matches both.
        # img_parent_tagb's one bbox is parent_label but carries only tag_b — no
        #   bbox on this image carries tag_a, so the bboxTag term excludes it.
        # img_both_labels' parent_label bbox also carries tag_a (its other
        #   bbox is other_label + tag_b) — matches both.
        assert set(result) == {seeded["img_parent"], seeded["img_both_labels"]}


# ──────────────────────────────────────────────────────────────────────────────
# require_bboxes interaction with a negated bbox-implying term
# ──────────────────────────────────────────────────────────────────────────────


class TestRequireBboxesParity:
    async def test_negated_term_with_require_bboxes(self, db_client, seeded):
        """A negated bboxTag term is satisfied vacuously by a zero-bbox image
        (no bbox implies no bbox carries the excluded tag). require_bboxes=True
        must still exclude that image under BOTH IN and EXISTS shapes."""
        params = ImageSearchParams(
            terms=[BBoxTagTerm(tag_uuids=[seeded["tag_a"]], negate=True)],
            require_bboxes=True,
        )
        result = await _assert_parity(db_client, params)
        assert seeded["img_no_bboxes"] not in result
        assert set(result) == {seeded["img_other"], seeded["img_parent_tagb"]}

    async def test_negated_term_without_require_bboxes_includes_bbox_free_image(self, db_client, seeded):
        """Sanity companion to the above: without require_bboxes, the same
        negated term DOES include the bbox-free image (vacuous truth)."""
        params = ImageSearchParams(terms=[BBoxTagTerm(tag_uuids=[seeded["tag_a"]], negate=True)])
        result = await _assert_parity(db_client, params)
        assert seeded["img_no_bboxes"] in result
