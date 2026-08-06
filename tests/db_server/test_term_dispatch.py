"""Selectivity dispatch (IN vs correlated EXISTS) over the term filter model.

Follow-up PR C. PR 1.4b deleted the flat ImageSearchParams fields the old
dispatch probed, leaving `_should_use_exists_dispatch` hardwired to False — a
documented performance regression with `test_exists_in_parity.py` guarding the
route back. This suite covers the decision itself; parity of the two query
shapes stays that file's job.

The decision is deliberately all-or-nothing: EXISTS only wins when SQLite can
walk the (sort_col, uuid) index and stop after LIMIT rows pass the filter,
which needs the CONJUNCTION of every predicate to be common. One rare or
unestimatable term turns the ordered scan into a full-table scan, so anything
this code cannot cheaply estimate must keep the IN shape.
"""

import uuid as uuidlib

import pytest

from apps.helpers.db.types import (
    UNVERIFIED_TAG_UUID,
    BBoxTagTerm,
    BBoxUpdate,
    BBoxVerificationTerm,
    DateTerm,
    FilenameTerm,
    ImageLabelState,
    ImageLabelTerm,
    ImageRecordUpdate,
    ImageSearchParams,
    LabelScope,
    ObjectLabelTerm,
)

# ──────────────────────────────────────────────────────────────────────────────
# Fixture — 10 images. `common_tag`/`common_label` are on 5 of them (50%, well
# clear of the 20% threshold); `rare_tag`/`rare_label` are on 1 (10%, below it).
# ──────────────────────────────────────────────────────────────────────────────

_TOTAL_IMAGES = 10
_COMMON_COUNT = 5


@pytest.fixture
async def seeded(db_client):
    parent_label = (await db_client.add_label(name="p", color="#111111")).uuid
    child_label = (await db_client.add_label(name="c", color="#222222", parent_uuid=parent_label)).uuid
    rare_label = (await db_client.add_label(name="r", color="#333333")).uuid
    image_label = (await db_client.add_label(name="imgscope", color="#444444", scope=LabelScope.IMAGE)).uuid

    common_tag = (await db_client.add_tag(name="common")).uuid
    rare_tag = (await db_client.add_tag(name="rare")).uuid

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

    # 5 images carry common_tag + the child label (so roll_up onto the parent is
    # also common), and the UNVERIFIED system tag for the bboxVerification case.
    for i in range(_COMMON_COUNT):
        await _make_image(f"common_{i}.jpg", [(child_label, [common_tag, UNVERIFIED_TAG_UUID])])
    # 1 image carries the rare tag + rare label.
    await _make_image("rare_0.jpg", [(rare_label, [rare_tag])])
    # 4 filler images with no bboxes at all.
    for i in range(_TOTAL_IMAGES - _COMMON_COUNT - 1):
        await _make_image(f"filler_{i}.jpg", [])

    return {
        "parent_label": parent_label,
        "child_label": child_label,
        "rare_label": rare_label,
        "image_label": image_label,
        "common_tag": common_tag,
        "rare_tag": rare_tag,
    }


def _params(*terms, **kwargs) -> ImageSearchParams:
    return ImageSearchParams(terms=list(terms), **kwargs)


# ──────────────────────────────────────────────────────────────────────────────
# Common predicates → EXISTS
# ──────────────────────────────────────────────────────────────────────────────


class TestCommonTermsChooseExists:
    async def test_common_bbox_tag(self, db_client, seeded):
        params = _params(BBoxTagTerm(tag_uuids=[seeded["common_tag"]]))
        assert await db_client._should_use_exists_dispatch(params) is True

    async def test_common_object_label(self, db_client, seeded):
        params = _params(ObjectLabelTerm(label_uuids=[seeded["child_label"]], roll_up=False))
        assert await db_client._should_use_exists_dispatch(params) is True

    async def test_common_object_label_via_roll_up(self, db_client, seeded):
        """The parent label has no bboxes of its own — only the descendant CTE
        makes it common, so the probe has to expand exactly like the query does."""
        direct = _params(ObjectLabelTerm(label_uuids=[seeded["parent_label"]], roll_up=False))
        assert await db_client._should_use_exists_dispatch(direct) is False

        rolled = _params(ObjectLabelTerm(label_uuids=[seeded["parent_label"]], roll_up=True))
        assert await db_client._should_use_exists_dispatch(rolled) is True

    async def test_common_bbox_verification(self, db_client, seeded):
        params = _params(BBoxVerificationTerm(state="unverified"))
        assert await db_client._should_use_exists_dispatch(params) is True

    async def test_two_common_terms(self, db_client, seeded):
        params = _params(
            BBoxTagTerm(tag_uuids=[seeded["common_tag"]]),
            ObjectLabelTerm(label_uuids=[seeded["child_label"]], roll_up=False),
        )
        assert await db_client._should_use_exists_dispatch(params) is True


# ──────────────────────────────────────────────────────────────────────────────
# Anything rare, negated or unestimatable → IN
# ──────────────────────────────────────────────────────────────────────────────


class TestFallbackToIn:
    async def test_no_terms(self, db_client, seeded):
        """An unfiltered listing has no restrictive predicate to be common."""
        assert await db_client._should_use_exists_dispatch(_params()) is False

    async def test_rare_bbox_tag(self, db_client, seeded):
        params = _params(BBoxTagTerm(tag_uuids=[seeded["rare_tag"]]))
        assert await db_client._should_use_exists_dispatch(params) is False

    async def test_rare_object_label(self, db_client, seeded):
        params = _params(ObjectLabelTerm(label_uuids=[seeded["rare_label"]], roll_up=False))
        assert await db_client._should_use_exists_dispatch(params) is False

    async def test_one_rare_term_vetoes_a_common_one(self, db_client, seeded):
        """All-or-nothing: the conjunction is as rare as its rarest predicate."""
        params = _params(
            BBoxTagTerm(tag_uuids=[seeded["common_tag"]]),
            ObjectLabelTerm(label_uuids=[seeded["rare_label"]], roll_up=False),
        )
        assert await db_client._should_use_exists_dispatch(params) is False

    async def test_negated_term(self, db_client, seeded):
        """A negated common predicate is not a common predicate."""
        params = _params(BBoxTagTerm(tag_uuids=[seeded["common_tag"]], negate=True))
        assert await db_client._should_use_exists_dispatch(params) is False

    async def test_match_mode_all_uses_the_rarest_member(self, db_client, seeded):
        """ "all" matches at most the smallest member, so a rare co-member vetoes."""
        both_common = _params(BBoxTagTerm(tag_uuids=[seeded["common_tag"], UNVERIFIED_TAG_UUID], match_mode="all"))
        assert await db_client._should_use_exists_dispatch(both_common) is True

        with_rare = _params(BBoxTagTerm(tag_uuids=[seeded["common_tag"], seeded["rare_tag"]], match_mode="all"))
        assert await db_client._should_use_exists_dispatch(with_rare) is False

    @pytest.mark.parametrize(
        "term",
        [
            pytest.param(FilenameTerm(patterns=["common"]), id="filename"),
            pytest.param(DateTerm(field="created", from_="2020-01-01T00:00:00"), id="date"),
        ],
    )
    async def test_unestimatable_term_alongside_a_common_one(self, db_client, seeded, term):
        """These have no cheap COUNT probe, so their conjunction with a common
        term can't be shown to be common — keep IN."""
        params = _params(BBoxTagTerm(tag_uuids=[seeded["common_tag"]]), term)
        assert await db_client._should_use_exists_dispatch(params) is False

    async def test_image_label_term_is_unestimatable(self, db_client, seeded):
        params = _params(
            BBoxTagTerm(tag_uuids=[seeded["common_tag"]]),
            ImageLabelTerm(label_uuid=seeded["image_label"], state=ImageLabelState.PRESENT),
        )
        assert await db_client._should_use_exists_dispatch(params) is False

    async def test_empty_uuid_list(self, db_client, seeded):
        """An empty list means "matches nothing" for match_mode=any and
        "matches everything" for "all" — neither is worth a probe, and getting
        it wrong is the catastrophic full-scan case."""
        assert await db_client._should_use_exists_dispatch(_params(BBoxTagTerm(tag_uuids=[]))) is False
        assert await db_client._should_use_exists_dispatch(_params(ObjectLabelTerm(label_uuids=[]))) is False

    async def test_empty_database(self, db_client):
        """No images at all: the fraction is undefined, and IN is the safe shape."""
        tag = (await db_client.add_tag(name="lonely")).uuid
        assert await db_client._should_use_exists_dispatch(_params(BBoxTagTerm(tag_uuids=[tag]))) is False


# ──────────────────────────────────────────────────────────────────────────────
# The dispatch must not change results — only the plan.
# ──────────────────────────────────────────────────────────────────────────────


class TestDispatchDoesNotChangeResults:
    async def test_search_matches_the_in_shape_when_exists_is_chosen(self, db_client, seeded):
        """End-to-end through list_images_search, which is what actually runs
        the dispatch. test_exists_in_parity.py covers the builder directly;
        this pins that the live path agrees with it."""
        params = _params(BBoxTagTerm(tag_uuids=[seeded["common_tag"]]), limit=50)
        assert await db_client._should_use_exists_dispatch(params) is True

        via_dispatch = [img.uuid for img in await db_client.list_images_search(params)]

        # Same filter, forced onto the IN shape by adding a term the dispatch
        # refuses to estimate but that matches everything seeded here.
        forced_in = _params(
            BBoxTagTerm(tag_uuids=[seeded["common_tag"]]),
            FilenameTerm(patterns=["common_"]),
            limit=50,
        )
        assert await db_client._should_use_exists_dispatch(forced_in) is False
        via_in = [img.uuid for img in await db_client.list_images_search(forced_in)]

        assert sorted(via_dispatch) == sorted(via_in)
        assert len(via_dispatch) == _COMMON_COUNT

    async def test_count_is_unaffected(self, db_client, seeded):
        """count_images_search deliberately always uses IN; the dispatch must
        not make the count disagree with the listing it accompanies."""
        params = _params(BBoxTagTerm(tag_uuids=[seeded["common_tag"]]), limit=50)
        listed = await db_client.list_images_search(params)
        assert await db_client.count_images_search(params) == len(listed)
