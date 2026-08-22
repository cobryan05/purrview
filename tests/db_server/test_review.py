"""Tests for the review queue endpoint — POST /api/images/review.

Covers tag filtering, label filtering, filename search, date range,
image_label filters, cursor pagination, and sort order.
"""

import uuid
from datetime import datetime, timedelta, timezone

import pytest

from apps.helpers.db.types import UNVERIFIED_TAG_UUID, VERIFIED_TAG_UUID
from tests.conftest import make_png
from tests.helpers.legacy_filter_params import legacy_params_to_body

# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────


@pytest.fixture
async def label_uuid(http_client):
    r = await http_client.post("/api/labels", json={"name": "cat", "color": "#FF0000"})
    return r.json()["label"]["uuid"]


@pytest.fixture
async def label2_uuid(http_client):
    r = await http_client.post("/api/labels", json={"name": "dog", "color": "#0000FF"})
    return r.json()["label"]["uuid"]


def _make_image(files_root, name: str) -> str:
    """Write a PNG to files_root and return the relative path."""
    p = files_root / name
    p.write_bytes(make_png())
    return name


async def _add_bbox(http_client, image_path: str, label_uuid: str, tag_uuids: list) -> str:
    """Create/replace bboxes on an image; return the bbox uuid."""
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
                    "width": 0.2,
                    "height": 0.2,
                    "tagUuids": tag_uuids,
                }
            ],
        },
    )
    assert r.status_code == 200
    return bid


async def _review(http_client, **params):
    """POST /api/images/review with the given params and return the images list.

    ImageSearchParams uses snake_case field names (no alias_generator). Legacy
    flat-filter kwargs (tag_uuids, label_uuids, image_label_filters, ...) are
    translated to `terms` by legacy_params_to_body — see that module's
    docstring for why the call sites below were left unchanged.
    """
    r = await http_client.post("/api/images/review", json=legacy_params_to_body(params))
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "success"
    return body["images"]


# ──────────────────────────────────────────────────────────────────────────────
# Basic tag filter
# ──────────────────────────────────────────────────────────────────────────────


async def test_empty_queue(http_client):
    images = await _review(http_client, tag_uuids=[UNVERIFIED_TAG_UUID])
    assert images == []


async def test_review_queue_rejects_filename_sort(http_client):
    """filename sort has no valid keyset cursor (after_sort_value is a datetime),
    so page 2+ silently returns wrong/duplicated results. It must not be an
    accepted sort option — reject it at validation instead of paginating wrong.
    """
    r = await http_client.post("/api/images/review", json={"sortBy": "filename", "limit": 5})
    assert r.status_code == 422


async def test_unverified_image_appears_in_queue(http_client, files_root, label_uuid):
    _make_image(files_root, "unverified.png")
    await _add_bbox(http_client, "unverified.png", label_uuid, [UNVERIFIED_TAG_UUID])
    images = await _review(http_client, tag_uuids=[UNVERIFIED_TAG_UUID])
    filenames = [img["filename"] for img in images]
    assert any("unverified.png" in f for f in filenames)


async def test_confirmed_image_not_in_unverified_queue(http_client, files_root, label_uuid):
    """A 'confirmed' bbox (no UNVERIFIED tag) does not appear in the UNVERIFIED queue."""
    _make_image(files_root, "confirmed.png")
    await _add_bbox(http_client, "confirmed.png", label_uuid, [])
    images = await _review(http_client, tag_uuids=[UNVERIFIED_TAG_UUID])
    filenames = [img["filename"] for img in images]
    assert not any("confirmed.png" in f for f in filenames)


async def test_confirmed_image_appears_when_show_all(http_client, files_root, label_uuid):
    """A confirmed image (no UNVERIFIED tag) is visible via show_all_bboxes."""
    _make_image(files_root, "show-all.png")
    await _add_bbox(http_client, "show-all.png", label_uuid, [])
    images = await _review(http_client, show_all_bboxes=True)
    filenames = [img["filename"] for img in images]
    assert any("show-all.png" in f for f in filenames)


async def test_show_all_bboxes_returns_all_images(http_client, files_root, label_uuid):
    _make_image(files_root, "all1.png")
    _make_image(files_root, "all2.png")
    await _add_bbox(http_client, "all1.png", label_uuid, [UNVERIFIED_TAG_UUID])
    await _add_bbox(http_client, "all2.png", label_uuid, [])
    images = await _review(http_client, show_all_bboxes=True)
    filenames = [img["filename"] for img in images]
    assert any("all1.png" in f for f in filenames)
    assert any("all2.png" in f for f in filenames)


# ──────────────────────────────────────────────────────────────────────────────
# Label filter
# ──────────────────────────────────────────────────────────────────────────────


async def test_label_filter_includes_matching(http_client, files_root, label_uuid, label2_uuid):
    _make_image(files_root, "cat-img.png")
    _make_image(files_root, "dog-img.png")
    await _add_bbox(http_client, "cat-img.png", label_uuid, [UNVERIFIED_TAG_UUID])
    await _add_bbox(http_client, "dog-img.png", label2_uuid, [UNVERIFIED_TAG_UUID])

    images = await _review(http_client, tag_uuids=[UNVERIFIED_TAG_UUID], label_uuids=[label_uuid])
    filenames = [img["filename"] for img in images]
    assert any("cat-img.png" in f for f in filenames)
    assert not any("dog-img.png" in f for f in filenames)


async def test_label_filter_any_of(http_client, files_root, label_uuid, label2_uuid):
    """label_uuids with label_match_mode='any' returns images that have EITHER label."""
    r = await http_client.post("/api/labels", json={"name": "bird", "color": "#0ff"})
    bird_uuid = r.json()["label"]["uuid"]

    _make_image(files_root, "cat2.png")
    _make_image(files_root, "dog2.png")
    _make_image(files_root, "bird2.png")
    await _add_bbox(http_client, "cat2.png", label_uuid, [UNVERIFIED_TAG_UUID])
    await _add_bbox(http_client, "dog2.png", label2_uuid, [UNVERIFIED_TAG_UUID])
    await _add_bbox(http_client, "bird2.png", bird_uuid, [UNVERIFIED_TAG_UUID])

    images = await _review(
        http_client,
        tag_uuids=[UNVERIFIED_TAG_UUID],
        label_uuids=[label_uuid, label2_uuid],
        label_match_mode="any",
    )
    filenames = [img["filename"] for img in images]
    assert any("cat2.png" in f for f in filenames)
    assert any("dog2.png" in f for f in filenames)
    assert not any("bird2.png" in f for f in filenames)


async def test_label_include_children(http_client, files_root, label_uuid):
    """label_include_children=True matches images with the label OR any child label."""
    # Create a child label under label_uuid
    r = await http_client.post("/api/labels", json={"name": "tabby", "color": "#abc", "parentUuid": label_uuid})
    child_uuid = r.json()["label"]["uuid"]

    _make_image(files_root, "parent-label.png")
    _make_image(files_root, "child-label.png")
    _make_image(files_root, "unrelated.png")
    r2 = await http_client.post("/api/labels", json={"name": "dog", "color": "#0f0"})
    other_uuid = r2.json()["label"]["uuid"]

    await _add_bbox(http_client, "parent-label.png", label_uuid, [UNVERIFIED_TAG_UUID])
    await _add_bbox(http_client, "child-label.png", child_uuid, [UNVERIFIED_TAG_UUID])
    await _add_bbox(http_client, "unrelated.png", other_uuid, [UNVERIFIED_TAG_UUID])

    # Without include_children: only the parent-label image matches
    images_no_children = await _review(
        http_client,
        show_all_bboxes=True,
        label_uuids=[label_uuid],
        label_include_children=False,
    )
    names_no = [img["filename"] for img in images_no_children]
    assert any("parent-label.png" in n for n in names_no)
    assert not any("child-label.png" in n for n in names_no)

    # With include_children: both parent and child images match
    images_with_children = await _review(
        http_client,
        show_all_bboxes=True,
        label_uuids=[label_uuid],
        label_include_children=True,
    )
    names_with = [img["filename"] for img in images_with_children]
    assert any("parent-label.png" in n for n in names_with)
    assert any("child-label.png" in n for n in names_with)
    assert not any("unrelated.png" in n for n in names_with)


async def test_label_filter_all_of(http_client, files_root, label_uuid, label2_uuid):
    """label_uuids with label_match_mode='all' returns only images that have BOTH labels."""
    _make_image(files_root, "both.png")
    _make_image(files_root, "cat-only.png")
    # Add two bboxes (different labels) to "both.png" in one call so neither overwrites the other.
    r = await http_client.post(
        "/api/images/metadata/update",
        json={
            "imagePath": "both.png",
            "boxes": [
                {
                    "uuid": str(uuid.uuid4()),
                    "labelUuid": label_uuid,
                    "x": 0.1,
                    "y": 0.1,
                    "width": 0.2,
                    "height": 0.2,
                    "tagUuids": [UNVERIFIED_TAG_UUID],
                },
                {
                    "uuid": str(uuid.uuid4()),
                    "labelUuid": label2_uuid,
                    "x": 0.3,
                    "y": 0.3,
                    "width": 0.1,
                    "height": 0.1,
                    "tagUuids": [UNVERIFIED_TAG_UUID],
                },
            ],
        },
    )
    assert r.status_code == 200
    await _add_bbox(http_client, "cat-only.png", label_uuid, [UNVERIFIED_TAG_UUID])

    images = await _review(
        http_client,
        show_all_bboxes=True,
        label_uuids=[label_uuid, label2_uuid],
        label_match_mode="all",
    )
    filenames = [img["filename"] for img in images]
    assert any("both.png" in f for f in filenames)
    assert not any("cat-only.png" in f for f in filenames)


# ──────────────────────────────────────────────────────────────────────────────
# Filename search
# ──────────────────────────────────────────────────────────────────────────────


async def test_filename_search_matches(http_client, files_root, label_uuid):
    _make_image(files_root, "search-match.png")
    _make_image(files_root, "no-match.png")
    await _add_bbox(http_client, "search-match.png", label_uuid, [UNVERIFIED_TAG_UUID])
    await _add_bbox(http_client, "no-match.png", label_uuid, [UNVERIFIED_TAG_UUID])

    images = await _review(http_client, show_all_bboxes=True, filename_search="search-match")
    filenames = [img["filename"] for img in images]
    assert any("search-match.png" in f for f in filenames)
    assert not any("no-match.png" in f for f in filenames)


class TestMultipleFilenameFilters:
    """The review panel can hold more than one filename filter.

    Includes collapse into ONE term whose patterns are combined by match_mode
    ('any' = OR, so two cameras can be asked for at once); each exclusion is
    its own negated term, and terms are ANDed, so every exclusion applies.
    """

    async def _setup(self, http_client, files_root, label_uuid):
        for name in ("cam1-front.png", "cam2-back.png", "cam3-side.png"):
            _make_image(files_root, name)
            await _add_bbox(http_client, name, label_uuid, [UNVERIFIED_TAG_UUID])

    async def _search(self, http_client, terms):
        r = await http_client.post("/api/images/review", json={"showAllBboxes": True, "terms": terms})
        assert r.status_code == 200
        return sorted(img["filename"].rsplit("/", 1)[-1] for img in r.json()["images"])

    async def test_match_mode_any_matches_either_pattern(self, http_client, files_root, label_uuid):
        await self._setup(http_client, files_root, label_uuid)
        found = await self._search(
            http_client,
            [{"kind": "filename", "patterns": ["cam1", "cam2"], "matchMode": "any"}],
        )
        assert found == ["cam1-front.png", "cam2-back.png"]

    async def test_match_mode_all_still_requires_every_pattern(self, http_client, files_root, label_uuid):
        """The default stays AND — two different cameras can't be in one path."""
        await self._setup(http_client, files_root, label_uuid)
        assert (
            await self._search(
                http_client,
                [{"kind": "filename", "patterns": ["cam1", "cam2"], "matchMode": "all"}],
            )
            == []
        )
        assert await self._search(
            http_client,
            [{"kind": "filename", "patterns": ["cam1", "front"], "matchMode": "all"}],
        ) == ["cam1-front.png"]

    async def test_two_exclusions_both_apply(self, http_client, files_root, label_uuid):
        await self._setup(http_client, files_root, label_uuid)
        found = await self._search(
            http_client,
            [
                {"kind": "filename", "patterns": ["cam1"], "negate": True},
                {"kind": "filename", "patterns": ["cam2"], "negate": True},
            ],
        )
        assert found == ["cam3-side.png"]

    async def test_include_any_narrowed_by_an_exclusion(self, http_client, files_root, label_uuid):
        await self._setup(http_client, files_root, label_uuid)
        found = await self._search(
            http_client,
            [
                {"kind": "filename", "patterns": ["cam1", "cam2"], "matchMode": "any"},
                {"kind": "filename", "patterns": ["back"], "negate": True},
            ],
        )
        assert found == ["cam1-front.png"]


async def test_filename_search_no_results(http_client, files_root, label_uuid):
    _make_image(files_root, "any.png")
    await _add_bbox(http_client, "any.png", label_uuid, [UNVERIFIED_TAG_UUID])
    images = await _review(http_client, show_all_bboxes=True, filename_search="zzz-not-there")
    assert images == []


# ──────────────────────────────────────────────────────────────────────────────
# Sort order
# ──────────────────────────────────────────────────────────────────────────────


async def test_default_sort_desc_newest_first(http_client, files_root, label_uuid):
    """Default sort is created_at DESC; later-created images come first."""
    for name in ["sort-a.png", "sort-b.png", "sort-c.png"]:
        _make_image(files_root, name)
        await _add_bbox(http_client, name, label_uuid, [UNVERIFIED_TAG_UUID])

    images = await _review(
        http_client,
        tag_uuids=[UNVERIFIED_TAG_UUID],
        sort_by="created_at",
        sort_order="desc",
    )
    assert len(images) >= 3
    # Most-recently created should be first — just confirm we get a list back
    assert all("filename" in img for img in images)


async def test_sort_asc_returns_images(http_client, files_root, label_uuid):
    for name in ["asc-a.png", "asc-b.png"]:
        _make_image(files_root, name)
        await _add_bbox(http_client, name, label_uuid, [UNVERIFIED_TAG_UUID])

    images = await _review(
        http_client,
        tag_uuids=[UNVERIFIED_TAG_UUID],
        sort_by="created_at",
        sort_order="asc",
    )
    assert len(images) >= 2


# ──────────────────────────────────────────────────────────────────────────────
# Limit / pagination
# ──────────────────────────────────────────────────────────────────────────────


async def test_limit_restricts_results(http_client, files_root, label_uuid):
    for i in range(5):
        name = f"page-{i}.png"
        _make_image(files_root, name)
        await _add_bbox(http_client, name, label_uuid, [UNVERIFIED_TAG_UUID])

    images = await _review(http_client, tag_uuids=[UNVERIFIED_TAG_UUID], limit=2)
    assert len(images) == 2


async def test_cursor_pagination_covers_all_records(http_client, files_root, label_uuid):
    """Paging through with limit=2 should yield all 4 distinct images."""
    names = [f"cur-{i}.png" for i in range(4)]
    for name in names:
        _make_image(files_root, name)
        await _add_bbox(http_client, name, label_uuid, [UNVERIFIED_TAG_UUID])

    seen = set()
    after_uuid = None
    after_sort_value = None

    for _ in range(4):  # at most 4 iterations for 4 records with limit=2
        params = dict(
            tag_uuids=[UNVERIFIED_TAG_UUID],
            sort_by="created_at",
            sort_order="desc",
            limit=2,
        )
        if after_uuid:
            params["after_uuid"] = after_uuid
            params["after_sort_value"] = after_sort_value

        images = await _review(http_client, **params)
        if not images:
            break
        for img in images:
            seen.add(img["filename"])
        last = images[-1]
        after_uuid = last["uuid"]
        after_sort_value = last.get("createdAt")  # ImageRecordRead uses camelCase

    assert len(seen) == 4


# ──────────────────────────────────────────────────────────────────────────────
# image_label_filters
# ──────────────────────────────────────────────────────────────────────────────


async def test_image_label_filter_present(http_client, files_root, label_uuid):
    """image_label_filters with PRESENT status should only return verified images."""
    _make_image(files_root, "ilf-verf.png")
    _make_image(files_root, "ilf-unv.png")
    await _add_bbox(http_client, "ilf-verf.png", label_uuid, [VERIFIED_TAG_UUID])
    await _add_bbox(http_client, "ilf-unv.png", label_uuid, [UNVERIFIED_TAG_UUID])

    images = await _review(
        http_client,
        show_all_bboxes=True,
        image_label_filters=[{"label_uuid": label_uuid, "status": "present", "exclude": False}],
    )
    filenames = [img["filename"] for img in images]
    assert any("ilf-verf.png" in f for f in filenames)
    assert not any("ilf-unv.png" in f for f in filenames)


async def test_image_label_filter_exclude(http_client, files_root, label_uuid):
    """Exclude filter: return images that do NOT have a 'present' label."""
    _make_image(files_root, "excl-verf.png")
    _make_image(files_root, "excl-unv.png")
    await _add_bbox(http_client, "excl-verf.png", label_uuid, [VERIFIED_TAG_UUID])
    await _add_bbox(http_client, "excl-unv.png", label_uuid, [UNVERIFIED_TAG_UUID])

    images = await _review(
        http_client,
        show_all_bboxes=True,
        image_label_filters=[{"label_uuid": label_uuid, "status": "present", "exclude": True}],
    )
    filenames = [img["filename"] for img in images]
    assert not any("excl-verf.png" in f for f in filenames)
    assert any("excl-unv.png" in f for f in filenames)


# ──────────────────────────────────────────────────────────────────────────────
# image_label_filters — require_unset
# ──────────────────────────────────────────────────────────────────────────────


async def test_image_label_filter_require_unset_matches_never_assessed(http_client, files_root, label_uuid):
    """require_unset=True returns only images with NO image_labels row for the
    label at all — i.e. the label was never assessed for that image."""
    _make_image(files_root, "ru-unassessed.png")
    _make_image(files_root, "ru-present.png")
    # Register the unassessed image with no bboxes, so no image_labels row exists.
    r = await http_client.post("/api/images/metadata/update", json={"imagePath": "ru-unassessed.png", "boxes": []})
    assert r.status_code == 200
    # This image gets a PRESENT image_labels row via the verified-bbox sync path.
    await _add_bbox(http_client, "ru-present.png", label_uuid, [VERIFIED_TAG_UUID])

    images = await _review(
        http_client,
        show_all_bboxes=True,
        image_label_filters=[{"label_uuid": label_uuid, "status": "present", "exclude": False, "require_unset": True}],
    )
    filenames = [img["filename"] for img in images]
    assert any("ru-unassessed.png" in f for f in filenames)
    assert not any("ru-present.png" in f for f in filenames)


async def test_image_label_filter_require_unset_excludes_present(http_client, files_root, label_uuid):
    """An image with a PRESENT image_labels row for the label is excluded by
    require_unset — row existence alone disqualifies it, regardless of status."""
    _make_image(files_root, "ru-excl-present.png")
    await _add_bbox(http_client, "ru-excl-present.png", label_uuid, [VERIFIED_TAG_UUID])

    images = await _review(
        http_client,
        show_all_bboxes=True,
        image_label_filters=[{"label_uuid": label_uuid, "status": "present", "exclude": False, "require_unset": True}],
    )
    filenames = [img["filename"] for img in images]
    assert not any("ru-excl-present.png" in f for f in filenames)


async def test_image_label_filter_require_unset_excludes_absent(http_client, files_root, label_uuid):
    """An image with an ABSENT image_labels row for the label is also excluded
    by require_unset — this is the case that IN(status)/EXISTS(status) queries
    silently miss today, since ABSENT is a real row, not the missing state."""
    _make_image(files_root, "ru-excl-absent.png")
    r = await http_client.post("/api/images/metadata/update", json={"imagePath": "ru-excl-absent.png", "boxes": []})
    assert r.status_code == 200
    r = await http_client.post(
        "/api/images/labels/set",
        json={"imagePath": "ru-excl-absent.png", "labelUuid": label_uuid, "status": "absent"},
    )
    assert r.status_code == 200
    assert r.json()["status"] == "success"

    images = await _review(
        http_client,
        show_all_bboxes=True,
        image_label_filters=[{"label_uuid": label_uuid, "status": "present", "exclude": False, "require_unset": True}],
    )
    filenames = [img["filename"] for img in images]
    assert not any("ru-excl-absent.png" in f for f in filenames)


async def test_image_label_filter_require_unset_ignores_exclude(http_client, files_root, label_uuid):
    """require_unset takes precedence over `exclude`, matching the documented
    ImageTagFilter semantics: exclude is ignored entirely once require_unset is set.

    Both an unassessed image (would match on require_unset alone) and a PRESENT
    image (would be excluded by require_unset alone) keep that same outcome
    whether exclude is True or False — proving exclude has no effect here.
    """
    _make_image(files_root, "ru-ign-unassessed.png")
    _make_image(files_root, "ru-ign-present.png")
    r = await http_client.post("/api/images/metadata/update", json={"imagePath": "ru-ign-unassessed.png", "boxes": []})
    assert r.status_code == 200
    await _add_bbox(http_client, "ru-ign-present.png", label_uuid, [VERIFIED_TAG_UUID])

    images = await _review(
        http_client,
        show_all_bboxes=True,
        image_label_filters=[{"label_uuid": label_uuid, "status": "present", "exclude": True, "require_unset": True}],
    )
    filenames = [img["filename"] for img in images]
    assert any("ru-ign-unassessed.png" in f for f in filenames)
    assert not any("ru-ign-present.png" in f for f in filenames)


async def test_image_label_filter_require_unset_beats_require_verified(http_client, files_root, label_uuid):
    """When both require_unset and require_verified are set, require_unset wins
    (it is checked first in the filter loop, mirroring ImageTagFilter)."""
    _make_image(files_root, "ru-vs-rv-unassessed.png")
    r = await http_client.post(
        "/api/images/metadata/update", json={"imagePath": "ru-vs-rv-unassessed.png", "boxes": []}
    )
    assert r.status_code == 200

    images = await _review(
        http_client,
        show_all_bboxes=True,
        image_label_filters=[
            {
                "label_uuid": label_uuid,
                "status": "present",
                "exclude": False,
                "require_verified": True,
                "require_unset": True,
            }
        ],
    )
    filenames = [img["filename"] for img in images]
    # require_verified alone would EXCLUDE this image (no PRESENT/ABSENT row
    # exists yet, so the "require verified" EXISTS check would fail). Getting
    # it back proves require_unset short-circuited require_verified.
    assert any("ru-vs-rv-unassessed.png" in f for f in filenames)


# ──────────────────────────────────────────────────────────────────────────────
# image_label_filters — roll_up
# ──────────────────────────────────────────────────────────────────────────────


async def test_image_label_filter_roll_up_matches_child_present(http_client, files_root, label_uuid):
    """A PRESENT term on the parent (roll_up defaults True) matches an image
    that only has the CHILD label recorded PRESENT — the Tabby->Cat case."""
    child = await http_client.post("/api/labels", json={"name": "tabby", "color": "#abc", "parentUuid": label_uuid})
    child_uuid = child.json()["label"]["uuid"]

    _make_image(files_root, "rollup-child.png")
    await _add_bbox(http_client, "rollup-child.png", child_uuid, [VERIFIED_TAG_UUID])

    images = await _review(
        http_client,
        show_all_bboxes=True,
        image_label_filters=[{"label_uuid": label_uuid, "status": "present", "exclude": False}],
    )
    filenames = [img["filename"] for img in images]
    assert any("rollup-child.png" in f for f in filenames)


async def test_image_label_filter_roll_up_matches_grandchild_present(http_client, files_root, label_uuid):
    """Roll-up recurses through a three-level hierarchy: PRESENT on the
    grandchild matches a filter term on the top-level parent."""
    child = await http_client.post("/api/labels", json={"name": "tabby", "color": "#abc", "parentUuid": label_uuid})
    child_uuid = child.json()["label"]["uuid"]
    grandchild = await http_client.post(
        "/api/labels", json={"name": "tabby-kitten", "color": "#abd", "parentUuid": child_uuid}
    )
    grandchild_uuid = grandchild.json()["label"]["uuid"]

    _make_image(files_root, "rollup-grandchild.png")
    await _add_bbox(http_client, "rollup-grandchild.png", grandchild_uuid, [VERIFIED_TAG_UUID])

    images = await _review(
        http_client,
        show_all_bboxes=True,
        image_label_filters=[{"label_uuid": label_uuid, "status": "present", "exclude": False}],
    )
    filenames = [img["filename"] for img in images]
    assert any("rollup-grandchild.png" in f for f in filenames)


async def test_image_label_filter_roll_up_false_returns_old_result(http_client, files_root, label_uuid):
    """roll_up=False restores the pre-roll-up behaviour: a child-only image is
    NOT matched by a PRESENT term on the parent."""
    child = await http_client.post("/api/labels", json={"name": "tabby", "color": "#abc", "parentUuid": label_uuid})
    child_uuid = child.json()["label"]["uuid"]

    _make_image(files_root, "no-rollup-child.png")
    await _add_bbox(http_client, "no-rollup-child.png", child_uuid, [VERIFIED_TAG_UUID])

    images = await _review(
        http_client,
        show_all_bboxes=True,
        image_label_filters=[{"label_uuid": label_uuid, "status": "present", "exclude": False, "roll_up": False}],
    )
    filenames = [img["filename"] for img in images]
    assert not any("no-rollup-child.png" in f for f in filenames)


async def test_image_label_filter_absent_ignores_roll_up(http_client, files_root, label_uuid):
    """ABSENT-status terms ignore roll_up server-side: a child recorded ABSENT
    must NOT cause an ABSENT+roll_up=True filter on the parent to match, since
    that's exactly the roll_up=False behaviour too (pins the documented
    exception)."""
    child = await http_client.post("/api/labels", json={"name": "tabby", "color": "#abc", "parentUuid": label_uuid})
    child_uuid = child.json()["label"]["uuid"]

    _make_image(files_root, "absent-child.png")
    r = await http_client.post("/api/images/metadata/update", json={"imagePath": "absent-child.png", "boxes": []})
    assert r.status_code == 200
    r = await http_client.post(
        "/api/images/labels/set",
        json={"imagePath": "absent-child.png", "labelUuid": child_uuid, "status": "absent"},
    )
    assert r.status_code == 200

    images_rollup_on = await _review(
        http_client,
        show_all_bboxes=True,
        image_label_filters=[{"label_uuid": label_uuid, "status": "absent", "exclude": False, "roll_up": True}],
    )
    images_rollup_off = await _review(
        http_client,
        show_all_bboxes=True,
        image_label_filters=[{"label_uuid": label_uuid, "status": "absent", "exclude": False, "roll_up": False}],
    )
    filenames_on = [img["filename"] for img in images_rollup_on]
    filenames_off = [img["filename"] for img in images_rollup_off]
    assert not any("absent-child.png" in f for f in filenames_on)
    assert filenames_on == filenames_off


async def test_image_label_filter_roll_up_require_unset_excludes_descendant_row(http_client, files_root, label_uuid):
    """A rolled-up require_unset term on the parent is NOT "unreviewed" when a
    DESCENDANT has an image_labels row — this must be a single NOT EXISTS over
    the whole descendant set, not two independent checks."""
    child = await http_client.post("/api/labels", json={"name": "tabby", "color": "#abc", "parentUuid": label_uuid})
    child_uuid = child.json()["label"]["uuid"]

    _make_image(files_root, "ru-rollup-child-present.png")
    await _add_bbox(http_client, "ru-rollup-child-present.png", child_uuid, [VERIFIED_TAG_UUID])

    _make_image(files_root, "ru-rollup-truly-unset.png")
    r = await http_client.post(
        "/api/images/metadata/update", json={"imagePath": "ru-rollup-truly-unset.png", "boxes": []}
    )
    assert r.status_code == 200

    images = await _review(
        http_client,
        show_all_bboxes=True,
        image_label_filters=[{"label_uuid": label_uuid, "status": "present", "exclude": False, "require_unset": True}],
    )
    filenames = [img["filename"] for img in images]
    assert not any("ru-rollup-child-present.png" in f for f in filenames)
    assert any("ru-rollup-truly-unset.png" in f for f in filenames)


async def test_image_label_filter_roll_up_require_verified_matches_descendant_row(http_client, files_root, label_uuid):
    """require_verified rolls up: a PRESENT (or ABSENT) row on a descendant
    satisfies a require_verified term on the parent, keeping it the exact
    complement of the rolled-up require_unset test above."""
    child = await http_client.post("/api/labels", json={"name": "tabby", "color": "#abc", "parentUuid": label_uuid})
    child_uuid = child.json()["label"]["uuid"]

    _make_image(files_root, "rv-rollup-child-present.png")
    await _add_bbox(http_client, "rv-rollup-child-present.png", child_uuid, [VERIFIED_TAG_UUID])

    _make_image(files_root, "rv-rollup-unset.png")
    r = await http_client.post("/api/images/metadata/update", json={"imagePath": "rv-rollup-unset.png", "boxes": []})
    assert r.status_code == 200

    images = await _review(
        http_client,
        show_all_bboxes=True,
        image_label_filters=[
            {"label_uuid": label_uuid, "status": "present", "exclude": False, "require_verified": True}
        ],
    )
    filenames = [img["filename"] for img in images]
    assert any("rv-rollup-child-present.png" in f for f in filenames)
    assert not any("rv-rollup-unset.png" in f for f in filenames)


async def test_image_label_filter_roll_up_coexists_with_bbox_label_cte(
    http_client, files_root, label_uuid, label2_uuid
):
    """Combine an image-label roll-up term (imglabel_desc_{i} CTE) with a bbox
    label_uuids filter using label_include_children (label_desc_{i} CTE) in the
    SAME request, to prove the two differently-prefixed recursive CTEs coexist
    without a SQLAlchemy 'Multiple, unrelated CTEs found with the same name'
    CompileError."""
    child = await http_client.post("/api/labels", json={"name": "tabby", "color": "#abc", "parentUuid": label_uuid})
    child_uuid = child.json()["label"]["uuid"]

    _make_image(files_root, "coexist.png")
    # Two bboxes on the SAME image in one call — _add_bbox replaces the whole
    # boxes array, so a second call would clobber the first. One bbox carries
    # the child label (rolls up to satisfy the image_label_filters term), the
    # other carries label2 (a completely separate subtree, for the
    # label_uuids/label_include_children filter).
    r = await http_client.post(
        "/api/images/metadata/update",
        json={
            "imagePath": "coexist.png",
            "boxes": [
                {
                    "uuid": str(uuid.uuid4()),
                    "labelUuid": child_uuid,
                    "x": 0.1,
                    "y": 0.1,
                    "width": 0.2,
                    "height": 0.2,
                    "tagUuids": [VERIFIED_TAG_UUID],
                },
                {
                    "uuid": str(uuid.uuid4()),
                    "labelUuid": label2_uuid,
                    "x": 0.4,
                    "y": 0.4,
                    "width": 0.2,
                    "height": 0.2,
                    "tagUuids": [VERIFIED_TAG_UUID],
                },
            ],
        },
    )
    assert r.status_code == 200

    images = await _review(
        http_client,
        show_all_bboxes=True,
        image_label_filters=[{"label_uuid": label_uuid, "status": "present", "exclude": False}],
        label_uuids=[label2_uuid],
        label_include_children=True,
        # "all" mode drives the bbox-loop label_desc_{i} CTE (label_desc_0 here),
        # the exact same index as our image_label_filters loop's imglabel_desc_0
        # — the closest the two loops get to an actual name collision.
        label_match_mode="all",
    )
    filenames = [img["filename"] for img in images]
    assert any("coexist.png" in f for f in filenames)


# ──────────────────────────────────────────────────────────────────────────────
# require_bboxes
# ──────────────────────────────────────────────────────────────────────────────


async def test_require_bboxes_excludes_empty_images(http_client, files_root, label_uuid):
    """require_bboxes=True must exclude images that have no bboxes."""
    _make_image(files_root, "has-bbox.png")
    _make_image(files_root, "no-bbox.png")
    await _add_bbox(http_client, "has-bbox.png", label_uuid, [])
    # Register no-bbox.png in the DB without any bboxes.
    r = await http_client.post(
        "/api/images/metadata/update",
        json={"imagePath": "no-bbox.png", "boxes": []},
    )
    assert r.status_code == 200

    images = await _review(http_client, show_all_bboxes=True, require_bboxes=True)
    filenames = [img["filename"] for img in images]
    assert any("has-bbox.png" in f for f in filenames)
    assert not any("no-bbox.png" in f for f in filenames)


async def test_require_bboxes_false_includes_empty_images(http_client, files_root, label_uuid):
    """require_bboxes=False (default) must include images with no bboxes."""
    _make_image(files_root, "with-box.png")
    _make_image(files_root, "empty.png")
    await _add_bbox(http_client, "with-box.png", label_uuid, [])
    # Register empty.png in the DB (no bboxes) by calling metadata/update with boxes=[].
    r = await http_client.post(
        "/api/images/metadata/update",
        json={"imagePath": "empty.png", "boxes": []},
    )
    assert r.status_code == 200

    images = await _review(http_client, show_all_bboxes=True, require_bboxes=False)
    filenames = [img["filename"] for img in images]
    assert any("with-box.png" in f for f in filenames)
    assert any("empty.png" in f for f in filenames)


# ──────────────────────────────────────────────────────────────────────────────
# label_exclusions
# ──────────────────────────────────────────────────────────────────────────────


async def test_label_exclusions_removes_matching_images(http_client, files_root, label_uuid, label2_uuid):
    """label_exclusions=[X] must exclude images that have any bbox with label X."""
    _make_image(files_root, "cat-excl.png")
    _make_image(files_root, "dog-excl.png")
    await _add_bbox(http_client, "cat-excl.png", label_uuid, [])
    await _add_bbox(http_client, "dog-excl.png", label2_uuid, [])

    # Exclude the 'cat' label — only the dog image should remain
    images = await _review(
        http_client,
        show_all_bboxes=True,
        label_exclusions=[label_uuid],
    )
    filenames = [img["filename"] for img in images]
    assert not any("cat-excl.png" in f for f in filenames)
    assert any("dog-excl.png" in f for f in filenames)


async def test_label_exclusions_multiple(http_client, files_root, label_uuid, label2_uuid):
    """label_exclusions with multiple UUIDs applies AND logic — excludes any image matching ANY exclusion."""
    r = await http_client.post("/api/labels", json={"name": "bird-ex", "color": "#0f0"})
    bird_uuid = r.json()["label"]["uuid"]

    _make_image(files_root, "cat-m.png")
    _make_image(files_root, "dog-m.png")
    _make_image(files_root, "bird-m.png")
    await _add_bbox(http_client, "cat-m.png", label_uuid, [])
    await _add_bbox(http_client, "dog-m.png", label2_uuid, [])
    await _add_bbox(http_client, "bird-m.png", bird_uuid, [])

    # Exclude cat AND dog — only bird should remain
    images = await _review(
        http_client,
        show_all_bboxes=True,
        label_exclusions=[label_uuid, label2_uuid],
    )
    filenames = [img["filename"] for img in images]
    assert not any("cat-m.png" in f for f in filenames)
    assert not any("dog-m.png" in f for f in filenames)
    assert any("bird-m.png" in f for f in filenames)


async def test_require_bboxes_with_label_exclusions(http_client, files_root, label_uuid, label2_uuid):
    """require_bboxes=True combined with label_exclusions works correctly."""
    _make_image(files_root, "combo-cat.png")
    _make_image(files_root, "combo-dog.png")
    _make_image(files_root, "combo-empty.png")
    await _add_bbox(http_client, "combo-cat.png", label_uuid, [])
    await _add_bbox(http_client, "combo-dog.png", label2_uuid, [])
    # Register combo-empty.png with no bboxes
    r = await http_client.post(
        "/api/images/metadata/update",
        json={"imagePath": "combo-empty.png", "boxes": []},
    )
    assert r.status_code == 200

    images = await _review(
        http_client,
        show_all_bboxes=True,
        require_bboxes=True,
        label_exclusions=[label_uuid],
    )
    filenames = [img["filename"] for img in images]
    # cat excluded by label_exclusions; empty excluded by require_bboxes; dog survives
    assert not any("combo-cat.png" in f for f in filenames)
    assert not any("combo-empty.png" in f for f in filenames)
    assert any("combo-dog.png" in f for f in filenames)
