"""Tests for the discriminated-union filter-term model (PR 1.4a).

`ImageSearchParams.terms` is additive — it sits alongside the existing flat
filter fields exercised in test_review.py. These tests cover:

- Round-trip serialisation, including the `from`/`from_` field naming and the
  `kind` discriminator (including rejection of an unknown kind).
- Each of the 9 term kinds evaluating correctly (positive + negative).
- `negate: true` inverting every kind.
- Multiple terms ANDing together.
- match_mode "any" vs "all" on objectLabel / bboxTag.
- imageLabel roll-up (present matches child; absent ignores roll_up).
- The term-loop CTEs coexisting with both pre-existing recursive-CTE loops in
  one statement (see db_client._label_uuids_with_descendants docstring).
- require_bboxes interacting correctly with a bbox-implying term.
"""

import uuid

import pytest

from apps.helpers.db.types import (
    UNVERIFIED_TAG_UUID,
    VERIFIED_TAG_UUID,
    DateTerm,
    ImageSearchParams,
)
from tests.conftest import make_png

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


@pytest.fixture
async def tag_uuid(http_client):
    r = await http_client.post("/api/tags/add", json={"name": "generic-a", "color": "#111111"})
    return r.json()["tag"]["uuid"]


@pytest.fixture
async def tag2_uuid(http_client):
    r = await http_client.post("/api/tags/add", json={"name": "generic-b", "color": "#222222"})
    return r.json()["tag"]["uuid"]


def _make_image(files_root, name: str) -> str:
    p = files_root / name
    p.write_bytes(make_png())
    return name


async def _add_bbox(http_client, image_path: str, label_uuid: str, tag_uuids: list) -> str:
    """Create/replace bboxes on an image with a single bbox; return its uuid."""
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


async def _add_bbox_with_extra(http_client, image_path: str, label_uuid: str, extra, tag_uuids: list = None) -> str:
    """Like _add_bbox, but carrying a metadata_json payload (`extra` on the wire).

    `extra=None` means "no metadata at all" — the unscored-box case.

    The metadata goes on via PATCH rather than the create call because
    BoundingBoxMetadataModel.extra is a Dict[str, str] (the canvas save path),
    which would reject the float confidences the task/preview-run paths write
    straight through db_client. Both shapes are real rows in a live DB, and this
    is the only way to produce the numeric one over HTTP.
    """
    bid = await _add_bbox(http_client, image_path, label_uuid, tag_uuids or [])
    if extra is not None:
        r = await http_client.patch(
            f"/api/images/bbox/{bid}",
            json={"tagUuids": tag_uuids or [], "metadataJson": extra},
        )
        assert r.status_code == 204
    return bid


async def _add_two_bboxes(http_client, image_path: str, first: tuple, second: tuple) -> None:
    """first/second are (label_uuid, tag_uuids) pairs; both land on the SAME image."""
    r = await http_client.post(
        "/api/images/metadata/update",
        json={
            "imagePath": image_path,
            "boxes": [
                {
                    "uuid": str(uuid.uuid4()),
                    "labelUuid": lbl,
                    "x": x,
                    "y": 0.1,
                    "width": 0.2,
                    "height": 0.2,
                    "tagUuids": tags,
                }
                for (lbl, tags), x in zip((first, second), (0.1, 0.5))
            ],
        },
    )
    assert r.status_code == 200


async def _register_image(http_client, image_path: str) -> None:
    r = await http_client.post("/api/images/metadata/update", json={"imagePath": image_path, "boxes": []})
    assert r.status_code == 200


async def _set_image_label(http_client, image_path, label_uuid, status):
    r = await http_client.post(
        "/api/images/labels/set",
        json={"imagePath": image_path, "labelUuid": label_uuid, "status": status},
    )
    assert r.status_code == 200
    assert r.json()["status"] == "success"


async def _set_image_tag_status(http_client, image_path, tag_uuid, status):
    r = await http_client.post(
        "/api/images/tags/set-status",
        json={"imagePath": image_path, "tagUuid": tag_uuid, "status": status},
    )
    assert r.status_code == 200
    assert r.json()["status"] == "success"


async def _review(http_client, **params):
    r = await http_client.post("/api/images/review", json=params)
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "success"
    return body["images"]


def _filenames(images):
    return [img["filename"] for img in images]


# ──────────────────────────────────────────────────────────────────────────────
# Round-trip serialisation
# ──────────────────────────────────────────────────────────────────────────────


class TestRoundTrip:
    def test_from_keyword_round_trips_through_alias(self):
        """`from` is a Python keyword; the field is `from_` in Python but must
        serialise/deserialise as "from" over the wire (JSON key), and still be
        constructible from the Python name via populate_by_name."""
        term = DateTerm(field="created", **{"from": "2020-01-01T00:00:00", "to": "2020-01-02T00:00:00"})
        dumped = term.model_dump(by_alias=True, mode="json")
        assert dumped["from"] == "2020-01-01T00:00:00"
        assert "from_" not in dumped

        # Round trip: dump -> parse -> dump again is stable.
        reparsed = DateTerm.model_validate(dumped)
        assert reparsed.from_ == term.from_
        assert reparsed.model_dump(by_alias=True, mode="json") == dumped

        # populate_by_name still accepts the Python attribute name.
        via_name = DateTerm(field="created", from_="2020-01-01T00:00:00")
        assert via_name.from_ == term.from_

    def test_terms_list_round_trips_all_kinds(self):
        raw_terms = [
            {"kind": "objectLabel", "labelUuids": ["a", "b"], "matchMode": "all", "rollUp": False},
            {"kind": "imageLabel", "labelUuid": "a", "state": "present"},
            {"kind": "bboxTag", "tagUuids": ["t1"], "matchMode": "any"},
            {"kind": "imageTag", "tagUuid": "t1", "state": "unreviewed"},
            {"kind": "bboxVerification", "state": "verified"},
            {"kind": "bboxConfidence", "min": 0.25, "max": 0.75, "includeMissing": True},
            {"kind": "labelVerification", "check": "all_verified", "labelUuids": []},
            {"kind": "date", "field": "updated", "from": "2020-01-01T00:00:00"},
            {"kind": "filename", "patterns": ["foo", "bar"]},
        ]
        params = ImageSearchParams(terms=raw_terms)
        assert [t.kind for t in params.terms] == [
            "objectLabel",
            "imageLabel",
            "bboxTag",
            "imageTag",
            "bboxVerification",
            "bboxConfidence",
            "labelVerification",
            "date",
            "filename",
        ]
        dumped = params.model_dump(by_alias=True, mode="json")
        reparsed = ImageSearchParams.model_validate(dumped)
        assert [t.kind for t in reparsed.terms] == [t.kind for t in params.terms]
        assert dumped["terms"][7]["from"] == "2020-01-01T00:00:00"

    def test_unknown_kind_rejected(self):
        with pytest.raises(Exception):
            ImageSearchParams(terms=[{"kind": "bogus"}])

    async def test_unknown_kind_rejected_via_api(self, http_client):
        r = await http_client.post("/api/images/review", json={"terms": [{"kind": "bogus"}]})
        assert r.status_code == 422

    async def test_terms_round_trip_via_review_endpoint(self, http_client, files_root):
        """A terms-bearing request is accepted and evaluated by the live endpoint
        (not just parsed) — end-to-end proof the API boundary round-trips."""
        _make_image(files_root, "rt.png")
        await _register_image(http_client, "rt.png")
        images = await _review(
            http_client,
            terms=[{"kind": "filename", "patterns": ["rt"]}],
        )
        assert any("rt.png" in f for f in _filenames(images))


# ──────────────────────────────────────────────────────────────────────────────
# objectLabel
# ──────────────────────────────────────────────────────────────────────────────


class TestObjectLabelTerm:
    async def test_matches_image_with_labeled_bbox(self, http_client, files_root, label_uuid, label2_uuid):
        _make_image(files_root, "has-cat.png")
        _make_image(files_root, "has-dog.png")
        await _add_bbox(http_client, "has-cat.png", label_uuid, [])
        await _add_bbox(http_client, "has-dog.png", label2_uuid, [])

        images = await _review(
            http_client, show_all_bboxes=True, terms=[{"kind": "objectLabel", "labelUuids": [label_uuid]}]
        )
        filenames = _filenames(images)
        assert any("has-cat.png" in f for f in filenames)
        assert not any("has-dog.png" in f for f in filenames)

    async def test_negate_inverts(self, http_client, files_root, label_uuid, label2_uuid):
        _make_image(files_root, "has-cat.png")
        _make_image(files_root, "has-dog.png")
        await _add_bbox(http_client, "has-cat.png", label_uuid, [])
        await _add_bbox(http_client, "has-dog.png", label2_uuid, [])

        images = await _review(
            http_client,
            show_all_bboxes=True,
            terms=[{"kind": "objectLabel", "labelUuids": [label_uuid], "negate": True}],
        )
        filenames = _filenames(images)
        assert not any("has-cat.png" in f for f in filenames)
        assert any("has-dog.png" in f for f in filenames)

    async def test_match_mode_any_vs_all(self, http_client, files_root, label_uuid, label2_uuid):
        _make_image(files_root, "both.png")
        _make_image(files_root, "cat-only.png")
        await _add_two_bboxes(http_client, "both.png", (label_uuid, []), (label2_uuid, []))
        await _add_bbox(http_client, "cat-only.png", label_uuid, [])

        any_images = _filenames(
            await _review(
                http_client,
                show_all_bboxes=True,
                terms=[{"kind": "objectLabel", "labelUuids": [label_uuid, label2_uuid], "matchMode": "any"}],
            )
        )
        assert any("both.png" in f for f in any_images)
        assert any("cat-only.png" in f for f in any_images)

        all_images = _filenames(
            await _review(
                http_client,
                show_all_bboxes=True,
                terms=[{"kind": "objectLabel", "labelUuids": [label_uuid, label2_uuid], "matchMode": "all"}],
            )
        )
        assert any("both.png" in f for f in all_images)
        assert not any("cat-only.png" in f for f in all_images)

    async def test_roll_up_matches_descendant_bbox(self, http_client, files_root, label_uuid):
        child = await http_client.post("/api/labels", json={"name": "tabby", "color": "#abc", "parentUuid": label_uuid})
        child_uuid = child.json()["label"]["uuid"]
        _make_image(files_root, "child-labeled.png")
        await _add_bbox(http_client, "child-labeled.png", child_uuid, [])

        rolled_up = await _review(
            http_client,
            show_all_bboxes=True,
            terms=[{"kind": "objectLabel", "labelUuids": [label_uuid], "rollUp": True}],
        )
        assert any("child-labeled.png" in f for f in _filenames(rolled_up))

        no_roll_up = await _review(
            http_client,
            show_all_bboxes=True,
            terms=[{"kind": "objectLabel", "labelUuids": [label_uuid], "rollUp": False}],
        )
        assert not any("child-labeled.png" in f for f in _filenames(no_roll_up))


# ──────────────────────────────────────────────────────────────────────────────
# imageLabel
# ──────────────────────────────────────────────────────────────────────────────


class TestImageLabelTerm:
    async def test_present_matches_verified_bbox(self, http_client, files_root, label_uuid):
        _make_image(files_root, "present.png")
        await _add_bbox(http_client, "present.png", label_uuid, [VERIFIED_TAG_UUID])

        images = await _review(
            http_client,
            show_all_bboxes=True,
            terms=[{"kind": "imageLabel", "labelUuid": label_uuid, "state": "present"}],
        )
        assert any("present.png" in f for f in _filenames(images))

    async def test_negate_inverts(self, http_client, files_root, label_uuid):
        _make_image(files_root, "present.png")
        _make_image(files_root, "no-row.png")
        await _add_bbox(http_client, "present.png", label_uuid, [VERIFIED_TAG_UUID])
        await _register_image(http_client, "no-row.png")

        images = await _review(
            http_client,
            show_all_bboxes=True,
            terms=[{"kind": "imageLabel", "labelUuid": label_uuid, "state": "present", "negate": True}],
        )
        filenames = _filenames(images)
        assert not any("present.png" in f for f in filenames)
        assert any("no-row.png" in f for f in filenames)

    async def test_unreviewed_and_any_reviewed(self, http_client, files_root, label_uuid):
        _make_image(files_root, "unset.png")
        _make_image(files_root, "assessed.png")
        await _register_image(http_client, "unset.png")
        await _register_image(http_client, "assessed.png")
        await _set_image_label(http_client, "assessed.png", label_uuid, "absent")

        unreviewed = _filenames(
            await _review(
                http_client,
                show_all_bboxes=True,
                terms=[{"kind": "imageLabel", "labelUuid": label_uuid, "state": "unreviewed"}],
            )
        )
        assert any("unset.png" in f for f in unreviewed)
        assert not any("assessed.png" in f for f in unreviewed)

        any_reviewed = _filenames(
            await _review(
                http_client,
                show_all_bboxes=True,
                terms=[{"kind": "imageLabel", "labelUuid": label_uuid, "state": "any_reviewed"}],
            )
        )
        assert any("assessed.png" in f for f in any_reviewed)
        assert not any("unset.png" in f for f in any_reviewed)

    async def test_roll_up_present_matches_child_but_absent_ignores_roll_up(self, http_client, files_root, label_uuid):
        child = await http_client.post("/api/labels", json={"name": "tabby", "color": "#abc", "parentUuid": label_uuid})
        child_uuid = child.json()["label"]["uuid"]

        _make_image(files_root, "child-present.png")
        await _add_bbox(http_client, "child-present.png", child_uuid, [VERIFIED_TAG_UUID])

        present_images = _filenames(
            await _review(
                http_client,
                show_all_bboxes=True,
                terms=[{"kind": "imageLabel", "labelUuid": label_uuid, "state": "present", "rollUp": True}],
            )
        )
        assert any("child-present.png" in f for f in present_images)

        # ABSENT must ignore roll_up even when roll_up=True: explicitly marking
        # the CHILD absent must NOT make the parent-label absent term match.
        _make_image(files_root, "child-absent.png")
        await _register_image(http_client, "child-absent.png")
        await _set_image_label(http_client, "child-absent.png", child_uuid, "absent")

        absent_images = _filenames(
            await _review(
                http_client,
                show_all_bboxes=True,
                terms=[{"kind": "imageLabel", "labelUuid": label_uuid, "state": "absent", "rollUp": True}],
            )
        )
        assert not any("child-absent.png" in f for f in absent_images)

        # But the parent's OWN absent status still matches (roll_up=False behavior
        # applies regardless of the roll_up flag for absent terms).
        _make_image(files_root, "parent-absent.png")
        await _register_image(http_client, "parent-absent.png")
        await _set_image_label(http_client, "parent-absent.png", label_uuid, "absent")
        absent_images2 = _filenames(
            await _review(
                http_client,
                show_all_bboxes=True,
                terms=[{"kind": "imageLabel", "labelUuid": label_uuid, "state": "absent", "rollUp": True}],
            )
        )
        assert any("parent-absent.png" in f for f in absent_images2)


# ──────────────────────────────────────────────────────────────────────────────
# bboxTag
# ──────────────────────────────────────────────────────────────────────────────


class TestBBoxTagTerm:
    async def test_matches_bbox_with_tag(self, http_client, files_root, label_uuid, tag_uuid, tag2_uuid):
        _make_image(files_root, "has-tag.png")
        _make_image(files_root, "no-tag.png")
        await _add_bbox(http_client, "has-tag.png", label_uuid, [tag_uuid])
        await _add_bbox(http_client, "no-tag.png", label_uuid, [tag2_uuid])

        images = await _review(http_client, show_all_bboxes=True, terms=[{"kind": "bboxTag", "tagUuids": [tag_uuid]}])
        filenames = _filenames(images)
        assert any("has-tag.png" in f for f in filenames)
        assert not any("no-tag.png" in f for f in filenames)

    async def test_negate_inverts(self, http_client, files_root, label_uuid, tag_uuid, tag2_uuid):
        _make_image(files_root, "has-tag.png")
        _make_image(files_root, "no-tag.png")
        await _add_bbox(http_client, "has-tag.png", label_uuid, [tag_uuid])
        await _add_bbox(http_client, "no-tag.png", label_uuid, [tag2_uuid])

        images = await _review(
            http_client,
            show_all_bboxes=True,
            terms=[{"kind": "bboxTag", "tagUuids": [tag_uuid], "negate": True}],
        )
        filenames = _filenames(images)
        assert not any("has-tag.png" in f for f in filenames)
        assert any("no-tag.png" in f for f in filenames)

    async def test_match_mode_any_vs_all(self, http_client, files_root, label_uuid, tag_uuid, tag2_uuid):
        _make_image(files_root, "both-tags.png")
        _make_image(files_root, "one-tag.png")
        await _add_two_bboxes(http_client, "both-tags.png", (label_uuid, [tag_uuid]), (label_uuid, [tag2_uuid]))
        await _add_bbox(http_client, "one-tag.png", label_uuid, [tag_uuid])

        any_images = _filenames(
            await _review(
                http_client,
                show_all_bboxes=True,
                terms=[{"kind": "bboxTag", "tagUuids": [tag_uuid, tag2_uuid], "matchMode": "any"}],
            )
        )
        assert any("both-tags.png" in f for f in any_images)
        assert any("one-tag.png" in f for f in any_images)

        all_images = _filenames(
            await _review(
                http_client,
                show_all_bboxes=True,
                terms=[{"kind": "bboxTag", "tagUuids": [tag_uuid, tag2_uuid], "matchMode": "all"}],
            )
        )
        assert any("both-tags.png" in f for f in all_images)
        assert not any("one-tag.png" in f for f in all_images)


# ──────────────────────────────────────────────────────────────────────────────
# imageTag
# ──────────────────────────────────────────────────────────────────────────────


class TestImageTagTerm:
    async def test_present_and_negate(self, http_client, files_root, tag_uuid):
        _make_image(files_root, "tag-present.png")
        _make_image(files_root, "tag-absent.png")
        await _register_image(http_client, "tag-present.png")
        await _register_image(http_client, "tag-absent.png")
        await _set_image_tag_status(http_client, "tag-present.png", tag_uuid, "present")
        await _set_image_tag_status(http_client, "tag-absent.png", tag_uuid, "absent")

        present = _filenames(
            await _review(
                http_client,
                show_all_bboxes=True,
                terms=[{"kind": "imageTag", "tagUuid": tag_uuid, "state": "present"}],
            )
        )
        assert any("tag-present.png" in f for f in present)
        assert not any("tag-absent.png" in f for f in present)

        negated = _filenames(
            await _review(
                http_client,
                show_all_bboxes=True,
                terms=[{"kind": "imageTag", "tagUuid": tag_uuid, "state": "present", "negate": True}],
            )
        )
        assert not any("tag-present.png" in f for f in negated)
        assert any("tag-absent.png" in f for f in negated)

    async def test_unreviewed_and_any_reviewed(self, http_client, files_root, tag_uuid):
        _make_image(files_root, "unset.png")
        _make_image(files_root, "assessed.png")
        await _register_image(http_client, "unset.png")
        await _register_image(http_client, "assessed.png")
        await _set_image_tag_status(http_client, "assessed.png", tag_uuid, "absent")

        unreviewed = _filenames(
            await _review(
                http_client,
                show_all_bboxes=True,
                terms=[{"kind": "imageTag", "tagUuid": tag_uuid, "state": "unreviewed"}],
            )
        )
        assert any("unset.png" in f for f in unreviewed)
        assert not any("assessed.png" in f for f in unreviewed)

        any_reviewed = _filenames(
            await _review(
                http_client,
                show_all_bboxes=True,
                terms=[{"kind": "imageTag", "tagUuid": tag_uuid, "state": "any_reviewed"}],
            )
        )
        assert any("assessed.png" in f for f in any_reviewed)
        assert not any("unset.png" in f for f in any_reviewed)


# ──────────────────────────────────────────────────────────────────────────────
# imageSource
# ──────────────────────────────────────────────────────────────────────────────


@pytest.fixture
async def source_uuids(http_client):
    """Two sources, so "only camera A" can be told apart from "any source"."""
    made = []
    for name in ("cam-a", "cam-b"):
        r = await http_client.post(
            "/api/sources/create",
            json={
                "sourceName": name,
                "imageProvider": "UrlImageProvider",
                "providerParams": {"url": f"http://example.com/{name}.jpg"},
            },
        )
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "success", body
        made.append(body["source"]["uuid"])
    return made


async def _register_image_from(http_client, image_path: str, source_uuid) -> None:
    r = await http_client.post(
        "/api/images/metadata/update",
        json={"imagePath": image_path, "boxes": [], "sourceUuid": source_uuid},
    )
    assert r.status_code == 200


class TestImageSourceTerm:
    async def test_selects_only_the_named_source(self, http_client, files_root, source_uuids):
        cam_a, cam_b = source_uuids
        for name in ("from-a.png", "from-b.png", "from-nowhere.png"):
            _make_image(files_root, name)
        await _register_image_from(http_client, "from-a.png", cam_a)
        await _register_image_from(http_client, "from-b.png", cam_b)
        await _register_image(http_client, "from-nowhere.png")

        only_a = _filenames(
            await _review(
                http_client,
                show_all_bboxes=True,
                terms=[{"kind": "imageSource", "sourceUuids": [cam_a]}],
            )
        )
        assert any("from-a.png" in f for f in only_a)
        assert not any("from-b.png" in f for f in only_a)
        assert not any("from-nowhere.png" in f for f in only_a)

        both = _filenames(
            await _review(
                http_client,
                show_all_bboxes=True,
                terms=[{"kind": "imageSource", "sourceUuids": [cam_a, cam_b]}],
            )
        )
        assert any("from-a.png" in f for f in both)
        assert any("from-b.png" in f for f in both)
        assert not any("from-nowhere.png" in f for f in both)

    async def test_include_unknown_reaches_sourceless_images(self, http_client, files_root, source_uuids):
        """NULL never satisfies IN, so images with no source are unreachable
        without the flag — and reachable with it, alongside a named source."""
        cam_a, _ = source_uuids
        for name in ("known.png", "orphan.png"):
            _make_image(files_root, name)
        await _register_image_from(http_client, "known.png", cam_a)
        await _register_image(http_client, "orphan.png")

        without = _filenames(
            await _review(
                http_client,
                show_all_bboxes=True,
                terms=[{"kind": "imageSource", "sourceUuids": [cam_a]}],
            )
        )
        assert not any("orphan.png" in f for f in without)

        unknown_only = _filenames(
            await _review(
                http_client,
                show_all_bboxes=True,
                terms=[{"kind": "imageSource", "sourceUuids": [], "includeUnknown": True}],
            )
        )
        assert any("orphan.png" in f for f in unknown_only)
        assert not any("known.png" in f for f in unknown_only)

        both = _filenames(
            await _review(
                http_client,
                show_all_bboxes=True,
                terms=[{"kind": "imageSource", "sourceUuids": [cam_a], "includeUnknown": True}],
            )
        )
        assert any("orphan.png" in f for f in both)
        assert any("known.png" in f for f in both)

    async def test_negate_excludes_that_source_but_keeps_sourceless(self, http_client, files_root, source_uuids):
        """A negated term is SQL NOT, and `source_uuid NOT IN (...)` is NULL —
        not true — for a sourceless image. "Not camera A" has to keep them."""
        cam_a, cam_b = source_uuids
        for name in ("neg-a.png", "neg-b.png", "neg-none.png"):
            _make_image(files_root, name)
        await _register_image_from(http_client, "neg-a.png", cam_a)
        await _register_image_from(http_client, "neg-b.png", cam_b)
        await _register_image(http_client, "neg-none.png")

        not_a = _filenames(
            await _review(
                http_client,
                show_all_bboxes=True,
                terms=[{"kind": "imageSource", "sourceUuids": [cam_a], "negate": True}],
            )
        )
        assert not any("neg-a.png" in f for f in not_a)
        assert any("neg-b.png" in f for f in not_a)
        assert any("neg-none.png" in f for f in not_a)

    async def test_empty_selection_is_a_no_op(self, http_client, files_root, source_uuids):
        """An empty Source filter must not silently drop every image — the
        panel emits no term at all, but a saved View could still carry one."""
        cam_a, _ = source_uuids
        for name in ("noop-a.png", "noop-none.png"):
            _make_image(files_root, name)
        await _register_image_from(http_client, "noop-a.png", cam_a)
        await _register_image(http_client, "noop-none.png")

        every = _filenames(
            await _review(
                http_client,
                show_all_bboxes=True,
                terms=[{"kind": "imageSource", "sourceUuids": []}],
            )
        )
        assert any("noop-a.png" in f for f in every)
        assert any("noop-none.png" in f for f in every)


# ──────────────────────────────────────────────────────────────────────────────
# bboxVerification
# ──────────────────────────────────────────────────────────────────────────────


class TestBBoxVerificationTerm:
    # Distinct, non-substring basenames per state — "verified.png" is (by
    # design) a literal substring of "unverified.png", so filename-based
    # assertions must never use overlapping name pairs.
    _STATE_FILES = {
        "verified": "state-ok.png",
        "unverified": "state-pending.png",
    }
    _STATE_TAGS = {
        "verified": VERIFIED_TAG_UUID,
        "unverified": UNVERIFIED_TAG_UUID,
    }

    @pytest.mark.parametrize("state", ["verified", "unverified"])
    async def test_matches_state(self, http_client, files_root, label_uuid, state):
        own_file = self._STATE_FILES[state]
        other_state = "verified" if state != "verified" else "unverified"
        other_file = self._STATE_FILES[other_state]
        _make_image(files_root, own_file)
        _make_image(files_root, other_file)
        await _add_bbox(http_client, own_file, label_uuid, [self._STATE_TAGS[state]])
        await _add_bbox(http_client, other_file, label_uuid, [self._STATE_TAGS[other_state]])

        images = _filenames(
            await _review(http_client, show_all_bboxes=True, terms=[{"kind": "bboxVerification", "state": state}])
        )
        assert any(own_file in f for f in images)
        assert not any(other_file in f for f in images)

    async def test_negate_inverts(self, http_client, files_root, label_uuid):
        _make_image(files_root, "state-ok.png")
        _make_image(files_root, "state-pending.png")
        await _add_bbox(http_client, "state-ok.png", label_uuid, [VERIFIED_TAG_UUID])
        await _add_bbox(http_client, "state-pending.png", label_uuid, [UNVERIFIED_TAG_UUID])

        images = _filenames(
            await _review(
                http_client,
                show_all_bboxes=True,
                terms=[{"kind": "bboxVerification", "state": "verified", "negate": True}],
            )
        )
        assert not any("state-ok.png" in f for f in images)
        assert any("state-pending.png" in f for f in images)

    async def test_two_verification_terms_and_within_an_image(self, http_client, files_root, label_uuid):
        """Both states at once means "this image has one of each", not "neither".

        The review UI's bulk-removal guard asks exactly this to find out how much
        human work is inside a delete set it has already scoped to unverified
        boxes: it ANDs a second `verified` term onto the caller's own filter. If
        the two terms collapsed into one bbox-level predicate (a box cannot be
        both) the answer would always be zero and the warning would never fire.
        """
        _make_image(files_root, "state-mixed.png")
        _make_image(files_root, "state-pending.png")
        # One call, not two: _add_bbox REPLACES an image's boxes.
        await _add_two_bboxes(
            http_client,
            "state-mixed.png",
            (label_uuid, [UNVERIFIED_TAG_UUID]),
            (label_uuid, [VERIFIED_TAG_UUID]),
        )
        await _add_bbox(http_client, "state-pending.png", label_uuid, [UNVERIFIED_TAG_UUID])

        images = _filenames(
            await _review(
                http_client,
                show_all_bboxes=True,
                terms=[
                    {"kind": "bboxVerification", "state": "unverified"},
                    {"kind": "bboxVerification", "state": "verified"},
                ],
            )
        )
        assert any("state-mixed.png" in f for f in images)
        assert not any("state-pending.png" in f for f in images)

    async def test_no_boxes_finds_images_no_tag_state_can_reach(self, http_client, files_root, label_uuid):
        """ "Unverified because there is nothing on it" is invisible to the tag states.

        A verification tag needs a box to hang off, so an image with no boxes at all
        matches neither `verified` nor `unverified` — it simply never appears. This
        state is the only way to see them.
        """
        _make_image(files_root, "state-empty.png")
        await _register_image(http_client, "state-empty.png")
        _make_image(files_root, "state-pending.png")
        await _add_bbox(http_client, "state-pending.png", label_uuid, [UNVERIFIED_TAG_UUID])

        images = _filenames(
            await _review(http_client, show_all_bboxes=True, terms=[{"kind": "bboxVerification", "state": "no_boxes"}])
        )
        assert any("state-empty.png" in f for f in images)
        assert not any("state-pending.png" in f for f in images)

        # And the state it is contrasted with really does miss it.
        unverified = _filenames(
            await _review(
                http_client, show_all_bboxes=True, terms=[{"kind": "bboxVerification", "state": "unverified"}]
            )
        )
        assert not any("state-empty.png" in f for f in unverified)

    async def test_no_boxes_does_not_narrow_the_query_to_images_that_have_boxes(
        self, http_client, files_root, label_uuid
    ):
        """The `bbox_implied` optimisation would make this term unsatisfiable.

        Every other verification state proves the image HAS a box, which lets the
        query narrow to those images. This one asks for the opposite, so setting the
        same flag would AND "has a bbox" onto "has no bbox" and return nothing —
        silently, and only for this state.
        """
        _make_image(files_root, "state-empty.png")
        await _register_image(http_client, "state-empty.png")

        images = _filenames(
            await _review(http_client, show_all_bboxes=True, terms=[{"kind": "bboxVerification", "state": "no_boxes"}])
        )
        assert any("state-empty.png" in f for f in images)

    async def test_no_boxes_negated_finds_images_that_have_boxes(self, http_client, files_root, label_uuid):
        _make_image(files_root, "state-empty.png")
        await _register_image(http_client, "state-empty.png")
        _make_image(files_root, "state-pending.png")
        await _add_bbox(http_client, "state-pending.png", label_uuid, [UNVERIFIED_TAG_UUID])

        images = _filenames(
            await _review(
                http_client,
                show_all_bboxes=True,
                terms=[{"kind": "bboxVerification", "state": "no_boxes", "negate": True}],
            )
        )
        assert any("state-pending.png" in f for f in images)
        assert not any("state-empty.png" in f for f in images)


# ──────────────────────────────────────────────────────────────────────────────
# bboxConfidence
# ──────────────────────────────────────────────────────────────────────────────


class TestBBoxConfidenceTerm:
    """Confidence lives in bboxes.metadata_json, under either of two keys, as
    either a float or a string — every one of those combinations has to filter.
    """

    async def _seed(self, http_client, files_root, label_uuid):
        """One image per confidence flavour; returns their basenames."""
        rows = {
            "conf-low.png": {"confidence": 0.15},
            "conf-mid.png": {"confidence": 0.55},
            "conf-high.png": {"confidence": 0.95},
            # The canvas writes String(detection.confidence) — same band, different
            # JSON type. See _bbox_confidence_expr's CAST.
            "conf-string.png": {"confidence": "0.60"},
            # ApplyFilterChainTask's add_bbox step writes this key instead.
            "conf-source-key.png": {"source_confidence": 0.65},
            # A hand-drawn box: no confidence at all.
            "conf-none.png": None,
        }
        for name, extra in rows.items():
            _make_image(files_root, name)
            await _add_bbox_with_extra(http_client, name, label_uuid, extra)
        return rows

    async def test_band_matches_both_keys_and_both_json_types(self, http_client, files_root, label_uuid):
        await self._seed(http_client, files_root, label_uuid)

        images = _filenames(
            await _review(
                http_client,
                show_all_bboxes=True,
                terms=[{"kind": "bboxConfidence", "min": 0.5, "max": 0.7}],
            )
        )
        assert any("conf-mid.png" in f for f in images)
        assert any("conf-string.png" in f for f in images)
        assert any("conf-source-key.png" in f for f in images)
        assert not any("conf-low.png" in f for f in images)
        assert not any("conf-high.png" in f for f in images)
        # An absent confidence is not 0.0 and must not fall into any band.
        assert not any("conf-none.png" in f for f in images)

    async def test_bounds_are_independently_optional_and_inclusive(self, http_client, files_root, label_uuid):
        await self._seed(http_client, files_root, label_uuid)

        min_only = _filenames(
            await _review(http_client, show_all_bboxes=True, terms=[{"kind": "bboxConfidence", "min": 0.95}])
        )
        assert any("conf-high.png" in f for f in min_only)  # inclusive on the bound itself
        assert not any("conf-mid.png" in f for f in min_only)

        max_only = _filenames(
            await _review(http_client, show_all_bboxes=True, terms=[{"kind": "bboxConfidence", "max": 0.15}])
        )
        assert any("conf-low.png" in f for f in max_only)
        assert not any("conf-mid.png" in f for f in max_only)

    async def test_include_missing_opts_unscored_boxes_in(self, http_client, files_root, label_uuid):
        await self._seed(http_client, files_root, label_uuid)

        images = _filenames(
            await _review(
                http_client,
                show_all_bboxes=True,
                terms=[{"kind": "bboxConfidence", "min": 0.5, "max": 0.7, "includeMissing": True}],
            )
        )
        assert any("conf-none.png" in f for f in images)
        assert any("conf-mid.png" in f for f in images)
        assert not any("conf-low.png" in f for f in images)

    async def test_no_bounds_means_has_any_confidence(self, http_client, files_root, label_uuid):
        await self._seed(http_client, files_root, label_uuid)

        images = _filenames(await _review(http_client, show_all_bboxes=True, terms=[{"kind": "bboxConfidence"}]))
        assert any("conf-mid.png" in f for f in images)
        assert not any("conf-none.png" in f for f in images)

    async def test_negate_inverts(self, http_client, files_root, label_uuid):
        await self._seed(http_client, files_root, label_uuid)

        images = _filenames(
            await _review(
                http_client,
                show_all_bboxes=True,
                terms=[{"kind": "bboxConfidence", "min": 0.5, "max": 0.7, "negate": True}],
            )
        )
        assert not any("conf-mid.png" in f for f in images)
        assert any("conf-low.png" in f for f in images)
        # Negation is "has no bbox in the band", so an unscored box passes.
        assert any("conf-none.png" in f for f in images)

    async def test_ands_with_bbox_verification(self, http_client, files_root, label_uuid):
        """The headline use case: low-confidence boxes that still need review."""
        _make_image(files_root, "pending-low.png")
        _make_image(files_root, "pending-high.png")
        _make_image(files_root, "ok-low.png")
        await _add_bbox_with_extra(
            http_client, "pending-low.png", label_uuid, {"confidence": 0.2}, [UNVERIFIED_TAG_UUID]
        )
        await _add_bbox_with_extra(
            http_client, "pending-high.png", label_uuid, {"confidence": 0.9}, [UNVERIFIED_TAG_UUID]
        )
        await _add_bbox_with_extra(http_client, "ok-low.png", label_uuid, {"confidence": 0.2}, [VERIFIED_TAG_UUID])

        images = _filenames(
            await _review(
                http_client,
                show_all_bboxes=True,
                terms=[
                    {"kind": "bboxVerification", "state": "unverified"},
                    {"kind": "bboxConfidence", "max": 0.5},
                ],
            )
        )
        assert any("pending-low.png" in f for f in images)
        assert not any("pending-high.png" in f for f in images)
        assert not any("ok-low.png" in f for f in images)


# ──────────────────────────────────────────────────────────────────────────────
# labelVerification
# ──────────────────────────────────────────────────────────────────────────────


class TestLabelVerificationTerm:
    async def test_all_verified_and_has_unverified(self, http_client, files_root, label_uuid):
        _make_image(files_root, "resolved-image.png")
        _make_image(files_root, "unchecked-image.png")
        await _register_image(http_client, "resolved-image.png")
        await _register_image(http_client, "unchecked-image.png")
        await _set_image_label(http_client, "resolved-image.png", label_uuid, "absent")
        # unchecked-image.png: no assessment row at all for label_uuid.

        all_verified = _filenames(
            await _review(
                http_client,
                show_all_bboxes=True,
                terms=[{"kind": "labelVerification", "check": "all_verified", "labelUuids": [label_uuid]}],
            )
        )
        assert any("resolved-image.png" in f for f in all_verified)
        assert not any("unchecked-image.png" in f for f in all_verified)

        has_unverified = _filenames(
            await _review(
                http_client,
                show_all_bboxes=True,
                terms=[{"kind": "labelVerification", "check": "has_unverified", "labelUuids": [label_uuid]}],
            )
        )
        assert not any("resolved-image.png" in f for f in has_unverified)
        assert any("unchecked-image.png" in f for f in has_unverified)

    async def test_negate_inverts(self, http_client, files_root, label_uuid):
        _make_image(files_root, "resolved-image.png")
        await _register_image(http_client, "resolved-image.png")
        await _set_image_label(http_client, "resolved-image.png", label_uuid, "absent")

        negated = _filenames(
            await _review(
                http_client,
                show_all_bboxes=True,
                terms=[
                    {
                        "kind": "labelVerification",
                        "check": "all_verified",
                        "labelUuids": [label_uuid],
                        "negate": True,
                    }
                ],
            )
        )
        assert not any("resolved-image.png" in f for f in negated)

    async def test_empty_label_uuids_falls_back_to_query_label_uuids(self, http_client, files_root, label_uuid):
        """Empty labelUuids on the term resolves against the flat
        ImageSearchParams.label_uuids field (today's verification_label_uuids ??
        label_uuids fallback), NOT against an empty/always-vacuous set."""
        _make_image(files_root, "resolved-image.png")
        # The flat label_uuids field is itself also a bbox-pool filter (image
        # must have a bbox with this label) — give it one so only the
        # verification predicate is under test here.
        await _add_bbox(http_client, "resolved-image.png", label_uuid, [])
        await _set_image_label(http_client, "resolved-image.png", label_uuid, "absent")

        images = _filenames(
            await _review(
                http_client,
                show_all_bboxes=True,
                label_uuids=[label_uuid],
                terms=[{"kind": "labelVerification", "check": "all_verified", "labelUuids": []}],
            )
        )
        assert any("resolved-image.png" in f for f in images)


# ──────────────────────────────────────────────────────────────────────────────
# date
# ──────────────────────────────────────────────────────────────────────────────


class TestDateTerm:
    async def test_from_to_range_and_negate(self, http_client, files_root):
        _make_image(files_root, "in-range.png")
        await _register_image(http_client, "in-range.png")

        # Wide range comfortably containing "now".
        images = _filenames(
            await _review(
                http_client,
                show_all_bboxes=True,
                terms=[
                    {
                        "kind": "date",
                        "field": "created",
                        "from": "2000-01-01T00:00:00",
                        "to": "2100-01-01T00:00:00",
                    }
                ],
            )
        )
        assert any("in-range.png" in f for f in images)

        # A range entirely in the past excludes it.
        excluded = _filenames(
            await _review(
                http_client,
                show_all_bboxes=True,
                terms=[
                    {
                        "kind": "date",
                        "field": "created",
                        "from": "1990-01-01T00:00:00",
                        "to": "1999-01-01T00:00:00",
                    }
                ],
            )
        )
        assert not any("in-range.png" in f for f in excluded)

        # negate: true on a wide (matching) range excludes the image.
        negated = _filenames(
            await _review(
                http_client,
                show_all_bboxes=True,
                terms=[
                    {
                        "kind": "date",
                        "field": "created",
                        "from": "2000-01-01T00:00:00",
                        "to": "2100-01-01T00:00:00",
                        "negate": True,
                    }
                ],
            )
        )
        assert not any("in-range.png" in f for f in negated)


# ──────────────────────────────────────────────────────────────────────────────
# filename
# ──────────────────────────────────────────────────────────────────────────────


class TestFilenameTerm:
    async def test_and_of_patterns_and_negate(self, http_client, files_root):
        _make_image(files_root, "alpha-beta.png")
        _make_image(files_root, "alpha-only.png")
        await _register_image(http_client, "alpha-beta.png")
        await _register_image(http_client, "alpha-only.png")

        images = _filenames(
            await _review(
                http_client,
                show_all_bboxes=True,
                terms=[{"kind": "filename", "patterns": ["alpha", "beta"]}],
            )
        )
        assert any("alpha-beta.png" in f for f in images)
        assert not any("alpha-only.png" in f for f in images)

        negated = _filenames(
            await _review(
                http_client,
                show_all_bboxes=True,
                terms=[{"kind": "filename", "patterns": ["alpha", "beta"], "negate": True}],
            )
        )
        assert not any("alpha-beta.png" in f for f in negated)
        assert any("alpha-only.png" in f for f in negated)


# ──────────────────────────────────────────────────────────────────────────────
# Multiple terms AND together
# ──────────────────────────────────────────────────────────────────────────────


class TestMultipleTermsAnd:
    async def test_two_terms_and_together(self, http_client, files_root, label_uuid, label2_uuid, tag_uuid):
        _make_image(files_root, "matches-both.png")
        _make_image(files_root, "matches-label-only.png")
        await _add_bbox(http_client, "matches-both.png", label_uuid, [tag_uuid])
        await _add_bbox(http_client, "matches-label-only.png", label_uuid, [])

        images = _filenames(
            await _review(
                http_client,
                show_all_bboxes=True,
                terms=[
                    {"kind": "objectLabel", "labelUuids": [label_uuid]},
                    {"kind": "bboxTag", "tagUuids": [tag_uuid]},
                ],
            )
        )
        assert any("matches-both.png" in f for f in images)
        assert not any("matches-label-only.png" in f for f in images)


# ──────────────────────────────────────────────────────────────────────────────
# CTE coexistence — trap #1: distinct cte_name prefixes must not collide.
# ──────────────────────────────────────────────────────────────────────────────


class TestCteCoexistence:
    async def test_term_ctes_coexist_with_both_existing_recursive_cte_loops(
        self, http_client, files_root, label_uuid, label2_uuid
    ):
        """One request combines THREE separate recursive-descendant-CTE code
        paths: the objectLabel term (term_desc_{i}), the image_label_filters
        loop (imglabel_desc_{i}), and the label_uuids/label_include_children
        "all" mode bbox loop (label_desc_{i}). If any two reused the same CTE
        name, SQLAlchemy would raise CompileError: 'Multiple, unrelated CTEs
        found with the same name.'"""
        child = await http_client.post("/api/labels", json={"name": "tabby", "color": "#abc", "parentUuid": label_uuid})
        child_uuid = child.json()["label"]["uuid"]
        child2 = await http_client.post("/api/labels", json={"name": "rex", "color": "#def", "parentUuid": label2_uuid})
        child2_uuid = child2.json()["label"]["uuid"]

        _make_image(files_root, "coexist.png")
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
                        "labelUuid": child2_uuid,
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
            label_match_mode="all",
            terms=[{"kind": "objectLabel", "labelUuids": [label_uuid], "rollUp": True}],
        )
        filenames = _filenames(images)
        assert any("coexist.png" in f for f in filenames)


# ──────────────────────────────────────────────────────────────────────────────
# require_bboxes interacting with a bbox-implying term
# ──────────────────────────────────────────────────────────────────────────────


class TestRequireBboxesInteraction:
    async def test_bbox_implying_term_satisfies_require_bboxes(self, http_client, files_root, label_uuid):
        _make_image(files_root, "has-bbox.png")
        _make_image(files_root, "no-bbox.png")
        await _add_bbox(http_client, "has-bbox.png", label_uuid, [])
        await _register_image(http_client, "no-bbox.png")

        images = _filenames(
            await _review(
                http_client,
                show_all_bboxes=True,
                require_bboxes=True,
                terms=[{"kind": "objectLabel", "labelUuids": [label_uuid]}],
            )
        )
        assert any("has-bbox.png" in f for f in images)
        assert not any("no-bbox.png" in f for f in images)

    async def test_negated_bbox_implying_term_does_not_short_circuit_require_bboxes(
        self, http_client, files_root, label_uuid, label2_uuid
    ):
        """A negated bbox-implying term ("no bbox with this label") is satisfied
        by an image with NO bboxes at all — so it must not set the internal
        _bbox_implied flag, or require_bboxes=True would wrongly skip its own
        subquery and let zero-bbox images through."""
        _make_image(files_root, "no-bbox.png")
        _make_image(files_root, "other-label.png")
        await _register_image(http_client, "no-bbox.png")
        await _add_bbox(http_client, "other-label.png", label2_uuid, [])

        images = _filenames(
            await _review(
                http_client,
                show_all_bboxes=True,
                require_bboxes=True,
                terms=[{"kind": "objectLabel", "labelUuids": [label_uuid], "negate": True}],
            )
        )
        # no-bbox.png does NOT have a label_uuid bbox, so the negated term is
        # satisfied, but require_bboxes=True must still exclude it for having
        # zero bboxes.
        assert not any("no-bbox.png" in f for f in images)
        assert any("other-label.png" in f for f in images)

    async def test_empty_label_uuids_all_mode_does_not_short_circuit_require_bboxes(
        self, http_client, files_root, label_uuid
    ):
        """An objectLabel term with match_mode="all" and an empty label_uuids
        list builds a vacuously-true and_() expression (matches every image,
        bbox or not) and must NOT set the internal bbox_implied flag, or
        require_bboxes=True would wrongly skip its own subquery and let
        zero-bbox images through."""
        _make_image(files_root, "has-bbox.png")
        _make_image(files_root, "no-bbox.png")
        await _add_bbox(http_client, "has-bbox.png", label_uuid, [])
        await _register_image(http_client, "no-bbox.png")

        images = _filenames(
            await _review(
                http_client,
                show_all_bboxes=True,
                require_bboxes=True,
                terms=[{"kind": "objectLabel", "labelUuids": [], "matchMode": "all"}],
            )
        )
        assert any("has-bbox.png" in f for f in images)
        assert not any("no-bbox.png" in f for f in images)


# ──────────────────────────────────────────────────────────────────────────────
# filename — matched against the FILES_ROOT-relative path
# ──────────────────────────────────────────────────────────────────────────────


class TestFilenameTermIsRelativeToFilesRoot:
    """The `filename` column stores absolute paths, but every record the API
    returns has been relativised (`Manager._image_rel_path`), and the review
    modals run the identical filter client-side over that relative value
    (`filterPredicate.js`'s `imageMatchesFilename`). So the term matches the
    relative path: a pattern that only exists in the FILES_ROOT prefix — which
    the user is never shown — must not match anything."""

    async def test_matches_relative_path_segments(self, http_client, files_root):
        (files_root / "cams").mkdir()
        _make_image(files_root, "cams/front.png")
        await _register_image(http_client, "cams/front.png")

        for pattern in ("front", "cams/", "cams/front.png", ".png"):
            images = _filenames(
                await _review(http_client, show_all_bboxes=True, terms=[{"kind": "filename", "patterns": [pattern]}])
            )
            assert any("cams/front.png" in f for f in images), pattern

    async def test_does_not_match_the_files_root_prefix(self, http_client, files_root):
        _make_image(files_root, "plain.png")
        await _register_image(http_client, "plain.png")

        # `files_root` is <tmp>/files — "files" appears in the stored absolute
        # path and in nothing the user can see. Pre-relativisation this matched
        # every image in the database.
        images = _filenames(
            await _review(http_client, show_all_bboxes=True, terms=[{"kind": "filename", "patterns": ["files"]}])
        )
        assert images == []

        # The full absolute path is likewise no longer a valid query.
        images = _filenames(
            await _review(
                http_client, show_all_bboxes=True, terms=[{"kind": "filename", "patterns": [str(files_root)]}]
            )
        )
        assert images == []

    async def test_returned_filename_is_what_the_pattern_matched(self, http_client, files_root):
        """Guards the invariant the whole change rests on: whatever the filter
        matched is exactly the string the client receives and re-filters."""
        (files_root / "cams").mkdir()
        _make_image(files_root, "cams/front.png")
        await _register_image(http_client, "cams/front.png")

        images = _filenames(
            await _review(http_client, show_all_bboxes=True, terms=[{"kind": "filename", "patterns": ["cams/"]}])
        )
        assert images == ["cams/front.png"]

    async def test_negate_uses_the_relative_path_too(self, http_client, files_root):
        (files_root / "cams").mkdir()
        _make_image(files_root, "cams/front.png")
        _make_image(files_root, "other.png")
        await _register_image(http_client, "cams/front.png")
        await _register_image(http_client, "other.png")

        images = _filenames(
            await _review(
                http_client,
                show_all_bboxes=True,
                terms=[{"kind": "filename", "patterns": ["cams"], "negate": True}],
            )
        )
        assert images == ["other.png"]

        # Negating a prefix-only pattern excludes nothing, since it matches nothing.
        images = _filenames(
            await _review(
                http_client,
                show_all_bboxes=True,
                terms=[{"kind": "filename", "patterns": ["files"], "negate": True}],
            )
        )
        assert sorted(images) == ["cams/front.png", "other.png"]

    async def test_row_outside_files_root_falls_back_to_stored_path(self, db_client, files_root):
        """Mirrors `_image_rel_path`'s fallback: a row whose filename isn't
        under FILES_ROOT (legacy import) keeps its stored value, and that is
        also what the client is shown — so it must stay matchable."""
        from pathlib import Path

        from apps.helpers.db.types import ImageSearchParams

        await db_client.add_image(Path("/elsewhere/legacy/orphan.png"))

        rows = await db_client.list_images_search(
            ImageSearchParams(terms=[{"kind": "filename", "patterns": ["legacy/orphan"]}], require_bboxes=False)
        )
        assert [r.filename for r in rows] == ["/elsewhere/legacy/orphan.png"]
