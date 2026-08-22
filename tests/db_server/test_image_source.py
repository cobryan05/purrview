"""Provenance on images: which source produced them (images.source_uuid).

The filter side lives in test_filter_terms.py (TestImageSourceTerm). This file
covers how the column gets written, and what happens to it when the source it
points at goes away.
"""

import pytest

from tests.conftest import make_png


@pytest.fixture
async def source_uuid(http_client):
    r = await http_client.post(
        "/api/sources/create",
        json={
            "sourceName": "cam-a",
            "imageProvider": "UrlImageProvider",
            "providerParams": {"url": "http://example.com/a.jpg"},
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "success", body
    return body["source"]["uuid"]


def _make_image(files_root, name: str) -> str:
    (files_root / name).write_bytes(make_png())
    return name


async def _save(http_client, image_path: str, **extra):
    r = await http_client.post(
        "/api/images/metadata/update",
        json={"imagePath": image_path, "boxes": [], **extra},
    )
    assert r.status_code == 200
    return r.json()


async def _source_of(db_client, filename: str):
    for img in await db_client.list_images(limit=1000):
        if img.filename.endswith(filename):
            return img.source_uuid
    raise AssertionError(f"{filename} was never registered")


class TestSourceAttribution:
    async def test_recorded_on_the_save_that_registers_the_image(self, http_client, db_client, files_root, source_uuid):
        _make_image(files_root, "attributed.png")
        await _save(http_client, "attributed.png", sourceUuid=source_uuid)
        assert await _source_of(db_client, "attributed.png") == source_uuid

    async def test_a_later_save_without_a_source_does_not_clear_it(
        self, http_client, db_client, files_root, source_uuid
    ):
        """Every annotation save re-enters add_image with no source. If an
        omitted source overwrote the stored one, drawing a single box on a
        collected image would erase where it came from."""
        _make_image(files_root, "annotated.png")
        await _save(http_client, "annotated.png", sourceUuid=source_uuid)
        await _save(http_client, "annotated.png")  # the canvas save
        assert await _source_of(db_client, "annotated.png") == source_uuid

    async def test_backfills_a_row_registered_without_one(self, http_client, db_client, files_root, source_uuid):
        """A path save can register the image before the collecting task reports
        its source; the later save has to be able to fill the blank in."""
        _make_image(files_root, "late.png")
        await _save(http_client, "late.png")
        assert await _source_of(db_client, "late.png") is None
        await _save(http_client, "late.png", sourceUuid=source_uuid)
        assert await _source_of(db_client, "late.png") == source_uuid

    async def test_an_unknown_source_still_registers_the_image(self, http_client, db_client, files_root):
        """A source deleted mid-run must cost the attribution, not the image —
        the column is a real FK, so the insert would otherwise be rejected."""
        _make_image(files_root, "orphaned.png")
        await _save(http_client, "orphaned.png", sourceUuid="00000000-0000-0000-0000-00000000dead")
        assert await _source_of(db_client, "orphaned.png") is None


class TestSourceDeletion:
    async def test_deleting_a_source_keeps_its_images(self, http_client, db_client, files_root, source_uuid):
        """ondelete=SET NULL. Deleting a camera must never cascade into the
        training data it collected."""
        _make_image(files_root, "survivor.png")
        await _save(http_client, "survivor.png", sourceUuid=source_uuid)

        r = await http_client.post("/api/sources/delete", json={"sourceUuids": [source_uuid]})
        assert r.status_code == 200
        assert r.json()["status"] == "success", r.json()

        assert await _source_of(db_client, "survivor.png") is None
