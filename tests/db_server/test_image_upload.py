"""Importing images from the user's machine (POST /api/images/upload).

Uploads are registered in the DB on arrival rather than left to lazy
registration: a record only appears on its own once something writes
annotations to an image, so an untouched upload would be invisible to Review —
which is the one place you would go looking for what you just imported.

Near-duplicates are reported, never rejected: re-importing an image you already
have is a real thing to want (a better scan of the same frame), so the caller
is told and decides.
"""

from pathlib import Path

import pytest

from tests.conftest import make_png


# ⚠️ Responses are camelCase: ApiModel carries alias_generator=to_camel, so
# duplicate_of arrives as duplicateOf (see CLAUDE.md, "Pydantic models").
def _upload(name: str = "photo.png", data: bytes = None):
    """A multipart file tuple as httpx wants it."""
    return ("files", (name, data if data is not None else make_png(), "image/png"))


async def _post(http_client, files, **form):
    return await http_client.post("/api/images/upload", files=files, data=form)


class TestHappyPath:
    async def test_writes_the_file_and_reports_where(self, http_client, files_root):
        r = await _post(http_client, [_upload("cat.png")])

        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "success"
        (result,) = body["results"]
        assert result["ok"] is True
        assert result["filename"] == "cat.png"
        assert (files_root / "cat.png").exists()
        # Path is FILES_ROOT-relative, which is what every other image API speaks.
        assert result["path"] == "cat.png"

    async def test_registers_the_image_so_review_can_find_it(self, http_client):
        """The whole reason not to leave this to lazy registration."""
        r = await _post(http_client, [_upload("findable.png")])
        uuid = r.json()["results"][0]["uuid"]

        assert uuid

        listed = await http_client.get("/api/images/list", params={"pattern": "*.png"})
        assert "findable.png" in [f["name"] for f in listed.json()["files"]]

    async def test_uploads_into_a_subdirectory(self, http_client, files_root):
        r = await _post(http_client, [_upload("kitten.png")], path="incoming/today")

        assert r.json()["results"][0]["path"] == str(Path("incoming/today/kitten.png"))
        assert (files_root / "incoming" / "today" / "kitten.png").exists()

    async def test_creates_a_destination_that_does_not_exist_yet(self, http_client, files_root):
        await _post(http_client, [_upload()], path="brand/new/folder")

        assert (files_root / "brand" / "new" / "folder").is_dir()


class TestMultipleFiles:
    async def test_uploads_a_batch(self, http_client, files_root):
        r = await _post(http_client, [_upload("a.png"), _upload("b.png"), _upload("c.png")])

        results = r.json()["results"]
        assert [x["ok"] for x in results] == [True, True, True]
        assert all((files_root / n).exists() for n in ("a.png", "b.png", "c.png"))

    async def test_one_bad_file_does_not_sink_the_batch(self, http_client, files_root):
        """Dropping twenty images and getting nothing because the nineteenth was
        a PDF would be worse than useless."""
        r = await _post(
            http_client,
            [_upload("good.png"), ("files", ("notes.pdf", b"%PDF-1.4", "application/pdf")), _upload("also_good.png")],
        )

        results = {x["filename"]: x for x in r.json()["results"]}
        assert results["good.png"]["ok"] is True
        assert results["also_good.png"]["ok"] is True
        assert results["notes.pdf"]["ok"] is False
        assert (files_root / "good.png").exists()
        assert not (files_root / "notes.pdf").exists()

    async def test_the_request_still_succeeds_when_a_file_is_rejected(self, http_client):
        """A partial failure is not a failed request — per-file results carry it."""
        r = await _post(http_client, [("files", ("x.pdf", b"%PDF", "application/pdf"))])

        assert r.status_code == 200
        assert r.json()["status"] == "success"
        assert r.json()["results"][0]["ok"] is False


class TestRejections:
    async def test_rejects_a_non_image_extension(self, http_client, files_root):
        r = await _post(http_client, [("files", ("script.sh", b"#!/bin/sh\nrm -rf /", "text/plain"))])

        result = r.json()["results"][0]
        assert result["ok"] is False
        assert "not an image" in result["message"]
        assert not (files_root / "script.sh").exists()

    async def test_rejects_bytes_that_are_not_a_decodable_image(self, http_client, files_root):
        """The extension is a claim, not evidence. An undecodable file under
        FILES_ROOT is worse than a rejected upload: every later scan trips on it."""
        r = await _post(http_client, [_upload("lies.png", b"this is not a png")])

        result = r.json()["results"][0]
        assert result["ok"] is False
        assert not (files_root / "lies.png").exists()

    async def test_a_traversing_filename_lands_in_the_destination(self, http_client, files_root):
        """Only the basename is honoured, so this cannot escape FILES_ROOT."""
        r = await _post(http_client, [_upload("../../escaped.png")])

        result = r.json()["results"][0]
        assert result["ok"] is True
        assert (files_root / "escaped.png").exists()
        assert not (files_root.parent / "escaped.png").exists()

    async def test_a_traversing_destination_is_refused(self, http_client, files_root):
        r = await _post(http_client, [_upload()], path="../../etc")

        result = r.json()["results"][0]
        assert result["ok"] is False
        assert "outside" in result["message"].lower()
        assert not (files_root.parent / "etc" / "photo.png").exists()


class TestNameCollisions:
    async def test_a_second_upload_of_the_same_name_is_uniquified(self, http_client, files_root):
        """Two cameras happily produce the same 'snapshot.jpg'. Clobbering the
        first loses both the file and every annotation made against that path."""
        await _post(http_client, [_upload("snapshot.png")])
        r = await _post(http_client, [_upload("snapshot.png")])

        result = r.json()["results"][0]
        assert result["ok"] is True
        assert result["renamedFrom"] == "snapshot.png"
        assert result["path"] == "snapshot_1.png"
        assert (files_root / "snapshot.png").exists()
        assert (files_root / "snapshot_1.png").exists()

    async def test_overwrite_replaces_in_place(self, http_client, files_root):
        await _post(http_client, [_upload("fixed.png")])
        r = await _post(http_client, [_upload("fixed.png")], overwrite="true")

        result = r.json()["results"][0]
        assert result["path"] == "fixed.png"
        assert result["renamedFrom"] is None
        assert not (files_root / "fixed_1.png").exists()

    async def test_a_first_upload_is_not_renamed(self, http_client):
        r = await _post(http_client, [_upload("original.png")])

        assert r.json()["results"][0]["renamedFrom"] is None


class TestDuplicateReporting:
    async def test_a_re_upload_is_flagged_but_still_saved(self, http_client, files_root):
        """Advisory, not a veto: the file lands either way and the caller decides."""
        png = make_png()
        first = await _post(http_client, [_upload("one.png", png)])
        second = await _post(http_client, [_upload("two.png", png)])

        original_uuid = first.json()["results"][0]["uuid"]
        result = second.json()["results"][0]

        assert result["ok"] is True
        assert (files_root / "two.png").exists()
        assert result["duplicateOf"] == original_uuid
        assert result["duplicateDistance"] == 0

    async def test_the_first_upload_of_an_image_is_not_a_duplicate_of_itself(self, http_client):
        r = await _post(http_client, [_upload("solo.png")])

        assert r.json()["results"][0]["duplicateOf"] is None
