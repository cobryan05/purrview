"""Importing images from the user's machine (POST /api/images/upload).

Uploads are registered in the DB on arrival rather than left to lazy
registration: a record only appears on its own once something writes
annotations to an image, so an untouched upload would be invisible to Review —
which is the one place you would go looking for what you just imported.

Near-duplicates are reported, never rejected: re-importing an image you already
have is a real thing to want (a better scan of the same frame), so the caller
is told and decides.
"""

import uuid
from pathlib import Path

import pytest

from apps.helpers.db.db_client import DbClient
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

    async def test_a_failed_overwrite_leaves_the_previous_file_intact(self, http_client, files_root, monkeypatch):
        """The rollback exists so a file the DB never learned about is not left
        behind. On an overwrite that reasoning inverts: the file that was there
        is already registered and may carry annotations, so deleting it on a
        failed re-upload loses the new file *and* the old one."""
        await _post(http_client, [_upload("kept.png")])
        original = (files_root / "kept.png").read_bytes()

        async def _boom(self, *args, **kwargs):
            raise RuntimeError("db went away mid-upload")

        monkeypatch.setattr(DbClient, "add_image", _boom)

        r = await _post(http_client, [_upload("kept.png", make_png(b"\x00\x00\x00"))], overwrite="true")

        assert r.json()["results"][0]["ok"] is False
        assert (files_root / "kept.png").exists(), "The pre-existing file must survive a failed overwrite"
        assert (files_root / "kept.png").read_bytes() == original

    async def test_a_failed_first_upload_leaves_no_orphan_file(self, http_client, files_root, monkeypatch):
        """The other half of the same rule: a brand-new file the DB never
        registered would show in the browser and be missing from every search."""

        async def _boom(self, *args, **kwargs):
            raise RuntimeError("db went away mid-upload")

        monkeypatch.setattr(DbClient, "add_image", _boom)

        r = await _post(http_client, [_upload("orphan.png")])

        assert r.json()["results"][0]["ok"] is False
        assert not (files_root / "orphan.png").exists()

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


def _patterned_png(seed: int) -> bytes:
    """A structured 32x32 PNG whose pHash actually depends on *seed*.

    make_png's 1x1 image is uniform, and a uniform image hashes the same
    whatever colour it is — so it cannot answer "did the stored hash follow the
    file?" at all.
    """
    import io

    from PIL import Image

    img = Image.new("RGB", (32, 32))
    px = img.load()
    for y in range(32):
        for x in range(32):
            px[x, y] = (255, 255, 255) if ((x // 4) + (y // 4) + seed) % 2 else (0, 0, 0)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class TestUploadPhash:
    """An upload's pHash is what the dedup index and similar-image search run on.

    It is computed on arrival (that is how the duplicate report in the response
    is produced), so the only question is whether it is *kept*.
    """

    async def test_the_phash_computed_on_arrival_is_persisted(self, http_client, db_client):
        """The in-memory index is rebuilt from images.phash on startup, so a hash
        that only ever reached the index is lost on the next db_server restart —
        and re-importing the same picture is then reported as new."""
        r = await _post(http_client, [_upload("hashed.png", _patterned_png(0))])
        uuid = r.json()["results"][0]["uuid"]

        image = await db_client.get_image_by_uuid(uuid)
        assert image.phash is not None, "the pHash was added to the index but never written to the row"

    async def test_an_overwrite_replaces_the_stored_phash(self, http_client, db_client):
        """add_image returns the *existing* record for an overwritten path, and
        update_image_metadata only recomputes a pHash that is None — so without
        an explicit refresh the row keeps describing the picture that was
        replaced, forever."""
        first = await _post(http_client, [_upload("swapped.png", _patterned_png(0))])
        uuid = first.json()["results"][0]["uuid"]
        original = (await db_client.get_image_by_uuid(uuid)).phash

        await _post(http_client, [_upload("swapped.png", _patterned_png(1))], overwrite="true")

        replaced = (await db_client.get_image_by_uuid(uuid)).phash
        assert replaced is not None
        assert replaced != original, "the row still holds the replaced file's pHash"

    async def test_an_overwritten_image_is_no_longer_a_duplicate_of_the_old_content(self, http_client, files_root):
        """The index half of the same bug: a stale entry keeps matching uploads
        against a picture that is no longer anywhere on disk."""
        old = _patterned_png(0)
        new = _patterned_png(1)
        await _post(http_client, [_upload("target.png", old)], overwrite="true")
        await _post(http_client, [_upload("target.png", new)], overwrite="true")

        # Upload the *old* content somewhere else: nothing on disk looks like it
        # any more, so it must not be reported as a duplicate of target.png.
        r = await _post(http_client, [_upload("elsewhere.png", old)])

        assert r.json()["results"][0]["duplicateOf"] is None, "matched a pHash for content that was overwritten"


class TestThumbnailFreshness:
    async def test_the_thumbnail_follows_an_overwritten_file(self, http_client):
        """The cache is keyed on path+width with no freshness check, so a
        re-upload that fixes an image keeps serving the one it replaced."""
        await _post(http_client, [_upload("shown.png", _patterned_png(0))], overwrite="true")
        before = await http_client.get("/api/images/shown.png/thumbnail", params={"width": 16})
        assert before.status_code == 200

        await _post(http_client, [_upload("shown.png", _patterned_png(1))], overwrite="true")
        after = await http_client.get("/api/images/shown.png/thumbnail", params={"width": 16})

        assert after.status_code == 200
        assert after.content != before.content, "the cache served the thumbnail of the replaced file"


class TestOverwriteBboxPhash:
    """A box's hash describes a *crop*, and an overwrite changes the crop too.

    The image pHash and the bbox thumbnails are both refreshed on an overwrite;
    `bboxes.phash` and its per-label index entries were not, so "find similar
    bboxes" kept matching crops of a picture that is no longer on disk — the
    same failure HashIndex.replace_image fixes one level up.
    """

    async def _image_with_a_box(self, http_client, name: str, data: bytes):
        await _post(http_client, [_upload(name, data)], overwrite="true")
        label = await http_client.post("/api/labels", json={"name": f"cat-{name}", "color": "#FF0000"})
        label_uuid = label.json()["label"]["uuid"]
        bbox_uuid = str(uuid.uuid4())
        r = await http_client.post(
            "/api/images/metadata/update",
            json={
                "imagePath": name,
                "boxes": [
                    {
                        "uuid": bbox_uuid,
                        "labelUuid": label_uuid,
                        "x": 0.1,
                        "y": 0.1,
                        "width": 0.6,
                        "height": 0.6,
                    }
                ],
            },
        )
        assert r.status_code == 200
        return bbox_uuid

    async def _bbox_phash(self, db_client, bbox_uuid: str):
        rows = await db_client.get_all_bbox_phashes()
        return next((r["phash"] for r in rows if r["uuid"] == bbox_uuid), None)

    async def test_an_overwrite_rehashes_the_boxes_on_the_image(self, http_client, db_client):
        bbox_uuid = await self._image_with_a_box(http_client, "boxed.png", _patterned_png(0))
        original = await self._bbox_phash(db_client, bbox_uuid)
        assert original is not None

        await _post(http_client, [_upload("boxed.png", _patterned_png(1))], overwrite="true")

        replaced = await self._bbox_phash(db_client, bbox_uuid)
        assert replaced is not None
        assert replaced != original, "the bbox row still holds a hash of the replaced picture"

    async def test_the_overwrite_does_not_clear_the_image_phash(self, http_client, db_client):
        """The bbox refresh writes through update_phashes, which sets the image
        hash in the same statement — passing the wrong value there would undo
        the image-level refresh that runs just before it."""
        bbox_uuid = await self._image_with_a_box(http_client, "kept-hash.png", _patterned_png(0))
        assert bbox_uuid

        r = await _post(http_client, [_upload("kept-hash.png", _patterned_png(1))], overwrite="true")
        uuid_ = r.json()["results"][0]["uuid"]

        image = await db_client.get_image_by_uuid(uuid_)
        assert image.phash is not None
