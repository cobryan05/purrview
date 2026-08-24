"""Tests for listing and re-pointing image records by directory.

These back the DatabaseMaintenanceTask "Relocate Images" phase, whose job is to empty a
directory of everything the database knows about so the directory itself can be removed.
The phase copies the bytes and then asks the DB to re-point; these cover the DB half.

All tests use the db_server conftest fixtures: db_client, files_root, http_client.
"""

import struct
import uuid as uuid_mod
import zlib
from pathlib import Path

import pytest

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _make_png() -> bytes:
    """Minimal 1×1 white PNG."""

    def chunk(tag, data):
        raw = tag + data
        return struct.pack(">I", len(data)) + raw + struct.pack(">I", zlib.crc32(raw) & 0xFFFFFFFF)

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(b"\x00\xff\xff\xff"))
        + chunk(b"IEND", b"")
    )


async def _register(http_client, files_root: Path, rel_path: str) -> str:
    """Write a PNG under files_root and register it. Returns the image uuid."""
    abs_path = files_root / rel_path
    abs_path.parent.mkdir(parents=True, exist_ok=True)
    abs_path.write_bytes(_make_png())
    r = await http_client.post("/api/images/metadata/update", json={"imagePath": rel_path, "boxes": []})
    assert r.status_code == 200, r.text
    sidecar = abs_path.with_suffix(".json")
    assert sidecar.exists(), f"expected a sidecar at {sidecar}"
    import json

    return json.loads(sidecar.read_text())["uuid"]


async def _list_under(http_client, directories) -> dict:
    r = await http_client.post("/api/images/list-by-directory", json={"directories": directories})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "success", body
    return {img["uuid"]: img["path"] for img in body["images"]}


async def _relocate(http_client, moves) -> dict:
    r = await http_client.post("/api/images/relocate", json={"moves": moves})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "success", body
    return body


# ──────────────────────────────────────────────────────────────────────────────
# Directory scoping
# ──────────────────────────────────────────────────────────────────────────────


class TestListImagesByDirectory:
    async def test_sibling_directory_sharing_a_prefix_is_not_included(self, http_client, files_root):
        """`timelapse` must not drag in `timelapse_old`.

        The scope is a path prefix, not a string prefix. Getting this wrong would move
        (or, via delete-by-directory, destroy) the contents of a directory the user
        never named — and the two share the matching rule deliberately.
        """
        inside = await _register(http_client, files_root, "timelapse/cam/2026-01-01/a.png")
        sibling = await _register(http_client, files_root, "timelapse_old/b.png")
        nested_deep = await _register(http_client, files_root, "timelapse/cam/2026-01-02/deep/c.png")

        found = await _list_under(http_client, ["timelapse"])

        assert inside in found
        assert nested_deep in found, "the scope is the whole subtree, not one level"
        assert sibling not in found

    async def test_unregistered_files_are_not_listed(self, http_client, files_root):
        """Only DB records count — the point of relocating from the database.

        A capture directory can hold a million frames of which a handful were annotated;
        listing the untracked ones would turn a small migration into a full tree copy.
        """
        tracked = await _register(http_client, files_root, "timelapse/tracked.png")
        (files_root / "timelapse" / "untracked.png").write_bytes(_make_png())

        found = await _list_under(http_client, ["timelapse"])

        assert list(found) == [tracked]

    async def test_no_directories_returns_nothing(self, http_client, files_root):
        await _register(http_client, files_root, "timelapse/a.png")
        assert await _list_under(http_client, []) == {}


# ──────────────────────────────────────────────────────────────────────────────
# Re-pointing records
# ──────────────────────────────────────────────────────────────────────────────


class TestRelocateImages:
    async def test_record_moves_out_of_the_source_scope(self, http_client, files_root):
        """After re-pointing, the source directory no longer claims the image.

        This is the condition the whole feature exists to reach: the directory can be
        unmounted because nothing in the database still points into it.
        """
        img = await _register(http_client, files_root, "timelapse/cam/2026-01-01/a.png")
        assert img in await _list_under(http_client, ["timelapse"])

        body = await _relocate(http_client, [{"uuid": img, "newPath": "archive/cam/2026-01-01/a.png"}])

        assert body["updated"] == 1
        assert body["errors"] == []
        assert img not in await _list_under(http_client, ["timelapse"])
        assert (await _list_under(http_client, ["archive"]))[img] == "archive/cam/2026-01-01/a.png"

    async def test_destination_held_by_another_image_is_rejected(self, http_client, files_root):
        """`images.filename` is UNIQUE, so a taken destination must fail loudly.

        Reported per-record rather than raised: the caller has already copied bytes for
        the rest of the batch and needs to know which single record did not land.
        """
        first = await _register(http_client, files_root, "timelapse/a.png")
        second = await _register(http_client, files_root, "keep/b.png")

        body = await _relocate(http_client, [{"uuid": first, "newPath": "keep/b.png"}])

        assert body["updated"] == 0
        assert [e["uuid"] for e in body["errors"]] == [first]
        # The loser keeps its original path — a rejected move must not half-apply.
        assert (await _list_under(http_client, ["timelapse"]))[first] == "timelapse/a.png"
        assert (await _list_under(http_client, ["keep"]))[second] == "keep/b.png"

    async def test_relocating_to_its_current_path_is_a_no_op_success(self, http_client, files_root):
        """Re-running an interrupted relocation must not fail on the records that landed."""
        img = await _register(http_client, files_root, "timelapse/a.png")

        body = await _relocate(http_client, [{"uuid": img, "newPath": "timelapse/a.png"}])

        assert body["errors"] == []
        assert body["updated"] == 1
        assert (await _list_under(http_client, ["timelapse"]))[img] == "timelapse/a.png"

    async def test_path_escaping_files_root_is_rejected(self, http_client, files_root):
        """A destination outside FILES_ROOT must not be written into the DB.

        `..` climbing is the interesting case: the record would name a file the app can
        neither serve nor sweep, and nothing else in the system writes outside the root.
        """
        img = await _register(http_client, files_root, "timelapse/a.png")
        before = (await _list_under(http_client, ["timelapse"]))[img]

        body = await _relocate(http_client, [{"uuid": img, "newPath": "../../etc/passwd"}])

        assert body["updated"] == 0
        assert len(body["errors"]) == 1
        assert (await _list_under(http_client, ["timelapse"]))[img] == before

    async def test_unknown_uuid_is_reported_not_raised(self, http_client, files_root):
        missing = str(uuid_mod.uuid4())
        body = await _relocate(http_client, [{"uuid": missing, "newPath": "archive/x.png"}])

        assert body["updated"] == 0
        assert [e["uuid"] for e in body["errors"]] == [missing]

    async def test_one_bad_move_does_not_block_the_good_ones(self, http_client, files_root):
        """A batch is not all-or-nothing — the copies that succeeded must still land."""
        good = await _register(http_client, files_root, "timelapse/good.png")
        bad = str(uuid_mod.uuid4())

        body = await _relocate(
            http_client,
            [
                {"uuid": good, "newPath": "archive/good.png"},
                {"uuid": bad, "newPath": "archive/bad.png"},
            ],
        )

        assert body["updated"] == 1
        assert [e["uuid"] for e in body["errors"]] == [bad]
        assert (await _list_under(http_client, ["archive"]))[good] == "archive/good.png"
