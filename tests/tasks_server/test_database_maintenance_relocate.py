"""Path mapping and file-copy safety for the DatabaseMaintenanceTask relocate phase.

The DB half of the phase lives in tests/db_server/test_image_relocate.py. What is left
here is the part that decides *where* a file lands and the part that refuses to destroy
one on the way — both pure enough to exercise without a running task.
"""

from pathlib import Path

import pytest

from apps.tasks_server.tasks import DatabaseMaintenanceTask as dmt_module
from apps.tasks_server.tasks.DatabaseMaintenanceTask import DatabaseMaintenanceTask


def _task(sources, destination, mode="copy") -> DatabaseMaintenanceTask:
    """A task instance carrying just the relocate config — no manager, no event loop."""
    task = object.__new__(DatabaseMaintenanceTask)
    task._relocate_sources = sources
    task._relocate_destination = destination
    task._relocate_mode = mode
    return task


# ──────────────────────────────────────────────────────────────────────────────
# Where a file lands
# ──────────────────────────────────────────────────────────────────────────────


class TestRelocationTarget:
    def test_structure_below_the_source_is_preserved(self):
        """Date/camera folders must survive the move.

        Flattening would collide the moment two day folders held a frame with the same
        name, which is the normal case for date-stamped captures — one image would
        silently overwrite another.
        """
        task = _task(["timelapse"], "archive")
        assert task._relocation_target("timelapse/cam/2026-01-01/a.png") == "archive/cam/2026-01-01/a.png"

    def test_file_directly_in_the_source_keeps_its_name(self):
        task = _task(["timelapse"], "archive")
        assert task._relocation_target("timelapse/a.png") == "archive/a.png"

    def test_path_outside_every_source_maps_nowhere(self):
        """Returning None is what makes the caller skip rather than invent a target."""
        task = _task(["timelapse"], "archive")
        assert task._relocation_target("run1/a.png") is None

    def test_sibling_sharing_a_string_prefix_maps_nowhere(self):
        """`timelapse_old` is not inside `timelapse` — a string-prefix check would say it is."""
        task = _task(["timelapse"], "archive")
        assert task._relocation_target("timelapse_old/a.png") is None

    def test_the_most_specific_source_wins(self):
        """Nested sources re-root at the deeper one, regardless of list order.

        Ordering by list position instead would make the result depend on the order the
        user happened to add the directories in.
        """
        task = _task(["timelapse", "timelapse/cam"], "archive")
        assert task._relocation_target("timelapse/cam/2026-01-01/a.png") == "archive/2026-01-01/a.png"

        reversed_order = _task(["timelapse/cam", "timelapse"], "archive")
        assert reversed_order._relocation_target("timelapse/cam/2026-01-01/a.png") == "archive/2026-01-01/a.png"

    def test_leading_and_trailing_slashes_do_not_change_the_result(self):
        """The directory picker returns `/timelapse`; a hand-typed value has no slash."""
        task = _task(["/timelapse/"], "/archive/")
        assert task._relocation_target("/timelapse/cam/a.png") == "archive/cam/a.png"


# ──────────────────────────────────────────────────────────────────────────────
# Copying without destroying anything
# ──────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def files_root(tmp_path, monkeypatch):
    """Point the task module's FILES_ROOT at a scratch tree."""
    root = tmp_path / "image_datasets"
    root.mkdir()
    monkeypatch.setattr(dmt_module, "FILES_ROOT", root)
    return root


def _write(root: Path, rel: str, body: bytes = b"image-bytes") -> Path:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)
    return path


class TestCopyOne:
    def test_image_and_sidecar_travel_together(self, files_root):
        """A frame without its sidecar loses the annotations the migration exists to save."""
        _write(files_root, "timelapse/a.png")
        _write(files_root, "timelapse/a.json", b'{"uuid": "x"}')

        task = _task(["timelapse"], "archive")
        ok, detail = task._copy_one_sync("timelapse/a.png", "archive/a.png")

        assert ok, detail
        assert (files_root / "archive/a.png").read_bytes() == b"image-bytes"
        assert (files_root / "archive/a.json").read_bytes() == b'{"uuid": "x"}'
        # Copy leaves the originals — that is what makes it safe to run before unmounting.
        assert (files_root / "timelapse/a.png").exists()
        assert (files_root / "timelapse/a.json").exists()

    def test_missing_sidecar_is_not_an_error(self, files_root):
        """Not every recorded image has one; the image is still worth relocating."""
        _write(files_root, "timelapse/a.png")

        ok, detail = _task(["timelapse"], "archive")._copy_one_sync("timelapse/a.png", "archive/a.png")

        assert ok, detail
        assert (files_root / "archive/a.png").exists()
        assert not (files_root / "archive/a.json").exists()

    def test_existing_destination_is_refused_not_overwritten(self, files_root):
        """The one outcome that loses data silently, so it must fail loudly instead."""
        _write(files_root, "timelapse/a.png", b"new")
        _write(files_root, "archive/a.png", b"ALREADY HERE")

        ok, detail = _task(["timelapse"], "archive")._copy_one_sync("timelapse/a.png", "archive/a.png")

        assert not ok
        assert "exists" in detail
        assert (files_root / "archive/a.png").read_bytes() == b"ALREADY HERE"

    def test_missing_source_is_reported(self, files_root):
        ok, detail = _task(["timelapse"], "archive")._copy_one_sync("timelapse/gone.png", "archive/gone.png")

        assert not ok
        assert "missing" in detail
        assert not (files_root / "archive/gone.png").exists()

    def test_destination_escaping_files_root_is_refused(self, files_root, tmp_path):
        """`..` climbing must not write outside the dataset tree."""
        _write(files_root, "timelapse/a.png")
        outside = tmp_path / "outside.png"

        ok, _detail = _task(["timelapse"], "archive")._copy_one_sync("timelapse/a.png", "../outside.png")

        assert not outside.exists()
        # Either refused outright, or safely re-rooted inside FILES_ROOT — never outside.
        if ok:
            assert not outside.exists()

    def test_deleting_removes_image_and_sidecar(self, files_root):
        """The move half: originals go only after the DB has been re-pointed."""
        _write(files_root, "timelapse/a.png")
        _write(files_root, "timelapse/a.json", b"{}")

        _task(["timelapse"], "archive", mode="move")._delete_one_sync("timelapse/a.png")

        assert not (files_root / "timelapse/a.png").exists()
        assert not (files_root / "timelapse/a.json").exists()
