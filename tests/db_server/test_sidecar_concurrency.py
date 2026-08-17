"""Concurrent sidecar writes for the same image must not fail or lose data.

Saving an annotation fires one `POST /api/images/labels/set` per changed label,
all in flight at once, and each one rewrites the *same* JSON sidecar. Two
independent races live in that path:

1. Every writer used a single fixed temp name (`<file>.json.tmp`), so writer A's
   `os.replace` consumed the file writer B had just written — B then raised
   `FileNotFoundError` and the UI reported "N label state(s) failed to save"
   even though the database row had landed correctly.

2. The merge is a read-modify-write. Two writers that both read the pre-existing
   file before either wrote it back would each persist their own slice on top of
   the *stale* copy, silently dropping the other's key from the file on disk.

Both are exercised here with real threads, because the writes really do happen
on the thread pool (`asyncio.to_thread` in `_write_image_label_sidecars`).
"""

import json
import threading
import time
from pathlib import Path

import pytest

from apps.helpers.db import db_client as db_client_module
from apps.helpers.db.db_client import DbClient


def _run_concurrently(fns):
    """Run `fns` on threads released together; return the exception per thread."""
    barrier = threading.Barrier(len(fns))
    errors = [None] * len(fns)

    def _wrap(i, fn):
        barrier.wait()
        try:
            fn()
        except Exception as e:  # noqa: BLE001 - recorded, asserted on below
            errors[i] = e

    threads = [threading.Thread(target=_wrap, args=(i, fn)) for i, fn in enumerate(fns)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    return errors


class TestConcurrentSidecarWrites:
    def test_parallel_writers_do_not_clobber_each_others_temp_file(self, tmp_path, monkeypatch):
        """Eight simultaneous writes to one sidecar all succeed.

        A delay between writing the temp file and renaming it widens the window
        the real thread pool hits by chance, so the shared-temp-name collision
        is reproduced every run rather than occasionally.
        """
        sidecar = tmp_path / "img.json"
        real_replace = db_client_module.os.replace

        def slow_replace(src, dst):
            time.sleep(0.05)
            return real_replace(src, dst)

        monkeypatch.setattr(db_client_module.os, "replace", slow_replace)

        client = DbClient.__new__(DbClient)  # sidecar writers touch no DB state
        rows = [{"labelUuid": f"label-{i}", "status": "absent"} for i in range(8)]
        errors = _run_concurrently([lambda r=r: client.write_image_labels_to_json(sidecar, [r]) for r in rows])

        assert errors == [None] * 8, f"concurrent sidecar writes failed: {errors}"
        assert sidecar.exists()
        # Exactly one writer wins the last write; the file must still be valid
        # JSON holding one of the values written, never a torn or empty file.
        written = json.loads(sidecar.read_text())
        assert written["image_labels"] in [[r] for r in rows]
        assert not list(Path(tmp_path).glob("*.tmp")), "temp files left behind"

    def test_merge_does_not_drop_a_concurrent_writers_key(self, tmp_path, monkeypatch):
        """`boxes` and `image_labels` written at once both survive.

        These are written by two different code paths that each own one top-level
        key, which is exactly the interleaving a save-plus-mark-absent produces.
        """
        sidecar = tmp_path / "img.json"
        real_dumps = db_client_module.json.dumps

        def slow_dumps(*args, **kwargs):
            # Serialisation happens after the existing file is read and before it
            # is written back — the exact window where a stale read gets baked in.
            out = real_dumps(*args, **kwargs)
            time.sleep(0.05)
            return out

        monkeypatch.setattr(db_client_module.json, "dumps", slow_dumps)

        client = DbClient.__new__(DbClient)
        errors = _run_concurrently(
            [
                lambda: client.write_image_labels_to_json(sidecar, [{"labelUuid": "l1", "status": "present"}]),
                lambda: DbClient._merge_write_sidecar(sidecar, {"boxes": [{"uuid": "b1"}]}),
            ]
        )

        assert errors == [None, None], f"concurrent sidecar writes failed: {errors}"
        written = json.loads(sidecar.read_text())
        assert written.get("image_labels") == [{"labelUuid": "l1", "status": "present"}]
        assert written.get("boxes") == [{"uuid": "b1"}]

    def test_failed_write_leaves_the_previous_sidecar_intact(self, tmp_path, monkeypatch):
        """A write that blows up mid-flight must not damage what was there."""
        sidecar = tmp_path / "img.json"
        sidecar.write_text(json.dumps({"boxes": [{"uuid": "original"}]}))

        def boom(src, dst):
            raise OSError("disk full")

        monkeypatch.setattr(db_client_module.os, "replace", boom)
        with pytest.raises(OSError):
            DbClient._merge_write_sidecar(sidecar, {"image_labels": [{"labelUuid": "l1"}]})

        assert json.loads(sidecar.read_text()) == {"boxes": [{"uuid": "original"}]}
        assert not list(Path(tmp_path).glob("*.tmp")), "temp file left behind after a failed write"
