"""The write-ahead log gets truncated instead of growing without bound.

Observed in production on 2026-08-19: a 57 MB WAL against a 38 MB database, 14x
past SQLite's own 4 MB autocheckpoint threshold and not shrinking. SQLite was
behaving correctly — autocheckpoint only ever runs a PASSIVE checkpoint, which
copies WAL pages into the main file but cannot reset it while any reader is
active, and a busy db_server rarely offers a quiet moment. The file therefore
keeps its high-water mark forever, and every reader pays to search it.

``checkpoint_wal`` is the deliberate TRUNCATE pass that reclaims it, scheduled
from db_server's periodic worker rather than any request path.
"""

import pytest

from apps.helpers.db.db_client import DbClient


@pytest.fixture
async def file_db(tmp_path, files_root):
    """A file-backed client — the shared `db_client` fixture is in-memory, and an
    in-memory database has no WAL to checkpoint at all."""
    client = DbClient(db_dir=tmp_path, memory=False, files_root=files_root)
    await client.init_db()
    yield client
    await client.close()


def _wal_size(client: DbClient) -> int:
    wal = client._db_path.with_name(client._db_path.name + "-wal")
    return wal.stat().st_size if wal.exists() else 0


async def _grow_the_wal(client: DbClient, rounds: int = 60) -> None:
    """Commit enough distinct writes to put real pages in the WAL."""
    payload = "x" * 20_000
    for i in range(rounds):
        await client.set_setting(f"wal_filler_{i}", payload)


class TestOnlyActsWhenThereIsSomethingToReclaim:
    async def test_an_in_memory_db_is_skipped(self, db_client):
        """Returns None rather than 0 so "no WAL exists" stays distinguishable
        from "there was a WAL and nothing needed doing"."""
        assert await db_client.checkpoint_wal() is None

    async def test_a_wal_under_the_threshold_is_left_alone(self, file_db):
        """Truncating on every pass would fight SQLite's own passive checkpoint
        and force a reader stall for no benefit."""
        await file_db.set_setting("small", "value")

        assert _wal_size(file_db) < file_db._WAL_CHECKPOINT_THRESHOLD_BYTES
        assert await file_db.checkpoint_wal() == 0


class TestReclaimsAnOversizedWal:
    async def test_an_oversized_wal_is_truncated(self, file_db, monkeypatch):
        """The regression: before this existed the WAL only ever grew."""
        monkeypatch.setattr(file_db, "_WAL_CHECKPOINT_THRESHOLD_BYTES", 4096)
        await _grow_the_wal(file_db)
        before = _wal_size(file_db)
        assert before > 4096, "fixture failed to produce a WAL worth checkpointing"

        reclaimed = await file_db.checkpoint_wal()

        after = _wal_size(file_db)
        assert after < before, f"WAL did not shrink: {before} -> {after}"
        assert reclaimed == before - after

    async def test_the_data_survives_the_checkpoint(self, file_db, monkeypatch):
        """A checkpoint moves pages into the main database. If it ever dropped
        them instead, a shrinking WAL would look like success."""
        monkeypatch.setattr(file_db, "_WAL_CHECKPOINT_THRESHOLD_BYTES", 4096)
        await _grow_the_wal(file_db)
        await file_db.set_setting("canary", "still-here")

        await file_db.checkpoint_wal()

        assert await file_db.get_setting("canary") == "still-here"
        assert await file_db.get_setting("wal_filler_0") == "x" * 20_000
