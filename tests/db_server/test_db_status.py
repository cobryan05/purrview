"""Tests for DbClient.get_filenames_in_db — including the large-IN-list edge case.

SQLite 3.32+ limits bind variables to 32 766 per statement.  A directory
listing with more files than that limit must not raise OperationalError.
The implementation must chunk the list and merge results.
"""
import pytest

from apps.helpers.db.db_client import DbClient
from tests.conftest import make_png


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
async def db_client(tmp_path):
    client = DbClient(db_dir=tmp_path, memory=True)
    await client.init_db()
    yield client
    await client.close()


@pytest.fixture
async def db_client_with_images(tmp_path):
    """DbClient pre-populated with three image records."""
    client = DbClient(db_dir=tmp_path, memory=True)
    await client.init_db()

    for name in ("a.jpg", "b.jpg", "c.jpg"):
        abs_path = tmp_path / name
        abs_path.write_bytes(make_png())
        await client.add_image(abs_path)

    yield client, tmp_path
    await client.close()


# ── Correctness: happy path ───────────────────────────────────────────────────


async def test_get_filenames_in_db_empty_input(db_client):
    result = await db_client.get_filenames_in_db([])
    assert result == {}


async def test_get_filenames_in_db_no_matches(db_client):
    result = await db_client.get_filenames_in_db(["/nonexistent/a.jpg", "/nonexistent/b.jpg"])
    assert result == {}


async def test_get_filenames_in_db_returns_registered_files(db_client_with_images):
    db_client, tmp_path = db_client_with_images

    abs_paths = [str(tmp_path / name) for name in ("a.jpg", "b.jpg", "c.jpg")]
    result = await db_client.get_filenames_in_db(abs_paths)

    assert set(result.keys()) == set(abs_paths)
    for path in abs_paths:
        assert "uuid" in result[path]
        assert "bbox_count" in result[path]
        assert result[path]["bbox_count"] == 0  # no bboxes added


async def test_get_filenames_in_db_ignores_missing_files(db_client_with_images):
    db_client, tmp_path = db_client_with_images

    paths = [str(tmp_path / "a.jpg"), "/does/not/exist.jpg"]
    result = await db_client.get_filenames_in_db(paths)

    assert str(tmp_path / "a.jpg") in result
    assert "/does/not/exist.jpg" not in result


# ── Edge case: list exceeding SQLite variable limit ───────────────────────────


async def test_get_filenames_in_db_large_list_does_not_crash(db_client):
    """Passing > 32 766 filenames must not raise sqlite3.OperationalError.

    SQLite 3.32+ limits bind variables to 32 766 per statement.  Without
    chunking, .where(filename.in_(filenames)) generates a single IN clause
    that exceeds this limit and raises OperationalError.

    After the fix (chunking in batches), the call must complete and return
    an empty dict (none of the fake paths exist in the DB).
    """
    # 33 001 paths — safely above the 32 766 limit.
    filenames = [f"/fake/image_{i:05d}.jpg" for i in range(33_001)]
    result = await db_client.get_filenames_in_db(filenames)
    assert result == {}
