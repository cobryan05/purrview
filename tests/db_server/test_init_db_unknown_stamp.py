"""A database stamped at a revision this build no longer ships still opens.

The hazard is the migration squash: collapsing the 73-revision chain into one
baseline deleted every revision any already-deployed database was stamped at, and
``alembic upgrade head`` resolves the *stored* revision before it does anything —
so it raises ``Can't locate revision identified by '<rev>'``, ``init_db`` fails,
``Manager._db_ready`` stays False and db_server answers nothing but ``/status``.

The schema such a database has is the schema the baseline describes, which is why
re-stamping is the right recovery rather than a workaround. A real upgrade must
still be an upgrade, so the second test is the half that keeps this from becoming
"stamp over anything we do not recognise".
"""

import sqlite3

import pytest
from alembic.script import ScriptDirectory

from apps.helpers.db.db_client import ALEMBIC_INI_PATH, AlembicConfig, DbClient


def _stamp(db_path, revision: str) -> None:
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("UPDATE alembic_version SET version_num = ?", (revision,))
        conn.commit()
    finally:
        conn.close()


def _stamped_at(db_path) -> str:
    conn = sqlite3.connect(db_path)
    try:
        return conn.execute("SELECT version_num FROM alembic_version").fetchone()[0]
    finally:
        conn.close()


def _head() -> str:
    cfg = AlembicConfig(ALEMBIC_INI_PATH)
    return ScriptDirectory.from_config(cfg).get_current_head()


@pytest.fixture
async def existing_db(tmp_path, files_root):
    """A file-backed database that has already been created and stamped once."""
    client = DbClient(db_dir=tmp_path, memory=False, files_root=files_root)
    await client.init_db()
    await client.set_setting("survives_the_restamp", "yes")
    await client.close()
    return tmp_path, files_root


class TestUnknownStamp:
    async def test_a_deleted_revision_does_not_stop_the_database_opening(self, existing_db):
        db_dir, files_root = existing_db
        # One of the revisions the baseline squash removed.
        _stamp(db_dir / "db.sqlite", "c9d1e4f7a2b3")

        client = DbClient(db_dir=db_dir, memory=False, files_root=files_root)
        await client.init_db()
        try:
            # The data is still there: this recovered the history, not the database.
            assert await client.get_setting("survives_the_restamp") == "yes"
        finally:
            await client.close()
        assert _stamped_at(db_dir / "db.sqlite") == _head()

    async def test_a_known_stamp_is_left_to_the_ordinary_upgrade(self, existing_db, monkeypatch):
        """The guard must not swallow a real migration.

        A database already at head has nothing to upgrade, so the observable is that
        the recovery path is the one not taken: a stamp here would be a stamp over
        whatever a future revision was about to do.
        """
        db_dir, files_root = existing_db
        from apps.helpers.db import db_client as module

        def _refuse(*args, **kwargs):
            raise AssertionError("re-stamped a database whose revision this build ships")

        monkeypatch.setattr(module.alembic_command, "stamp", _refuse)

        client = DbClient(db_dir=db_dir, memory=False, files_root=files_root)
        await client.init_db()
        await client.close()
        assert _stamped_at(db_dir / "db.sqlite") == _head()
