"""Tests for the Alembic migration path.

All DBs are in-memory (sqlite:///:memory: + StaticPool), matching the pattern
used everywhere else in this test suite.  Alembic is driven via the low-level
EnvironmentContext API so it runs on our own engine rather than creating a
separate file-based connection.

A session-scoped fixture builds the pre-redesign schema once; per-test copies
are made in ~1 ms via sqlite3.Connection.backup(), so the ~80 ms initial build
is paid once per session regardless of how many tests run.
"""

import sqlite3
from pathlib import Path

import pytest
from alembic.config import Config as AlembicConfig
from alembic.runtime.environment import EnvironmentContext
from alembic.script import ScriptDirectory
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool
from sqlmodel import SQLModel

ALEMBIC_INI = Path(__file__).parent.parent.parent / "apps/helpers/db/alembic.ini"

# Revision just before the model_type / architecture / model_config_node_uuid redesign.
_PRE_REDESIGN_REV = "307a345162a9"


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _make_engine():
    """Fresh in-memory SQLite engine with a persistent single connection."""
    return create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )


def _run_migrations(engine, target: str = "head") -> None:
    """Run alembic migrations up to *target* on *engine* without touching any file."""
    cfg = AlembicConfig(str(ALEMBIC_INI))
    script = ScriptDirectory.from_config(cfg)

    def upgrade_fn(rev, ctx):
        return script._upgrade_revs(target, rev)

    with engine.connect() as conn:
        with EnvironmentContext(cfg, script, fn=upgrade_fn) as env_ctx:
            env_ctx.configure(
                conn,
                target_metadata=SQLModel.metadata,
                render_as_batch=True,
                compare_type=True,
            )
            with env_ctx.begin_transaction():
                env_ctx.run_migrations()


def _backup(src_engine, dst_engine) -> None:
    """Copy the entire in-memory DB from src to dst via sqlite3.backup()."""
    sp = src_engine.raw_connection()
    dp = dst_engine.raw_connection()
    sp.driver_connection.backup(dp.driver_connection)
    sp.close()  # checks back into StaticPool; does NOT close the sqlite3 conn
    dp.close()


def _columns(engine, table: str) -> set[str]:
    with engine.connect() as conn:
        rows = conn.exec_driver_sql(f"PRAGMA table_info({table})").fetchall()
    return {r[1] for r in rows}


def _tables(engine) -> set[str]:
    with engine.connect() as conn:
        rows = conn.exec_driver_sql("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    return {r[0] for r in rows}


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def _pre_redesign_engine():
    """Build the schema at 307a345162a9 exactly once per test session (~80 ms).

    Running to this revision implicitly validates that the full migration chain
    from an empty DB up to this point is correct.  Per-test fixtures copy it
    via backup() rather than re-running migrations.
    """
    engine = _make_engine()
    _run_migrations(engine, _PRE_REDESIGN_REV)
    return engine


@pytest.fixture
def pre_redesign_copy(_pre_redesign_engine):
    """Per-test in-memory copy of the pre-redesign schema (~1 ms via backup)."""
    engine = _make_engine()
    _backup(_pre_redesign_engine, engine)
    return engine


# ──────────────────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────────────────


class TestMigrationPath:
    def test_upgrade_from_pre_redesign_adds_new_columns(self, pre_redesign_copy):
        """Running migrations from 307a345162a9 → head must:
        - add model_type + architecture to model_config_nodes (drop model_typename)
        - add model_config_node_uuid to training_configs (drop trainer_type + model_config_uuid)
        - create model_bundles table
        - drop training_config_label_links

        The session fixture validated the full chain from an empty DB up to
        307a345162a9; this test covers the remaining steps — together they
        exercise every migration without duplicating the slow initial build.
        """
        engine = pre_redesign_copy

        # ── old schema ─────────────────────────────────────────────────────────
        old_node_cols = _columns(engine, "model_config_nodes")
        assert "model_typename" in old_node_cols, "Old schema missing model_typename"
        assert "model_type" not in old_node_cols, "Old schema should not have model_type yet"

        old_tc_cols = _columns(engine, "training_configs")
        assert "trainer_type" in old_tc_cols, "Old schema missing trainer_type"
        assert "model_config_node_uuid" not in old_tc_cols, "Old schema should not have model_config_node_uuid yet"

        assert "training_config_label_links" in _tables(engine)
        assert "model_bundles" not in _tables(engine)

        # ── apply remaining migrations ──────────────────────────────────────────
        _run_migrations(engine, "head")

        # ── new schema ─────────────────────────────────────────────────────────
        new_node_cols = _columns(engine, "model_config_nodes")
        assert "model_type" in new_node_cols, "model_config_nodes missing model_type after migration"
        assert "architecture" in new_node_cols, "model_config_nodes missing architecture after migration"
        assert "model_typename" not in new_node_cols, "Deprecated model_typename still present"

        new_tc_cols = _columns(engine, "training_configs")
        assert "model_config_node_uuid" in new_tc_cols, "training_configs missing model_config_node_uuid"
        assert "trainer_type" not in new_tc_cols, "Deprecated trainer_type still present"
        assert "model_config_uuid" not in new_tc_cols, "Deprecated model_config_uuid still present"

        assert "model_bundles" in _tables(engine), "model_bundles table was not created"
        assert "training_config_label_links" not in _tables(
            engine
        ), "training_config_label_links should have been dropped"
