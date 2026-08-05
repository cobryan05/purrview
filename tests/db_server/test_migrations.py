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


def _run_downgrade(engine, target: str) -> None:
    """Run alembic downgrades back to *target* — the mirror of _run_migrations,
    which only ever moves forward (``_upgrade_revs`` silently no-ops when asked
    to go backwards)."""
    cfg = AlembicConfig(str(ALEMBIC_INI))
    script = ScriptDirectory.from_config(cfg)

    def downgrade_fn(rev, ctx):
        return script._downgrade_revs(target, rev)

    with engine.connect() as conn:
        with EnvironmentContext(cfg, script, fn=downgrade_fn) as env_ctx:
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


def _indexes(engine, table: str) -> set[str]:
    with engine.connect() as conn:
        rows = conn.exec_driver_sql(f"PRAGMA index_list({table})").fetchall()
    return {r[1] for r in rows}


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


class TestPartialMigrationRecovery:
    """f93545d9a50b must be idempotent w.r.t. partial prior application.

    The original migration had drop_constraint(None) which raised ValueError
    inside the training_configs batch block.  SQLite's non-transactional DDL
    meant the first steps (CREATE model_bundles, DROP training_config_label_links,
    ALTER model_config_nodes) were committed before the failure, leaving the DB
    stamped at e1a2b3c4d5e6 but physically partially migrated.  On the next
    startup the fixed migration reruns and must not fail on already-done steps.
    """

    def _partial_state(self, engine) -> None:
        """Simulate a partial application of f93545d9a50b: model_bundles exists,
        training_config_label_links is gone, model_config_nodes already altered,
        but training_configs is still in the old shape."""
        with engine.begin() as conn:
            conn.exec_driver_sql("""
                CREATE TABLE model_bundles (
                    created_at DATETIME NOT NULL,
                    updated_at DATETIME NOT NULL,
                    uuid VARCHAR NOT NULL,
                    name VARCHAR NOT NULL,
                    description VARCHAR,
                    model_config_uuid VARCHAR NOT NULL,
                    bundle_path VARCHAR NOT NULL,
                    manifest_json JSON,
                    training_config_uuids JSON,
                    PRIMARY KEY (uuid),
                    FOREIGN KEY(model_config_uuid) REFERENCES model_configs (uuid) ON DELETE RESTRICT
                )
            """)
            conn.exec_driver_sql("CREATE UNIQUE INDEX ix_model_bundles_name ON model_bundles (name)")
            conn.exec_driver_sql("DROP TABLE training_config_label_links")
            # Simulate the model_config_nodes batch: add model_type/architecture, drop model_typename
            conn.exec_driver_sql("""
                CREATE TABLE model_config_nodes_new (
                    uuid VARCHAR NOT NULL PRIMARY KEY,
                    model_config_uuid VARCHAR NOT NULL,
                    label_uuid VARCHAR,
                    model_type VARCHAR NOT NULL DEFAULT 'object_detection',
                    architecture VARCHAR NOT NULL DEFAULT 'yolo_v8_n',
                    FOREIGN KEY(model_config_uuid) REFERENCES model_configs (uuid) ON DELETE CASCADE,
                    FOREIGN KEY(label_uuid) REFERENCES labels (uuid) ON DELETE CASCADE
                )
            """)
            conn.exec_driver_sql(
                "INSERT INTO model_config_nodes_new SELECT uuid, model_config_uuid, label_uuid, "
                "'object_detection', 'yolo_v8_n' FROM model_config_nodes"
            )
            conn.exec_driver_sql("DROP TABLE model_config_nodes")
            conn.exec_driver_sql("ALTER TABLE model_config_nodes_new RENAME TO model_config_nodes")

    def test_upgrade_from_partial_application_does_not_raise(self, pre_redesign_copy):
        """Running to head from a partially-applied f93545d9a50b must not raise."""
        engine = pre_redesign_copy
        _run_migrations(engine, "e1a2b3c4d5e6")
        self._partial_state(engine)
        # Must not raise OperationalError: table model_bundles already exists
        _run_migrations(engine, "head")

    def test_upgrade_from_partial_completes_training_configs(self, pre_redesign_copy):
        """After recovery the training_configs schema must be fully migrated."""
        engine = pre_redesign_copy
        _run_migrations(engine, "e1a2b3c4d5e6")
        self._partial_state(engine)
        _run_migrations(engine, "head")
        tc_cols = _columns(engine, "training_configs")
        assert "model_config_node_uuid" in tc_cols
        assert "trainer_type" not in tc_cols
        assert "model_config_uuid" not in tc_cols

    def test_upgrade_from_partial_preserves_model_bundles(self, pre_redesign_copy):
        """model_bundles must survive the f93545d9a50b recovery intact.

        It does not survive to head — 9c62f3502b03 drops the table once the
        bundle concept was removed — so this stops at f93545d9a50b, which is
        the revision whose recovery behaviour is actually under test here.
        """
        engine = pre_redesign_copy
        _run_migrations(engine, "e1a2b3c4d5e6")
        self._partial_state(engine)
        _run_migrations(engine, "f93545d9a50b")
        assert "model_bundles" in _tables(engine)

    def test_upgrade_from_minimal_partial_only_model_bundles(self, pre_redesign_copy):
        """Even if only model_bundles was pre-created the migration must succeed.

        Runs all the way to head, so it also covers 9c62f3502b03 dropping a
        table that arrived via the partial-recovery path rather than a clean
        create.
        """
        engine = pre_redesign_copy
        _run_migrations(engine, "e1a2b3c4d5e6")
        with engine.begin() as conn:
            conn.exec_driver_sql("""
                CREATE TABLE model_bundles (
                    uuid VARCHAR NOT NULL PRIMARY KEY,
                    name VARCHAR NOT NULL,
                    created_at DATETIME NOT NULL,
                    updated_at DATETIME NOT NULL,
                    model_config_uuid VARCHAR NOT NULL,
                    bundle_path VARCHAR NOT NULL
                )
            """)
        _run_migrations(engine, "head")
        assert "model_bundles" not in _tables(engine)

    def test_upgrade_with_tmp_model_config_nodes_and_original_present(self, pre_redesign_copy):
        """_alembic_tmp_model_config_nodes left behind alongside the original.

        Interrupted between CREATE _alembic_tmp and DROP original.  The migration
        must drop the stale temp and redo the batch alter cleanly.
        """
        engine = pre_redesign_copy
        _run_migrations(engine, "e1a2b3c4d5e6")
        with engine.begin() as conn:
            # Simulate alembic batch leaving a temp table while original still exists
            conn.exec_driver_sql("""
                CREATE TABLE _alembic_tmp_model_config_nodes (
                    uuid VARCHAR NOT NULL PRIMARY KEY,
                    model_config_uuid VARCHAR NOT NULL,
                    label_uuid VARCHAR,
                    model_type VARCHAR NOT NULL DEFAULT 'object_detection',
                    architecture VARCHAR NOT NULL DEFAULT 'yolo_v8_n'
                )
            """)
        _run_migrations(engine, "head")
        new_cols = _columns(engine, "model_config_nodes")
        assert "model_type" in new_cols
        assert "model_typename" not in new_cols
        assert "_alembic_tmp_model_config_nodes" not in _tables(engine)

    def test_upgrade_with_tmp_model_config_nodes_original_gone(self, pre_redesign_copy):
        """_alembic_tmp_model_config_nodes exists but original was already dropped.

        Interrupted between DROP original and RENAME temp.  Migration must complete
        the rename instead of re-running the batch alter.
        """
        engine = pre_redesign_copy
        _run_migrations(engine, "e1a2b3c4d5e6")
        with engine.begin() as conn:
            # Simulate batch got past DROP original but crashed before RENAME
            conn.exec_driver_sql("""
                CREATE TABLE _alembic_tmp_model_config_nodes (
                    uuid VARCHAR NOT NULL PRIMARY KEY,
                    model_config_uuid VARCHAR NOT NULL,
                    label_uuid VARCHAR,
                    model_type VARCHAR NOT NULL DEFAULT 'object_detection',
                    architecture VARCHAR NOT NULL DEFAULT 'yolo_v8_n',
                    FOREIGN KEY(model_config_uuid) REFERENCES model_configs (uuid) ON DELETE CASCADE
                )
            """)
            conn.exec_driver_sql("DROP TABLE model_config_nodes")
        _run_migrations(engine, "head")
        assert "model_config_nodes" in _tables(engine)
        assert "_alembic_tmp_model_config_nodes" not in _tables(engine)

    def test_upgrade_with_tmp_training_configs_and_original_present(self, pre_redesign_copy):
        """_alembic_tmp_training_configs left behind alongside the original.

        Mirrors the model_config_nodes scenario for the training_configs batch step.
        """
        engine = pre_redesign_copy
        _run_migrations(engine, "e1a2b3c4d5e6")
        with engine.begin() as conn:
            conn.exec_driver_sql("""
                CREATE TABLE _alembic_tmp_training_configs (
                    uuid VARCHAR NOT NULL PRIMARY KEY,
                    name VARCHAR NOT NULL,
                    created_at DATETIME NOT NULL,
                    updated_at DATETIME NOT NULL,
                    model_config_node_uuid VARCHAR
                )
            """)
        _run_migrations(engine, "head")
        tc_cols = _columns(engine, "training_configs")
        assert "model_config_node_uuid" in tc_cols
        assert "trainer_type" not in tc_cols
        assert "_alembic_tmp_training_configs" not in _tables(engine)


class TestDropModelBundles:
    """9c62f3502b03 removes the dead model_bundles table.

    SQLite's DDL is non-transactional, so a half-applied run can leave the
    table already gone; re-running must not raise (same reason f93545d9a50b
    guards every one of its steps).
    """

    def test_table_is_gone_at_head(self, pre_redesign_copy):
        engine = pre_redesign_copy
        _run_migrations(engine, "head")
        assert "model_bundles" not in _tables(engine)

    def test_drop_is_safe_when_the_table_is_already_absent(self, pre_redesign_copy):
        engine = pre_redesign_copy
        _run_migrations(engine, "c1e5a83f7d92")
        assert "model_bundles" in _tables(engine)
        with engine.begin() as conn:
            conn.exec_driver_sql("DROP TABLE model_bundles")
        # Must not raise OperationalError: no such table
        _run_migrations(engine, "head")
        assert "model_bundles" not in _tables(engine)

    def test_downgrade_recreates_the_table(self, pre_redesign_copy):
        """The downgrade is the only way back; it must actually rebuild the
        table rather than silently no-op."""
        engine = pre_redesign_copy
        _run_migrations(engine, "head")
        assert "model_bundles" not in _tables(engine)
        _run_downgrade(engine, "c1e5a83f7d92")
        assert "model_bundles" in _tables(engine)
        assert "bundle_path" in _columns(engine, "model_bundles")


class TestMigrationPath:
    def test_upgrade_from_pre_redesign_adds_new_columns(self, pre_redesign_copy):
        """Running migrations from 307a345162a9 → head must:
        - add model_type + architecture to model_config_nodes (drop model_typename)
        - add model_config_node_uuid to training_configs (drop trainer_type + model_config_uuid)
        - create model_bundles table, then drop it again in 9c62f3502b03
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

        # Created by f93545d9a50b, dropped again by 9c62f3502b03 — at head the
        # bundle concept is gone entirely.
        assert "model_bundles" not in _tables(engine), "model_bundles table should have been dropped"
        assert "training_config_label_links" not in _tables(
            engine
        ), "training_config_label_links should have been dropped"


class TestPerformanceIndexes:
    """Guard the label reverse-lookup indexes added by d8e2f1a4b6c9.

    SQLite batch_alter_table recreates the whole table, which silently drops any
    index the batch op is unaware of — this already happened to the earlier
    migration-only (image_uuid, label_uuid) composite. These indexes must survive
    all the way to head so the dataset/label filters keep their covering seek.
    """

    def test_label_reverse_indexes_present_at_head(self, pre_redesign_copy):
        engine = pre_redesign_copy
        _run_migrations(engine, "head")

        bboxes_idx = _indexes(engine, "bboxes")
        assert "ix_bboxes_label_uuid_image_uuid" in bboxes_idx, "composite label→image index missing at head"
        # Superseded by the composite's leading column; must be dropped.
        assert "ix_bboxes_label_uuid" not in bboxes_idx, "redundant single-column label index still present"
        assert "ix_image_labels_label_uuid" in _indexes(engine, "image_labels"), "image_labels label index missing"

    def test_image_label_composite_present_at_head(self, pre_redesign_copy):
        """The (image_uuid, label_uuid) composite from b1d4e8f2a093 was silently
        dropped by the phash batch_alter_table migration; it must be restored and
        survive to head — the correlated-EXISTS image probes depend on it."""
        engine = pre_redesign_copy
        _run_migrations(engine, "head")

        bboxes_idx = _indexes(engine, "bboxes")
        assert "ix_bboxes_image_uuid_label_uuid" in bboxes_idx, "composite image→label index missing at head"
        # Superseded by the composite's leading column; must be dropped.
        assert "ix_bboxes_image_uuid" not in bboxes_idx, "redundant single-column image index still present"

    def test_image_label_composite_present_on_fresh_create_all(self):
        """Fresh installs use create_all + stamp head (migrations never replay),
        so the composite must also live on the model, not just in a migration."""
        engine = _make_engine()
        SQLModel.metadata.create_all(engine)

        bboxes_idx = _indexes(engine, "bboxes")
        assert "ix_bboxes_image_uuid_label_uuid" in bboxes_idx, "composite image→label index missing from model"
        assert "ix_bboxes_image_uuid" not in bboxes_idx, "redundant single-column image index still on model"

    def test_images_sort_indexes_present_at_head_and_on_create_all(self, pre_redesign_copy):
        """(created_at, uuid) / (updated_at, uuid) on images let the correlated
        EXISTS search dispatch stop at LIMIT instead of full-scan + temp-sort."""
        migrated = pre_redesign_copy
        _run_migrations(migrated, "head")
        fresh = _make_engine()
        SQLModel.metadata.create_all(fresh)

        for engine, origin in ((migrated, "migration head"), (fresh, "fresh create_all")):
            images_idx = _indexes(engine, "images")
            assert "ix_images_created_at_uuid" in images_idx, f"created_at sort index missing on {origin}"
            assert "ix_images_updated_at_uuid" in images_idx, f"updated_at sort index missing on {origin}"


class TestLegacyDatasetFilterJson:
    """b7d31c904ae2 converts pre-1.4b flat filter_json into the term list.

    PR 1.4b made DatasetFilterCriteria extra="forbid" without migrating stored
    rows, and DatasetRead validates filter_json on read — so a single legacy row
    (every install has one: the seeded "Whole Database" dataset) fails the whole
    GET /api/datasets listing, not just that row, and every dataset picker in
    the app goes with it.
    """

    LEGACY = {
        "tag_uuids": ["tag-a", "tag-b"],
        "tag_exclude_uuids": ["tag-x"],
        "tag_match_mode": "all",
        "label_exclude_uuids": ["lbl-x"],
        "label_match_mode": "all",
        "label_include_children": True,
        "date_from": "2026-01-01T00:00:00+00:00",
        "date_to": None,
        "filename_search": "cam3",
        "verification_check": "has_unverified",
        "image_tag_conditions": [{"tag_uuid": "tag-c", "state": "unset"}],
        "image_label_conditions": [{"label_uuid": "lbl-c", "state": "uncertain"}],
    }

    def _insert(self, engine, uuid: str, filter_json: dict) -> None:
        import json as _json

        with engine.connect() as conn:
            conn.exec_driver_sql(
                "INSERT INTO datasets (uuid, name, description, filter_json, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, '2026-01-01 00:00:00', '2026-01-01 00:00:00')",
                (uuid, f"ds-{uuid}", None, _json.dumps(filter_json)),
            )
            conn.commit()

    def _filter_json(self, engine, uuid: str) -> dict:
        import json as _json

        with engine.connect() as conn:
            raw = conn.exec_driver_sql("SELECT filter_json FROM datasets WHERE uuid = ?", (uuid,)).scalar()
        return _json.loads(raw) if isinstance(raw, str) else raw

    def test_legacy_row_validates_against_the_current_model_after_upgrade(self, pre_redesign_copy):
        """The bug itself: before this migration the row 500s every listing."""
        from apps.helpers.db.types import DatasetFilterCriteria

        self._insert(pre_redesign_copy, "legacy-1", self.LEGACY)
        _run_migrations(pre_redesign_copy)

        # Must not raise — this is exactly what DatasetRead does on every read.
        DatasetFilterCriteria(**self._filter_json(pre_redesign_copy, "legacy-1"))

    def test_flat_fields_become_equivalent_terms(self, pre_redesign_copy):
        self._insert(pre_redesign_copy, "legacy-2", self.LEGACY)
        _run_migrations(pre_redesign_copy)
        fj = self._filter_json(pre_redesign_copy, "legacy-2")

        by_kind: dict[str, list] = {}
        for term in fj["terms"]:
            by_kind.setdefault(term["kind"], []).append(term)

        # Included tags keep their match mode; each excluded tag is its own negated term.
        included = [t for t in by_kind["bboxTag"] if not t["negate"]]
        excluded = [t for t in by_kind["bboxTag"] if t["negate"]]
        assert included[0]["tag_uuids"] == ["tag-a", "tag-b"]
        assert included[0]["match_mode"] == "all"
        assert [t["tag_uuids"] for t in excluded] == [["tag-x"]]

        assert by_kind["date"][0]["field"] == "created"
        assert by_kind["date"][0]["from_"] == "2026-01-01T00:00:00+00:00"
        assert by_kind["filename"][0]["patterns"] == ["cam3"]

        # The two "never assessed" spellings both become 'unreviewed'.
        assert by_kind["imageTag"][0] == {
            "kind": "imageTag",
            "tag_uuid": "tag-c",
            "state": "unreviewed",
            "negate": False,
        }
        assert by_kind["imageLabel"][0]["label_uuid"] == "lbl-c"
        assert by_kind["imageLabel"][0]["state"] == "unreviewed"
        # Pre-1.4b conditions meant "this exact label", so roll-up must stay off
        # or a migrated filter silently starts matching more images than before.
        assert by_kind["imageLabel"][0]["roll_up"] is False

    def test_label_modifier_fields_survive(self, pre_redesign_copy):
        """These are not legacy — they modify Dataset.label_uuids, a separate column."""
        self._insert(pre_redesign_copy, "legacy-3", self.LEGACY)
        _run_migrations(pre_redesign_copy)
        fj = self._filter_json(pre_redesign_copy, "legacy-3")

        assert fj["label_exclude_uuids"] == ["lbl-x"]
        assert fj["label_match_mode"] == "all"
        assert fj["label_include_children"] is True
        assert fj["verification_check"] == "has_unverified"

    def test_vacuous_legacy_row_becomes_an_empty_filter(self, pre_redesign_copy):
        """The seeded "Whole Database" row: all-empty flat fields, no filters."""
        vacuous = {
            "tag_uuids": [],
            "tag_exclude_uuids": [],
            "tag_match_mode": "any",
            "label_exclude_uuids": [],
            "label_match_mode": "any",
            "label_include_children": False,
            "date_from": None,
            "date_to": None,
            "filename_search": None,
            "verification_check": None,
            "image_tag_conditions": [],
            "image_label_conditions": [],
        }
        self._insert(pre_redesign_copy, "legacy-whole-db", vacuous)
        _run_migrations(pre_redesign_copy)

        fj = self._filter_json(pre_redesign_copy, "legacy-whole-db")
        assert fj["terms"] == []
        assert not any(k in fj for k in ("tag_uuids", "date_from", "image_label_conditions"))

    def test_already_migrated_rows_are_left_alone(self, pre_redesign_copy):
        current = {
            "label_exclude_uuids": [],
            "label_match_mode": "any",
            "label_include_children": False,
            "verification_check": None,
            "terms": [{"kind": "filename", "patterns": ["already"], "negate": False}],
        }
        self._insert(pre_redesign_copy, "current-1", current)
        _run_migrations(pre_redesign_copy)

        assert self._filter_json(pre_redesign_copy, "current-1") == current

    def test_empty_filter_json_is_untouched(self, pre_redesign_copy):
        self._insert(pre_redesign_copy, "empty-1", {})
        _run_migrations(pre_redesign_copy)
        assert self._filter_json(pre_redesign_copy, "empty-1") == {}
