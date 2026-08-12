"""Tests for Manager.get_status() notices, specifically the _build_phase
message variants introduced alongside the on_phase build callback.

Also covers the startup-gate middleware: while _db_ready is False, all
non-/status routes must return 503 so that DB requests cannot arrive while
alembic migration is still running.
"""

import pytest
from httpx import ASGITransport, AsyncClient

from apps.db_server.manager import Manager
from apps.db_server.web import WebApp


@pytest.fixture
async def manager(db_client, files_root):
    """Uninitialised Manager — _hash_index_ready stays False."""
    return Manager(db_client=db_client, files_root=files_root)


@pytest.fixture
async def unready_client(db_client, files_root):
    """HTTP client backed by a WebApp whose manager has _db_ready=False (migration in progress)."""
    manager = Manager(db_client=db_client, files_root=files_root)
    # Deliberately do NOT set manager._db_ready = True
    web_app = WebApp(app_name="test", manager=manager)
    transport = ASGITransport(app=web_app.app())
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


class TestGetStatusBuildPhase:
    """Phase-specific notices emitted while the hash index is building."""

    async def test_build_notice_present_before_ready(self, manager):
        status = await manager.get_status()
        keys = [n["key"] for n in status["notices"]]
        assert "hash_build" in keys

    async def test_loading_phase_message(self, manager):
        manager._build_phase = "loading"
        status = await manager.get_status()
        notice = next(n for n in status["notices"] if n["key"] == "hash_build")
        assert "loading hashes from DB" in notice["message"]

    async def test_constructing_phase_message(self, manager):
        manager._build_phase = "constructing"
        status = await manager.get_status()
        notice = next(n for n in status["notices"] if n["key"] == "hash_build")
        assert "constructing search trees" in notice["message"]

    async def test_unknown_phase_falls_back_to_generic_message(self, manager):
        manager._build_phase = "unknown_phase"
        status = await manager.get_status()
        notice = next(n for n in status["notices"] if n["key"] == "hash_build")
        assert "loading" not in notice["message"]
        assert "constructing" not in notice["message"]
        # Generic fallback still contains the base message
        assert "similarity index" in notice["message"]

    async def test_no_build_notice_after_ready(self, manager):
        manager._hash_index_ready = True
        status = await manager.get_status()
        keys = [n["key"] for n in status["notices"]]
        assert "hash_build" not in keys

    async def test_rebuild_notice_when_rebuild_in_progress(self, manager):
        manager._hash_index_ready = True
        manager._rebuild_in_progress = True
        status = await manager.get_status()
        keys = [n["key"] for n in status["notices"]]
        assert "hash_rebuild" in keys
        notice = next(n for n in status["notices"] if n["key"] == "hash_rebuild")
        assert notice["level"] == "warning"


class TestGetStatusMigrationPhase:
    """Notices during alembic migration and on initialization failure."""

    async def test_migrating_phase_shows_db_migrating_notice(self, manager):
        manager._build_phase = "migrating"
        status = await manager.get_status()
        keys = [n["key"] for n in status["notices"]]
        assert "db_migrating" in keys
        assert "hash_build" not in keys, "hash_build must not appear while migrating"

    async def test_migrating_phase_notice_is_info_level(self, manager):
        manager._build_phase = "migrating"
        status = await manager.get_status()
        notice = next(n for n in status["notices"] if n["key"] == "db_migrating")
        assert notice["level"] == "info"

    async def test_init_error_shows_error_notice(self, manager):
        manager._init_error = "alembic upgrade failed: table already exists"
        status = await manager.get_status()
        keys = [n["key"] for n in status["notices"]]
        assert "db_init_error" in keys
        assert "hash_build" not in keys

    async def test_init_error_notice_is_error_level(self, manager):
        manager._init_error = "alembic upgrade failed: table already exists"
        status = await manager.get_status()
        notice = next(n for n in status["notices"] if n["key"] == "db_init_error")
        assert notice["level"] == "error"

    async def test_init_error_message_contains_reason(self, manager):
        reason = "alembic upgrade failed: table already exists"
        manager._init_error = reason
        status = await manager.get_status()
        notice = next(n for n in status["notices"] if n["key"] == "db_init_error")
        assert reason in notice["message"]

    async def test_init_error_takes_priority_over_migrating_phase(self, manager):
        """If both _init_error and _build_phase are set (shouldn't happen normally),
        the error notice must win so the user sees the failure."""
        manager._init_error = "boom"
        manager._build_phase = "migrating"
        status = await manager.get_status()
        keys = [n["key"] for n in status["notices"]]
        assert "db_init_error" in keys
        assert "db_migrating" not in keys


class TestStartupGateMiddleware:
    """While _db_ready is False the server must reject non-/status requests with 503."""

    async def test_status_returns_ok_while_not_ready(self, unready_client):
        r = await unready_client.get("/status")
        assert r.status_code == 200

    async def test_api_request_returns_503_while_not_ready(self, unready_client):
        r = await unready_client.get("/api/labels")
        assert r.status_code == 503

    async def test_503_body_is_json_with_detail(self, unready_client):
        r = await unready_client.post("/api/labels", json={"name": "cat", "color": "#f00"})
        assert r.status_code == 503
        body = r.json()
        assert "detail" in body


class TestGetStatusDetails:
    """The Service Status panel's expandable detail view for db_server."""

    async def test_summary_reports_the_indexed_image_count(self, manager):
        status = await manager.get_status()
        assert status["summary"].startswith("index: ")

    async def test_summary_flags_a_needed_rebuild(self, manager):
        manager._hash_index._needs_rebuild = True
        status = await manager.get_status()
        assert "rebuild" in status["summary"]

    async def test_index_section_reports_phase_and_readiness(self, manager):
        manager._hash_index_ready = True
        manager._build_phase = "constructing"
        status = await manager.get_status()
        rows = {r["label"]: r["value"] for r in status["details"][0]["rows"]}
        assert rows["Build phase"] == "constructing"
        assert rows["Ready"] == "yes"
        assert rows["Needs rebuild"] == "no"

    async def test_bbox_trees_are_aggregated_not_listed_by_uuid(self, manager):
        """Trees are keyed by label uuid, and resolving those to names would
        cost a DB round-trip on every notices poll — so the section reports
        totals rather than rows of raw uuids, which tell a reader nothing."""
        manager._hash_index._bbox_trees = {"cat-uuid": None, "dog-uuid": None}
        manager._hash_index._valid_bboxes = {"cat-uuid": {"a"}, "dog-uuid": {"b", "c", "d"}}
        status = await manager.get_status()
        rows = {r["label"]: r["value"] for r in status["details"][0]["rows"]}
        assert rows["Bbox trees"] == "2 labels, 4 entries (0 dead)"
        assert rows["Largest bbox tree"] == "3"
        assert "cat-uuid" not in str(status["details"])

    async def test_largest_bbox_tree_is_a_dash_when_there_are_none(self, manager):
        status = await manager.get_status()
        rows = {r["label"]: r["value"] for r in status["details"][0]["rows"]}
        assert rows["Largest bbox tree"] == "—"
        assert rows["Bbox trees"] == "0 labels, 0 entries (0 dead)"
