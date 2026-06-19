"""Tests for Manager.get_status() notices, specifically the _build_phase
message variants introduced alongside the on_phase build callback.
"""

import pytest

from apps.db_server.manager import Manager


@pytest.fixture
async def manager(db_client, files_root):
    """Uninitialised Manager — _hash_index_ready stays False."""
    return Manager(db_client=db_client, files_root=files_root)


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
