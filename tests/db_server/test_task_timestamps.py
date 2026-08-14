"""Tests for task-instance timestamps and log persistence (Phase 2A).

Exercises the PATCH /api/task-instances/{uuid}/timestamps endpoint and verifies
that started_at, ended_at, and log_json are stored and returned in list responses.
"""

from datetime import datetime, timezone

import pytest

from apps.helpers.consts import TaskStatus

# ── Helpers ─────────────────────────────────────────────────────────────────────


async def _create_config(http_client, name="ts-task") -> str:
    r = await http_client.post(
        "/api/task-configs",
        json={"name": name, "typename": "TestTask", "params": {}},
    )
    assert r.status_code == 200
    return r.json()["config"]["uuid"]


async def _create_instance(http_client, config_uuid: str) -> str:
    r = await http_client.post(
        "/api/task-instances",
        json={"configUuid": config_uuid},
    )
    assert r.status_code == 200
    return r.json()["instance"]["uuid"]


# ── Tests ────────────────────────────────────────────────────────────────────────


async def test_set_timestamps_started_at(http_client):
    """PATCH /timestamps sets started_at on the instance."""
    cfg = await _create_config(http_client)
    inst = await _create_instance(http_client, cfg)

    started = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    r = await http_client.patch(
        f"/api/task-instances/{inst}/timestamps",
        json={"startedAt": started.isoformat()},
    )
    assert r.status_code == 200
    assert r.json()["status"] == "success"

    # Verify via list
    r2 = await http_client.get("/api/task-instances", params={"uuids": [inst]})
    assert r2.status_code == 200
    instances = r2.json()["instances"]
    assert len(instances) == 1
    assert instances[0]["startedAt"] is not None


async def test_set_timestamps_ended_at_and_logs(http_client):
    """PATCH /timestamps sets ended_at and log_json together."""
    cfg = await _create_config(http_client, "ts-task-2")
    inst = await _create_instance(http_client, cfg)

    started = datetime(2026, 1, 1, 10, 0, 0, tzinfo=timezone.utc)
    ended = datetime(2026, 1, 1, 10, 5, 0, tzinfo=timezone.utc)
    logs = ["Starting up", "Running inference", "Completed successfully"]

    # Set started_at first
    r = await http_client.patch(
        f"/api/task-instances/{inst}/timestamps",
        json={"startedAt": started.isoformat()},
    )
    assert r.json()["status"] == "success"

    # Then set ended_at + logs
    r = await http_client.patch(
        f"/api/task-instances/{inst}/timestamps",
        json={"endedAt": ended.isoformat(), "logJson": logs},
    )
    assert r.status_code == 200
    assert r.json()["status"] == "success"

    # Verify
    r2 = await http_client.get("/api/task-instances", params={"uuids": [inst]})
    assert r2.status_code == 200
    instance = r2.json()["instances"][0]
    assert instance["startedAt"] is not None
    assert instance["endedAt"] is not None
    assert instance["logLines"] == logs


async def test_set_timestamps_partial_update_preserves_other_fields(http_client):
    """Setting only ended_at does not clear started_at."""
    cfg = await _create_config(http_client, "ts-task-3")
    inst = await _create_instance(http_client, cfg)

    started = datetime(2026, 1, 1, 9, 0, 0, tzinfo=timezone.utc)
    await http_client.patch(
        f"/api/task-instances/{inst}/timestamps",
        json={"startedAt": started.isoformat()},
    )

    ended = datetime(2026, 1, 1, 9, 30, 0, tzinfo=timezone.utc)
    await http_client.patch(
        f"/api/task-instances/{inst}/timestamps",
        json={"endedAt": ended.isoformat()},
    )

    r = await http_client.get("/api/task-instances", params={"uuids": [inst]})
    instance = r.json()["instances"][0]
    assert instance["startedAt"] is not None
    assert instance["endedAt"] is not None


async def test_set_timestamps_unknown_instance_is_noop(http_client):
    """PATCH /timestamps with an unknown UUID returns success (noop, not 404)."""
    r = await http_client.patch(
        "/api/task-instances/00000000-0000-0000-0000-000000000099/timestamps",
        json={"startedAt": "2026-01-01T00:00:00+00:00"},
    )
    assert r.status_code == 200
    assert r.json()["status"] == "success"


async def test_log_json_roundtrip(http_client):
    """log_json stores and returns an arbitrary string list."""
    cfg = await _create_config(http_client, "ts-task-logs")
    inst = await _create_instance(http_client, cfg)

    logs = ["line 1", "line 2 with unicode: 🐱", "line 3"]
    r = await http_client.patch(
        f"/api/task-instances/{inst}/timestamps",
        json={"logJson": logs},
    )
    assert r.json()["status"] == "success"

    r2 = await http_client.get("/api/task-instances", params={"uuids": [inst]})
    assert r2.json()["instances"][0]["logLines"] == logs


# ── Persisted status-widget snapshot ────────────────────────────────────────────


class TestStatusDataPersistence:
    """A task's status widgets (histograms, stat grids, source tables) read from
    get_status_data(), which lives on the task object and dies with the run. The
    last snapshot is persisted alongside the logs so a finished card can still
    render what the run produced."""

    async def test_status_data_round_trips(self, http_client):
        cfg = await _create_config(http_client, "sd-task")
        inst = await _create_instance(http_client, cfg)

        snapshot = {
            "stats": {"images_saved": 306, "images_processed": 5672},
            "dedup_image_hist": {"bins": [{"label": "0", "count": 12, "marked": True}], "total": 12},
            "sources": [{"name": "CatCloset", "health": "ok", "last_error": "—"}],
        }
        r = await http_client.patch(
            f"/api/task-instances/{inst}/timestamps",
            json={"statusDataJson": snapshot},
        )
        assert r.status_code == 200
        assert r.json()["status"] == "success"

        r2 = await http_client.get("/api/task-instances", params={"uuids": [inst]})
        assert r2.json()["instances"][0]["statusData"] == snapshot

    async def test_a_later_write_without_status_data_does_not_blank_it(self, http_client):
        """Timestamps are written at points where the snapshot isn't being sent
        (e.g. the started_at write at launch); those must leave it intact."""
        cfg = await _create_config(http_client, "sd-task-keep")
        inst = await _create_instance(http_client, cfg)
        snapshot = {"stats": {"images_saved": 7}}

        await http_client.patch(f"/api/task-instances/{inst}/timestamps", json={"statusDataJson": snapshot})
        await http_client.patch(
            f"/api/task-instances/{inst}/timestamps",
            json={"endedAt": datetime(2026, 1, 1, tzinfo=timezone.utc).isoformat()},
        )

        instance = (await http_client.get("/api/task-instances", params={"uuids": [inst]})).json()["instances"][0]
        assert instance["statusData"] == snapshot
        assert instance["endedAt"] is not None

    async def test_absent_status_data_is_null(self, http_client):
        cfg = await _create_config(http_client, "sd-task-none")
        inst = await _create_instance(http_client, cfg)
        instance = (await http_client.get("/api/task-instances", params={"uuids": [inst]})).json()["instances"][0]
        assert instance["statusData"] is None


class TestRestartClearsEndedAt:
    """A new run's started_at write must retire the previous run's ended_at.

    Resuming or auto-restarting an instance reuses the same row, so an ended_at
    left over from the previous run outlives it — producing an instance that is
    running while claiming to have ended, with ended_at *before* started_at.
    """

    async def test_started_at_clears_a_previous_runs_ended_at(self, http_client):
        cfg = await _create_config(http_client, "ts-task-restart")
        inst = await _create_instance(http_client, cfg)

        first_run_end = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        await http_client.patch(
            f"/api/task-instances/{inst}/timestamps",
            json={"endedAt": first_run_end.isoformat()},
        )

        second_run_start = datetime(2026, 1, 2, 9, 0, 0, tzinfo=timezone.utc)
        r = await http_client.patch(
            f"/api/task-instances/{inst}/timestamps",
            json={"startedAt": second_run_start.isoformat()},
        )
        assert r.status_code == 200

        instance = (await http_client.get("/api/task-instances", params={"uuids": [inst]})).json()["instances"][0]
        assert instance["startedAt"] is not None
        assert instance["endedAt"] is None, "a restarted instance must not keep the previous run's ended_at"

    async def test_ended_at_still_persists_when_written_after_started_at(self, http_client):
        """Clearing on start must not stop a run from recording its own end."""
        cfg = await _create_config(http_client, "ts-task-restart-end")
        inst = await _create_instance(http_client, cfg)

        await http_client.patch(
            f"/api/task-instances/{inst}/timestamps",
            json={"startedAt": datetime(2026, 1, 2, 9, 0, 0, tzinfo=timezone.utc).isoformat()},
        )
        await http_client.patch(
            f"/api/task-instances/{inst}/timestamps",
            json={"endedAt": datetime(2026, 1, 2, 10, 0, 0, tzinfo=timezone.utc).isoformat()},
        )

        instance = (await http_client.get("/api/task-instances", params={"uuids": [inst]})).json()["instances"][0]
        assert instance["endedAt"] is not None
