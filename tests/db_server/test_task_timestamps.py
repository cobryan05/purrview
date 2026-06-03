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
    assert instances[0]["started_at"] is not None


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
    assert instance["started_at"] is not None
    assert instance["ended_at"] is not None
    assert instance["log_json"] == logs


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
    assert instance["started_at"] is not None
    assert instance["ended_at"] is not None


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
    assert r2.json()["instances"][0]["log_json"] == logs
