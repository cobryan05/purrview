"""Tests for task-instance error_message persistence.

The message explaining a failure used to live only in result_json, which is
one-shot — fetched once, then deleted — so the reason a task died disappeared
from the UI as soon as anything read it. It is now written to the (long-
existing but previously unused) task_instances.error_message column alongside
the ERROR status, via PATCH /api/task-instances/{uuid}/status.
"""

from apps.helpers.consts import TaskStatus

# ── Helpers ─────────────────────────────────────────────────────────────────────


async def _create_config(http_client, name="err-task") -> str:
    r = await http_client.post(
        "/api/task-configs",
        json={"name": name, "typename": "TestTask", "params": {}},
    )
    assert r.status_code == 200
    return r.json()["config"]["uuid"]


async def _create_instance(http_client, config_uuid: str) -> str:
    r = await http_client.post("/api/task-instances", json={"configUuid": config_uuid})
    assert r.status_code == 200
    return r.json()["instance"]["uuid"]


async def _read_instance(http_client, inst: str) -> dict:
    r = await http_client.get("/api/task-instances", params={"uuids": [inst]})
    assert r.status_code == 200
    instances = r.json()["instances"]
    assert len(instances) == 1
    return instances[0]


async def _set_status(http_client, inst: str, status: str, error_message: str | None = None):
    payload: dict = {"status": status}
    if error_message is not None:
        payload["errorMessage"] = error_message
    r = await http_client.patch(f"/api/task-instances/{inst}/status", json=payload)
    assert r.status_code == 200
    assert r.json()["status"] == "success"


class TestTaskErrorMessage:
    async def test_error_status_persists_the_message(self, http_client):
        cfg = await _create_config(http_client)
        inst = await _create_instance(http_client, cfg)

        msg = "URLError: <urlopen error _ssl.c:999: The handshake operation timed out>"
        await _set_status(http_client, inst, TaskStatus.ERROR, msg)

        instance = await _read_instance(http_client, inst)
        assert instance["status"] == TaskStatus.ERROR
        assert instance["errorMessage"] == msg

    async def test_message_survives_a_result_fetch(self, http_client):
        """The whole point: unlike result_json, this isn't consumed by reading it."""
        cfg = await _create_config(http_client, "err-task-survives")
        inst = await _create_instance(http_client, cfg)
        await _set_status(http_client, inst, TaskStatus.ERROR, "boom")

        r = await http_client.post("/api/task-instances/results", json={"taskUuids": [inst]})
        assert r.status_code == 200

        assert (await _read_instance(http_client, inst))["errorMessage"] == "boom"

    async def test_error_without_a_message_is_null(self, http_client):
        cfg = await _create_config(http_client, "err-task-nomsg")
        inst = await _create_instance(http_client, cfg)
        await _set_status(http_client, inst, TaskStatus.ERROR)

        assert (await _read_instance(http_client, inst))["errorMessage"] is None

    async def test_a_non_error_status_clears_a_previous_message(self, http_client):
        """A resumed instance that is running again must not still show why it
        failed last time — the card would be reporting a state that isn't true."""
        cfg = await _create_config(http_client, "err-task-cleared")
        inst = await _create_instance(http_client, cfg)
        await _set_status(http_client, inst, TaskStatus.ERROR, "transient camera timeout")
        assert (await _read_instance(http_client, inst))["errorMessage"] == "transient camera timeout"

        await _set_status(http_client, inst, TaskStatus.RUNNING)

        instance = await _read_instance(http_client, inst)
        assert instance["status"] == TaskStatus.RUNNING
        assert instance["errorMessage"] is None
