"""Tests for keeping resume_data off the task-instance listing payload.

A task's checkpoint (resume_data) is internal state — only the two tasks_server
paths that actually restart a task ever read it. It is also by far the heaviest
field on an instance: a finished TrainModelTask carries ~500 KB of epoch history.

GET /api/task-instances is polled twice a second by the task SSE stream, so
shipping checkpoints on every frame turned a listing into megabytes of JSON that
three services had to serialize, parse and re-serialize. The listing therefore
omits resume_data unless a caller explicitly asks for it with
include_resume_data=true.
"""

from apps.helpers.consts import TaskStatus

# ── Helpers ─────────────────────────────────────────────────────────────────────


async def _create_config(http_client, name="resume-task") -> str:
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


async def _set_resume_data(http_client, inst: str, resume_data: dict) -> None:
    r = await http_client.patch(
        f"/api/task-instances/{inst}/resume-data",
        json={"resumeData": resume_data},
    )
    assert r.status_code == 200
    assert r.json()["status"] == "success"


async def _list(http_client, inst: str, **params) -> dict:
    r = await http_client.get("/api/task-instances", params={"uuids": [inst], **params})
    assert r.status_code == 200
    instances = r.json()["instances"]
    assert len(instances) == 1
    return instances[0]


CHECKPOINT = {"epoch_history": [{"epoch": n, "loss": 0.5} for n in range(50)]}


class TestResumeDataOnListing:
    async def test_listing_omits_resume_data_by_default(self, http_client):
        cfg = await _create_config(http_client)
        inst = await _create_instance(http_client, cfg)
        await _set_resume_data(http_client, inst, CHECKPOINT)

        assert (await _list(http_client, inst))["resumeData"] is None

    async def test_listing_includes_resume_data_when_asked(self, http_client):
        """The resume paths must still be able to read the checkpoint."""
        cfg = await _create_config(http_client, "resume-task-optin")
        inst = await _create_instance(http_client, cfg)
        await _set_resume_data(http_client, inst, CHECKPOINT)

        instance = await _list(http_client, inst, include_resume_data="true")
        assert instance["resumeData"] == CHECKPOINT

    async def test_omitting_it_does_not_delete_the_checkpoint(self, http_client):
        """A lean read must not be a destructive read — the next resume needs it."""
        cfg = await _create_config(http_client, "resume-task-nondestructive")
        inst = await _create_instance(http_client, cfg)
        await _set_resume_data(http_client, inst, CHECKPOINT)

        assert (await _list(http_client, inst))["resumeData"] is None

        assert (await _list(http_client, inst, include_resume_data="true"))["resumeData"] == CHECKPOINT

    async def test_other_fields_are_unaffected(self, http_client):
        cfg = await _create_config(http_client, "resume-task-otherfields")
        inst = await _create_instance(http_client, cfg)
        await _set_resume_data(http_client, inst, CHECKPOINT)
        r = await http_client.patch(
            f"/api/task-instances/{inst}/status",
            json={"status": TaskStatus.PAUSED},
        )
        assert r.status_code == 200

        instance = await _list(http_client, inst)
        assert instance["status"] == TaskStatus.PAUSED
        assert instance["uuid"] == inst
        assert instance["configUuid"] == cfg

    async def test_resumable_filter_still_sees_the_checkpoint(self, http_client):
        """resumable= filters in SQL on the column, so it must keep working even
        though the column is no longer serialized into the response."""
        cfg = await _create_config(http_client, "resume-task-filter")
        inst = await _create_instance(http_client, cfg)
        await _set_resume_data(http_client, inst, CHECKPOINT)

        r = await http_client.get("/api/task-instances", params={"uuids": [inst], "resumable": "true"})
        assert r.status_code == 200
        assert [i["uuid"] for i in r.json()["instances"]] == [inst]

        r = await http_client.get("/api/task-instances", params={"uuids": [inst], "resumable": "false"})
        assert r.status_code == 200
        assert r.json()["instances"] == []
