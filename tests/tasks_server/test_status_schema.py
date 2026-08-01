import asyncio

from tests.tasks_server.conftest import (
    CONFIG_UUID,
    STATUS_SCHEMA_CONFIG_UUID,
    STATUS_SCHEMA_TASK_UUID,
    TASK_UUID,
)


class TestTaskTypeStatusSchema:
    """POST /api/tasks/types/status-schema"""

    async def test_default_schema_matches_todays_card(self, http_client):
        """A task overriding nothing must reproduce today's card exactly:
        progress bar + status message + log tail, all summary-sourced."""
        resp = await http_client.post(
            "/api/tasks/types/status-schema",
            json={"taskTypenames": ["TestTask"]},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"
        assert data["schemas"]["TestTask"] == [
            {"id": "progress", "type": "progress_bar", "source": "summary"},
            {"id": "statusMessage", "type": "text", "source": "summary"},
            {"id": "log", "type": "log_tail", "source": "summary", "lines": 5},
        ]

    async def test_overridden_schema_round_trips(self, http_client):
        resp = await http_client.post(
            "/api/tasks/types/status-schema",
            json={"taskTypenames": ["StatusSchemaTask"]},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"
        schema = data["schemas"]["StatusSchemaTask"]
        assert schema[0] == {"id": "progress", "type": "progress_bar", "source": "summary"}
        # "source" defaults to "status" when absent, and unknown widget types must
        # round-trip untouched -- the backend never rejects them, the frontend skips them.
        assert schema[1]["id"] == "counter"
        assert schema[1]["type"] == "stat_grid"
        assert "source" not in schema[1]
        assert schema[2] == {"id": "mystery_widget", "type": "some_future_widget_type"}

    async def test_unknown_typename_returns_failure(self, http_client):
        resp = await http_client.post(
            "/api/tasks/types/status-schema",
            json={"taskTypenames": ["NonExistentTask"]},
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "failure"


class TestTaskInstanceStatusData:
    """POST /api/tasks/instances/status-data"""

    async def test_status_data_for_running_task(self, http_client):
        await http_client.post(
            "/api/tasks/instances/start",
            json={"configUuid": STATUS_SCHEMA_CONFIG_UUID},
        )
        await asyncio.sleep(0)

        resp = await http_client.post(
            "/api/tasks/instances/status-data",
            json={"taskUuids": [STATUS_SCHEMA_TASK_UUID]},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"
        assert data["statusData"] == {STATUS_SCHEMA_TASK_UUID: {"counter": 0}}

    async def test_unknown_uuid_is_omitted_not_erroring(self, http_client):
        resp = await http_client.post(
            "/api/tasks/instances/status-data",
            json={"taskUuids": ["does-not-exist"]},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"
        assert data["statusData"] == {}

    async def test_not_running_task_is_omitted(self, http_client):
        # TASK_UUID's config was never started in this test, so it's not in
        # _running_tasks -- must be omitted rather than raising.
        resp = await http_client.post(
            "/api/tasks/instances/status-data",
            json={"taskUuids": [TASK_UUID]},
        )
        assert resp.status_code == 200
        assert resp.json()["statusData"] == {}


class TestTaskCommand:
    """POST /api/tasks/instances/command"""

    async def test_command_on_task_without_handle_command_returns_clean_failure(self, http_client):
        """Base Task.handle_command() raises NotImplementedError; the endpoint must
        translate that into a clean failure response, never an unhandled 500."""
        await http_client.post(
            "/api/tasks/instances/start",
            json={"configUuid": CONFIG_UUID},
        )
        await asyncio.sleep(0)

        resp = await http_client.post(
            "/api/tasks/instances/command",
            json={"taskUuid": TASK_UUID, "command": "capture_now", "params": {}},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "failure"
        assert data["message"]

    async def test_command_on_task_that_implements_it_sets_flag(self, http_client):
        await http_client.post(
            "/api/tasks/instances/start",
            json={"configUuid": STATUS_SCHEMA_CONFIG_UUID},
        )
        await asyncio.sleep(0)

        resp = await http_client.post(
            "/api/tasks/instances/command",
            json={"taskUuid": STATUS_SCHEMA_TASK_UUID, "command": "capture_now", "params": {}},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"

        status_resp = await http_client.post(
            "/api/tasks/instances/status-data",
            json={"taskUuids": [STATUS_SCHEMA_TASK_UUID]},
        )
        assert status_resp.json()["statusData"] == {STATUS_SCHEMA_TASK_UUID: {"counter": 1}}

    async def test_command_on_unrecognized_command_returns_clean_failure(self, http_client):
        await http_client.post(
            "/api/tasks/instances/start",
            json={"configUuid": STATUS_SCHEMA_CONFIG_UUID},
        )
        await asyncio.sleep(0)

        resp = await http_client.post(
            "/api/tasks/instances/command",
            json={"taskUuid": STATUS_SCHEMA_TASK_UUID, "command": "not_a_real_command", "params": {}},
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "failure"

    async def test_command_on_unknown_task_uuid_returns_clean_failure(self, http_client):
        resp = await http_client.post(
            "/api/tasks/instances/command",
            json={"taskUuid": "does-not-exist", "command": "capture_now", "params": {}},
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "failure"
