"""The Service Status panel's detail view for the tasks server.

The count alone says the service is busy; the detail rows say *what with*.
"""

import uuid
from unittest.mock import MagicMock

import db_client.models as db_models

from apps.helpers.consts import TaskStatus
from apps.tasks_server.types import TaskInfo


def _running(manager, name: str, typename: str = "TestTask", *, progress=0.25, message="", paused=False):
    config_uuid, instance_uuid = str(uuid.uuid4()), str(uuid.uuid4())
    config = db_models.TaskConfigRead(
        uuid=config_uuid,
        name=name,
        typename=typename,
        params={},
        marked_for_delete=False,
        description=None,
    )
    instance = db_models.TaskInstanceRead(
        uuid=instance_uuid,
        config_uuid=config_uuid,
        status=TaskStatus.RUNNING,
        resume_data=None,
        result=None,
        error_message=None,
    )
    task = MagicMock()
    task.get_notices.return_value = []
    task.get_progress.return_value = progress
    task.get_status_message.return_value = message
    task.is_paused_in_place.return_value = paused
    manager._running_tasks[instance_uuid] = TaskInfo(config=config, instance=instance, task=task)


class TestTasksStatusDetails:
    def test_no_running_tasks_reports_an_empty_section(self, manager):
        status = manager.get_status()

        assert status["tasks"]["running"] == 0
        assert status["details"][0]["rows"] == []
        assert status["details"][0]["empty"] == "No tasks running"

    def test_running_tasks_are_named_with_their_type(self, manager):
        _running(manager, "nightly-import", typename="ImportTask")

        rows = manager.get_status()["details"][0]["rows"]

        assert rows[0]["label"] == "nightly-import (ImportTask)"

    def test_rows_are_sorted_by_name(self, manager):
        _running(manager, "zebra")
        _running(manager, "Alpha")

        rows = manager.get_status()["details"][0]["rows"]

        assert [r["label"].split(" ")[0] for r in rows] == ["Alpha", "zebra"]

    def test_row_carries_status_progress_and_message(self, manager):
        _running(manager, "scan", progress=0.5, message="halfway")

        value = manager.get_status()["details"][0]["rows"][0]["value"]

        assert TaskStatus.RUNNING in value
        assert "50%" in value
        assert "halfway" in value

    def test_indeterminate_progress_is_not_rendered_as_zero_percent(self, manager):
        """get_progress() returning None means indeterminate, not 0% — showing
        "0%" would claim a task is stuck when it simply doesn't report."""
        _running(manager, "indeterminate", progress=None)

        value = manager.get_status()["details"][0]["rows"][0]["value"]

        assert "%" not in value

    def test_paused_tasks_are_flagged_in_the_row_and_the_summary(self, manager):
        _running(manager, "paused-one", paused=True)
        _running(manager, "busy-one")

        status = manager.get_status()
        rows = {r["label"]: r["value"] for r in status["details"][0]["rows"]}

        assert "paused" in rows["paused-one (TestTask)"]
        assert status["tasks"]["paused"] == 1
        assert status["summary"] == "running: 2 (1 paused)"

    def test_summary_is_the_running_count_when_nothing_is_paused(self, manager):
        _running(manager, "busy-one")

        assert manager.get_status()["summary"] == "running: 1"

    async def test_status_endpoint_serves_the_details(self, http_client, manager):
        _running(manager, "nightly-import")

        resp = await http_client.get("/status")

        assert resp.status_code == 200
        data = resp.json()
        assert data["service"] == "tasks_server"
        assert data["details"][0]["rows"][0]["label"].startswith("nightly-import")
