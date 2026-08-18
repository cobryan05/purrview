"""Re-attaching to a remote training job across restarts and network blips.

A training runs on the training server — possibly on another host — and this
task only watches it. Everything here is about that asymmetry:

  - the watcher must record which job it is watching *before* it can be
    interrupted, or the job is orphaned with nothing pointing at it;
  - a dropped SSE stream means the watcher lost the connection, not that the
    training failed;
  - "the training server did not answer" is not "the job does not exist".
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from apps.tasks_server.tasks._training_job_runner import _StreamDropped
from apps.tasks_server.tasks.TrainModelTask import TrainModelTask


def _make_task(manager=None) -> TrainModelTask:
    if manager is None:
        # A caller that passes its own manager has already configured it — don't
        # overwrite the very mock the test is asserting on.
        manager = MagicMock()
        manager.update_task_heartbeat = AsyncMock()
        manager.save_task_resume_data = AsyncMock()
    task = TrainModelTask(task_config_uuid="cfg-uuid", params={}, manager=manager)
    task.set_instance_uuid("inst-uuid")
    task._api_client = MagicMock()
    task._training_base_url = "http://training:8005"
    task._model_name = "m1"
    task._manifest = []
    task._config = {"trainer_type": "binary_image_classifier"}
    task._epoch_history = []
    task._job_id = None
    return task


class TestCheckpointOnSubmit:
    async def test_job_id_is_persisted_before_the_job_can_start(self):
        """The window between submit and the next 60s checkpoint pass is exactly
        when a restart orphans a real training job, so the write must be
        immediate rather than flagged for later collection."""
        manager = MagicMock()
        manager.update_task_heartbeat = AsyncMock()
        manager.save_task_resume_data = AsyncMock()
        task = _make_task(manager)

        seen_at_push: list[str] = []

        async def _push_images(job_id, _manifest):
            # Whatever was written by now is what a restart here would find.
            seen_at_push.extend(c.args[1].get("job_id", "") for c in manager.save_task_resume_data.await_args_list)

        task._push_images = _push_images
        task._consume_sse = AsyncMock(return_value={"job_id": "job-1"})

        with patch("apps.tasks_server.tasks._training_job_runner.TrainingApi") as api_cls:
            api_cls.return_value.submit_training_job.return_value = MagicMock(job_id="job-1")
            await task._run_one_job()

        assert seen_at_push == ["job-1"]

    async def test_a_failed_checkpoint_write_falls_back_to_the_periodic_pass(self):
        """db_server being briefly down must not fail the training run."""
        manager = MagicMock()
        manager.update_task_heartbeat = AsyncMock()
        manager.save_task_resume_data = AsyncMock(side_effect=ConnectionError("db_server restarting"))
        task = _make_task(manager)
        task._push_images = AsyncMock()
        task._consume_sse = AsyncMock(return_value={"job_id": "job-1"})

        with patch("apps.tasks_server.tasks._training_job_runner.TrainingApi") as api_cls:
            api_cls.return_value.submit_training_job.return_value = MagicMock(job_id="job-1")
            await task._run_one_job()

        assert task._resume_data["job_id"] == "job-1"
        assert task.is_data_ready(), "the periodic checkpoint pass must still pick this up"


class TestRemoteStatusErrors:
    async def test_an_unreachable_server_is_not_reported_as_a_missing_job(self):
        """ "not_found" makes the task resubmit the whole job from scratch. A
        connection error means we don't know — those must not be the same
        answer, least of all with the training server on another host."""
        task = _make_task()

        with patch("apps.tasks_server.tasks._training_job_runner.TrainingApi") as api_cls:
            api_cls.return_value.get_training_job.side_effect = httpx.ConnectError("no route to host")
            status = await task._check_remote_job_status("job-1")

        assert status == "unreachable"

    async def test_a_404_is_reported_as_not_found(self):
        task = _make_task()

        class _ApiError(Exception):
            status = 404

        with patch("apps.tasks_server.tasks._training_job_runner.TrainingApi") as api_cls:
            api_cls.return_value.get_training_job.side_effect = _ApiError()
            status = await task._check_remote_job_status("job-1")

        assert status == "not_found"

    async def test_resume_waits_out_an_unreachable_server_instead_of_resubmitting(self):
        """Resubmitting on a blip would push the whole image set again and leave
        the original job training in the background."""
        task = _make_task()
        task._job_id = "job-1"
        statuses = iter(["unreachable", "unreachable", "running"])
        task._check_remote_job_status = AsyncMock(side_effect=lambda _job_id: next(statuses))
        task._consume_sse = AsyncMock(return_value={"job_id": "job-1"})
        submitted = MagicMock()

        with patch("apps.tasks_server.tasks._training_job_runner.TrainingApi", submitted):
            with patch("apps.tasks_server.tasks._training_job_runner.asyncio.sleep", new=AsyncMock()):
                result = await task._run_one_job()

        assert result == {"job_id": "job-1"}
        submitted.return_value.submit_training_job.assert_not_called()

    async def test_cancelling_while_the_server_is_unreachable_does_not_resubmit(self):
        """Give up quietly rather than starting a second copy of the training."""
        task = _make_task()
        task._job_id = "job-1"
        task._check_remote_job_status = AsyncMock(return_value="unreachable")
        task._cancel_flag.set()
        submitted = MagicMock()

        with patch("apps.tasks_server.tasks._training_job_runner.TrainingApi", submitted):
            with patch("apps.tasks_server.tasks._training_job_runner.asyncio.sleep", new=AsyncMock()):
                result = await task._run_one_job()

        assert result["cancelled"] is True
        submitted.return_value.submit_training_job.assert_not_called()


class TestSseReconnect:
    async def test_a_dropped_stream_reconnects_while_the_job_is_still_running(self):
        """The stream ending without a terminal event means the connection went
        away — the job on the other host is unaffected."""
        task = _make_task()
        task._job_id = "job-1"
        attempts = {"n": 0}

        async def _stream_once(_job_id):
            attempts["n"] += 1
            if attempts["n"] == 1:
                raise httpx.ReadError("connection reset")
            return {"job_id": "job-1", "model_name": "m1"}

        task._stream_job_events = _stream_once
        task._check_remote_job_status = AsyncMock(return_value="running")

        with patch("apps.tasks_server.tasks._training_job_runner.asyncio.sleep", new=AsyncMock()):
            result = await task._consume_sse("job-1")

        assert result["job_id"] == "job-1"
        assert attempts["n"] == 2

    async def test_a_job_that_finished_while_disconnected_is_collected_not_failed(self):
        task = _make_task()
        task._job_id = "job-1"
        task._stream_job_events = AsyncMock(side_effect=httpx.ReadError("connection reset"))
        task._check_remote_job_status = AsyncMock(return_value="done")
        task._save_artifact = AsyncMock()

        with patch("apps.tasks_server.tasks._training_job_runner.asyncio.sleep", new=AsyncMock()):
            result = await task._consume_sse("job-1")

        task._save_artifact.assert_awaited_once_with("job-1")
        assert result["job_id"] == "job-1"

    async def test_an_interrupted_job_is_restarted_on_the_training_server(self):
        """The training server restarted: its own job state says so, and the
        job resumes from its last epoch checkpoint."""
        task = _make_task()
        task._job_id = "job-1"
        streams = {"n": 0}

        async def _stream(_job_id):
            streams["n"] += 1
            if streams["n"] == 1:
                raise httpx.ReadError("training server went away")
            return {"job_id": "job-1"}

        task._stream_job_events = _stream
        task._check_remote_job_status = AsyncMock(return_value="interrupted")
        task._restart_remote_job = AsyncMock()

        with patch("apps.tasks_server.tasks._training_job_runner.asyncio.sleep", new=AsyncMock()):
            result = await task._consume_sse("job-1")

        task._restart_remote_job.assert_awaited_once_with("job-1")
        assert result == {"job_id": "job-1"}

    async def test_a_terminal_failure_still_fails_the_task(self):
        """Reconnecting must not turn a genuinely failed job into a hang."""
        task = _make_task()
        task._job_id = "job-1"
        task._stream_job_events = AsyncMock(side_effect=httpx.ReadError("connection reset"))
        task._check_remote_job_status = AsyncMock(return_value="error")

        with patch("apps.tasks_server.tasks._training_job_runner.asyncio.sleep", new=AsyncMock()):
            with pytest.raises(RuntimeError, match="error"):
                await task._consume_sse("job-1")

    async def test_reconnecting_gives_up_eventually(self):
        """An unreachable training server must not leave the task retrying forever."""
        task = _make_task()
        task._job_id = "job-1"
        task._stream_job_events = AsyncMock(side_effect=httpx.ConnectError("no route to host"))
        task._check_remote_job_status = AsyncMock(return_value="unreachable")

        with patch("apps.tasks_server.tasks._training_job_runner.asyncio.sleep", new=AsyncMock()):
            with pytest.raises(RuntimeError):
                await task._consume_sse("job-1")

        assert task._check_remote_job_status.await_count <= TrainModelTask.MAX_STREAM_RECONNECTS + 1

    async def test_cancelling_during_a_reconnect_wait_cancels_the_remote_job(self):
        task = _make_task()
        task._job_id = "job-1"
        task._stream_job_events = AsyncMock(side_effect=httpx.ReadError("connection reset"))
        task._check_remote_job_status = AsyncMock(return_value="running")
        task._cancel_remote_job = AsyncMock()
        task._cancel_flag.set()

        with patch("apps.tasks_server.tasks._training_job_runner.asyncio.sleep", new=AsyncMock()):
            result = await task._consume_sse("job-1")

        assert result["cancelled"] is True
        task._cancel_remote_job.assert_awaited_once_with("job-1")


class _FakeResponse:
    """An httpx.Response stand-in whose line stream the test drives."""

    def __init__(self, lines):
        self._lines = lines

    async def aiter_lines(self):
        for line in self._lines:
            if callable(line):
                await line()
                continue
            yield line


class _FakeStream:
    def __init__(self, response):
        self._response = response

    async def __aenter__(self):
        return self._response

    async def __aexit__(self, *exc):
        return False


class _FakeClient:
    def __init__(self, response):
        self._response = response

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def stream(self, *_args, **_kwargs):
        return _FakeStream(self._response)


def _patch_stream(response):
    return patch(
        "apps.tasks_server.tasks._training_job_runner.httpx.AsyncClient",
        lambda *a, **kw: _FakeClient(response),
    )


class TestQuietStream:
    """A job queued behind another emits nothing until its turn comes.

    The training server runs one job at a time, so the wait is however long the
    job ahead of it takes — hours on a real training. Everything this task does
    between events happens on the SSE loop, so a loop that simply blocks on the
    next line is a task that neither beats nor reads its cancel flag: it is
    reported as not responding, and Stop never reaches it.
    """

    async def test_a_silent_stream_still_beats(self):
        task = _make_task()
        task.STREAM_IDLE_WAKEUP = 0.01
        task.HEARTBEAT_INTERVAL = 0  # every pass, so the test needn't wait 30s

        beats = {"n": 0}

        async def _quiet():
            # Enough dead air for several wake-ups, then let the stream end so
            # the call returns instead of hanging the suite.
            while beats["n"] < 3:
                await asyncio.sleep(0.01)

        async def _beat(*_args, **_kwargs):
            beats["n"] += 1

        task._manager.update_task_heartbeat = AsyncMock(side_effect=_beat)

        with _patch_stream(_FakeResponse([_quiet])):
            with pytest.raises(_StreamDropped):
                await task._stream_job_events("job-1")

        assert beats["n"] >= 3, "a queued job's watcher must keep beating through the wait"

    async def test_stop_reaches_a_task_waiting_on_a_silent_stream(self):
        task = _make_task()
        task.STREAM_IDLE_WAKEUP = 0.01
        task._cancel_remote_job = AsyncMock()

        async def _quiet():
            task._cancel_flag.set()
            # The stream never says anything again; only the idle wake-up can
            # notice the flag.
            await asyncio.sleep(30)

        with _patch_stream(_FakeResponse([_quiet])):
            result = await asyncio.wait_for(task._stream_job_events("job-1"), timeout=5)

        assert result["cancelled"] is True
        task._cancel_remote_job.assert_awaited_once_with("job-1")

    async def test_events_still_arrive_and_are_handled(self):
        """The idle path must not cost the stream its actual job."""
        task = _make_task()
        task.STREAM_IDLE_WAKEUP = 0.01
        task._save_artifact = AsyncMock()

        lines = [
            "event: progress",
            'data: {"progress": 0.5, "log": "epoch 1"}',
            "",
            "event: done",
            'data: {"val_accuracy": 0.9}',
            "",
        ]
        task._job_id = "job-1"

        with _patch_stream(_FakeResponse(lines)):
            result = await task._stream_job_events("job-1")

        assert result["job_id"] == "job-1"
        assert result["val_accuracy"] == 0.9

    async def test_a_read_error_mid_stream_still_surfaces(self):
        """Relaying lines through a queue must not swallow the failure that the
        reconnect logic exists to catch."""
        task = _make_task()
        task.STREAM_IDLE_WAKEUP = 0.01

        async def _boom():
            raise httpx.ReadError("connection reset")

        with _patch_stream(_FakeResponse(["event: progress", _boom])):
            with pytest.raises(httpx.ReadError):
                await task._stream_job_events("job-1")
