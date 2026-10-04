"""The Service Status panel's detail view for the training server.

/status used to be a bare {"status": "ok"} — the panel could not tell a machine
that had been training for two hours from an idle one.
"""

import asyncio
from datetime import datetime, timedelta, timezone

from apps.training_server.job_queue import Job


def _job(manager, job_id: str, status: str, **kwargs) -> Job:
    job = Job(
        job_id=job_id,
        status=status,
        config={"trainer_type": kwargs.pop("trainer_type", "binary_image_classifier"), **kwargs.pop("config", {})},
        cancel_event=kwargs.pop("cancel_event", None),
        created_at=datetime.now(timezone.utc),
        **kwargs,
    )
    manager._queue._jobs[job_id] = job
    return job


class TestTrainingStatusDetails:
    def test_idle_server_says_so(self, manager):
        status = manager.get_status()

        assert status["status"] == "ok"
        assert status["summary"] == "idle"
        assert status["training"] == {"running": 0, "queued": 0, "awaiting_data": 0}
        assert status["details"][0]["empty"] == "No training running"

    def test_running_job_reports_its_epoch_and_metrics(self, manager):
        job = _job(manager, "j1", "running", config={"model_name": "cats-v3"})
        job.started_at = datetime.now(timezone.utc) - timedelta(minutes=7)
        job.last_progress = {
            "progress": 0.4,
            "epoch": 4,
            "total_epochs": 10,
            "train_loss": 0.123,
            "val_loss": 0.456,
            "val_accuracy": 0.91,
        }

        row = manager.get_status()["details"][0]["rows"][0]

        assert row["label"] == "cats-v3"
        assert "epoch 4/10" in row["value"]
        assert "40%" in row["value"]
        assert "loss 0.123" in row["value"]
        assert "val_loss 0.456" in row["value"]
        assert "val_acc 0.91" in row["value"]
        assert "elapsed 7m" in row["value"]

    def test_zero_val_loss_is_omitted(self, manager):
        """The detection trainers don't expose val_loss and send a hard 0.0.
        Rendering that as a metric would report a perfect model."""
        job = _job(manager, "j1", "running", config={"model_name": "det"})
        job.last_progress = {"epoch": 1, "total_epochs": 5, "train_loss": 2.0, "val_loss": 0.0, "val_accuracy": 0.1}

        assert "val_loss" not in manager.get_status()["details"][0]["rows"][0]["value"]

    def test_job_with_no_progress_yet_reads_as_starting(self, manager):
        _job(manager, "j1", "running", config={"model_name": "fresh"})

        assert "starting" in manager.get_status()["details"][0]["rows"][0]["value"]

    def test_summary_counts_running_queued_and_awaiting(self, manager):
        _job(manager, "j1", "running", config={"model_name": "a"})
        _job(manager, "j2", "queued", config={"model_name": "b"})
        _job(manager, "j3", "awaiting_data", config={"model_name": "c"})

        status = manager.get_status()

        assert status["training"] == {"running": 1, "queued": 1, "awaiting_data": 1}
        assert "training: 1" in status["summary"]
        assert "1 queued" in status["summary"]
        assert "1 awaiting data" in status["summary"]

    def test_pending_jobs_get_their_own_section(self, manager):
        _job(manager, "j2", "queued", config={"model_name": "b"})

        sections = manager.get_status()["details"]

        assert [s["title"] for s in sections] == ["Training now (0)", "Waiting (1)"]
        assert sections[1]["rows"][0]["label"] == "b"

    def test_finished_jobs_do_not_appear(self, manager):
        _job(manager, "j1", "done", config={"model_name": "old"})

        status = manager.get_status()

        assert status["summary"] == "idle"
        assert status["details"] == [
            {"title": "Training now (0)", "rows": [], "empty": "No training running"},
        ]

    def test_unnamed_job_falls_back_to_its_id(self, manager):
        _job(manager, "abcdef123456", "running")

        assert manager.get_status()["details"][0]["rows"][0]["label"] == "abcdef12"

    async def test_status_endpoint_serves_the_details(self, http_client, manager):
        _job(manager, "j1", "running", config={"model_name": "cats-v3"})

        resp = await http_client.get("/status")

        assert resp.status_code == 200
        data = resp.json()
        assert data["service"] == "training_server"
        assert data["details"][0]["rows"][0]["label"] == "cats-v3"


class TestJobProgressCapture:
    """The status view reads Job.last_progress/started_at rather than replaying
    the SSE event log, so the worker has to record them as training runs."""

    async def test_worker_records_latest_progress_and_start_time(self, manager):
        started = asyncio.Event()
        release = asyncio.Event()

        class FakeTrainer:
            async def train(self, config, progress_cb, cancel_event):
                progress_cb({"progress": 0.1, "epoch": 1, "total_epochs": 2, "train_loss": 9.0})
                progress_cb({"progress": 0.5, "epoch": 2, "total_epochs": 2, "train_loss": 1.5})
                started.set()
                await release.wait()
                return {"model_path": "/tmp/model.onnx"}

        queue = manager._queue
        queue.register_trainer("fake", FakeTrainer())
        queue.start()
        try:
            job = await queue.submit({"trainer_type": "fake", "model_name": "fake-model"})
            ok, _ = queue.start_job(job.job_id, [])
            assert ok
            await asyncio.wait_for(started.wait(), timeout=5)

            assert job.started_at is not None
            # The *latest* progress, not the first.
            assert job.last_progress["epoch"] == 2

            status = manager.get_status()
            row = status["details"][0]["rows"][0]
            assert row["label"] == "fake-model"
            assert "epoch 2/2" in row["value"]
        finally:
            release.set()
            await queue.shutdown()


class TestTheCpuFallbackNotice:
    """A job that fell back to the CPU is 10-50x slower and otherwise looks
    normal — the only prior signals were a log line and a status message on the
    job's own card, both of which need you to already be looking at that job.
    This puts it in the app-wide notices bar instead."""

    def test_a_cpu_job_raises_a_warning_notice(self, manager, monkeypatch):
        monkeypatch.setenv("TRAINING_EXPECTS_GPU", "1")
        job = _job(manager, "j1", "running", config={"model_name": "cats-v3"})
        job.device_warning = "WARNING: no GPU detected"

        notices = manager.get_status()["notices"]

        assert len(notices) == 1
        assert notices[0]["level"] == "warning"
        assert notices[0]["key"] == "training_cpu_only_j1"
        assert "cats-v3" in notices[0]["message"]

    def test_a_gpu_job_raises_nothing(self, manager, monkeypatch):
        """The distinguishing half: a running job on its own must not warn, or
        the bar would carry a permanent notice through every healthy run."""
        monkeypatch.setenv("TRAINING_EXPECTS_GPU", "1")
        job = _job(manager, "j1", "running", config={"model_name": "cats-v3"})
        job.last_progress = {"epoch": 3, "total_epochs": 10}

        assert manager.get_status()["notices"] == []

    def test_a_deployment_that_chose_cpu_is_not_nagged(self, manager, monkeypatch):
        """setup.sh writes 0 when the operator answered "continue anyway" on a
        host with no GPU. CPU training is then what they asked for."""
        monkeypatch.setenv("TRAINING_EXPECTS_GPU", "0")
        job = _job(manager, "j1", "running", config={"model_name": "cats-v3"})
        job.device_warning = "WARNING: no GPU detected"

        assert manager.get_status()["notices"] == []

    def test_an_unset_flag_still_warns(self, manager, monkeypatch):
        """A container built before the variable existed must not go quiet — an
        absent answer is 'expects a GPU', so the failure stays loud."""
        monkeypatch.delenv("TRAINING_EXPECTS_GPU", raising=False)
        job = _job(manager, "j1", "running", config={"model_name": "cats-v3"})
        job.device_warning = "WARNING: no GPU detected"

        assert len(manager.get_status()["notices"]) == 1

    def test_the_notice_clears_when_the_job_stops(self, manager, monkeypatch):
        """Drawn from the running jobs only: a finished job's warning must not
        outlive it, even though the job object still carries it."""
        monkeypatch.setenv("TRAINING_EXPECTS_GPU", "1")
        job = _job(manager, "j1", "running", config={"model_name": "cats-v3"})
        job.device_warning = "WARNING: no GPU detected"
        assert len(manager.get_status()["notices"]) == 1

        job.status = "done"

        assert manager.get_status()["notices"] == []

    def test_two_cpu_jobs_keep_their_own_notices(self, manager, monkeypatch):
        """Keyed per job — the bar counts notices, and collapsing two jobs into
        one key would under-report."""
        monkeypatch.setenv("TRAINING_EXPECTS_GPU", "1")
        for jid in ("j1", "j2"):
            job = _job(manager, jid, "running", config={"model_name": jid})
            job.device_warning = "WARNING: no GPU detected"

        keys = {n["key"] for n in manager.get_status()["notices"]}

        assert keys == {"training_cpu_only_j1", "training_cpu_only_j2"}

    async def test_the_notice_survives_the_epochs_that_follow_it(self, manager, monkeypatch):
        """The half the hand-set cases above cannot reach: `warn_if_cpu` emits the
        warning in one event at the top of `train()` and nothing repeats it, so a
        notice read off the newest progress payload lives for one epoch tick."""
        monkeypatch.setenv("TRAINING_EXPECTS_GPU", "1")
        released = asyncio.Event()
        warned = asyncio.Event()

        class CpuTrainer:
            async def train(self, config, progress_cb, cancel_event):
                progress_cb({"log": "no GPU detected", "device_warning": "WARNING: no GPU detected"})
                progress_cb({"progress": 0.1, "epoch": 1, "total_epochs": 100, "train_loss": 9.0})
                warned.set()
                await released.wait()
                return {"model_path": "/tmp/model.onnx"}

        queue = manager._queue
        queue.register_trainer("cpu-fake", CpuTrainer())
        queue.start()
        try:
            job = await queue.submit({"trainer_type": "cpu-fake", "model_name": "slow-model"})
            ok, _ = queue.start_job(job.job_id, [])
            assert ok
            await asyncio.wait_for(warned.wait(), timeout=5)

            notices = manager.get_status()["notices"]

            assert [n["key"] for n in notices] == [f"training_cpu_only_{job.job_id}"]
            assert "slow-model" in notices[0]["message"]
        finally:
            released.set()
            await queue.shutdown()
