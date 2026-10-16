"""Resuming a run the server died in the middle of.

The bug this covers shipped, and it was silent in the worst way: `restart_job`'s own
docstring said *"training resumes from checkpoint"*, `forget()` refused to drop an
interrupted job because it "can still be restarted from its checkpoint", and the
training modal's comments said the same — while the trainer unconditionally loaded fresh
pretrained weights and never passed `resume=` to libreyolo. A 20-epoch run that got
killed went back to epoch 1, and the only visible symptom was the loss climbing back to
its opening value, about an hour of GPU time after the fact.

So what is worth protecting is the *wiring*, in both directions:

* A resume reaches libreyolo as `resume=True` **and** loads the run's own `last.pt`.
  Either half alone is wrong — weights without `resume=` is a warm start that re-runs
  the full epoch budget with a fresh optimizer and LR schedule.
* "Start over" really does start over. It is the escape hatch for a changed dataset or
  architecture, and a resume forced on those is refused by libreyolo or, worse, trains
  something subtly wrong.
* The epoch the UI offers survives the crash. It is persisted per epoch precisely
  because `emit` only persists on status transitions, and a crash is the case where no
  transition follows.

The trainer is driven through a fake model class rather than libreyolo: what is being
asserted is which arguments the app chooses, and a real 300 MB checkpoint would assert
nothing extra about that.
"""

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from apps.training_server.job_queue import Job, JobQueue


def _job(**overrides) -> Job:
    from datetime import datetime, timezone

    fields: dict[str, Any] = {
        "job_id": "job-1",
        "status": "interrupted",
        "config": {"trainer_type": "object_detection", "model_name": "primary_model"},
        "cancel_event": asyncio.Event(),
        "created_at": datetime.now(timezone.utc),
    }
    fields.update(overrides)
    return Job(**fields)


class TestRestartCarriesTheResumeChoice:
    """`restart_job` is the only way back into the queue, so the choice has to ride on
    the config — that is what reaches the trainer and what `_persist` already writes."""

    def test_a_restart_asks_the_trainer_to_resume_by_default(self):
        """The default is the whole fix for existing callers: nothing else in the app
        passes this flag, and every one of them already believed resume was happening."""
        queue = JobQueue()
        job = _job()
        queue._jobs[job.job_id] = job

        assert queue.restart_job(job.job_id) is True
        assert job.config["resume"] is True

    def test_start_over_asks_it_not_to(self):
        queue = JobQueue()
        job = _job()
        queue._jobs[job.job_id] = job

        assert queue.restart_job(job.job_id, resume=False) is True
        assert job.config["resume"] is False

    def test_the_rest_of_the_config_survives(self):
        """The config is the submitted job — class names, manifest, hyperparams. Replacing
        it rather than merging would restart the run as a different job."""
        queue = JobQueue()
        job = _job(config={"trainer_type": "object_detection", "max_epochs": 100, "imgsz": 640})
        queue._jobs[job.job_id] = job

        queue.restart_job(job.job_id)

        assert job.config["max_epochs"] == 100
        assert job.config["imgsz"] == 640

    def test_a_job_that_is_not_interrupted_is_refused(self):
        """A running job has a worker on it; re-queueing would train it twice."""
        queue = JobQueue()
        job = _job(status="running")
        queue._jobs[job.job_id] = job

        assert queue.restart_job(job.job_id) is False
        assert "resume" not in job.config, "a refused restart must not leave the flag behind"


class TestTheEpochSurvivesTheCrash:
    def test_an_epoch_report_is_written_to_disk_immediately(self, tmp_path):
        """`emit` persists only on status transitions, and a crash is exactly the case
        where no transition follows — so without a write here the state file still says
        whatever it did when the job went `running`."""
        job = _job(status="running", state_path=tmp_path / "job_state.json")
        queue = JobQueue()
        queue._jobs[job.job_id] = job

        job.last_epoch = 20
        job._persist()

        assert json.loads((tmp_path / "job_state.json").read_text())["last_epoch"] == 20

    def test_the_offer_names_the_next_epoch(self):
        """The checkpoint holds the epoch just finished; the user is offered the one they
        would watch start."""
        job = _job(status="interrupted", last_epoch=20)
        assert job.to_dict()["resumable_epoch"] == 21

    def test_a_job_with_no_epochs_yet_offers_no_number(self):
        """Killed during dataset prep, before epoch 1. A resume is still possible in
        principle, but there is no epoch to name and inventing one would be a lie."""
        job = _job(status="interrupted", last_epoch=None)
        assert job.to_dict()["resumable_epoch"] is None

    def test_a_running_job_offers_nothing(self):
        """The conditional that scopes the offer: only an interrupted job can be resumed,
        and a Resume button on a job that is training is a way to train it twice."""
        job = _job(status="running", last_epoch=20)
        assert job.to_dict()["resumable_epoch"] is None

    def test_a_done_job_offers_nothing(self):
        job = _job(status="done", last_epoch=100)
        assert job.to_dict()["resumable_epoch"] is None


class TestTheTrainerFindsTheCheckpoint:
    def _weights(self, tmp_path: Path, model_name: str = "primary_model") -> Path:
        weights = tmp_path / "runs" / model_name / "weights"
        weights.mkdir(parents=True)
        return weights

    def test_the_last_checkpoint_is_what_a_resume_loads(self, tmp_path):
        """`last.pt`, not `best.pt`: resuming continues the run, and the best epoch may
        be far behind the last one."""
        from apps.training_server.trainers.object_detection_trainer import _resume_checkpoint

        weights = self._weights(tmp_path)
        (weights / "last.pt").write_bytes(b"x")
        (weights / "best.pt").write_bytes(b"x")

        assert _resume_checkpoint(tmp_path, "primary_model") == weights / "last.pt"

    def test_a_run_with_only_a_best_checkpoint_cannot_be_resumed(self, tmp_path):
        """The conditional against the test above. `best.pt` is the deliverable — it
        survives a restart that overwrites `last.pt` — but it carries no optimizer or
        epoch state, so resuming from it would silently be a warm start."""
        from apps.training_server.trainers.object_detection_trainer import _resume_checkpoint

        weights = self._weights(tmp_path)
        (weights / "best.pt").write_bytes(b"x")

        assert _resume_checkpoint(tmp_path, "primary_model") is None

    def test_a_fresh_job_has_no_checkpoint(self, tmp_path):
        from apps.training_server.trainers.object_detection_trainer import _resume_checkpoint

        assert _resume_checkpoint(tmp_path, "primary_model") is None

    def test_the_checkpoint_is_looked_up_under_the_run_s_own_name(self, tmp_path):
        """Two models train into the same artifact root under different `name`s, so a
        lookup that ignored the name would resume one run from another's weights."""
        from apps.training_server.trainers.object_detection_trainer import _resume_checkpoint

        weights = self._weights(tmp_path, "other_model")
        (weights / "last.pt").write_bytes(b"x")

        assert _resume_checkpoint(tmp_path, "primary_model") is None


class TestTheEndpointPassesTheChoiceThrough:
    async def test_resume_defaults_to_true(self, http_client, manager, monkeypatch):
        seen = {}

        def _restart(job_id, resume=True):
            seen["job_id"] = job_id
            seen["resume"] = resume
            return True

        monkeypatch.setattr(manager, "restart_job", _restart)
        monkeypatch.setattr(manager, "get_job", lambda job_id: _job(job_id=job_id))

        response = await http_client.post("/api/training/jobs/job-1/restart")

        assert response.status_code == 200
        assert seen == {"job_id": "job-1", "resume": True}

    async def test_resume_false_reaches_the_manager(self, http_client, manager, monkeypatch):
        seen = {}
        monkeypatch.setattr(manager, "restart_job", lambda job_id, resume=True: seen.update(resume=resume) or True)
        monkeypatch.setattr(manager, "get_job", lambda job_id: _job(job_id=job_id))

        response = await http_client.post("/api/training/jobs/job-1/restart?resume=false")

        assert response.status_code == 200
        assert seen["resume"] is False

    async def test_a_job_the_server_refuses_answers_409(self, http_client, manager, monkeypatch):
        monkeypatch.setattr(manager, "restart_job", lambda job_id, resume=True: False)
        monkeypatch.setattr(manager, "get_job", lambda job_id: _job(job_id=job_id, status="running"))

        response = await http_client.post("/api/training/jobs/job-1/restart")

        assert response.status_code == 409
        assert "running" in response.json()["detail"]

    async def test_an_unknown_job_answers_404(self, http_client, manager, monkeypatch):
        monkeypatch.setattr(manager, "get_job", lambda job_id: None)

        response = await http_client.post("/api/training/jobs/nope/restart")

        assert response.status_code == 404


class TestTheTrainerActuallyResumes:
    """The half the original bug lived in.

    Everything above could pass while the trainer still threw the checkpoint away, which
    is exactly what happened: the flag was set, the file was there, the docstrings said
    resume — and `model_cls(weight_filename)` loaded COCO weights and `model.train()`
    never saw `resume=`. These assert the two arguments that decide it.
    """

    @pytest.fixture
    def base_config(self, tmp_path):
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        artifact_dir = tmp_path / "artifacts"
        artifact_dir.mkdir()
        manifest = []
        for i in range(3):
            uuid = f"uuid-{i}"
            (cache_dir / uuid).write_bytes(b"\xff\xd8\xff\xd9")
            manifest.append({"image_uuid": uuid, "filename": f"{uuid}.jpg", "status": "present"})
        return {
            "architecture": "yolo_v9_m",
            "class_names": ["cat", "dog"],
            "annotations": {},
            "model_name": "primary_model",
            "cache_dir": str(cache_dir),
            "artifact_dir": str(artifact_dir),
            "manifest": manifest,
            "max_epochs": 3,
            "imgsz": 320,
            "batch_size": 4,
            "patience": 5,
        }

    @pytest.fixture
    def spy(self, monkeypatch):
        """A stand-in model class that records how it was constructed and trained."""
        seen: dict[str, Any] = {}

        class FakeModel:
            FILENAME_PREFIX = "yolo9-"
            WEIGHT_EXT = ".pt"

            def __init__(self, model_path, size=None, **kwargs):
                seen["loaded_from"] = model_path
                seen["size"] = size

            def train(self, data_yaml, **kwargs):
                seen["train_kwargs"] = kwargs
                return {"best_mAP50": 0.5}

            def export(self, *a, **k):
                return None

        monkeypatch.setattr(
            "apps.training_server.trainers.object_detection_trainer._get_model_class",
            lambda name: FakeModel,
        )
        return seen

    def _checkpoint(self, config, name="last.pt") -> Path:
        weights = Path(config["artifact_dir"]) / "runs" / config["model_name"] / "weights"
        weights.mkdir(parents=True, exist_ok=True)
        path = weights / name
        path.write_bytes(b"x")
        return path

    async def _run(self, config):
        from apps.training_server.trainers.object_detection_trainer import ObjectDetectionTrainer

        notes: list = []
        try:
            await ObjectDetectionTrainer().train(
                config, progress_cb=lambda e: notes.append(e), cancel_event=asyncio.Event()
            )
        except Exception:
            # The epilogue exports and validates ONNX, which the fake model does not
            # produce. Everything under test happens before that.
            pass
        return notes

    async def test_a_resume_loads_the_checkpoint_and_tells_libreyolo_to_resume(self, base_config, spy):
        """Both halves. Weights without `resume=` is a warm start that re-runs the whole
        epoch budget with a fresh optimizer and LR schedule — which looks like training
        and silently discards the run's position in it."""
        checkpoint = self._checkpoint(base_config)
        base_config["resume"] = True

        await self._run(base_config)

        assert spy["loaded_from"] == str(checkpoint)
        assert spy["train_kwargs"]["resume"] is True

    async def test_start_over_loads_pretrained_weights_and_does_not_resume(self, base_config, spy):
        """The conditional: the same job, the same checkpoint on disk, opposite wiring."""
        self._checkpoint(base_config)
        base_config["resume"] = False

        await self._run(base_config)

        assert spy["loaded_from"] == "yolo9-m.pt", spy["loaded_from"]
        assert spy["train_kwargs"]["resume"] is False

    async def test_a_job_that_never_asked_to_resume_does_not(self, base_config, spy):
        """A fresh submit carries no `resume` key at all, and must behave as it always
        did — this is the path every first run takes."""
        self._checkpoint(base_config)

        await self._run(base_config)

        assert spy["loaded_from"] == "yolo9-m.pt"
        assert spy["train_kwargs"]["resume"] is False

    async def test_a_resume_with_no_checkpoint_starts_over_and_says_so(self, base_config, spy):
        """Killed before the first epoch wrote anything. Starting from scratch is the
        only option, and the user is owed that before they watch the loss climb back to
        its opening value."""
        base_config["resume"] = True

        notes = await self._run(base_config)

        assert spy["loaded_from"] == "yolo9-m.pt"
        assert spy["train_kwargs"]["resume"] is False
        assert any("starts from the beginning" in str(n.get("log", "")) for n in notes), notes

    async def test_only_best_pt_is_not_enough_to_resume(self, base_config, spy):
        """`best.pt` carries no optimizer or epoch state, so resuming from it would be a
        warm start wearing a resume's name."""
        self._checkpoint(base_config, name="best.pt")
        base_config["resume"] = True

        await self._run(base_config)

        assert spy["loaded_from"] == "yolo9-m.pt"
        assert spy["train_kwargs"]["resume"] is False
