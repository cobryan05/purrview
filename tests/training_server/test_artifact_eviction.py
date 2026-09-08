"""Tests for artifact eviction — the bound on what finished training runs leave behind.

Every completed job used to keep its whole artifact directory (weights plus every
checkpoint, around a gigabyte for a detector) forever, and nothing removed them.
What matters now is that the sweep removes the *right* directories: never one a
job still needs, and never the newest ones.
"""

import asyncio
import shutil
import time
from datetime import datetime, timedelta, timezone

import pytest


async def _make_job(manager, *, status: str, age_minutes: int, size: int = 1000):
    """Register a job with an artifact directory holding one file."""
    job = await manager._queue.submit(config={"trainer_type": "test"}, manifest=[])
    job.status = status
    job.created_at = datetime.now(timezone.utc) - timedelta(minutes=age_minutes)
    job_dir = manager._artifacts_dir / job.job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "primary_model.onnx").write_bytes(b"x" * size)
    job.artifact_dir = job_dir
    return job


class TestEvictArtifacts:
    async def test_keeps_the_newest_finished_runs_and_removes_the_rest(self, manager):
        jobs = [await _make_job(manager, status="done", age_minutes=m) for m in (50, 40, 30, 20, 10)]

        result = await manager.evict_artifacts(keep=2)

        surviving = [j for j in jobs if j.artifact_dir.exists()]
        assert [j.job_id for j in surviving] == [jobs[3].job_id, jobs[4].job_id]
        assert len(result["removed"]) == 3
        assert result["freed_bytes"] == 3000

    async def test_a_job_that_is_not_finished_is_never_a_candidate(self, manager):
        """Its directory is where the run is still writing, or restarting from."""
        live = [
            await _make_job(manager, status=status, age_minutes=100)
            for status in ("running", "queued", "awaiting_data", "interrupted")
        ]
        await _make_job(manager, status="done", age_minutes=1)

        await manager.evict_artifacts(keep=0)

        assert all(job.artifact_dir.exists() for job in live)

    async def test_evicted_jobs_stop_being_listed(self, manager):
        """A job whose artifacts are gone can neither restart nor be downloaded."""
        old = await _make_job(manager, status="done", age_minutes=60)
        await _make_job(manager, status="done", age_minutes=1)

        await manager.evict_artifacts(keep=1)

        assert manager.get_job(old.job_id) is None

    async def test_interrupted_jobs_are_still_listed_after_a_sweep(self, manager):
        interrupted = await _make_job(manager, status="interrupted", age_minutes=99)
        await manager.evict_artifacts(keep=0)
        assert manager.get_job(interrupted.job_id) is not None

    async def test_keeping_more_than_exist_removes_nothing(self, manager):
        job = await _make_job(manager, status="done", age_minutes=5)
        result = await manager.evict_artifacts(keep=10)
        assert result["removed"] == []
        assert job.artifact_dir.exists()

    async def test_cancelled_and_errored_runs_are_swept_like_completed_ones(self, manager):
        failed = await _make_job(manager, status="error", age_minutes=60)
        cancelled = await _make_job(manager, status="cancelled", age_minutes=50)
        await _make_job(manager, status="done", age_minutes=1)

        await manager.evict_artifacts(keep=1)

        assert not failed.artifact_dir.exists()
        assert not cancelled.artifact_dir.exists()


class TestEvictArtifactsEndpoint:
    async def test_endpoint_reports_what_it_removed(self, manager, http_client):
        await _make_job(manager, status="done", age_minutes=60)
        await _make_job(manager, status="done", age_minutes=1)

        r = await http_client.post("/api/training/artifacts/evict", params={"keep": 1})

        assert r.status_code == 200, r.text
        body = r.json()
        assert len(body["removed"]) == 1
        assert body["freed_bytes"] == 1000
        assert body["jobs"] == 1

    async def test_default_keep_applies_when_the_caller_names_no_number(self, manager, http_client):
        for m in range(6):
            await _make_job(manager, status="done", age_minutes=60 - m)

        r = await http_client.post("/api/training/artifacts/evict")

        assert r.status_code == 200, r.text
        assert r.json()["jobs"] == manager.DEFAULT_ARTIFACTS_KEEP


class TestConcurrentSweeps:
    """tasks_server evicts after every training run, so two runs finishing close
    together produce two overlapping sweeps against the same job list."""

    async def test_two_sweeps_at_once_do_not_both_delete_the_same_directories(self, manager, monkeypatch):
        doomed = [await _make_job(manager, status="done", age_minutes=m) for m in (60, 50, 40)]
        await _make_job(manager, status="done", age_minutes=1)

        # Hold each rmtree open long enough that an unserialised second sweep
        # would have selected its victims before the first one deleted them.
        real_rmtree = shutil.rmtree
        rmtree_calls: list[str] = []

        def slow_rmtree(path, *a, **kw):
            rmtree_calls.append(str(path))
            time.sleep(0.05)
            return real_rmtree(path, *a, **kw)

        monkeypatch.setattr(shutil, "rmtree", slow_rmtree)

        first, second = await asyncio.gather(
            manager.evict_artifacts(keep=1),
            manager.evict_artifacts(keep=1),
        )

        # Each directory is walked and deleted exactly once, by exactly one sweep.
        assert sorted(rmtree_calls) == sorted(str(j.artifact_dir) for j in doomed)
        assert len(first["removed"]) + len(second["removed"]) == 3
        # ...and the bytes are reported once, not once per sweep. Without the lock
        # both sweeps size the same files and each claims all 3000.
        assert first["freed_bytes"] + second["freed_bytes"] == 3000

    async def test_a_job_submitted_during_a_sweep_is_not_swept(self, manager, monkeypatch):
        """The sweep acts on the victims it chose, not on whatever exists when it
        finishes — a run submitted while a gigabyte is being deleted is newer than
        everything it selected."""
        await _make_job(manager, status="done", age_minutes=60)
        keeper = await _make_job(manager, status="done", age_minutes=1)

        real_rmtree = shutil.rmtree
        started = asyncio.Event()
        loop = asyncio.get_running_loop()

        def slow_rmtree(path, *a, **kw):
            loop.call_soon_threadsafe(started.set)
            time.sleep(0.05)
            return real_rmtree(path, *a, **kw)

        monkeypatch.setattr(shutil, "rmtree", slow_rmtree)

        sweep = asyncio.create_task(manager.evict_artifacts(keep=1))
        await started.wait()
        latecomer = await _make_job(manager, status="done", age_minutes=0)
        await sweep

        assert latecomer.artifact_dir.exists()
        assert keeper.artifact_dir.exists()
        assert manager.get_job(latecomer.job_id) is not None
