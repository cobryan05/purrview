"""Pushing a job's images to the training server.

The training server may be on this host or another one, and that single fact
decides the whole shape of the push: co-located, it reads FILES_ROOT itself and
no bytes move; remote, every image crosses as tar and the local-ingest probe
could only ever answer "not found". Asking anyway cost a round trip and a stat
of every UUID per batch.

The tar build reads its files concurrently, so the other thing worth pinning is
that a batch still packs each UUID with its own bytes.
"""

from __future__ import annotations

import io
import tarfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from apps.tasks_server.tasks._training_job_runner import _TAR_BATCH_SIZE
from apps.tasks_server.tasks.TrainModelTask import TrainModelTask


def _make_task(base_url: str) -> TrainModelTask:
    manager = MagicMock()
    manager.update_task_heartbeat = AsyncMock()
    manager.save_task_resume_data = AsyncMock()
    task = TrainModelTask(task_config_uuid="cfg-uuid", params={}, manager=manager)
    task.set_instance_uuid("inst-uuid")
    task._api_client = MagicMock()
    task._training_base_url = base_url
    task._model_name = "m1"
    task._manifest = []
    task._config = {}
    task._epoch_history = []
    task._job_id = None
    return task


class _Response:
    def __init__(self, received: int):
        self._received = received

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict:
        return {"received": self._received}


class _RecordingClient:
    """Stands in for httpx.AsyncClient, recording the UUIDs of every tar POSTed."""

    def __init__(self, sink: list[str]):
        self._sink = sink

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        return False

    async def post(self, _url, content, headers):
        with tarfile.open(fileobj=io.BytesIO(content), mode="r") as tf:
            names = [m.name for m in tf.getmembers()]
        self._sink.extend(names)
        return _Response(len(names))


def _write_images(root, count: int) -> dict[str, str]:
    """Create *count* images whose bytes differ, and return {uuid: filename}."""
    mapping = {}
    for i in range(count):
        uuid = f"uuid-{i:04d}"
        name = f"{uuid}.jpg"
        (root / name).write_bytes(f"image-bytes-{i}".encode() * 8)
        mapping[uuid] = name
    return mapping


def _manifest(mapping: dict[str, str]) -> list[dict]:
    return [{"image_uuid": u, "filename": f} for u, f in mapping.items()]


class TestLocalIngestProbeGate:
    """Whether the zero-copy probe is worth asking is decided by the URL alone.

    It can only succeed when the training server shares this filesystem, so
    against another host it is a round trip and a stat of every UUID per batch
    that always answers not_found.
    """

    async def _push(self, base_url: str, tmp_path, image_count: int):
        mapping = _write_images(tmp_path, image_count)
        task = _make_task(base_url)
        posted: list[str] = []

        api = MagicMock()
        api.get_missing_images.return_value = MagicMock(uuids=list(mapping))
        # If it is asked, the probe claims everything — so anything still tarred
        # afterwards proves the probe was skipped rather than merely unhelpful.
        api.ingest_local_images.return_value = MagicMock(not_found=[], linked=image_count, copied=0)

        with (
            patch("apps.tasks_server.tasks._training_job_runner.FILES_ROOT", tmp_path),
            patch("apps.tasks_server.tasks._training_job_runner.TrainingApi", return_value=api),
            patch(
                "apps.tasks_server.tasks._training_job_runner.httpx.AsyncClient",
                return_value=_RecordingClient(posted),
            ),
        ):
            await task._push_images("job-1", _manifest(mapping))

        return api, posted, mapping

    async def test_a_remote_training_server_is_never_asked_to_ingest_locally(self, tmp_path):
        api, posted, mapping = await self._push("http://192.0.2.10:8005", tmp_path, 5)

        api.ingest_local_images.assert_not_called()
        assert sorted(posted) == sorted(mapping)

    async def test_a_colocated_training_server_still_takes_the_zero_copy_path(self, tmp_path):
        api, posted, _ = await self._push("http://localhost:8005", tmp_path, 5)

        api.ingest_local_images.assert_called()
        assert posted == []

    async def test_a_remote_push_spans_batches_without_losing_images(self, tmp_path):
        """The remote path builds the next batch while the current one is in
        flight, so an off-by-one in the prefetch would drop or double a batch."""
        count = _TAR_BATCH_SIZE * 2 + 1
        _, posted, mapping = await self._push("http://192.0.2.10:8005", tmp_path, count)

        assert sorted(posted) == sorted(mapping)


class TestBuildTar:
    def test_every_image_lands_under_its_own_uuid(self, tmp_path):
        """The reads are concurrent; a batch that paired a UUID with another
        image's bytes would train the model on quietly mislabelled data."""
        mapping = _write_images(tmp_path, 64)
        task = _make_task("http://192.0.2.10:8005")

        with patch("apps.tasks_server.tasks._training_job_runner.FILES_ROOT", tmp_path):
            tar_bytes, added, skipped = task._build_tar(list(mapping), mapping)

        assert (added, skipped) == (64, 0)
        with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r") as tf:
            assert {m.name for m in tf.getmembers()} == set(mapping)
            for uuid, filename in mapping.items():
                assert tf.extractfile(uuid).read() == (tmp_path / filename).read_bytes()

    def test_a_missing_file_is_skipped_rather_than_failing_the_batch(self, tmp_path):
        mapping = _write_images(tmp_path, 4)
        (tmp_path / mapping["uuid-0002"]).unlink()
        task = _make_task("http://192.0.2.10:8005")

        with patch("apps.tasks_server.tasks._training_job_runner.FILES_ROOT", tmp_path):
            tar_bytes, added, skipped = task._build_tar(list(mapping), mapping)

        assert (added, skipped) == (3, 1)
        with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r") as tf:
            assert "uuid-0002" not in {m.name for m in tf.getmembers()}
