"""The StreamWatcher's live view — `get_live_preview`.

This is a debugging surface before it is anything else: it exists to distinguish
"inference returned nothing" from "the tracker dropped everything inference returned"
from "the rule never fired", which are indistinguishable from the status card alone.
The tests therefore assert that detections and tracks are reported *separately*, not
just that a frame comes back.

The other thing under test is what the view costs. Encoding is on demand, cached per
frame, and skippable via `since_seq`; a poll that outruns the camera must not re-encode.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np

from apps.helpers.tracking.tracker import Detection, Track
from apps.helpers.watch.engine import WatchEngine
from apps.helpers.watch.geometry import Zone
from apps.tasks_server.tasks._watch_pipeline import WatchPipeline
from apps.tasks_server.tasks.StreamWatcherTask import StreamWatcherTask

CAT = "label-cat"


class _NullTracker:
    """Returns whatever it is told to, so a test can make tracks and detections differ."""

    def __init__(self, tracks=()):
        self.tracks = list(tracks)

    def update(self, detections, now):
        return list(self.tracks)


def _pipeline(
    *,
    source_uuid: str = "src-1",
    name: str = "Litter cam",
    tracks=(),
    zones=(),
) -> WatchPipeline:
    return WatchPipeline(
        source_uuid=source_uuid,
        source_name=name,
        provider=SimpleNamespace(),
        tracker=_NullTracker(tracks),
        engine=WatchEngine(rules=[], zones=zones, source_uuid=source_uuid),
        model_names=["yolov8n"],
        analysis_fps=2.0,
        analysis_width=64,
    )


def _task(pipelines) -> StreamWatcherTask:
    """A task with only what `get_live_preview` touches — no manager, DB or profile."""
    task = StreamWatcherTask.__new__(StreamWatcherTask)
    task._pipelines = list(pipelines)
    task._profile = SimpleNamespace(uuid="profile-1", name="Litter box")
    task._dry_run = False
    task._preview_cache = {}
    return task


def _frame(width: int = 64, height: int = 48) -> np.ndarray:
    # Non-uniform so the JPEG has something to compress and a wrong-frame bug would
    # not be masked by two identical encodings.
    return np.random.default_rng(0).integers(0, 255, (height, width, 3), dtype=np.uint8)


def _detection(x1=1.0, y1=2.0, x2=10.0, y2=12.0, conf=0.9) -> Detection:
    return Detection(bbox=(x1, y1, x2, y2), label_uuid=CAT, label_name="Cat", confidence=conf)


def _track(track_id=1, now=100.0) -> Track:
    return Track(
        track_id=track_id,
        bbox=(1.0, 2.0, 10.0, 12.0),
        label_uuid=CAT,
        label_name="Cat",
        confidence=0.9,
        first_seen=now - 5,
        last_seen=now,
        hits=3,
        confirmed=True,
    )


class TestLivePreviewContent:
    async def test_reports_sources_before_any_frame_arrives(self):
        """A watcher that has not captured yet must still populate the source picker."""
        task = _task([_pipeline(source_uuid="a", name="A"), _pipeline(source_uuid="b", name="B")])

        preview = await task.get_live_preview()

        assert [s["sourceUuid"] for s in preview["sources"]] == ["a", "b"]
        assert preview["sourceUuid"] == "a"  # defaults to the first source
        assert preview["frame"] is None

    async def test_selects_the_requested_source(self):
        task = _task([_pipeline(source_uuid="a", name="A"), _pipeline(source_uuid="b", name="B")])

        preview = await task.get_live_preview(source_uuid="b")

        assert preview["sourceUuid"] == "b"
        assert preview["sourceName"] == "B"

    async def test_unknown_source_returns_no_frame_rather_than_the_wrong_one(self):
        task = _task([_pipeline(source_uuid="a")])

        preview = await task.get_live_preview(source_uuid="nope")

        assert preview["sourceUuid"] is None
        assert preview["frame"] is None

    async def test_returns_the_analysed_frame_and_its_geometry(self):
        pipeline = _pipeline()
        pipeline.prepare_frame(_frame(64, 48))
        task = _task([pipeline])

        preview = await task.get_live_preview()

        assert preview["frameWidth"] == 64
        assert preview["frameHeight"] == 48
        assert isinstance(preview["frame"], str) and preview["frame"]

    async def test_detections_and_tracks_are_reported_separately(self):
        """The whole diagnostic value of this view.

        A tracker that discards everything leaves tracks empty while detections are
        full — collapsing the two into one list would make that case look identical to
        a model that detected nothing.
        """
        pipeline = _pipeline(tracks=[])  # tracker swallows the detection
        pipeline.prepare_frame(_frame())
        pipeline.step([_detection()], now=100.0)
        task = _task([pipeline])

        preview = await task.get_live_preview()

        assert len(preview["detections"]) == 1
        assert preview["detections"][0]["labelName"] == "Cat"
        assert preview["tracks"] == []

    async def test_tracks_carry_age(self):
        pipeline = _pipeline(tracks=[_track()])
        pipeline.prepare_frame(_frame())
        pipeline.step([_detection()], now=100.0)
        task = _task([pipeline])

        preview = await task.get_live_preview()

        assert preview["tracks"][0]["trackId"] == 1
        assert preview["tracks"][0]["ageS"] >= 0

    async def test_zones_are_included_in_normalised_coordinates(self):
        """Zones are drawn over the frame client-side, so they travel as stored."""
        zone = Zone(uuid="z1", name="Box", kind="box", points=[(0.1, 0.1), (0.5, 0.5)])
        pipeline = _pipeline(zones=[zone])
        pipeline.prepare_frame(_frame())
        task = _task([pipeline])

        preview = await task.get_live_preview()

        assert preview["zones"] == [{"uuid": "z1", "name": "Box", "kind": "box", "points": [[0.1, 0.1], [0.5, 0.5]]}]

    async def test_surfaces_the_source_error_that_explains_a_frozen_frame(self):
        pipeline = _pipeline()
        pipeline.prepare_frame(_frame())
        pipeline.health.record_failure(100.0, "connection refused")
        task = _task([pipeline])

        preview = await task.get_live_preview()

        assert preview["healthy"] is False
        assert preview["lastError"] == "connection refused"

    async def test_reports_when_a_source_has_no_models(self):
        """ "Nothing is detected" has a boring explanation often enough to show it."""
        pipeline = _pipeline()
        pipeline.model_names = []
        pipeline.prepare_frame(_frame())
        task = _task([pipeline])

        preview = await task.get_live_preview()

        assert preview["modelNames"] == []


class TestLivePreviewCost:
    async def test_frame_seq_advances_once_per_analysed_frame(self):
        pipeline = _pipeline()
        assert pipeline.frame_seq == 0
        pipeline.prepare_frame(_frame())
        pipeline.prepare_frame(_frame())
        assert pipeline.frame_seq == 2

    async def test_since_seq_skips_the_image_but_keeps_the_metadata(self):
        pipeline = _pipeline(tracks=[_track()])
        pipeline.prepare_frame(_frame())
        pipeline.step([_detection()], now=100.0)
        task = _task([pipeline])

        preview = await task.get_live_preview(since_seq=pipeline.frame_seq)

        assert preview["frameUnchanged"] is True
        assert preview["frame"] is None
        # Tracks age between frames, so the caller still needs them.
        assert len(preview["tracks"]) == 1

    async def test_a_new_frame_defeats_since_seq(self):
        pipeline = _pipeline()
        pipeline.prepare_frame(_frame())
        task = _task([pipeline])
        stale_seq = pipeline.frame_seq

        pipeline.prepare_frame(_frame())
        preview = await task.get_live_preview(since_seq=stale_seq)

        assert preview.get("frameUnchanged") is not True
        assert preview["frame"]

    async def test_the_encode_is_cached_per_frame(self, monkeypatch):
        """Two viewers of one camera must cost one encode, not two."""
        pipeline = _pipeline()
        pipeline.prepare_frame(_frame())
        task = _task([pipeline])

        calls = []
        real_encode = StreamWatcherTask._encode_preview

        def _counting(frame, quality):
            calls.append(quality)
            return real_encode(frame, quality)

        monkeypatch.setattr(StreamWatcherTask, "_encode_preview", staticmethod(_counting))

        first = await task.get_live_preview()
        second = await task.get_live_preview()

        assert len(calls) == 1
        assert first["frame"] == second["frame"]

        # ...but a new frame is a new encode, or the view would freeze.
        pipeline.prepare_frame(_frame())
        await task.get_live_preview()
        assert len(calls) == 2

    async def test_a_different_quality_is_not_served_from_cache(self):
        pipeline = _pipeline()
        pipeline.prepare_frame(_frame())
        task = _task([pipeline])

        low = await task.get_live_preview(quality=20)
        high = await task.get_live_preview(quality=95)

        assert low["frame"] != high["frame"]

    async def test_encodes_a_read_only_frame(self):
        """ffmpeg's provider hands out a read-only view over its pipe buffer."""
        frame = _frame()
        frame.setflags(write=False)
        pipeline = _pipeline()
        pipeline.prepare_frame(frame)
        task = _task([pipeline])

        preview = await task.get_live_preview()

        assert preview["frame"]


class TestLivePreviewHook:
    async def test_a_task_type_without_a_live_view_returns_none(self):
        """The base hook is what lets the endpoint say "not available" rather than 500."""
        from apps.tasks_server.tasks.PreviewSourceTask import PreviewSourceTask

        task = PreviewSourceTask.__new__(PreviewSourceTask)

        assert await task.get_live_preview() is None


def _register_running(manager, task) -> str:
    """Put a task into the manager's running set, as a started task would be."""
    from apps.tasks_server.types import TaskInfo

    uuid = "task-live-1"
    manager._running_tasks[uuid] = TaskInfo(config=MagicMock(), instance=MagicMock(), task=task)
    return uuid


class TestLivePreviewEndpoint:
    async def test_serves_the_preview_of_a_running_task(self, http_client, manager):
        pipeline = _pipeline()
        pipeline.prepare_frame(_frame())
        uuid = _register_running(manager, _task([pipeline]))

        resp = await http_client.get(f"/api/tasks/instances/{uuid}/live-preview")

        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "success"
        assert body["preview"]["frame"]
        assert body["preview"]["sourceUuid"] == "src-1"

    async def test_passes_since_seq_through(self, http_client, manager):
        pipeline = _pipeline()
        pipeline.prepare_frame(_frame())
        uuid = _register_running(manager, _task([pipeline]))

        resp = await http_client.get(
            f"/api/tasks/instances/{uuid}/live-preview", params={"since_seq": pipeline.frame_seq}
        )

        assert resp.json()["preview"]["frameUnchanged"] is True

    async def test_a_stopped_watcher_fails_cleanly_rather_than_500ing(self, http_client):
        """The ordinary case: the view is open when the watcher is stopped."""
        resp = await http_client.get("/api/tasks/instances/not-running/live-preview")

        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "failure"
        assert "not currently running" in body["message"]

    async def test_a_task_type_with_no_live_view_says_so(self, http_client, manager):
        from apps.tasks_server.tasks.PreviewSourceTask import PreviewSourceTask

        uuid = _register_running(manager, PreviewSourceTask.__new__(PreviewSourceTask))

        body = (await http_client.get(f"/api/tasks/instances/{uuid}/live-preview")).json()

        assert body["status"] == "failure"
        assert body["preview"] is None
