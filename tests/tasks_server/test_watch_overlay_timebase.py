"""The clock a clip overlay's boxes are filed under.

Clip overlays are drawn by mapping a sample's wallclock onto the clip by subtraction
(`detectionTrack.build_overlay`), so a sample stamped later than the picture it
describes puts its boxes that far past the action — the boxes trail the cat across the
frame. The watcher shipped doing exactly that: it re-read the clock *after* awaiting
inference and stamped the sample with that, so every box was dated by however long the
model took, and the frame ring — which files the same frame at arrival — disagreed with
the boxes drawn over it.

Both facts below are about one frame's two records agreeing. Neither can be satisfied
by a stamp taken after inference returns, which is what makes them regression tests
rather than restatements of the code.
"""

import asyncio
import time
from types import SimpleNamespace

from apps.helpers.tracking.tracker import Detection
from apps.tasks_server.tasks._watch_pipeline import SourceHealth
from apps.tasks_server.tasks.StreamWatcherTask import StreamWatcherTask

#: Long enough that a post-inference stamp is unmistakable against scheduler noise,
#: short enough not to slow the suite.
INFERENCE_DELAY_S = 0.25


class _Recorder:
    """Stands in for the frame ring and the detection track, which share a `push`."""

    def __init__(self) -> None:
        self.stamps: list = []

    def push(self, at, *args, **kwargs) -> None:
        self.stamps.append(at)


def _frame():
    """One provided frame. `image` only needs a `.shape` and to survive prepare_frame."""
    import numpy as np

    return SimpleNamespace(image=np.zeros((4, 4, 3), dtype=np.uint8), encoded_b64=None)


class _Provider:
    def __init__(self) -> None:
        self._frames = [_frame()]

    def is_ready(self) -> bool:
        return True

    def is_connecting(self) -> bool:
        return False

    async def getNextImage(self):
        return self._frames.pop(0) if self._frames else None


def _harness():
    """A watcher and pipeline carrying only what one successful pass touches.

    The ring and the track are the same recorder type on purpose: the point of the
    test is that one frame reaches both of them under one timestamp.
    """
    task = StreamWatcherTask.__new__(StreamWatcherTask)
    task._failed_sources = set()
    task._log_lines = []
    task.log = task._log_lines.append
    task._frames_analysed = 0
    task._wallclock_offset = time.time() - time.monotonic()
    task._FETCH_TIMEOUT_S = 10.0

    ring, track = _Recorder(), _Recorder()
    task._ring_for = lambda pipeline: ring
    task._detection_track_for = lambda pipeline: track
    task._record_inference_time = lambda ms: None

    async def _inference(*args, **kwargs):
        # The whole point: something that takes real, measurable time between the
        # frame arriving and the boxes existing.
        await asyncio.sleep(INFERENCE_DELAY_S)
        return [Detection(bbox=(0.0, 0.0, 1.0, 1.0), label_uuid="label-cat", label_name="Cat", confidence=0.9)]

    task._run_inference = _inference

    async def _noop(*args, **kwargs):
        return None

    task._dispatch = _noop
    task._service_drain = _noop
    task._dispatch_frame = _noop
    task._promote_open_events = _noop

    pipeline = SimpleNamespace(
        source_uuid="src-1",
        source_name="Kitchen",
        provider=_Provider(),
        health=SourceHealth(),
        model_names=["cat-model"],
        detections=[],
        tracks=[],
    )
    pipeline.schedule_next = lambda now: None
    pipeline.prepare_frame = lambda image: image
    pipeline.step = lambda detections, now: []
    return task, pipeline, ring, track


class TestOverlayTimeBase:
    async def test_the_boxes_and_the_frame_are_filed_under_one_clock(self, monkeypatch):
        """A ring-cut clip and its overlay must be built on the same time base.

        They were not: the frame went in at arrival and its boxes went in after
        inference, so the sidecar's `t` values were offset from the very frames they
        were drawn over, by an amount that varied with model latency.
        """
        task, pipeline, ring, track = _harness()
        monkeypatch.setattr(StreamWatcherTask, "_record_source_failure", lambda *a, **k: None, raising=False)

        await task._service_source(pipeline, inference_api=None)

        assert len(ring.stamps) == 1 and len(track.stamps) == 1
        assert ring.stamps[0] == track.stamps[0]

    async def test_a_sample_is_not_dated_by_how_long_the_model_took(self, monkeypatch):
        """The stamp belongs to the picture, not to the moment the boxes came back.

        With a post-inference stamp this gap is the model's latency — a quarter of a
        second here, and seconds on a loaded GPU — and it lands the boxes that far
        past the action in every clip.
        """
        task, pipeline, ring, track = _harness()
        monkeypatch.setattr(StreamWatcherTask, "_record_source_failure", lambda *a, **k: None, raising=False)

        # Wallclock, because that is the base both rings are stamped in — the loop
        # runs on a monotonic clock and the offset is added on the way in.
        before = time.time()
        await task._service_source(pipeline, inference_api=None)
        after = time.time()

        assert after - before >= INFERENCE_DELAY_S  # the delay really happened
        assert track.stamps[0] < before + INFERENCE_DELAY_S
