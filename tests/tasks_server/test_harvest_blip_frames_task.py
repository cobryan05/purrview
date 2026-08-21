"""Turning tracking blips into training frames.

Two things here are worth protecting and neither raises when it breaks:

* **Only detector faults may be harvested.** A blip the *tracker* caused is a frame the
  detector already got right, and filing it pads the training set with cases the model
  handles. That is the failure the whole fault split exists to prevent, and from the
  outside a poisoned harvest looks exactly like a successful one.
* **A frame must match the box offered with it.** The proposal comes from the sidecar at
  a given instant; a bare keyframe seek can land over a second away, which would pair a
  picture with a box describing somewhere the subject no longer is. That is the precise
  failure this feature exists to remove, so re-introducing it in the harvester would be
  particularly bleak.

The API round-trips are faked. The decode is not — it runs against a real file written
by cv2, because seek behaviour is the thing under test and a mock would assert nothing.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import cv2
import numpy as np
import pytest

from apps.tasks_server.tasks.HarvestBlipFramesTask import HarvestBlipFramesTask, _grab_frame

_MOCK_MANAGER = MagicMock()
_MOCK_MANAGER.update_task_heartbeat = AsyncMock()

FPS = 30.0
CLIP_SECONDS = 4


def _make_task(**overrides) -> HarvestBlipFramesTask:
    params = {"blip_uuids": ["blip-1"]}
    params.update(overrides)
    task = HarvestBlipFramesTask(task_config_uuid="cfg", params=params, manager=_MOCK_MANAGER)
    task.set_instance_uuid("22222222-2222-2222-2222-222222222222")
    return task


def _blip(uuid="blip-1", *, harvestable=True, fault="detector", kind="identity_flip", **overrides):
    fields = {
        "uuid": uuid,
        "event_uuid": "event-abcdef12",
        "recording_uuid": "rec-1",
        "kind": kind,
        "track_id": 7,
        "start_s": 1.0,
        "end_s": 2.0,
        "severity": 0.6,
        "fault": fault,
        "harvestable": harvestable,
        "state": "new",
        "detail_json": {"modalLabel": "cat_a"},
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


def _write_clip(path, *, seconds=CLIP_SECONDS, fps=FPS):
    """A clip whose every frame states its own index in pixel values.

    Frame *i* is a flat grey of `i * 5`, so a decoded frame can be read back as a frame
    number — the only way to tell whether a seek landed where it was asked to.

    The stride of 5 rather than 1 is what gives the assertions teeth: adjacent frames
    have to differ by more than the codec's own drift, or an off-by-one seek and a bit
    of mp4v noise would be indistinguishable.
    """
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (64, 48))
    total = int(seconds * fps)
    for i in range(total):
        writer.write(np.full((48, 64, 3), (i * 5) % 256, dtype=np.uint8))
    writer.release()
    return total


def _mean_by_sequential_decode(clip, index):
    """Ground truth: the same frame reached by decoding from the start, no seeking.

    Compared against rather than the frame's nominal grey level, because mp4v is lossy
    and the encoded value drifts — which would otherwise make codec noise and a seek
    landing on the wrong frame indistinguishable.
    """
    capture = cv2.VideoCapture(str(clip))
    frame = None
    for _ in range(index + 1):
        ok, frame = capture.read()
        assert ok, f"the clip has no frame {index}"
    capture.release()
    return float(frame[:, :, 0].mean())


class TestPullingTheRightFrame:
    @pytest.mark.parametrize("at_s", [0.5, 2.0, 3.0])
    def test_a_decoded_frame_comes_from_the_moment_that_was_asked_for(self, tmp_path, at_s):
        """The box offered with a frame is the sidecar's box at that instant. A seek that
        lands on a neighbouring frame would pair it with a box describing somewhere the
        subject no longer is."""
        clip = tmp_path / "clip.mp4"
        _write_clip(clip)

        jpeg = _grab_frame(str(clip), at_s)
        assert jpeg is not None

        decoded = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        expected = _mean_by_sequential_decode(clip, int(round(at_s * FPS)))

        assert abs(float(decoded[:, :, 0].mean()) - expected) < 1.0

    def test_two_different_moments_return_two_different_frames(self, tmp_path):
        """Guards the degenerate pass where every seek returns the same keyframe — which
        would look like a working harvest producing one picture over and over."""
        clip = tmp_path / "clip.mp4"
        _write_clip(clip)

        early = cv2.imdecode(np.frombuffer(_grab_frame(str(clip), 0.5), np.uint8), cv2.IMREAD_COLOR)
        late = cv2.imdecode(np.frombuffer(_grab_frame(str(clip), 3.0), np.uint8), cv2.IMREAD_COLOR)

        assert abs(float(early[:, :, 0].mean()) - float(late[:, :, 0].mean())) > 10

    def test_an_unreadable_clip_returns_nothing_rather_than_raising(self, tmp_path):
        missing = tmp_path / "not-here.mp4"
        assert _grab_frame(str(missing), 1.0) is None

    def test_a_moment_past_the_end_returns_nothing(self, tmp_path):
        clip = tmp_path / "clip.mp4"
        _write_clip(clip, seconds=1)

        assert _grab_frame(str(clip), 30.0) is None


class TestWhatMayBeHarvested:
    async def test_a_tracker_fault_is_refused_and_reported(self, monkeypatch):
        """The frames are ones the detector already got right. Harvesting them would
        teach the model nothing and skew the set toward cases it handles."""
        task = _make_task(blip_uuids=["blip-1"])
        await task._init(task._params, None)
        monkeypatch.setattr(task, "_fetch_blips", AsyncMock(return_value=[_blip(harvestable=False, fault="tracker")]))
        created = AsyncMock()
        monkeypatch.setattr(task, "_create_run", created)

        result = await task._run()

        assert result["skippedNotHarvestable"] == 1
        assert result["framesHarvested"] == 0
        created.assert_not_awaited(), "no preview run should be opened for nothing to review"

    async def test_a_detector_fault_is_accepted(self, monkeypatch):
        """The conditional half: same shape of blip, different fault, opposite outcome."""
        task = _make_task(blip_uuids=["blip-1"])
        await task._init(task._params, None)
        monkeypatch.setattr(task, "_fetch_blips", AsyncMock(return_value=[_blip(harvestable=True)]))
        monkeypatch.setattr(task, "_create_run", AsyncMock(return_value="run-1"))
        harvested = AsyncMock()
        monkeypatch.setattr(task, "_harvest_recording", harvested)

        result = await task._run()

        assert result["skippedNotHarvestable"] == 0
        harvested.assert_awaited_once()

    async def test_a_mixed_selection_harvests_only_the_detector_faults(self, monkeypatch):
        task = _make_task(blip_uuids=["blip-1", "blip-2"])
        await task._init(task._params, None)
        monkeypatch.setattr(
            task,
            "_fetch_blips",
            AsyncMock(return_value=[_blip("blip-1", harvestable=True), _blip("blip-2", harvestable=False)]),
        )
        monkeypatch.setattr(task, "_create_run", AsyncMock(return_value="run-1"))
        seen: list = []

        async def _capture(recording_uuid, group):
            seen.extend(b.uuid for b in group)

        monkeypatch.setattr(task, "_harvest_recording", _capture)

        result = await task._run()

        assert seen == ["blip-1"]
        assert result["skippedNotHarvestable"] == 1

    async def test_a_blip_naming_no_recording_is_counted_as_failed_not_crashed(self, monkeypatch):
        """A blip whose clip has been swept has nothing to pull from. It must not take
        the rest of the run with it."""
        task = _make_task(blip_uuids=["blip-1", "blip-2"])
        await task._init(task._params, None)
        monkeypatch.setattr(
            task,
            "_fetch_blips",
            AsyncMock(return_value=[_blip("blip-1", recording_uuid=None), _blip("blip-2", recording_uuid="rec-2")]),
        )
        monkeypatch.setattr(task, "_create_run", AsyncMock(return_value="run-1"))
        reached: list = []

        async def _capture(recording_uuid, group):
            reached.append(recording_uuid)

        monkeypatch.setattr(task, "_harvest_recording", _capture)

        result = await task._run()

        assert result["failed"] == 1
        assert reached == ["rec-2"], "the healthy blip is still harvested"


class TestRefusingToGuess:
    async def test_a_blip_with_no_label_to_propose_files_nothing(self, monkeypatch):
        """Without a modal label there is no correction to suggest, and an image with no
        proposal is clutter in the review browser rather than a frame worth keeping."""
        task = _make_task()
        await task._init(task._params, None)
        blip = _blip(detail_json={})

        grabbed = MagicMock()
        monkeypatch.setattr("apps.tasks_server.tasks.HarvestBlipFramesTask._grab_frame", grabbed)

        proposals = await task._harvest_blip(blip, "/clip.mp4", {"frames": []})

        assert proposals == []
        grabbed.assert_not_called(), "the clip should not even be opened"

    async def test_a_moment_with_no_honest_geometry_pulls_no_frame(self, monkeypatch):
        """`proposed_box_at` returns None when it would have to invent a box. The frame
        is skipped rather than filed without a suggestion."""
        task = _make_task()
        await task._init(task._params, None)
        grabbed = MagicMock()
        monkeypatch.setattr("apps.tasks_server.tasks.HarvestBlipFramesTask._grab_frame", grabbed)

        # An empty document knows about no tracks at all.
        proposals = await task._harvest_blip(_blip(), "/clip.mp4", {"frames": []})

        assert proposals == []
        assert task._no_proposal > 0
        grabbed.assert_not_called()
