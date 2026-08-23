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

from apps.helpers.db.types import HARVESTED_TAG_UUID
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


class TestFindingTheFramesAgain:
    """The frames are the point, and until now nothing pointed at them.

    A harvest writes its proposals into a preview run, and a preview run is a
    temporary address: apply it or delete it and the frames become ordinary images
    in `blip_harvest`, indistinguishable from anything else. Two things fix that —
    a durable image tag, and actions on the finished job card that lead back.
    """

    async def _upload_result(self, ok=True, duplicate_of=None, path="blip_harvest/frame.jpg"):
        return SimpleNamespace(ok=ok, duplicate_of=duplicate_of, path=path, uuid="img-1", message=None)

    async def test_a_harvested_frame_is_tagged(self, monkeypatch):
        """Without this the only record that an image came from a harvest is the
        preview run, which the user is expected to delete when they are done."""
        task = _make_task()
        await task._init(task._params, None)
        tagged = AsyncMock()
        monkeypatch.setattr(task, "_tag_harvested", tagged)

        images = MagicMock()
        images.upload_images.return_value = SimpleNamespace(
            status="success",
            results=[SimpleNamespace(ok=True, duplicate_of=None, path="p.jpg", uuid="u", message=None)],
        )
        monkeypatch.setattr(task, "_api", lambda cls: images)

        await task._upload(b"jpeg", _blip(), 1.0)

        tagged.assert_awaited_once()

    async def test_a_duplicate_is_not_tagged_again(self, monkeypatch):
        """A re-run must stay close to free. The image the duplicate matched already
        carries the tag from the run that first filed it, and a second tag write per
        skipped frame is work for no change."""
        task = _make_task()
        await task._init(task._params, None)
        tagged = AsyncMock()
        monkeypatch.setattr(task, "_tag_harvested", tagged)

        images = MagicMock()
        images.upload_images.return_value = SimpleNamespace(
            status="success",
            results=[SimpleNamespace(ok=True, duplicate_of="other", path="p.jpg", uuid="u", message=None)],
        )
        monkeypatch.setattr(task, "_api", lambda cls: images)

        assert await task._upload(b"jpeg", _blip(), 1.0) is None
        tagged.assert_not_awaited()

    async def test_a_failed_tag_does_not_fail_the_harvest(self, monkeypatch):
        """The frame and its proposal are already filed by the time the tag is
        written. Losing the run over a bookkeeping call would throw away the work
        that actually matters."""
        task = _make_task()
        await task._init(task._params, None)
        images = MagicMock()
        images.set_image_tag.side_effect = RuntimeError("tags are down")
        monkeypatch.setattr(task, "_api", lambda cls: images)

        await task._tag_harvested(SimpleNamespace(path="blip_harvest/frame.jpg"))  # must not raise

    async def test_the_finished_job_offers_a_way_into_the_run(self, monkeypatch):
        task = _make_task()
        await task._init(task._params, None)
        task._preview_run_uuid = "run-1"

        kinds = {a["type"]: a for a in task._result_actions()}

        assert kinds["open_preview_run"]["payload"]["run_uuid"] == "run-1"

    async def test_the_tag_action_survives_a_run_that_never_opened(self, monkeypatch):
        """The conditional. With no preview run there is nothing to open — but the
        tag filter is still a valid destination, so the card must not end up with no
        actions at all."""
        task = _make_task()
        await task._init(task._params, None)
        task._preview_run_uuid = None

        kinds = {a["type"] for a in task._result_actions()}

        assert "open_preview_run" not in kinds
        assert "open_review" in kinds

    def test_status_actions_are_safe_before_init_has_run(self):
        """get_status_actions() is reachable during the PENDING window, before _init
        has assigned anything. An AttributeError here breaks the whole Jobs card."""
        task = HarvestBlipFramesTask(task_config_uuid="cfg", params={"blip_uuids": ["b"]}, manager=_MOCK_MANAGER)

        assert isinstance(task.get_status_actions(), list)

    async def test_the_review_action_filters_on_images_not_bboxes(self, monkeypatch):
        """A harvested frame has no bbox until its run is applied, so a bbox-tag
        filter would match nothing at exactly the moment the user follows the link."""
        task = _make_task()
        await task._init(task._params, None)

        review = next(a for a in task._result_actions() if a["type"] == "open_review")

        assert review["payload"]["image_tag_uuids"] == [HARVESTED_TAG_UUID]
        assert review["payload"]["tag_uuids"] == []
