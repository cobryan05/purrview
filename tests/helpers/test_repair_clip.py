"""Concealing a corrupt clip, and the guards that stop it making things worse.

The repair is a decode-and-re-encode, so the ways it can go wrong are all ways of
damaging a file that was merely ugly: losing it, shortening it, or replacing it with
something a player cannot open. The sidecar's box offsets are written against the
original timeline, so a repair that changed the duration would slide every box — which
is the exact fault the clip pipeline was just fixed to stop doing.

So the tests that matter here are the refusals, not the happy path.
"""

import asyncio
import subprocess
from pathlib import Path

import pytest

from apps.helpers.recordings import clipRepairAdvice, repairClip
from apps.helpers.recordings.repairClip import repair_if_corrupt


def _clip(path: Path, seconds: int = 2) -> Path:
    """A real, playable mp4 with nothing wrong with it.

    `-f mp4` named explicitly because these are also written to `<name>.mp4.repair`,
    and ffmpeg picks its muxer from the extension — the same trap `concat_parts`
    documents for its own `.part` file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"testsrc=size=160x120:rate=10:duration={seconds}",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-f",
            "mp4",
            str(path),
        ],
        capture_output=True,
        check=True,
        timeout=60,
    )
    return path


def _damaged_clip(path: Path, seconds: int = 3) -> Path:
    """A real mp4 whose compressed frame data has been corrupted in place.

    Byte-flipped rather than re-encoded: the container, the timestamps and the duration
    all stay intact, so what the decoder meets is exactly the shape a lossy camera
    produces — a frame it cannot reconstruct inside an otherwise sound file. Every other
    test here patches `_conceal`, so this is the only fixture that reaches the real
    ffmpeg command.

    The corruption is deterministic: a fixed stride over a fixed window a quarter of the
    way in, which lands in the mdat and leaves the moov alone.
    """
    _clip(path, seconds)
    data = bytearray(path.read_bytes())
    start = len(data) // 4
    for index in range(start, start + 2000, 7):
        data[index] ^= 0xFF
    path.write_bytes(bytes(data))
    return path


class TestTheConcealCommandTheInstalledFfmpegActuallyAccepts:
    """The one path every other test in this file patches out.

    `_conceal` is stubbed by the overlay-invalidation tests and never reached by the
    leave-well-alone ones, so an option this image's ffmpeg does not have disabled the
    whole feature for every camera that opted in, and the suite stayed green. These run
    the real command.
    """

    async def test_a_damaged_clip_is_actually_repaired(self, tmp_path):
        clip = _damaged_clip(tmp_path / "damaged.mp4")
        before = repairClip._count_errors(clip)
        assert before >= repairClip.MIN_ERRORS_TO_REPAIR, "fixture must be damaged enough to be worth repairing"

        result = await repair_if_corrupt(clip)

        assert result["repaired"] is True, f"reason: {result.get('reason')}"
        assert result["decodeErrorsAfter"] < before, "a repair that conceals nothing is not a repair"

    def test_a_failed_conceal_reports_ffmpegs_reason_not_its_version_banner(self, tmp_path):
        """The log keeps only the tail of stderr, so an unsuppressed banner is the
        difference between "Unrecognized option" and a list of library versions. The
        real failure read as the latter for two days."""
        proc = repairClip._conceal(tmp_path / "does-not-exist.mp4", tmp_path / "out.mp4")

        assert proc.returncode != 0
        tail = proc.stderr[-300:].decode("utf-8", "replace")
        assert "No such file" in tail
        assert "libavformat" not in tail, f"the banner is crowding out the reason: {tail!r}"

    async def test_the_repair_keeps_the_clip_the_same_length(self, tmp_path):
        """The overlay places boxes by offset into the video, so a re-encode that moves
        the duration slides every box. `repair_if_corrupt` discards such a repair — this
        asserts the real command does not produce one, which is what makes the feature
        usable rather than merely safe."""
        clip = _damaged_clip(tmp_path / "damaged.mp4")
        before = repairClip._probe_duration(clip)

        result = await repair_if_corrupt(clip)

        assert result["repaired"] is True, f"reason: {result.get('reason')}"
        after = repairClip._probe_duration(clip)
        assert abs(after - before) <= repairClip.DURATION_TOLERANCE_S


class TestItLeavesWellAloneClipsAlone:
    async def test_a_disabled_camera_is_not_scanned(self, tmp_path):
        """The scan is a full decode pass. A camera that has not asked for this must not
        pay for one on every clip."""
        scanned = []
        clip = _clip(tmp_path / "a.mp4")

        result = await repair_if_corrupt(clip, enabled=False)

        assert result == {"repaired": False, "reason": "disabled"}
        assert scanned == []

    async def test_a_missing_file_is_not_an_error(self, tmp_path):
        """A clip whose extraction failed is absent, and this runs after extraction."""
        result = await repair_if_corrupt(tmp_path / "never-written.mp4")

        assert result["repaired"] is False
        assert result["reason"] == "missing"

    async def test_an_undamaged_clip_is_not_re_encoded(self, tmp_path):
        """Most clips are clean, and re-encoding one costs CPU for an identical picture
        — on the box also serving the live app."""
        clip = _clip(tmp_path / "clean.mp4")
        before = clip.read_bytes()

        result = await repair_if_corrupt(clip)

        assert result["repaired"] is False
        assert result["reason"] == "clean"
        assert clip.read_bytes() == before, "a clean clip was rewritten"


class TestItRefusesARepairThatWouldInvalidateTheOverlay:
    """The detection sidecar is written against the clip's measured length, so a repair
    that moved the duration would put every box at the wrong moment. Keeping a corrupt
    clip whose boxes are right beats a clean one whose boxes are wrong.
    """

    async def test_a_repair_that_shortens_the_clip_is_discarded(self, tmp_path, monkeypatch):
        clip = _clip(tmp_path / "c.mp4", seconds=3)
        original = clip.read_bytes()
        monkeypatch.setattr(repairClip, "_count_errors", lambda p: 50)

        def _shorter(src, dest):
            _clip(Path(dest), seconds=1)
            return subprocess.CompletedProcess([], 0, b"", b"")

        monkeypatch.setattr(repairClip, "_conceal", _shorter)

        result = await repair_if_corrupt(clip)

        assert result["repaired"] is False
        assert result["reason"] == "duration moved"
        assert clip.read_bytes() == original, "the original was replaced by a shorter clip"

    async def test_a_repair_keeping_the_duration_is_accepted(self, tmp_path, monkeypatch):
        """The positive half — without it the test above passes against a function that
        refuses every repair, and the feature would silently never work."""
        clip = _clip(tmp_path / "d.mp4", seconds=2)
        monkeypatch.setattr(repairClip, "_count_errors", lambda p: 50)

        def _same_length(src, dest):
            _clip(Path(dest), seconds=2)
            return subprocess.CompletedProcess([], 0, b"", b"")

        monkeypatch.setattr(repairClip, "_conceal", _same_length)

        result = await repair_if_corrupt(clip)

        assert result["repaired"] is True
        assert result["decodeErrors"] == 50

    async def test_a_failed_encode_keeps_the_original(self, tmp_path, monkeypatch):
        clip = _clip(tmp_path / "e.mp4")
        original = clip.read_bytes()
        monkeypatch.setattr(repairClip, "_count_errors", lambda p: 50)
        monkeypatch.setattr(
            repairClip,
            "_conceal",
            lambda src, dest: subprocess.CompletedProcess([], 1, b"", b"ffmpeg fell over"),
        )

        result = await repair_if_corrupt(clip)

        assert result["repaired"] is False
        assert result["reason"] == "encode failed"
        assert clip.read_bytes() == original

    async def test_the_half_written_repair_is_never_left_behind(self, tmp_path, monkeypatch):
        """`.repair` beside a clip would be picked up by nothing and swept by nothing."""
        clip = _clip(tmp_path / "f.mp4")
        monkeypatch.setattr(repairClip, "_count_errors", lambda p: 50)
        monkeypatch.setattr(
            repairClip,
            "_conceal",
            lambda src, dest: subprocess.CompletedProcess([], 1, b"", b"boom"),
        )

        await repair_if_corrupt(clip)

        assert list(tmp_path.glob("*.repair")) == []


class TestTheErrorScan:
    def test_decoder_errors_are_counted_not_just_detected(self, tmp_path):
        """The count drives whether a re-encode is worth it, so it has to be a number
        rather than a boolean."""
        assert repairClip._count_errors(_clip(tmp_path / "g.mp4")) == 0

    @pytest.mark.parametrize(
        "line",
        [
            "[h264 @ 0x55] error while decoding MB 38 7, bytestream -7",
            "[h264 @ 0x55] concealing 120 DC, 120 AC, 120 MV errors in P frame",
            "[h264 @ 0x55] Invalid NAL unit size",
        ],
    )
    def test_the_damage_ffmpeg_reports_is_recognised(self, line):
        """These are the shapes ffmpeg actually prints for a lossy stream; missing one
        means a corrupt clip is reported clean and never repaired."""
        assert repairClip._ERROR_PATTERN.search(line)

    def test_an_unrelated_warning_is_not_counted_as_damage(self, tmp_path):
        """Counting ordinary chatter would re-encode every clip."""
        assert not repairClip._ERROR_PATTERN.search("[mp4 @ 0x55] track 0 has no keyframes")


class TestTheCameraCarriesTheChoiceNotTheAction:
    """Concealment is a property of the camera's stream, so the switch lives on the
    source and reaches the action through its context. An action cannot be configured
    for it: two clip actions on one lossy camera could otherwise disagree, and anything
    else that re-encodes that camera's video would have no way to ask.
    """

    @staticmethod
    def _action(**kwargs):
        from apps.helpers.actions.recordClipAction import RecordClipAction

        return RecordClipAction(**kwargs)

    @staticmethod
    def _context(*, repair: bool, source_uuid: str = "src-1"):
        from apps.helpers.actions.action import ActionContext

        return ActionContext(
            profile_uuid="p",
            profile_name="P",
            source_uuid=source_uuid,
            source_name="Garage",
            repair_corruption=repair,
        )

    @staticmethod
    async def _settle(action):
        """Close the action and let the worker finish whatever it had started."""
        await asyncio.sleep(0)
        await action.close()

    @pytest.fixture(autouse=True)
    def _fresh_advice(self):
        """Probe bookkeeping is process-wide, so a test must not inherit another's."""
        clipRepairAdvice.reset()
        yield
        clipRepairAdvice.reset()

    async def test_the_action_cannot_be_configured_for_it(self):
        """The knob is gone from the action: a per-camera fact must have exactly one
        place to be set, or two actions on one camera can disagree."""
        from apps.helpers.actions.recordClipAction import RecordClipAction

        assert "repair_corruption" not in RecordClipAction.params_schema()

    async def test_a_source_that_opted_in_is_repaired(self, tmp_path, monkeypatch):
        repaired, probed = [], []
        monkeypatch.setattr(repairClip, "submit_repair", lambda p: repaired.append(p) or True)
        monkeypatch.setattr(repairClip, "submit_probe", lambda p, **kw: probed.append(p) or True)

        action = self._action()
        action._conceal_damage({}, tmp_path / "clip.mp4", self._context(repair=True))
        await self._settle(action)

        assert repaired == [tmp_path / "clip.mp4"]
        assert probed == []

    async def test_a_source_that_did_not_is_only_scanned(self, tmp_path, monkeypatch):
        """The conditional that matters: a camera which has not opted in never gets a
        re-encode, only the far cheaper scan that can suggest one."""
        repaired, probed = [], []
        monkeypatch.setattr(repairClip, "submit_repair", lambda p: repaired.append(p) or True)
        monkeypatch.setattr(repairClip, "submit_probe", lambda p, **kw: probed.append(p) or True)

        action = self._action()
        action._conceal_damage({}, tmp_path / "clip.mp4", self._context(repair=False))
        await self._settle(action)

        assert repaired == []
        assert probed == [tmp_path / "clip.mp4"]

    async def test_an_answered_source_is_neither_repaired_nor_scanned(self, tmp_path, monkeypatch):
        """Once the user has declined the suggestion the scan stops too — otherwise a
        camera they said no to pays a decode pass on every clip forever."""
        repaired, probed = [], []
        monkeypatch.setattr(repairClip, "submit_repair", lambda p: repaired.append(p) or True)
        monkeypatch.setattr(repairClip, "submit_probe", lambda p, **kw: probed.append(p) or True)
        clipRepairAdvice.decide("src-1")

        action = self._action()
        action._conceal_damage({}, tmp_path / "clip.mp4", self._context(repair=False))
        await self._settle(action)

        assert repaired == []
        assert probed == []

    async def test_the_caller_does_not_wait_for_the_repair(self, tmp_path, monkeypatch):
        """The watcher awaits its actions inside the frame loop, so the minutes a decode
        scan and re-encode take must not be spent there. The clip is queued and the
        repair happens after the action has returned.
        """
        started, release = asyncio.Event(), asyncio.Event()
        seen = {}

        async def _fake(path, *, enabled=True):
            seen["path"] = path
            started.set()
            await release.wait()
            return {"repaired": True, "decodeErrors": 42}

        monkeypatch.setattr("apps.helpers.recordings.repairClip.repair_if_corrupt", _fake)
        action = self._action()

        action._conceal_damage({}, tmp_path / "clip.mp4", self._context(repair=True))
        # Generous on purpose: the budget only has to outlast the queue consumer being
        # scheduled on a box running eight pytest workers. A repair that genuinely never
        # starts then hangs until pytest's own limit instead of failing in a second that
        # says nothing (FLAKY_TESTS.md 2026-10-09).
        await asyncio.wait_for(started.wait(), 30)

        # The repair is in flight and the action has already returned — the assertion
        # that fails if it is ever awaited inline again.
        assert not release.is_set()
        assert seen["path"] == tmp_path / "clip.mp4"

        release.set()
        await self._settle(action)

    async def test_two_clips_are_repaired_one_at_a_time(self, tmp_path, monkeypatch):
        """Queued rather than a task per clip: a busy camera closes several events at
        once, and concurrent 1080p re-encodes are what the niceness exists to prevent.
        """
        concurrent, peak, release, entered = 0, 0, asyncio.Event(), asyncio.Event()

        async def _fake(path, *, enabled=True):
            nonlocal concurrent, peak
            concurrent += 1
            peak = max(peak, concurrent)
            entered.set()
            await release.wait()
            concurrent -= 1
            return {"repaired": False}

        monkeypatch.setattr("apps.helpers.recordings.repairClip.repair_if_corrupt", _fake)
        action = self._action()
        context = self._context(repair=True)

        action._conceal_damage({}, tmp_path / "a.mp4", context)
        action._conceal_damage({}, tmp_path / "b.mp4", context)
        # Wait for the first repair to be *inside* the fake, not for a fixed number of
        # loop iterations: releasing before the consumer got there leaves peak at 0 and
        # nothing ever overlaps, so the test passes or fails on scheduling order
        # (FLAKY_TESTS.md 2026-10-09).
        await asyncio.wait_for(entered.wait(), 30)
        release.set()
        await self._settle(action)

        assert peak == 1

    async def test_a_scan_never_runs_beside_a_re_encode(self, tmp_path, monkeypatch):
        """Probes share the repair queue rather than getting their own: a scan that ran
        alongside a 1080p re-encode would defeat the point of serialising at all.
        """
        concurrent, peak, release, entered = 0, 0, asyncio.Event(), asyncio.Event()

        async def _busy(*args, **kwargs):
            nonlocal concurrent, peak
            concurrent += 1
            peak = max(peak, concurrent)
            entered.set()
            await release.wait()
            concurrent -= 1
            return {"repaired": False}

        async def _busy_probe(path):
            await _busy(path)
            return 0

        monkeypatch.setattr("apps.helpers.recordings.repairClip.repair_if_corrupt", _busy)
        monkeypatch.setattr("apps.helpers.recordings.repairClip.probe_for_corruption", _busy_probe)
        action = self._action()

        action._conceal_damage({}, tmp_path / "a.mp4", self._context(repair=True, source_uuid="opted-in"))
        action._conceal_damage({}, tmp_path / "b.mp4", self._context(repair=False, source_uuid="undecided"))
        # The first worker item must be inside the fake before it is released, or
        # nothing overlaps and peak stays 0 whatever the queue does.
        await asyncio.wait_for(entered.wait(), 30)
        release.set()
        await self._settle(action)

        assert peak == 1

    async def test_a_ring_cut_clip_is_skipped(self, tmp_path, monkeypatch):
        """A ring-cut clip is encoded from whole JPEGs, so it has no damaged slices to
        conceal — scanning one would be a decode pass for a guaranteed zero. True of
        the probe as much as the repair: there is nothing for either to find."""
        repaired, probed = [], []
        monkeypatch.setattr(repairClip, "submit_repair", lambda p: repaired.append(p) or True)
        monkeypatch.setattr(repairClip, "submit_probe", lambda p, **kw: probed.append(p) or True)

        action = self._action()
        action._conceal_damage({"lowFps": True}, tmp_path / "c.mp4", self._context(repair=True))
        action._conceal_damage({"lowFps": True}, tmp_path / "d.mp4", self._context(repair=False))
        await self._settle(action)

        assert repaired == []
        assert probed == []

    async def test_a_full_backlog_leaves_the_clip_as_recorded(self, tmp_path, monkeypatch):
        """A repair that never runs leaves the file exactly as the recorder wrote it,
        which is what every failure path in this module already settles for — so the
        backlog is bounded rather than allowed to grow without limit.
        """
        monkeypatch.setattr(repairClip, "MAX_PENDING_REPAIRS", 2)
        release, entered = asyncio.Event(), asyncio.Event()

        async def _fake(path, *, enabled=True):
            entered.set()
            await release.wait()
            return {"repaired": False}

        monkeypatch.setattr("apps.helpers.recordings.repairClip.repair_if_corrupt", _fake)
        worker = repairClip._RepairWorker()
        worker.retain()

        # The first submission starts the worker; waiting for it to be *in* the repair
        # is what empties the queue slot, so the two that follow fill the bound and the
        # fourth is refused. A bare yield only sometimes gets that far.
        assert worker.submit(tmp_path / "0.mp4") is True
        await asyncio.wait_for(entered.wait(), 30)
        accepted = [worker.submit(tmp_path / f"{n}.mp4") for n in (1, 2, 3)]

        assert accepted == [True, True, False]

        release.set()
        await worker.release()
