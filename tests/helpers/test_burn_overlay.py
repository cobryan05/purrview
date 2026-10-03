"""The annotated-export renderer: where the overlay changes, and what ends up burnt in."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from apps.helpers.recordings.burnOverlay import (
    DEFAULT_STALENESS_S,
    burn_overlay,
    frame_times,
    overlay_segments,
    probe_video,
    sample_at,
)

HAS_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def doc(frames, *, event=None, staleness=DEFAULT_STALENESS_S, version=2):
    document = {"version": version, "maxStalenessS": staleness, "frames": frames}
    if event:
        document["event"] = event
    return document


def box(label="Cat", x=0.25, y=0.25, w=0.2, h=0.2):
    return {"x": x, "y": y, "w": w, "h": h, "label": label, "labelUuid": "u-1", "conf": 0.9}


class TestSampleAt:
    def test_a_sample_is_held_until_the_next_one(self):
        d = doc([{"t": 0.0, "boxes": [box()]}, {"t": 1.0, "boxes": []}], staleness=5.0)
        assert sample_at(d, 0.5)["t"] == 0.0
        assert sample_at(d, 1.5)["t"] == 1.0

    def test_nothing_is_drawn_once_a_sample_is_stale(self):
        # The distinction that matters: a gap in the record must render as a gap, not as
        # a box frozen where the subject was when the watcher stopped analysing.
        d = doc([{"t": 0.0, "boxes": [box()]}], staleness=1.0)
        assert sample_at(d, 0.9) is not None
        assert sample_at(d, 1.5) is None

    def test_nothing_is_drawn_before_the_first_sample(self):
        assert sample_at(doc([{"t": 2.0, "boxes": [box()]}]), 1.0) is None


class TestOverlaySegments:
    def test_the_event_window_opens_and_closes_on_its_own_boundaries(self):
        segments = overlay_segments(doc([], event={"startS": 2.0, "endS": 4.0}), 6.0)
        active = [(s["start"], s["end"]) for s in segments if s["eventActive"]]
        assert active == [(2.0, 4.0)]

    def test_a_stale_stretch_is_one_empty_segment_however_long_it_is(self):
        # Merging is what keeps a quiet clip cheap: without it every staleness boundary
        # would be its own rendered still for footage in which nothing is drawn at all.
        d = doc([{"t": 0.0, "boxes": [box()]}], staleness=0.5)
        segments = overlay_segments(d, 10.0)
        assert [s["sample"] is None for s in segments] == [False, True]
        assert segments[-1]["end"] == 10.0

    def test_two_samples_holding_equal_boxes_stay_two_segments(self):
        # Equal content, different moments. Merging on equality instead of identity
        # would collapse them and the second sample's hold would be attributed to the
        # first — invisible here, wrong the moment a staleness gap falls between them.
        d = doc([{"t": 0.0, "boxes": [box()]}, {"t": 1.0, "boxes": [box()]}], staleness=5.0)
        starts = [s["start"] for s in overlay_segments(d, 2.0)]
        assert 1.0 in starts

    def test_segments_cover_the_whole_clip_without_gaps(self):
        d = doc(
            [{"t": 0.3, "boxes": [box()]}, {"t": 1.9, "boxes": []}],
            event={"startS": 0.8, "endS": 1.4},
            staleness=0.4,
        )
        segments = overlay_segments(d, 3.0)
        assert segments[0]["start"] == 0.0
        assert segments[-1]["end"] == 3.0
        for before, after in zip(segments, segments[1:]):
            assert before["end"] == after["start"]

    def test_a_zero_length_clip_renders_nothing(self):
        assert overlay_segments(doc([{"t": 0.0, "boxes": [box()]}]), 0.0) == []


class TestSegmentsFollowTheVideosOwnFrames:
    """The half-second lag, and why segmenting on sample times caused it.

    A sidecar sample is stamped when the watcher's copy of a frame came back out of its
    decoder, which is a few milliseconds past that frame's own position in the clip.
    ffmpeg's `overlay` takes up a new image only on the next main frame at or after its
    timestamp, so a segment starting at the sample time landed on the *following* frame —
    at the couple of frames a second these cameras produce, visibly late.
    """

    #: A real clip's frame times and the sidecar written beside it, abridged. The gap in
    #: each pair is the whole bug: 6.6821 describes the frame at 6.68, not the one at 7.48.
    FRAMES = [0.0, 1.2, 6.68, 7.48, 8.12, 8.48, 9.12, 9.44, 9.64, 10.12]
    SAMPLES = [1.2, 6.6821, 7.4844, 8.1028, 8.4916, 9.1274, 9.4248, 9.6275]

    def _doc(self):
        return doc(
            [{"t": t, "boxes": [box(label=f"at-{t}")]} for t in self.SAMPLES],
            staleness=5.0,
        )

    def _segment_at(self, segments, t):
        for segment in segments:
            if segment["start"] <= t < segment["end"]:
                return segment
        return None

    def test_a_sample_is_drawn_on_the_frame_it_was_taken_from(self):
        segments = overlay_segments(self._doc(), 11.0, self.FRAMES)
        drawn = self._segment_at(segments, 6.68)["sample"]
        assert drawn["t"] == 6.6821

    def test_every_segment_begins_on_a_real_frame(self):
        # What makes the correction stick: `overlay` changes images on main frames, so a
        # boundary anywhere else is rounded forward to the next one.
        segments = overlay_segments(self._doc(), 11.0, self.FRAMES)
        assert all(segment["start"] in self.FRAMES for segment in segments)

    def test_the_event_border_opens_on_the_frame_the_rule_fired_on(self):
        document = doc(
            [{"t": t, "boxes": []} for t in self.SAMPLES],
            event={"startS": 6.6821, "endS": 9.6275},
            staleness=5.0,
        )
        segments = overlay_segments(document, 11.0, self.FRAMES)
        assert self._segment_at(segments, 6.68)["eventActive"]
        assert not self._segment_at(segments, 1.2)["eventActive"]

    def test_a_stale_sample_still_drops_out(self):
        # The correction must not turn into "hold forever": staleness is still measured
        # against the sample's own time.
        document = doc([{"t": 1.2, "boxes": [box()]}], staleness=1.0)
        segments = overlay_segments(document, 11.0, self.FRAMES)
        assert self._segment_at(segments, 1.2)["sample"] is not None
        assert self._segment_at(segments, 6.68)["sample"] is None


def _make_clip(dest: Path, seconds=1.0, width=64, height=48):
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"color=c=black:s={width}x{height}:r=10:d={seconds}",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(dest),
        ],
        capture_output=True,
        check=True,
        timeout=120,
    )


def _painted_pixels(clip: Path, at_s: float, tmp_path: Path) -> int:
    """How many pixels of the frame at `at_s` are visibly not the black source.

    Thresholded rather than compared exactly: the render is lossy yuv420p, so x264
    bleeds values like (1, 0, 0) into frames well after the last drawn box. An exact
    "is it still pure black" test fails on that noise while a frame with a real box in
    it has thousands of pixels far above any threshold.
    """
    from PIL import Image

    out = tmp_path / f"frame-{at_s}.png"
    subprocess.run(
        ["ffmpeg", "-nostdin", "-y", "-ss", f"{at_s:.3f}", "-i", str(clip), "-frames:v", "1", str(out)],
        capture_output=True,
        check=True,
        timeout=120,
    )
    with Image.open(out) as img:
        return sum(count for count, colour in img.convert("RGB").getcolors(maxcolors=1 << 20) if max(colour) > 32)


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg/ffprobe are not on PATH")
class TestBurnOverlay:
    def test_the_boxes_land_inside_the_event_window_and_nowhere_else(self, tmp_path):
        clip = tmp_path / "clip.mp4"
        _make_clip(clip, seconds=3.0)
        document = doc(
            [{"t": 1.0, "boxes": [box()]}],
            event={"startS": 1.0, "endS": 2.0, "ruleName": "cat-in-yard"},
            staleness=1.0,
        )
        dest = tmp_path / "annotated.mp4"
        burn_overlay(clip, document, dest)

        assert dest.is_file()
        width, height, duration = probe_video(dest)
        # The whole clip survives: an overlay that runs out must not truncate the video.
        assert (width, height) == (64, 48)
        assert duration == pytest.approx(3.0, abs=0.3)

        # The source is uniformly black, so anything in the picture was burnt in — and
        # the pre-roll and post-roll must come back clean, or the red event frame is
        # telling the viewer the rule fired when it had not.
        assert _painted_pixels(dest, 0.3, tmp_path) == 0
        assert _painted_pixels(dest, 1.5, tmp_path) > 100
        assert _painted_pixels(dest, 2.6, tmp_path) == 0

    def test_the_frame_times_come_back_from_a_real_file(self, tmp_path):
        clip = tmp_path / "clip.mp4"
        _make_clip(clip, seconds=1.0)
        times = frame_times(clip)
        # 10 fps for a second, and ascending — the order the segmenter relies on.
        assert len(times) == 10
        assert times == sorted(times)
        assert times[0] == pytest.approx(0.0, abs=0.01)

    def test_the_frame_times_of_an_unreadable_file_are_empty_not_an_error(self, tmp_path):
        # The fallback path has to be reachable: a clip ffprobe cannot enumerate should
        # export one frame late, not fail to export at all.
        broken = tmp_path / "broken.mp4"
        broken.write_bytes(b"not an mp4")
        assert frame_times(broken) == []

    def test_a_sidecar_from_a_newer_writer_is_refused(self, tmp_path):
        clip = tmp_path / "clip.mp4"
        _make_clip(clip)
        with pytest.raises(ValueError):
            burn_overlay(clip, doc([{"t": 0.0, "boxes": [box()]}], version=99), tmp_path / "out.mp4")

    def test_an_overlay_covering_none_of_the_clip_is_refused(self, tmp_path):
        # The clip is a second long and the sidecar describes a moment a minute in.
        # Rendering a bare re-encode and calling it annotated would be worse than
        # failing: the user would file "the boxes are missing" against a 200.
        clip = tmp_path / "clip.mp4"
        _make_clip(clip)
        with pytest.raises(ValueError):
            burn_overlay(clip, doc([]), tmp_path / "out.mp4")

    def test_the_sidecar_as_written_on_disk_renders(self, tmp_path):
        # Guards the one seam the unit tests above cannot see: `json.loads` of a real
        # sidecar produces string keys and floats, and the renderer reads it unmapped.
        clip = tmp_path / "clip.mp4"
        _make_clip(clip)
        sidecar = tmp_path / "clip.mp4.boxes.json"
        sidecar.write_text(json.dumps(doc([{"t": 0.0, "boxes": [box()]}], event={"startS": 0.0, "endS": 1.0})))
        dest = tmp_path / "out.mp4"
        burn_overlay(clip, json.loads(sidecar.read_text()), dest)
        assert _painted_pixels(dest, 0.4, tmp_path) > 100
