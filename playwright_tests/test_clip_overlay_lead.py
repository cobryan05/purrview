"""The clip overlay's correction for an untrimmed head.

MediaMTX cannot cut a clip at an arbitrary instant: it starts at the keyframe at or
before the one asked for and marks the excess with a negative PTS plus an mp4 edit list.
Real clips here carried 4.2 s and 8.8 s of that lead.

Whether the lead is *played* is up to the browser. ffmpeg and Chromium apply the edit
list, so their t=0 is the instant requested and the sidecar lines up; a browser that does
not starts at the earlier keyframe, and every box is drawn against a picture from `lead`
seconds earlier. That was the bug: on Firefox the overlay ran ~4 s ahead of the video,
while on Chromium the same clip measured ~0.3 s out — the same files, the same sidecar.

So the offset is measured from the two durations rather than assumed, and these pin that
arithmetic. They are deliberately about the pure function: mocking a browser into
mis-handling an edit list is not something a test can arrange, but the decision it drives
is one line and is what actually broke.

Run with:
    pytest playwright_tests/test_clip_overlay_lead.py -p no:asyncio -v
"""

import pytest
from playwright.sync_api import Page

_SEEK_CALL = """
    async ([times, fps, lead]) => {
        const { seekTargetFor } = await import('/app-static/js/app/nvr/events/clipOverlay.js');
        return times.map(t => Math.floor((seekTargetFor(t, lead) - lead) * fps));
    }
"""

_CALL = """
    async ([declared, actual]) => {
        const { untrimmedLead } = await import('/app-static/js/app/nvr/events/clipOverlay.js');
        return untrimmedLead(declared, actual);
    }
"""


@pytest.fixture
def landed_on(page: Page, base_url: str):
    """The frame index a seek for each sidecar time actually presents."""
    page.goto(base_url)
    page.wait_for_load_state("networkidle")
    return lambda times, fps, lead=0: page.evaluate(_SEEK_CALL, [times, fps, lead])


@pytest.fixture
def lead(page: Page, base_url: str):
    page.goto(base_url)
    page.wait_for_load_state("networkidle")
    return lambda declared, actual: page.evaluate(_CALL, [declared, actual])


class TestUntrimmedLead:
    def test_a_browser_that_trimmed_needs_no_correction(self, lead):
        """Chromium's reading of a real clip: 29.019 played against 29.153 declared."""
        assert lead(29.1531, 29.019) == 0

    def test_the_surplus_is_the_offset(self, lead):
        """A browser playing the 4.2 s head reports the clip that much longer, and that
        surplus is exactly how far ahead of the picture the boxes would be drawn."""
        assert lead(29.1531, 29.019 + 4.2) == pytest.approx(4.066, abs=0.01)

    def test_muxer_rounding_is_not_treated_as_a_lead(self, lead):
        """Durations disagree slightly for ordinary reasons — the final frame's own
        duration, rounding in the muxer. Shifting the overlay by that noise is worse
        than leaving it alone."""
        assert lead(29.1531, 29.30) == 0

    def test_a_file_shorter_than_declared_is_left_alone(self, lead):
        """A chunk-assembled clip whose duration was summed arithmetically rather than
        measured: one real clip declared 246.4 s against a 231.2 s file. Its boxes drift
        progressively, which a constant shift cannot repair and would only disguise."""
        assert lead(246.3969, 231.164) == 0

    @pytest.mark.parametrize("declared,actual", [(0, 30), (-1, 30), (None, 30), (30, None)])
    def test_unusable_durations_yield_no_correction(self, lead, declared, actual):
        """`video.duration` is NaN until metadata arrives, and the overlay draws before
        then. Guessing an offset from a missing number would move every box on the first
        paint and then move them back."""
        assert lead(declared, actual) == 0


class TestSeekTargetFor:
    """Seeking to the frame a sidecar sample describes.

    `detectionTrack.py` writes sample times to four decimals, so a time generated as
    `index / fps` can round to just below the boundary of its own frame — 1/12 is
    written 0.0833 — and a seek there presents the *previous* frame. Playback never
    shows it, because the overlay looks a sample up from the true `currentTime`; the
    label-correction harvest seeks to the stated time, and on a timelapse one frame is
    five seconds of real time, so it saved a picture the boxes never described.
    """

    @staticmethod
    def _sidecar_times(count, fps):
        """As `_round(index / fps)` writes them."""
        return [round(i / fps, 4) for i in range(count)]

    def test_every_sample_lands_on_its_own_frame(self, landed_on):
        """The whole point: sample i must present frame i, for every i. Against the
        unnudged seek this failed for 10 of these 29 — including both frames the
        correction actually harvested."""
        times = self._sidecar_times(29, 12)
        assert landed_on(times, 12) == list(range(29))

    def test_holds_at_the_rates_clips_are_written_at(self, landed_on):
        """A timelapse is the damaging case, but the same rounding applies to every
        sidecar; a rate whose reciprocal is exact must not be pushed forward a frame."""
        for fps in (5, 10, 12, 15, 25, 30):
            times = self._sidecar_times(40, fps)
            assert landed_on(times, fps) == list(range(40)), fps

    def test_an_untrimmed_head_shifts_the_seek_without_moving_the_frame(self, landed_on):
        """The lead and the nudge are independent corrections: one puts the playhead in
        the right part of the file, the other inside the right frame of it."""
        times = self._sidecar_times(29, 12)
        assert landed_on(times, 12, 4.2) == list(range(29))
