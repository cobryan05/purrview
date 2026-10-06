"""Suggesting clip repair for a camera that is visibly losing packets.

`repair_corruption` stays a switch the user throws — a re-encode costs real CPU and
only they know whether a camera's picture bothers them. What the app can do is stop
making them guess the one thing it already measures.

The behaviour worth pinning is all about *restraint*: the probe is a full decode pass,
so the tests here are the ones that fail if it ever starts running more often than the
question needs answering.
"""

import json

import pytest

from apps.helpers.recordings import clipRepairAdvice


@pytest.fixture(autouse=True)
def _fresh():
    clipRepairAdvice.reset()
    yield
    clipRepairAdvice.reset()


class TestWhenItIsWorthScanningAClip:
    def test_an_unknown_source_is_scanned(self):
        assert clipRepairAdvice.wants_probe("src-1") is True

    def test_a_source_with_no_uuid_is_not(self):
        """Ad-hoc providers built outside SourceHelper have no source row to advise on."""
        assert clipRepairAdvice.wants_probe(None) is False

    def test_scanning_stops_once_the_damage_is_known(self):
        """One finding is the whole output. Re-measuring a camera already queued up as a
        suggestion spends a decode pass to learn nothing."""
        clipRepairAdvice.record_probe("src-1", 999, "a.mp4")

        assert clipRepairAdvice.wants_probe("src-1") is False

    def test_scanning_stops_after_the_cap(self):
        """A clean camera must not be scanned for the life of the process: the question
        is "is this camera lossy", which a few clips answer either way."""
        for n in range(clipRepairAdvice.MAX_PROBES_PER_SOURCE):
            assert clipRepairAdvice.wants_probe("src-1") is True
            clipRepairAdvice.record_probe("src-1", 0, f"{n}.mp4")

        assert clipRepairAdvice.wants_probe("src-1") is False

    def test_one_source_reaching_the_cap_does_not_silence_another(self):
        for n in range(clipRepairAdvice.MAX_PROBES_PER_SOURCE):
            clipRepairAdvice.record_probe("src-1", 0, f"{n}.mp4")

        assert clipRepairAdvice.wants_probe("src-2") is True

    def test_an_answered_source_is_never_scanned_again(self):
        clipRepairAdvice.decide("src-1")

        assert clipRepairAdvice.wants_probe("src-1") is False


class TestWhenADamagedClipBecomesASuggestion:
    def test_a_few_errors_are_not_worth_interrupting_for(self):
        """A handful of damaged macroblocks in one frame is a flicker nobody sees, and
        the suggestion costs the user attention — so it sits above that floor."""
        assert clipRepairAdvice.record_probe("src-1", clipRepairAdvice.MIN_ERRORS_TO_SUGGEST - 1, "a.mp4") is False
        assert clipRepairAdvice.findings() == []

    def test_a_badly_damaged_clip_is(self):
        assert clipRepairAdvice.record_probe("src-1", clipRepairAdvice.MIN_ERRORS_TO_SUGGEST, "a.mp4") is True

        (finding,) = clipRepairAdvice.findings()
        assert finding["sourceUuid"] == "src-1"
        assert finding["errors"] == clipRepairAdvice.MIN_ERRORS_TO_SUGGEST

    def test_damage_found_on_a_source_the_user_declined_is_not_resurfaced(self):
        """The decline has to survive the next damaged clip, or it means nothing."""
        clipRepairAdvice.decide("src-1")

        assert clipRepairAdvice.record_probe("src-1", 999, "a.mp4") is False
        assert clipRepairAdvice.findings() == []

    def test_answering_removes_the_outstanding_suggestion(self):
        clipRepairAdvice.record_probe("src-1", 999, "a.mp4")
        clipRepairAdvice.decide("src-1")

        assert clipRepairAdvice.findings() == []


class TestTheDeclineSurvivesARestart:
    """The suggestion is raised by the watcher, which restarts far more often than the
    user changes their mind — so "no" is persisted and read back, while "yes" needs no
    record beyond the source row it just wrote.
    """

    def test_a_decline_round_trips_through_the_settings_store(self):
        clipRepairAdvice.decide("src-1")
        stored = clipRepairAdvice.dismissed_json()

        clipRepairAdvice.reset()
        assert clipRepairAdvice.wants_probe("src-1") is True

        clipRepairAdvice.load_dismissed(stored)
        assert clipRepairAdvice.wants_probe("src-1") is False

    def test_a_fresh_install_has_nothing_to_load(self):
        clipRepairAdvice.load_dismissed(None)

        assert clipRepairAdvice.wants_probe("src-1") is True

    @pytest.mark.parametrize("raw", ["not json", '{"not": "a list"}', "[]", ""])
    def test_an_unusable_stored_value_does_not_stop_the_watcher_starting(self, raw):
        """Read during task startup. A settings value that cannot be parsed must cost a
        log line, not every camera on the profile."""
        clipRepairAdvice.load_dismissed(raw)

        assert clipRepairAdvice.wants_probe("src-1") is True


class TestTheNoticeItRenders:
    def test_both_answers_are_offered_and_are_same_origin_api_paths(self):
        """`noticesBar.js` refuses any action url that is not a relative /api/ path, so a
        notice whose urls drift off that shape renders as text with no way to answer it.
        """
        clipRepairAdvice.record_probe("src-1", 99, "a.mp4", source_name="Garage")
        notice = clipRepairAdvice.notice_for(clipRepairAdvice.findings()[0])

        urls = [a["url"] for a in notice["actions"]]
        assert len(urls) == 2
        assert all(u.startswith("/api/") and not u.startswith("//") for u in urls)
        assert all("src-1" in u for u in urls)

    def test_the_notice_names_the_camera_the_user_would_recognise(self):
        """Keyed by uuid, but the message has to name the camera the way the Sources
        pane does — a uuid tells the user nothing about which picture is breaking up."""
        clipRepairAdvice.record_probe("src-1", 99, "a.mp4", source_name="Garage")
        notice = clipRepairAdvice.notice_for(clipRepairAdvice.findings()[0])

        assert "Garage" in notice["message"]
        assert "99" in notice["message"]

    def test_a_source_with_no_name_still_reads_as_a_sentence(self):
        clipRepairAdvice.record_probe("src-1", 99, "a.mp4")
        notice = clipRepairAdvice.notice_for(clipRepairAdvice.findings()[0])

        assert notice["message"].startswith("A camera dropped")

    def test_each_camera_gets_its_own_notice_row(self):
        """The bar keys rows by `key`; a shared one would collapse two cameras into one
        suggestion and lose whichever was found second."""
        clipRepairAdvice.record_probe("src-1", 99, "a.mp4")
        clipRepairAdvice.record_probe("src-2", 99, "b.mp4")

        keys = {clipRepairAdvice.notice_for(f)["key"] for f in clipRepairAdvice.findings()}
        assert len(keys) == 2
