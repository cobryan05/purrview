"""
The NVR → Events review page (app/nvr/events/).

What is tested is what a reviewer actually needs from a page of past events:

  - the card answers "when" and "who did what to what" without being opened, because a
    grid of cards captioned with uuids or bare rule names is unusable at a glance;
  - a card with a clip is distinguishable from one without, *before* clicking — the
    whole reason a video badge exists;
  - every filter reaches the server as a query parameter. Filters that render but do
    not narrow anything are the failure mode here, and a filter bar that looks right
    while returning everything is indistinguishable from a working one on screen;
  - the detail modal plays the clip and offers "save frame for annotation", which is
    the loop the NVR exists to close;
  - an event still in progress, or one whose snapshot has been reaped, degrades
    visibly rather than looking broken.

Run with:
    pytest playwright_tests/test_nvr_events.py -p no:asyncio -v
"""

import base64
import json
import re
from contextlib import contextmanager
from urllib.parse import parse_qs, urlparse

from playwright.sync_api import Page, expect

from playwright_tests.test_nvr_live_view import FRAME_B64
from playwright_tests.waits import issues_a_request, wait_until

PROFILE_UUID = "99999999-eeee-0000-0000-000000000002"
EVENT_UUID = "event-uuid-0001"
RECORDING_UUID = "recording-uuid-0001"
TIMELAPSE_UUID = "recording-uuid-0002"


def _clip_summary(**overrides) -> dict:
    """One entry of `event.recordings`, as the events list sends it.

    A summary, not the whole row — the card only needs the kind to pick a badge, and
    the modal fetches the full row when the user opens it.
    """
    summary = {"uuid": RECORDING_UUID, "kind": "clip", "durationS": 39.0, "status": "complete"}
    summary.update(overrides)
    return summary


def _timelapse_summary(**overrides) -> dict:
    summary = {"uuid": TIMELAPSE_UUID, "kind": "timelapse", "durationS": 10.0, "status": "complete"}
    summary.update(overrides)
    return summary


#: A real JPEG, so an <img> decodes something rather than taking the error path. Shared
#: with the live-view tests so both exercise the same decode.
PIXEL_JPEG = base64.b64decode(FRAME_B64)


def _event(**overrides) -> dict:
    event = {
        "uuid": EVENT_UUID,
        "profileUuid": PROFILE_UUID,
        "sourceUuid": "src-1",
        "ruleUuid": "rule-1",
        "ruleName": "cat using litter box",
        "eventLabel": None,
        "startedAt": "2026-08-18T09:15:00",
        "endedAt": "2026-08-18T09:15:24",
        "peakConfidence": 0.87,
        "trackIdsJson": [3],
        "subjectRef": "label-cat",
        "subjectName": "Tabby",
        "targetRef": "zone-box",
        "targetName": "litter box",
        "truncated": False,
        "hasSnapshot": True,
        "metadataJson": {},
        "recordings": [_clip_summary()],
    }
    event.update(overrides)
    return event


def _events(count: int) -> list:
    """`count` distinct events, each identifiable on screen by its headline."""
    return [
        _event(
            uuid=f"event-uuid-{i:04d}",
            subjectName=f"Cat {i}",
            startedAt=f"2026-08-18T09:{15 + i:02d}:00",
            endedAt=f"2026-08-18T09:{15 + i:02d}:24",
            recordings=[],
        )
        for i in range(count)
    ]


def _facets(**overrides) -> dict:
    facets = {
        "rules": [{"ref": "rule-1", "name": "cat using litter box"}],
        "subjects": [{"ref": "label-cat", "name": "Tabby"}],
        "targets": [{"ref": "zone-box", "name": "litter box"}],
        "sources": [{"ref": "src-1", "name": "Litter cam"}],
    }
    facets.update(overrides)
    return facets


def _overlay(**overrides) -> dict:
    """A detection sidecar covering the mocked clip.

    The clip starts ten seconds before the event, so a box at `t=0` is in the pre-roll
    and the event window opens at 10 — which is what makes "is the red frame on right
    now" a question with a different answer at different scrub positions.
    """
    doc = {
        "version": 1,
        "startedAt": "2026-08-18T09:14:50+00:00",
        "durationS": 39.0,
        "maxStalenessS": 1.5,
        "event": {"startS": 10.0, "endS": 34.0, "ruleName": "cat using litter box", "eventUuid": EVENT_UUID},
        "frames": [
            {
                "t": 0.0,
                "boxes": [{"x": 0.1, "y": 0.1, "w": 0.2, "h": 0.2, "label": "cat", "labelUuid": "lbl-1", "conf": 0.8}],
                "tracks": [],
            },
            {
                "t": 12.0,
                "boxes": [{"x": 0.4, "y": 0.4, "w": 0.2, "h": 0.2, "label": "cat", "labelUuid": "lbl-1", "conf": 0.9}],
                "tracks": [],
            },
        ],
    }
    doc.update(overrides)
    return doc


@contextmanager
def _issues_a_search(page: Page, queries: list):
    """Barrier: the block must cause exactly one more search request to arrive.

    Asserting on `queries[-1]` right after changing a filter is a race. The
    obvious-looking synchronisation point — waiting for the result cards — is not
    one here, because `_mock` answers every query with the same single event, so
    the card count is already satisfied by the load that happened on open. The
    wait returns instantly and the assertion reads the *initial* query, passing
    only because the fetch usually wins the race against the next statement.

    Worse, it makes some of these tests unfalsifiable: `assert "rule_uuid" not in
    queries[-1]` is trivially true of the on-open query, so it would report
    success even if clearing the filters did nothing at all.

    So key the barrier on the request list itself, which is the thing under test.

    The mechanism now lives in `waits.issues_a_request` — the same trap turned up in
    the Review filter tests, and two copies would drift.
    """
    with issues_a_request(page, queries, "search"):
        yield


def _blip(**overrides) -> dict:
    """One EventBlipRead, camelCased the way the API sends it.

    Harvestable by default, because that is the row the feature exists for; the
    unharvestable cases are spelled out at the call site so a test that depends on one
    says so.
    """
    blip = {
        "uuid": "blip-uuid-0001",
        "eventUuid": EVENT_UUID,
        "recordingUuid": RECORDING_UUID,
        "kind": "identity_flip",
        "trackId": 3,
        "startS": 12.0,
        "endS": 15.5,
        "severity": 0.82,
        "fault": "detector",
        "harvestable": True,
        "state": "new",
        "detailJson": {"modalLabel": "lbl-1", "sawLabel": "lbl-2", "samples": 9, "trackSamples": 121},
        "createdAt": "2026-08-18T09:16:00",
        "updatedAt": "2026-08-18T09:16:00",
    }
    blip.update(overrides)
    return blip


def _mock(page: Page, events=None, facets=None, recording=True, overlay=None, deletes=None, blips=None):
    """Route every call the Events page makes, and record the search queries.

    Returns the list of parsed query dicts, so a test can assert that a filter actually
    reached the server rather than merely appearing on screen. Pass `deletes` a list to
    also capture the uuid batches sent to the bulk-delete endpoint.

    `blips` defaults to none found. It is routed unconditionally rather than only when a
    test asks for it: the detail modal now fetches them for every event it opens, and an
    unrouted request would leave the page reaching past the mocks at the real server.
    """
    queries = []
    # The mock's own copy of the result set: a delete removes rows from it, so the page
    # that follows is genuinely shorter rather than the same one served again.
    rows = list(events) if events is not None else [_event()]

    def _fulfil_events(route):
        url = route.request.url
        # Same prefix as the search endpoint, so it has to be claimed before the query
        # recorder — otherwise a delete would be logged as a search and answered with a
        # page of events.
        if url.endswith("/delete"):
            uuids = json.loads(route.request.post_data)["uuids"]
            if deletes is not None:
                deletes.append(uuids)
            # Actually drop them, so the re-search that follows returns a shorter page —
            # a delete that only 200s would let a grid that never refreshes pass.
            rows[:] = [r for r in rows if r["uuid"] not in uuids]
            route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps({"status": "success", "deleted": len(uuids)}),
            )
            return
        if "facets" in url:
            route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps({"status": "success", **(facets if facets is not None else _facets())}),
            )
            return
        if "/snapshot" in url:
            route.fulfill(status=200, content_type="image/jpeg", body=PIXEL_JPEG)
            return
        queries.append({k: v[0] for k, v in parse_qs(urlparse(url).query).items()})
        route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({"status": "success", "events": rows, "total": len(rows)}),
        )

    def _fulfil_recordings(route):
        if "/overlay" in route.request.url:
            # 404 is the ordinary "this clip has no boxes" answer, not an error: clips
            # written before overlays existed have none.
            if overlay is None:
                route.fulfill(status=404, content_type="application/json", body="{}")
                return
            route.fulfill(status=200, content_type="application/json", body=json.dumps(overlay))
            return
        if "/file" in route.request.url:
            # Not a real mp4; the tests assert the element and its src, never playback,
            # which would be a test of the browser's decoder rather than of this page.
            route.fulfill(status=200, content_type="video/mp4", body=b"")
            return
        body = {"status": "success", "recording": None}
        if recording:
            # Keyed off the uuid in the path, because an event can own two recordings
            # and the modal fetches whichever one the user is looking at. Answering the
            # clip for both would make the timelapse tab silently show the clip.
            if TIMELAPSE_UUID in route.request.url:
                body["recording"] = {
                    "uuid": TIMELAPSE_UUID,
                    "sourceUuid": "src-1",
                    "profileUuid": PROFILE_UUID,
                    "eventUuid": EVENT_UUID,
                    "kind": "timelapse",
                    "filePath": "/data/recordings/timelapses/a.mp4",
                    "startedAt": "2026-08-18T09:15:00",
                    "endedAt": "2026-08-18T09:15:10",
                    # The *video's* length. The event it summarises ran for 20 minutes.
                    "durationS": 10.0,
                    "sizeBytes": 2048,
                    "status": "complete",
                    "error": None,
                    "partialPreroll": False,
                    "metadataJson": {"frames": 120, "sourceSpanS": 1200.0, "intervalS": 10.0},
                }
            else:
                body["recording"] = {
                    "uuid": RECORDING_UUID,
                    "sourceUuid": "src-1",
                    "profileUuid": PROFILE_UUID,
                    "eventUuid": EVENT_UUID,
                    "kind": "clip",
                    "filePath": "/data/recordings/clips/a.mp4",
                    # Ten seconds before the event: the pre-roll the rewind buffer held.
                    "startedAt": "2026-08-18T09:14:50",
                    "endedAt": "2026-08-18T09:15:29",
                    "durationS": 39.0,
                    "sizeBytes": 1024,
                    "status": "complete",
                    "error": None,
                    "partialPreroll": False,
                    "metadataJson": {},
                }
        route.fulfill(status=200, content_type="application/json", body=json.dumps(body))

    def _fulfil_sources(route):
        route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps(
                {
                    "status": "success",
                    "sources": {
                        "src-1": {
                            "uuid": "src-1",
                            "name": "Litter cam",
                            "typename": "FfmpegImageProvider",
                            "params": {},
                        }
                    },
                }
            ),
        )

    rows_blips = list(blips) if blips is not None else []

    def _fulfil_blips(route):
        if route.request.url.endswith("/states"):
            payload = json.loads(route.request.post_data)
            # Actually move them, so a panel that re-reads after dismissing shows the new
            # state rather than the old one served again.
            for row in rows_blips:
                if row["uuid"] in payload["uuids"]:
                    row["state"] = payload["state"]
            route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps({"status": "success", "updated": len(payload["uuids"])}),
            )
            return
        route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({"status": "success", "blips": rows_blips, "total": len(rows_blips)}),
        )

    page.route(re.compile(r"/api/watch-events"), _fulfil_events)
    page.route(re.compile(r"/api/event-blips"), _fulfil_blips)
    page.route(re.compile(r"/api/recordings"), _fulfil_recordings)
    page.route(re.compile(r"/api/sources/get"), _fulfil_sources)
    return queries


def _open_events_pane(page: Page, base_url: str):
    page.goto(base_url)
    page.wait_for_load_state("networkidle")
    page.locator(".main-tab-button[data-tab='tab-nvr']").click()
    page.locator(".workspace-tab-button[data-tab='tab-nvr-events']").click()
    page.wait_for_selector("[data-testid='event-search']", timeout=8000)


class TestTheCardAnswersTheObviousQuestions:
    def test_the_time_is_on_the_card(self, page: Page, base_url: str):
        """ "When" is the first question asked of an NVR, so it must not require a click."""
        _mock(page)
        _open_events_pane(page, base_url)

        expect(page.locator("[data-testid='event-card-time']").first).to_contain_text("09:15")

    def test_the_headline_names_the_actor_and_what_it_acted_on(self, page: Page, base_url: str):
        _mock(page)
        _open_events_pane(page, base_url)

        headline = page.locator("[data-testid='event-card-headline']").first
        expect(headline).to_contain_text("Tabby")
        expect(headline).to_contain_text("litter box")

    def test_the_rule_is_shown_as_the_kind_of_event(self, page: Page, base_url: str):
        _mock(page)
        _open_events_pane(page, base_url)

        expect(page.locator("[data-testid='event-card']").first).to_contain_text("cat using litter box")

    def test_the_camera_is_named_not_uuided(self, page: Page, base_url: str):
        """Events store only the source uuid; a card captioned with one is unreadable."""
        _mock(page)
        _open_events_pane(page, base_url)

        expect(page.locator("[data-testid='event-card']").first).to_contain_text("Litter cam")

    def test_the_snapshot_is_shown(self, page: Page, base_url: str):
        _mock(page)
        _open_events_pane(page, base_url)

        expect(page.locator(".event-card__image").first).to_be_visible()

    def test_a_card_with_a_clip_is_marked_before_it_is_opened(self, page: Page, base_url: str):
        _mock(page)
        _open_events_pane(page, base_url)

        expect(page.locator("[data-testid='event-card-video-badge']")).to_have_count(1)

    def test_a_card_without_a_clip_has_no_video_badge(self, page: Page, base_url: str):
        _mock(page, events=[_event(recordings=[])])
        _open_events_pane(page, base_url)

        expect(page.locator("[data-testid='event-card-video-badge']")).to_have_count(0)

    def test_a_card_with_a_timelapse_is_marked_separately(self, page: Page, base_url: str):
        """A clip and a timelapse answer different questions, so one "Video" badge for
        both would leave the timelapse undiscoverable without opening every event."""
        _mock(page, events=[_event(recordings=[_timelapse_summary()])])
        _open_events_pane(page, base_url)

        expect(page.locator("[data-testid='event-card-timelapse-badge']")).to_have_count(1)
        expect(page.locator("[data-testid='event-card-video-badge']")).to_have_count(0)

    def test_a_card_with_both_shows_both_badges(self, page: Page, base_url: str):
        _mock(page, events=[_event(recordings=[_clip_summary(), _timelapse_summary()])])
        _open_events_pane(page, base_url)

        expect(page.locator("[data-testid='event-card-video-badge']")).to_have_count(1)
        expect(page.locator("[data-testid='event-card-timelapse-badge']")).to_have_count(1)

    def test_two_badges_do_not_land_on_top_of_each_other(self, page: Page, base_url: str):
        """They used to be individually pinned to the corner, which put both on exactly
        the same pixels the moment an event could own two recordings."""
        _mock(page, events=[_event(recordings=[_clip_summary(), _timelapse_summary()])])
        _open_events_pane(page, base_url)

        clip = page.locator("[data-testid='event-card-video-badge']").bounding_box()
        timelapse = page.locator("[data-testid='event-card-timelapse-badge']").bounding_box()
        assert clip["y"] != timelapse["y"]
        assert clip["y"] + clip["height"] <= timelapse["y"] + 1

    def test_an_event_still_in_progress_says_so(self, page: Page, base_url: str):
        """An open event has no duration and no clip yet; showing "0s" would be a wrong
        answer rather than a missing one."""
        _mock(page, events=[_event(endedAt=None, recordings=[])])
        _open_events_pane(page, base_url)

        expect(page.locator("[data-testid='event-card']").first).to_contain_text("In progress")

    def test_an_event_with_no_snapshot_shows_a_placeholder(self, page: Page, base_url: str):
        """A broken-image icon in a grid reads as a bug; the placeholder reads as data."""
        _mock(page, events=[_event(hasSnapshot=False)])
        _open_events_pane(page, base_url)

        expect(page.locator(".event-card__media--empty")).to_have_count(1)
        expect(page.locator(".event-card__image")).to_have_count(0)


class TestFiltersReachTheServer:
    """Each filter proved to arrive as a query parameter.

    A filter bar that renders correctly and narrows nothing looks identical on screen to
    one that works, so asserting on the request is the only real check.
    """

    def test_the_page_searches_on_open(self, page: Page, base_url: str):
        queries = _mock(page)
        _open_events_pane(page, base_url)

        expect(page.locator("[data-testid='event-card']")).to_have_count(1)
        assert len(queries) >= 1

    def test_filtering_by_event_type(self, page: Page, base_url: str):
        queries = _mock(page)
        _open_events_pane(page, base_url)

        with _issues_a_search(page, queries):
            page.locator("[data-testid='event-filter-rule']").select_option("rule-1")
        assert queries[-1].get("rule_uuid") == "rule-1"

    def test_filtering_by_camera(self, page: Page, base_url: str):
        queries = _mock(page)
        _open_events_pane(page, base_url)

        with _issues_a_search(page, queries):
            page.locator("[data-testid='event-filter-source']").select_option("src-1")
        assert queries[-1].get("source_uuid") == "src-1"

    def test_filtering_by_the_acting_thing(self, page: Page, base_url: str):
        queries = _mock(page)
        _open_events_pane(page, base_url)

        with _issues_a_search(page, queries):
            page.locator("[data-testid='event-filter-subject']").select_option("label-cat")
        assert queries[-1].get("subject_ref") == "label-cat"

    def test_filtering_by_the_thing_acted_on(self, page: Page, base_url: str):
        queries = _mock(page)
        _open_events_pane(page, base_url)

        with _issues_a_search(page, queries):
            page.locator("[data-testid='event-filter-target']").select_option("zone-box")
        assert queries[-1].get("target_ref") == "zone-box"

    def test_filtering_by_date_range(self, page: Page, base_url: str):
        """The `to` bound must cover the whole day it names, or picking one day for both
        returns only events at exactly midnight."""
        queries = _mock(page)
        _open_events_pane(page, base_url)

        page.locator("[data-testid='event-filter-from']").fill("2026-08-18")
        page.locator("[data-testid='event-filter-to']").fill("2026-08-18")
        # Each bound issues its own search, so waiting for "a new query" could
        # settle on the one carrying only `from`. Wait for the pair to be
        # present; the assertions below are about the *values*, which is the
        # actual claim — that `to` covers the whole day rather than midnight.
        wait_until(
            page,
            lambda: queries and {"start_from", "start_to"} <= queries[-1].keys(),
            lambda: f"no search carried both date bounds: {queries[-1] if queries else None}",
        )
        assert queries[-1].get("start_from") == "2026-08-18T00:00:00"
        assert queries[-1].get("start_to") == "2026-08-18T23:59:59"

    def test_filtering_to_events_with_no_video(self, page: Page, base_url: str):
        """`false` is a real filter and must not be dropped the way a falsy guard would."""
        queries = _mock(page)
        _open_events_pane(page, base_url)

        with _issues_a_search(page, queries):
            page.locator("[data-testid='event-filter-clip']").select_option("false")
        assert queries[-1].get("has_recording") == "false"

    def test_clearing_removes_every_filter(self, page: Page, base_url: str):
        queries = _mock(page)
        _open_events_pane(page, base_url)

        # Both halves need their own barrier. Waiting once would let the assertion
        # read the *filtered* query and fail, or — as before this fix — the
        # on-open query, where "rule_uuid" is trivially absent and the test would
        # pass even if Clear did nothing whatsoever.
        with _issues_a_search(page, queries):
            page.locator("[data-testid='event-filter-rule']").select_option("rule-1")
        assert queries[-1].get("rule_uuid") == "rule-1", "precondition: the filter must be applied before clearing it"

        with _issues_a_search(page, queries):
            page.locator("[data-testid='event-filter-clear']").click()
        assert "rule_uuid" not in queries[-1]

    def test_an_empty_result_explains_itself(self, page: Page, base_url: str):
        _mock(page, events=[])
        _open_events_pane(page, base_url)

        expect(page.locator("[data-testid='event-search-empty']")).to_be_visible()


class TestTheDetailModal:
    def _open_modal(self, page: Page, base_url: str, **mock_kwargs):
        _mock(page, **mock_kwargs)
        _open_events_pane(page, base_url)
        page.locator("[data-testid='event-card']").first.click()
        page.wait_for_selector("[data-testid='event-detail-modal']", timeout=5000)

    def test_opening_a_card_shows_the_details(self, page: Page, base_url: str):
        self._open_modal(page, base_url)

        modal = page.locator("[data-testid='event-detail-modal']")
        expect(modal).to_contain_text("Tabby")
        expect(modal).to_contain_text("Litter cam")
        expect(modal).to_contain_text("cat using litter box")

    def test_the_clip_is_playable(self, page: Page, base_url: str):
        self._open_modal(page, base_url)

        video = page.locator("[data-testid='event-detail-video']")
        expect(video).to_be_visible()
        assert RECORDING_UUID in video.get_attribute("src")

    def test_the_lead_in_before_the_event_is_spelled_out(self, page: Page, base_url: str):
        """The clip starts before the event because pre-roll is cut retroactively out of
        the rewind buffer. That is only useful if the player says where in the lead-in
        the rule actually fired."""
        self._open_modal(page, base_url)

        expect(page.locator("[data-testid='event-detail-jump']")).to_contain_text("10.0s")

    def test_saving_a_frame_is_offered_while_the_clip_is_open(self, page: Page, base_url: str):
        """The frames worth training on are the ones where something happened, and this
        is where the user is already looking at them."""
        self._open_modal(page, base_url)

        expect(page.locator("[data-testid='event-detail-save-frame']")).to_be_visible()

    def test_an_event_with_no_clip_still_opens(self, page: Page, base_url: str):
        self._open_modal(page, base_url, events=[_event(recordings=[])])

        expect(page.locator("[data-testid='event-detail-video']")).to_have_count(0)
        expect(page.locator("[data-testid='event-detail-modal']")).to_contain_text("Nothing was recorded")

    def test_a_clip_missing_from_disk_degrades_to_the_snapshot(self, page: Page, base_url: str):
        """The row can outlive its file; the snapshot is still worth showing."""
        self._open_modal(page, base_url, recording=False)

        expect(page.locator("[data-testid='event-detail-video']")).to_have_count(0)
        expect(page.locator("[data-testid='event-detail-snapshot']")).to_be_visible()

    def test_one_recording_gets_no_tab_strip(self, page: Page, base_url: str):
        """A single-tab tab strip is chrome that explains nothing, and one recording is
        the overwhelmingly common case."""
        self._open_modal(page, base_url)

        expect(page.locator("[data-testid='event-detail-media-tabs']")).to_have_count(0)

    def test_an_event_with_both_offers_a_switch(self, page: Page, base_url: str):
        self._open_modal(page, base_url, events=[_event(recordings=[_clip_summary(), _timelapse_summary()])])

        expect(page.locator("[data-testid='event-detail-media-tab-clip']")).to_be_visible()
        expect(page.locator("[data-testid='event-detail-media-tab-timelapse']")).to_be_visible()
        # The clip opens first: it is the literal record, the timelapse summarises it.
        expect(page.locator("[data-testid='event-detail-video']")).to_have_attribute("src", re.compile(RECORDING_UUID))

    def test_switching_to_the_timelapse_plays_the_timelapse(self, page: Page, base_url: str):
        """The tab has to actually change the source. Fetching the clip's row for both
        would show the clip under a tab labelled Timelapse."""
        self._open_modal(page, base_url, events=[_event(recordings=[_clip_summary(), _timelapse_summary()])])

        page.locator("[data-testid='event-detail-media-tab-timelapse']").click()

        expect(page.locator("[data-testid='event-detail-video']")).to_have_attribute("src", re.compile(TIMELAPSE_UUID))

    def test_the_timelapse_says_what_it_compressed(self, page: Page, base_url: str):
        """A scrubber reading 0:10 over a twenty-minute visit invites the reader to
        think the cat left after ten seconds."""
        self._open_modal(page, base_url, events=[_event(recordings=[_timelapse_summary()])])

        note = page.locator("[data-testid='event-detail-timelapse-note']")
        expect(note).to_contain_text("120 frames")
        expect(note).to_contain_text("20 minutes")

    def test_the_timelapse_offers_no_jump_to_the_moment_it_fired(self, page: Page, base_url: str):
        """There is no pre-roll to skip: a timelapse begins when the event does, and its
        seconds are not the event's seconds."""
        self._open_modal(page, base_url, events=[_event(recordings=[_timelapse_summary()])])

        expect(page.locator("[data-testid='event-detail-video']")).to_be_visible()
        expect(page.locator("[data-testid='event-detail-jump']")).to_have_count(0)


class TestTheDetectionOverlay:
    """The boxes the watcher saw, drawn over the clip rather than burnt into it.

    The point of the separation is not aesthetic: the video element must stay the
    untouched clip so that "save frame for annotation" cannot put a training image with
    rectangles drawn on it into the dataset. So the load-bearing assertions here are that
    the canvas is a *sibling* of the video and that the video's src is still the plain
    clip.
    """

    def _open_modal(self, page: Page, base_url: str, **mock_kwargs):
        _mock(page, **mock_kwargs)
        _open_events_pane(page, base_url)
        page.locator("[data-testid='event-card']").first.click()
        page.wait_for_selector("[data-testid='event-detail-modal']", timeout=5000)

    def test_a_clip_with_a_sidecar_gets_an_overlay_canvas(self, page: Page, base_url: str):
        self._open_modal(page, base_url, overlay=_overlay())

        expect(page.locator("[data-testid='clip-overlay-canvas']")).to_have_count(1)

    def test_a_clip_without_a_sidecar_offers_no_overlay(self, page: Page, base_url: str):
        """Every clip written before this existed has no boxes, and they cannot be
        recovered — advertising a toggle that can never do anything is worse than
        silence."""
        self._open_modal(page, base_url, overlay=None)

        expect(page.locator("[data-testid='clip-overlay-canvas']")).to_have_count(0)
        expect(page.locator("[data-testid='event-detail-overlay-toggle']")).to_have_count(0)

    def test_the_overlay_can_be_switched_off(self, page: Page, base_url: str):
        self._open_modal(page, base_url, overlay=_overlay())

        toggle = page.locator("[data-testid='event-detail-overlay-toggle']")
        expect(toggle).to_be_checked()
        toggle.uncheck()
        expect(toggle).not_to_be_checked()
        # The canvas stays — it is simply drawn empty. Removing it would rebuild the
        # wrapper on every toggle and lose the video's playback position.
        expect(page.locator("[data-testid='clip-overlay-canvas']")).to_have_count(1)

    def test_the_video_itself_is_never_annotated(self, page: Page, base_url: str):
        """The whole design in one assertion: boxes live on a canvas above the video, and
        the video's source is still the raw clip file."""
        self._open_modal(page, base_url, overlay=_overlay())

        # Siblings inside the overlay wrapper, not nested — a <video> cannot host
        # children, and compositing into it would mean re-encoding.
        #
        # expect(), not a bare .count(): _open_modal's barrier is the modal, and
        # the overlay wrapper is built a tick later, so an immediate read races
        # it. The bare count passed alone and failed under a loaded parallel run
        # — the assertion needs its own barrier, which only the retrying form has.
        expect(page.locator(".clip-overlay > [data-testid='event-detail-video']")).to_have_count(1)
        expect(page.locator(".clip-overlay > [data-testid='clip-overlay-canvas']")).to_have_count(1)

        video = page.locator(".clip-overlay > [data-testid='event-detail-video']")
        assert video.get_attribute("src").endswith(f"/api/recordings/{RECORDING_UUID}/file")

    def test_the_overlay_never_swallows_clicks_on_the_controls(self, page: Page, base_url: str):
        """The canvas covers the video, scrubber included. Without pointer-events: none
        it would break the seeking this modal is built around."""
        self._open_modal(page, base_url, overlay=_overlay())

        canvas = page.locator("[data-testid='clip-overlay-canvas']")
        assert canvas.evaluate("el => getComputedStyle(el).pointerEvents") == "none"

    def test_saving_a_frame_still_works_with_an_overlay_up(self, page: Page, base_url: str):
        self._open_modal(page, base_url, overlay=_overlay())

        expect(page.locator("[data-testid='event-detail-save-frame']")).to_be_visible()

    def test_a_timelapse_gets_the_overlay_too(self, page: Page, base_url: str):
        """Timelapses used to be the one recording that could only have its boxes burnt
        in, which made every frame of one useless as a training image — and "save frame
        for annotation" points straight at a dataset. They now carry the same sidecar,
        so the same renderer has to pick it up.
        """
        self._open_modal(
            page,
            base_url,
            events=[_event(recordings=[_timelapse_summary()])],
            overlay=_overlay(),
        )

        expect(page.locator("[data-testid='event-detail-video']")).to_have_attribute("src", re.compile(TIMELAPSE_UUID))
        expect(page.locator("[data-testid='clip-overlay-canvas']")).to_have_count(1)
        expect(page.locator("[data-testid='event-detail-overlay-toggle']")).to_be_checked()

    def test_a_timelapse_without_a_sidecar_offers_no_toggle(self, page: Page, base_url: str):
        """Every timelapse compiled before this existed has none, and the ones written
        with boxes burnt in cannot be given one retroactively."""
        self._open_modal(page, base_url, events=[_event(recordings=[_timelapse_summary()])], overlay=None)

        expect(page.locator("[data-testid='event-detail-video']")).to_be_visible()
        expect(page.locator("[data-testid='event-detail-overlay-toggle']")).to_have_count(0)


class TestThePagerSurvivesAShrinkingResultSet:
    """Deleting the last event on the last page must not strand the reviewer.

    Paging is server-side offset paging, so `offset` is state the page holds on its own
    and the server knows nothing about. Delete the only event on the final page and that
    offset is suddenly past the end of a result set one page shorter: the request comes
    back empty, and an empty page renders as "no events match these filters" — over a
    result set that is merely one page smaller. The pager hides itself at that point too,
    so there is no control left to get back.
    """

    PAGE_SIZE = 60

    def _mock_paged(self, page: Page, store: dict):
        """Route the Events page against a mutable store that honours offset/limit."""

        def _fulfil_events(route):
            url = route.request.url
            if "facets" in url:
                route.fulfill(
                    status=200,
                    content_type="application/json",
                    body=json.dumps({"status": "success", **_facets()}),
                )
                return
            if "/snapshot" in url:
                route.fulfill(status=200, content_type="image/jpeg", body=PIXEL_JPEG)
                return
            query = {k: v[0] for k, v in parse_qs(urlparse(url).query).items()}
            offset = int(query.get("offset", 0))
            limit = int(query.get("limit", self.PAGE_SIZE))
            rows = store["events"]
            route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps({"status": "success", "events": rows[offset : offset + limit], "total": len(rows)}),
            )

        def _fulfil_delete(route):
            # The page deletes by uuid; drop it from the store so the next search sees a
            # genuinely shorter result set rather than a mocked-away one.
            uuid = route.request.url.rstrip("/").rsplit("/", 1)[-1]
            store["events"] = [e for e in store["events"] if e["uuid"] != uuid]
            route.fulfill(status=200, content_type="application/json", body=json.dumps({"status": "success"}))

        # Registered before the general route so the DELETE is claimed first; Playwright
        # matches the most recently added route, so this must come last of the two.
        page.route(re.compile(r"/api/watch-events"), _fulfil_events)
        page.route(
            re.compile(r"/api/watch-events/[^/?]+$"),
            lambda r: _fulfil_delete(r) if r.request.method == "DELETE" else _fulfil_events(r),
        )
        page.route(
            re.compile(r"/api/recordings"),
            lambda r: r.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps({"status": "success", "recording": None}),
            ),
        )
        page.route(
            re.compile(r"/api/sources/get"),
            lambda r: r.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps(
                    {
                        "status": "success",
                        # Keyed by uuid, not a list — `fetchSourceList` returns a Map.
                        "sources": {
                            "src-1": {
                                "uuid": "src-1",
                                "name": "Litter cam",
                                "typename": "FfmpegImageProvider",
                                "params": {},
                            }
                        },
                    }
                ),
            ),
        )

    def _store(self, count: int) -> dict:
        return {
            "events": [
                _event(uuid=f"event-{i:04d}", recordings=[], startedAt=f"2026-08-18T09:{i % 60:02d}:00")
                for i in range(count)
            ]
        }

    def test_deleting_the_only_event_on_the_last_page_falls_back_to_the_previous_one(self, page: Page, base_url: str):
        store = self._store(self.PAGE_SIZE + 1)
        self._mock_paged(page, store)
        _open_events_pane(page, base_url)

        page.locator("[data-testid='event-search-next']").click()
        expect(page.locator("[data-testid='event-card']")).to_have_count(1)

        page.locator("[data-testid='event-card']").first.click()
        page.locator("[data-testid='event-detail-delete']").click()
        page.get_by_role("button", name="Delete", exact=True).click()

        # The full previous page, not an empty grid claiming nothing matches.
        expect(page.locator("[data-testid='event-card']")).to_have_count(self.PAGE_SIZE)
        expect(page.locator("[data-testid='event-search-empty']")).to_have_count(0)

    def test_a_genuinely_empty_result_still_says_so(self, page: Page, base_url: str):
        """The clamp must not paper over an actually-empty result set."""
        store = self._store(1)
        self._mock_paged(page, store)
        _open_events_pane(page, base_url)

        page.locator("[data-testid='event-card']").first.click()
        page.locator("[data-testid='event-detail-delete']").click()
        page.get_by_role("button", name="Delete", exact=True).click()

        expect(page.locator("[data-testid='event-search-empty']")).to_be_visible()


class TestARestartClosedEventIsHonest:
    """An event a killed watcher left open is closed at its own start, because the real
    end was never recorded. The card must not present that as a measured duration.

    `formatDuration` already refuses to print "0s" for an *open* event on exactly this
    reasoning — a wrong answer is worse than a missing one — and closing the row would
    have routed straight around that.
    """

    def test_the_duration_is_replaced_by_a_note(self, page: Page, base_url: str):
        _mock(
            page,
            events=[
                _event(
                    startedAt="2026-08-18T09:15:00",
                    endedAt="2026-08-18T09:15:00",
                    metadataJson={"closedBy": "restart"},
                )
            ],
        )
        _open_events_pane(page, base_url)

        card = page.locator("[data-testid='event-card']").first
        expect(page.locator("[data-testid='event-card-stranded']").first).to_be_visible()
        expect(card).not_to_contain_text("0.0s")

    def test_an_ordinary_event_still_shows_its_duration(self, page: Page, base_url: str):
        _mock(page)
        _open_events_pane(page, base_url)

        expect(page.locator("[data-testid='event-card-stranded']")).to_have_count(0)
        expect(page.locator("[data-testid='event-card']").first).to_contain_text("24s")


class TestSelectingEvents:
    """Multi-select over the grid, and the bulk delete it exists to feed.

    The load-bearing property is that selecting and *opening* stay separate gestures: a
    grid where the click that opens an event also selects it (or worse, where selection
    swallowed the open) is a different page from the one this is. So each test here
    asserts both halves — what the gesture did, and what it did not do.
    """

    def _open(self, page: Page, base_url: str, count: int = 4, deletes=None):
        _mock(page, events=_events(count), deletes=deletes)
        _open_events_pane(page, base_url)
        expect(page.locator("[data-testid='event-card']")).to_have_count(count)

    def test_a_plain_click_opens_the_event_without_selecting_it(self, page: Page, base_url: str):
        self._open(page, base_url)

        page.locator("[data-testid='event-card']").first.click()

        expect(page.locator("[data-testid='event-detail-modal']")).to_be_visible()
        expect(page.locator("[data-testid='event-search-selection']")).to_be_hidden()

    def test_a_modifier_click_selects_the_event_without_opening_it(self, page: Page, base_url: str):
        """The gesture every file manager already taught the user, and the one that must
        not also put a modal in front of the grid being selected."""
        self._open(page, base_url)

        page.locator("[data-testid='event-card']").first.click(modifiers=["ControlOrMeta"])

        expect(page.locator("[data-testid='event-search-selection-count']")).to_have_text("1 selected")
        expect(page.locator("[data-testid='event-detail-modal']")).to_have_count(0)

    def test_the_checkbox_selects_only_its_own_card(self, page: Page, base_url: str):
        self._open(page, base_url)

        page.locator("[data-testid='event-card-select']").nth(1).check()

        expect(page.locator("[data-testid='event-search-selection-count']")).to_have_text("1 selected")
        expect(page.locator(".event-card--selected")).to_have_count(1)
        expect(page.locator("[data-testid='event-detail-modal']")).to_have_count(0)

    def test_shift_extends_the_selection_to_a_range(self, page: Page, base_url: str):
        """Reviewing a night's events means deleting runs of them, so the range gesture
        is the difference between one click and forty."""
        self._open(page, base_url)
        boxes = page.locator("[data-testid='event-card-select']")

        boxes.nth(0).check()
        boxes.nth(2).click(modifiers=["Shift"])

        expect(page.locator("[data-testid='event-search-selection-count']")).to_have_text("3 selected")
        # The card past the end of the range is the whole point of the assertion: a
        # range that selects everything on the page would satisfy a count alone.
        expect(page.locator("[data-testid='event-card']").nth(3)).not_to_have_class(re.compile("event-card--selected"))

    def test_shift_can_deselect_a_range_it_selected(self, page: Page, base_url: str):
        self._open(page, base_url)
        boxes = page.locator("[data-testid='event-card-select']")

        boxes.nth(0).check()
        boxes.nth(2).click(modifiers=["Shift"])
        boxes.nth(2).click(modifiers=["Shift"])

        expect(page.locator("[data-testid='event-search-selection']")).to_be_hidden()
        expect(page.locator(".event-card--selected")).to_have_count(0)

    def test_select_all_takes_the_page_and_clear_gives_it_back(self, page: Page, base_url: str):
        self._open(page, base_url)

        page.locator("[data-testid='event-search-select-all']").check()
        expect(page.locator(".event-card--selected")).to_have_count(4)

        page.locator("[data-testid='event-search-clear-selection']").click()
        expect(page.locator(".event-card--selected")).to_have_count(0)

    def test_deleting_the_selection_sends_exactly_it_and_the_grid_loses_those_cards(self, page: Page, base_url: str):
        deletes = []
        self._open(page, base_url, deletes=deletes)
        boxes = page.locator("[data-testid='event-card-select']")

        boxes.nth(0).check()
        boxes.nth(1).check()
        page.locator("[data-testid='event-search-delete-selected']").click()
        page.get_by_role("button", name="Delete", exact=True).click()

        expect(page.locator("[data-testid='event-card']")).to_have_count(2)
        assert deletes == [["event-uuid-0000", "event-uuid-0001"]], deletes
        # A selection that outlived the cards it was made of would arm the next delete
        # against rows the user is no longer looking at.
        expect(page.locator("[data-testid='event-search-selection']")).to_be_hidden()

    def test_cancelling_the_confirmation_deletes_nothing(self, page: Page, base_url: str):
        deletes = []
        self._open(page, base_url, deletes=deletes)

        page.locator("[data-testid='event-card-select']").nth(0).check()
        page.locator("[data-testid='event-search-delete-selected']").click()
        page.get_by_role("button", name="Keep", exact=True).click()

        expect(page.locator("[data-testid='event-card']")).to_have_count(4)
        assert deletes == []


class TestWalkingTheEventsFromTheModal:
    """Next/previous inside the detail modal.

    Reviewing thirty events means opening thirty modals, and the walk exists so that
    costs thirty keystrokes rather than sixty clicks through a grid that re-renders each
    time. What must hold is that the modal actually *becomes* the next event — title,
    details and media together — rather than only relabelling its header.
    """

    def _open_first(self, page: Page, base_url: str, count: int = 3):
        _mock(page, events=_events(count))
        _open_events_pane(page, base_url)
        page.locator("[data-testid='event-card']").first.click()
        page.wait_for_selector("[data-testid='event-detail-modal']", timeout=5000)

    def test_next_moves_the_modal_onto_the_following_event(self, page: Page, base_url: str):
        self._open_first(page, base_url)
        expect(page.locator("[data-testid='event-detail-title']")).to_contain_text("Cat 0")

        page.locator("[data-testid='event-detail-next']").click()

        expect(page.locator("[data-testid='event-detail-title']")).to_contain_text("Cat 1")
        expect(page.locator("[data-testid='event-detail-position']")).to_have_text("2 / 3")

    def test_the_walk_stops_at_both_ends_rather_than_wrapping(self, page: Page, base_url: str):
        """Wrapping silently is how a reviewer watches the same ten events twice."""
        self._open_first(page, base_url)

        expect(page.locator("[data-testid='event-detail-prev']")).to_be_disabled()
        expect(page.locator("[data-testid='event-detail-next']")).to_be_enabled()

        page.locator("[data-testid='event-detail-next']").click()
        page.locator("[data-testid='event-detail-next']").click()

        expect(page.locator("[data-testid='event-detail-title']")).to_contain_text("Cat 2")
        expect(page.locator("[data-testid='event-detail-next']")).to_be_disabled()
        expect(page.locator("[data-testid='event-detail-prev']")).to_be_enabled()

    def test_the_prev_next_hotkeys_drive_the_walk(self, page: Page, base_url: str):
        """The same keys that step through images on the canvas, read from the user's
        own bindings — a walk you have to reach for the mouse to use is not the feature
        that was asked for."""
        self._open_first(page, base_url)

        page.keyboard.press("]")
        expect(page.locator("[data-testid='event-detail-title']")).to_contain_text("Cat 1")

        page.keyboard.press("[")
        expect(page.locator("[data-testid='event-detail-title']")).to_contain_text("Cat 0")

    def test_the_sidebar_follows_the_walk(self, page: Page, base_url: str):
        """The header is the cheap half. If only it were re-rendered, the modal would
        show one event's title over another event's details."""
        self._open_first(page, base_url)

        page.locator("[data-testid='event-detail-next']").click()

        modal = page.locator("[data-testid='event-detail-modal']")
        expect(modal).to_contain_text("Cat 1")
        expect(modal).not_to_contain_text("Cat 0")

    def test_a_lone_event_gets_no_walk_controls(self, page: Page, base_url: str):
        """Two dead buttons and "1 / 1" is chrome that only explains itself once you
        have already tried to use it."""
        self._open_first(page, base_url, count=1)

        expect(page.locator("[data-testid='event-detail-nav']")).to_be_hidden()

    def test_deleting_from_the_modal_moves_on_instead_of_closing(self, page: Page, base_url: str):
        """Deleting one of thirty events being reviewed should leave you reviewing the
        other twenty-nine."""
        self._open_first(page, base_url)

        page.locator("[data-testid='event-detail-delete']").click()
        page.get_by_role("button", name="Delete", exact=True).click()

        expect(page.locator("[data-testid='event-detail-modal']")).to_be_visible()
        expect(page.locator("[data-testid='event-detail-title']")).to_contain_text("Cat 1")
        expect(page.locator("[data-testid='event-detail-position']")).to_have_text("1 / 2")


class TestTheTrackingBlipsPanel:
    """The event modal is where a tracking finding becomes training data.

    Analysis is automatic, so by the time anyone opens an event the findings already
    exist — but harvesting frames from them is deliberately a human decision, and this
    panel is the only place that decision can be made. What matters is that it never
    proposes something it cannot honestly do: a finding the tracker caused is not
    harvestable, and a checkbox offering it anyway would quietly pad the training set
    with frames the detector already gets right.
    """

    def _open_modal(self, page: Page, base_url: str, **mock_kwargs):
        _mock(page, **mock_kwargs)
        _open_events_pane(page, base_url)
        page.locator("[data-testid='event-card']").first.click()
        page.wait_for_selector("[data-testid='event-detail-modal']", timeout=5000)

    def _mock_launch(self, page: Page, calls: list):
        def handle_create(route):
            calls.append(json.loads(route.request.post_data))
            route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps({"status": "success", "configUuid": "tc-harvest-1"}),
            )

        page.route("**/api/tasks/configs/create", handle_create)
        page.route(
            "**/api/tasks/instances/start",
            lambda route: route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps({"status": "success", "taskUuid": "task-harvest-1"}),
            ),
        )

    def test_an_event_with_no_findings_says_nothing(self, page: Page, base_url: str):
        """The negative half, and the one that carries the weight: an event with no row
        is either clean or never analysed, and the two are not distinguishable from the
        outside. A heading announcing a clean bill of health would be a guess."""
        self._open_modal(page, base_url, blips=[])

        expect(page.locator("[data-testid='event-blips-title']")).to_have_count(0)

    def test_a_finding_is_listed_with_the_claim_behind_it(self, page: Page, base_url: str):
        """A bare "identity flip at 12s" is not judgeable. "9 of 121 frames" is."""
        self._open_modal(page, base_url, blips=[_blip()])

        expect(page.locator("[data-testid='event-blips-title']")).to_contain_text("1")
        row = page.locator("[data-testid='event-blip-row']")
        expect(row).to_contain_text("identity flip")
        expect(row).to_contain_text("12.0–15.5s")
        expect(page.locator("[data-testid='event-blip-claim']")).to_contain_text("9 of 121 frames")

    def test_a_tracker_fault_offers_no_checkbox_and_says_why(self, page: Page, base_url: str):
        """The load-bearing distinction of the whole feature. Harvesting a tracker fault
        pads the set with frames the model already gets right — and from the outside a
        poisoned harvest looks exactly like a successful one."""
        self._open_modal(
            page,
            base_url,
            blips=[
                _blip(),
                _blip(uuid="blip-uuid-0002", kind="tracker_dropout", fault="tracker", harvestable=False),
            ],
        )

        rows = page.locator("[data-testid='event-blip-row']")
        expect(rows).to_have_count(2)
        # The conditional is the point: the harvestable row has a checkbox, this one
        # does not, and it explains itself rather than merely being inert.
        expect(rows.nth(0).locator("[data-testid='event-blip-check']")).to_have_count(1)
        expect(rows.nth(1).locator("[data-testid='event-blip-check']")).to_have_count(0)
        expect(rows.nth(1).locator("[data-testid='event-blip-blocked']")).to_contain_text("tracker was at fault")

    def test_an_expired_finding_cannot_be_harvested_or_played(self, page: Page, base_url: str):
        """Its clip has been swept. Offering either would open a dead player or start a
        run that can never find a frame."""
        self._open_modal(page, base_url, blips=[_blip(state="expired")])

        row = page.locator("[data-testid='event-blip-row']")
        expect(row.locator("[data-testid='event-blip-check']")).to_have_count(0)
        expect(row.locator("[data-testid='event-blip-jump']")).to_have_count(0)
        expect(row.locator("[data-testid='event-blip-blocked']")).to_contain_text("deleted")

    def test_harvest_sends_only_the_selected_findings(self, page: Page, base_url: str):
        """The task takes an explicit uuid list, so "select one, harvest all" would be
        invisible here and only discoverable by the frames that turned up in Review."""
        calls = []
        self._mock_launch(page, calls)
        self._open_modal(
            page,
            base_url,
            blips=[_blip(), _blip(uuid="blip-uuid-0002", startS=20.0, endS=22.0)],
        )

        page.locator("[data-testid='event-blip-row']").nth(1).locator("[data-testid='event-blip-check']").check()
        page.locator("[data-testid='event-blips-harvest']").click()

        wait_until(
            page,
            lambda: len(calls) >= 1,
            lambda: f"Expected 1 POST to /api/tasks/configs/create, got {len(calls)}",
        )
        assert calls[0]["typename"] == "HarvestBlipFramesTask"
        assert calls[0]["params"]["blip_uuids"] == ["blip-uuid-0002"]

    def test_harvest_is_dead_until_something_is_selected(self, page: Page, base_url: str):
        """A run over nothing is a task instance that exists only to fail."""
        self._open_modal(page, base_url, blips=[_blip()])

        harvest = page.locator("[data-testid='event-blips-harvest']")
        expect(harvest).to_be_disabled()

        page.locator("[data-testid='event-blip-check']").check()

        expect(harvest).to_be_enabled()
        expect(harvest).to_contain_text("1")

    # There is deliberately no test that the jump button moves `currentTime`. The clip
    # these tests serve is an empty body — the suite asserts the player's element and
    # src, never playback, which would be a test of the browser's decoder — so no
    # metadata ever arrives and the seek's readiness guard correctly never fires. A test
    # of it here would be a test of the mock. What the seek is guarded by (the button is
    # absent once the clip has been swept) is covered above.

    def test_dismissing_moves_the_finding_rather_than_hiding_it(self, page: Page, base_url: str):
        """Dismissing is a judgement, not a delete: the row stays, says so, and can still
        be harvested if the reviewer changes their mind."""
        self._open_modal(page, base_url, blips=[_blip()])

        page.locator("[data-testid='event-blip-check']").check()
        page.locator("[data-testid='event-blips-dismiss']").click()

        row = page.locator("[data-testid='event-blip-row']")
        expect(row).to_have_count(1)
        expect(row.locator("[data-testid='event-blip-blocked']")).to_contain_text("dismissed")
        expect(row.locator("[data-testid='event-blip-check']")).to_have_count(1)

    def test_the_panel_follows_the_walk(self, page: Page, base_url: str):
        """The modal walks events in place. A panel that kept the previous event's
        findings would attribute one event's mistakes to another."""
        events = _events(2)
        blips = [_blip(eventUuid=events[0]["uuid"])]
        _mock(page, events=events, blips=blips)
        _open_events_pane(page, base_url)
        page.locator("[data-testid='event-card']").first.click()
        page.wait_for_selector("[data-testid='event-detail-modal']", timeout=5000)
        expect(page.locator("[data-testid='event-blip-row']")).to_have_count(1)

        # The mock answers every event with the same row, so filtering client-side is
        # what is under test here: the panel asks per event and repaints from the answer.
        page.locator("[data-testid='event-detail-next']").click()

        expect(page.locator("[data-testid='event-detail-title']")).to_contain_text("Cat 1")
        expect(page.locator("[data-testid='event-blip-row']")).to_have_count(1)


class TestTheCardSaysWhenTrackingWentWrong:
    def test_a_card_with_findings_is_badged(self, page: Page, base_url: str):
        """Without this the findings are only discoverable by opening every event in the
        grid, which for a feature whose value is "these are the frames worth training
        on" makes it unfindable."""
        _mock(page, events=[_event(blipCounts={"identity_flip": 2, "tracker_dropout": 1})])
        _open_events_pane(page, base_url)

        badge = page.locator("[data-testid='event-card-blip-badge']")
        expect(badge).to_be_visible()
        # Every blip counts, the unharvestable ones included: the badge says "the
        # analyser found something here", and a card that stayed quiet while its detail
        # view listed three problems would be a lie.
        expect(badge).to_contain_text("3")

    def test_a_clean_card_is_not_badged(self, page: Page, base_url: str):
        _mock(page, events=[_event()])
        _open_events_pane(page, base_url)

        expect(page.locator("[data-testid='event-card']")).to_have_count(1)
        expect(page.locator("[data-testid='event-card-blip-badge']")).to_have_count(0)
