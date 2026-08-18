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
from urllib.parse import parse_qs, urlparse

from playwright.sync_api import Page, expect

from playwright_tests.test_nvr_live_view import FRAME_B64

PROFILE_UUID = "99999999-eeee-0000-0000-000000000002"
EVENT_UUID = "event-uuid-0001"
RECORDING_UUID = "recording-uuid-0001"

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
        "recordingUuid": RECORDING_UUID,
    }
    event.update(overrides)
    return event


def _facets(**overrides) -> dict:
    facets = {
        "rules": [{"ref": "rule-1", "name": "cat using litter box"}],
        "subjects": [{"ref": "label-cat", "name": "Tabby"}],
        "targets": [{"ref": "zone-box", "name": "litter box"}],
        "sources": [{"ref": "src-1", "name": "Litter cam"}],
    }
    facets.update(overrides)
    return facets


def _mock(page: Page, events=None, facets=None, recording=True):
    """Route every call the Events page makes, and record the search queries.

    Returns the list of parsed query dicts, so a test can assert that a filter actually
    reached the server rather than merely appearing on screen.
    """
    queries = []

    def _fulfil_events(route):
        url = route.request.url
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
        rows = events if events is not None else [_event()]
        route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({"status": "success", "events": rows, "total": len(rows)}),
        )

    def _fulfil_recordings(route):
        if "/file" in route.request.url:
            # Not a real mp4; the tests assert the element and its src, never playback,
            # which would be a test of the browser's decoder rather than of this page.
            route.fulfill(status=200, content_type="video/mp4", body=b"")
            return
        body = {"status": "success", "recording": None}
        if recording:
            body["recording"] = {
                "uuid": RECORDING_UUID,
                "sourceUuid": "src-1",
                "profileUuid": PROFILE_UUID,
                "eventUuid": EVENT_UUID,
                "filePath": "/data/recordings/clips/a.mp4",
                # Ten seconds before the event: the pre-roll the rewind buffer held.
                "startedAt": "2026-08-18T09:14:50",
                "endedAt": "2026-08-18T09:15:29",
                "durationS": 39.0,
                "sizeBytes": 1024,
                "status": "complete",
                "error": None,
                "partialPreroll": False,
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

    page.route(re.compile(r"/api/watch-events"), _fulfil_events)
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
        _mock(page, events=[_event(recordingUuid=None)])
        _open_events_pane(page, base_url)

        expect(page.locator("[data-testid='event-card-video-badge']")).to_have_count(0)

    def test_an_event_still_in_progress_says_so(self, page: Page, base_url: str):
        """An open event has no duration and no clip yet; showing "0s" would be a wrong
        answer rather than a missing one."""
        _mock(page, events=[_event(endedAt=None, recordingUuid=None)])
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

        page.locator("[data-testid='event-filter-rule']").select_option("rule-1")
        page.wait_for_function("() => true")
        expect(page.locator("[data-testid='event-card']")).to_have_count(1)
        assert queries[-1].get("rule_uuid") == "rule-1"

    def test_filtering_by_camera(self, page: Page, base_url: str):
        queries = _mock(page)
        _open_events_pane(page, base_url)

        page.locator("[data-testid='event-filter-source']").select_option("src-1")
        expect(page.locator("[data-testid='event-card']")).to_have_count(1)
        assert queries[-1].get("source_uuid") == "src-1"

    def test_filtering_by_the_acting_thing(self, page: Page, base_url: str):
        queries = _mock(page)
        _open_events_pane(page, base_url)

        page.locator("[data-testid='event-filter-subject']").select_option("label-cat")
        expect(page.locator("[data-testid='event-card']")).to_have_count(1)
        assert queries[-1].get("subject_ref") == "label-cat"

    def test_filtering_by_the_thing_acted_on(self, page: Page, base_url: str):
        queries = _mock(page)
        _open_events_pane(page, base_url)

        page.locator("[data-testid='event-filter-target']").select_option("zone-box")
        expect(page.locator("[data-testid='event-card']")).to_have_count(1)
        assert queries[-1].get("target_ref") == "zone-box"

    def test_filtering_by_date_range(self, page: Page, base_url: str):
        """The `to` bound must cover the whole day it names, or picking one day for both
        returns only events at exactly midnight."""
        queries = _mock(page)
        _open_events_pane(page, base_url)

        page.locator("[data-testid='event-filter-from']").fill("2026-08-18")
        page.locator("[data-testid='event-filter-to']").fill("2026-08-18")
        expect(page.locator("[data-testid='event-card']")).to_have_count(1)
        assert queries[-1].get("start_from") == "2026-08-18T00:00:00"
        assert queries[-1].get("start_to") == "2026-08-18T23:59:59"

    def test_filtering_to_events_with_no_video(self, page: Page, base_url: str):
        """`false` is a real filter and must not be dropped the way a falsy guard would."""
        queries = _mock(page)
        _open_events_pane(page, base_url)

        page.locator("[data-testid='event-filter-clip']").select_option("false")
        expect(page.locator("[data-testid='event-card']")).to_have_count(1)
        assert queries[-1].get("has_recording") == "false"

    def test_clearing_removes_every_filter(self, page: Page, base_url: str):
        queries = _mock(page)
        _open_events_pane(page, base_url)

        page.locator("[data-testid='event-filter-rule']").select_option("rule-1")
        page.locator("[data-testid='event-filter-clear']").click()
        expect(page.locator("[data-testid='event-card']")).to_have_count(1)
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
        self._open_modal(page, base_url, events=[_event(recordingUuid=None)])

        expect(page.locator("[data-testid='event-detail-video']")).to_have_count(0)
        expect(page.locator("[data-testid='event-detail-modal']")).to_contain_text("No clip")

    def test_a_clip_missing_from_disk_degrades_to_the_snapshot(self, page: Page, base_url: str):
        """The row can outlive its file; the snapshot is still worth showing."""
        self._open_modal(page, base_url, recording=False)

        expect(page.locator("[data-testid='event-detail-video']")).to_have_count(0)
        expect(page.locator("[data-testid='event-detail-snapshot']")).to_be_visible()


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
                _event(uuid=f"event-{i:04d}", recordingUuid=None, startedAt=f"2026-08-18T09:{i % 60:02d}:00")
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
