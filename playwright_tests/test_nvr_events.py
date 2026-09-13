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
from datetime import date
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from playwright.sync_api import Page, Route, expect

from playwright_tests.test_nvr_live_view import FRAME_B64, _mock_nvr, _open_live_view
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

#: A one-second 64x48 WebM, for the tests that need a clip the browser will really
#: decode (see `_mock(playable=True)`). Committed rather than generated at run time: it
#: is 700 bytes, and building it needs ffmpeg with libvpx, which the test box is not
#: required to have. Regenerate with:
#:   ffmpeg -f lavfi -i "color=c=red:s=64x48:d=1:r=5" -c:v libvpx -b:v 50k tiny_clip.webm
TINY_WEBM = (Path(__file__).parent / "fixtures" / "tiny_clip.webm").read_bytes()

#: Does a serialised timestamp state its zone? Mirrors `HAS_OFFSET` in `shared/time.js`.
_HAS_OFFSET = re.compile(r"(?:Z|[+-]\d{2}:?\d{2})$")


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


class _Query(dict):
    """One recorded search query: the first value per key, with every value kept too.

    Nearly every assertion here reads a single value (`q.get("limit") == "60"`), which a
    bare dict-of-lists would break. But the facet filters send a repeated key per chosen
    value, and flattening loses all but the first — so the full lists stay available on
    `.all` for the tests that care how many arrived.
    """

    def __init__(self, raw: dict):
        super().__init__({k: v[0] for k, v in raw.items()})
        self.all = raw


def _query_values(query, key: str) -> list:
    """Every value sent for `key`, not just the first."""
    return getattr(query, "all", {}).get(key, [])


def _pick_facet(page: Page, testid: str, name: str, exclude: bool = False):
    """Include (or, with `exclude`, negate) one option of a facet filter dropdown.

    The facet filters are multi-select dropdowns rather than `<select>`s, so there is no
    `select_option` to call: the panel has to be opened and the row's checkbox — or its
    ≠ button — clicked.

    The panel is then closed again, because it floats over the rest of the filter row:
    left open, it swallows the next click a test makes on any other control.
    """
    field = page.locator(f"[data-testid='{testid}']")
    trigger = field.locator(".ms-dropdown__trigger")
    trigger.click()
    row = field.locator(".ms-item", has_text=name)
    if exclude:
        row.locator(".ms-item__not-btn").click()
    else:
        row.locator("input[type='checkbox']").click()
    trigger.click()


def _mock(
    page: Page,
    events=None,
    facets=None,
    recording=True,
    overlay=None,
    deletes=None,
    blips=None,
    sessions=None,
    review_states=None,
    playable=False,
):
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
        # ⚠️ Both image endpoints, and both before the list handler at the bottom. This
        # route matches on the `/api/watch-events` *prefix*, so an unhandled per-event
        # path does not 404 — it falls through and is answered with the events list,
        # which appends to `queries`. Every "the page asked for X" assertion in this file
        # then sees one bogus query per card, and the card gets JSON for its picture.
        if "/snapshot" in url or "/crop" in url:
            route.fulfill(status=200, content_type="image/jpeg", body=PIXEL_JPEG)
            return
        # ⚠️ Both derived from `rows`, never hand-written. The three endpoints answer the
        # same question about the same events, so a mock whose groups could disagree with
        # its own list would let exactly the drift these views must not have pass a test.
        if "/groups" in url:
            queries.append(_Query(parse_qs(urlparse(url).query)))
            route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps({"status": "success", "groups": _group_rows(rows), "total": len(rows)}),
            )
            return
        if "/histogram" in url:
            queries.append(_Query(parse_qs(urlparse(url).query)))
            route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps({"status": "success", "bucket": "hour", "buckets": _histogram_rows(rows)}),
            )
            return
        if "/review-states" in url:
            payload = json.loads(route.request.post_data or "{}")
            if review_states is not None:
                review_states.append((payload.get("state"), payload.get("uuids", [])))
            route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps({"status": "success", "updated": len(payload.get("uuids", []))}),
            )
            return
        if "/sessions" in url:
            queries.append(_Query(parse_qs(urlparse(url).query)))
            # A fixed payload, not a re-derivation of `rows`. Which events fold into one
            # visit is decided by a window function and is covered exhaustively in
            # tests/db_server/; what needs a browser is only that the page asks for
            # sessions and renders what comes back.
            route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps(
                    {"status": "success", "sessions": sessions or [], "total": len(sessions or []), "gapS": 120.0}
                ),
            )
            return
        queries.append(_Query(parse_qs(urlparse(url).query)))
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
            if playable:
                # A real (tiny) WebM, for the handful of tests that need the browser to
                # actually *decode* a frame — `captureVideoFrame` reads `videoWidth`, so
                # everything downstream of "save this frame" is unreachable without one.
                # WebM rather than H.264: Chromium ships no H.264 decoder.
                route.fulfill(status=200, content_type="video/webm", body=TINY_WEBM)
                return
            # Not a real mp4; most tests assert the element and its src, never playback,
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
        # The panel now asks for `harvestable=true` (production filters tracker-fault
        # blips out server-side) — honoring it here keeps the mock's shape true to what
        # the real endpoint would answer, rather than a mock that always returns
        # everything and would silently hide a regression in that filter.
        harvestable_param = parse_qs(urlparse(route.request.url).query).get("harvestable", [None])[0]
        rows = rows_blips
        if harvestable_param is not None:
            want = harvestable_param == "true"
            rows = [b for b in rows_blips if bool(b.get("harvestable")) == want]
        route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({"status": "success", "blips": rows, "total": len(rows)}),
        )

    page.route(re.compile(r"/api/watch-events"), _fulfil_events)
    page.route(re.compile(r"/api/event-blips"), _fulfil_blips)
    page.route(re.compile(r"/api/recordings"), _fulfil_recordings)
    page.route(re.compile(r"/api/sources/get"), _fulfil_sources)
    return queries


def _group_rows(rows):
    """Collapse mocked events to groups exactly as the backend does."""
    groups = {}
    for row in rows:
        key = (row.get("ruleUuid"), row.get("subjectRef"), row.get("targetRef"), row.get("sourceUuid"))
        group = groups.setdefault(
            key,
            {
                "ruleUuid": row.get("ruleUuid"),
                "ruleName": row.get("ruleName"),
                "subjectRef": row.get("subjectRef"),
                "subjectName": row.get("subjectName"),
                "targetRef": row.get("targetRef"),
                "targetName": row.get("targetName"),
                "sourceUuid": row.get("sourceUuid"),
                "count": 0,
                "firstSeen": row.get("startedAt"),
                "lastSeen": row.get("startedAt"),
                "totalDurationS": 0.0,
                "representativeUuid": row["uuid"],
                "withRecording": 0,
                "withBlips": 0,
            },
        )
        group["count"] += 1
        if row.get("recordings"):
            group["withRecording"] += 1
        if row.get("startedAt") and row["startedAt"] > (group["lastSeen"] or ""):
            group["lastSeen"] = row["startedAt"]
            group["representativeUuid"] = row["uuid"]
    return sorted(groups.values(), key=lambda g: -g["count"])


def _histogram_rows(rows):
    """One bucket per (hour, identity), same derivation as the group rows above."""
    buckets = {}
    for row in rows:
        started = row.get("startedAt") or ""
        key = (
            f"{started[:13]}:00:00" if len(started) >= 13 else started,
            row.get("ruleUuid"),
            row.get("subjectRef"),
            row.get("targetRef"),
            row.get("sourceUuid"),
        )
        buckets[key] = buckets.get(key, 0) + 1
    return [
        {
            "bucketStart": bucket_start,
            "ruleUuid": rule_uuid,
            "subjectRef": subject_ref,
            "targetRef": target_ref,
            "sourceUuid": source_uuid,
            "count": count,
        }
        for (bucket_start, rule_uuid, subject_ref, target_ref, source_uuid), count in buckets.items()
    ]


def _open_events_pane(page: Page, base_url: str, view: str = "all", date_range: str = "", session_gap: str = "0"):
    """Open NVR → Events on a named view, date range, and session-gap grouping.

    All three are chosen explicitly rather than left to the default. The view and gap are
    both remembered in localStorage, so a test inheriting whatever the previous one picked
    would depend on execution order — and most of these tests are about the flat grid of
    individual events specifically, which is not what "all" renders once the gap default
    (5 minutes) is in effect. `session_gap` defaults to `"0"` (**Every event**) so these
    tests keep exercising that flat grid; tests that are about session grouping itself set
    it to whatever they're testing.

    `date_range` defaults to `""` — **All time** — because the fixtures here are dated at
    fixed points in 2026 while the page now opens on the last 7 days. Leaving the default
    would make almost every test depend on how long ago those fixture dates were, i.e. it
    would pass today and start failing on its own next week. Tests that are *about* the
    range set it themselves.

    `session_gap` is seeded into localStorage *before* the page reads it, rather than
    selected on the control afterwards — selecting it would fire a second, asynchronous
    search that a test's own `issues_a_request`-style barrier could then mistake for the
    one its own interaction produced, since that barrier only waits for *a* new request,
    not a particular one.
    """
    page.goto(base_url)
    page.evaluate("gap => localStorage.setItem('nvr.events.sessionGapS', gap)", session_gap)
    page.wait_for_load_state("networkidle")
    page.locator(".main-tab-button[data-tab='tab-nvr']").click()
    page.locator(".workspace-tab-button[data-tab='tab-nvr-events']").click()
    page.wait_for_selector("[data-testid='event-search']")
    page.locator("[data-testid='event-filter-range-select']").select_option(date_range)
    page.locator(f"[data-testid='event-view-{view}']").click()
    page.wait_for_selector(f"[data-testid='event-view-{view}'].event-search__view--active")


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
            _pick_facet(page, "event-filter-rule", "cat using litter box")
        assert queries[-1].get("rule_uuids") == "rule-1"

    def test_filtering_by_camera(self, page: Page, base_url: str):
        queries = _mock(page)
        _open_events_pane(page, base_url)

        with _issues_a_search(page, queries):
            _pick_facet(page, "event-filter-source", "Litter cam")
        assert queries[-1].get("source_uuids") == "src-1"

    def test_filtering_by_the_acting_thing(self, page: Page, base_url: str):
        queries = _mock(page)
        _open_events_pane(page, base_url)

        with _issues_a_search(page, queries):
            _pick_facet(page, "event-filter-subject", "Tabby")
        assert queries[-1].get("subject_refs") == "label-cat"

    def test_filtering_by_the_thing_acted_on(self, page: Page, base_url: str):
        queries = _mock(page)
        _open_events_pane(page, base_url)

        with _issues_a_search(page, queries):
            _pick_facet(page, "event-filter-target", "litter box")
        assert queries[-1].get("target_refs") == "zone-box"

    def test_excluding_the_acting_thing(self, page: Page, base_url: str):
        """The ≠ toggle has to send the value as an *exclusion*, not as a selection.

        Wiring it to the same parameter as the checkbox would look identical in the
        dropdown — struck-through row, "not Tabby" on the trigger — while returning
        exactly the events the user asked to be rid of.
        """
        queries = _mock(page)
        _open_events_pane(page, base_url)

        with _issues_a_search(page, queries):
            _pick_facet(page, "event-filter-subject", "Tabby", exclude=True)
        assert queries[-1].get("exclude_subject_refs") == "label-cat"
        assert "subject_refs" not in queries[-1], "an excluded value must not also be sent as an inclusion"

    def test_excluding_then_including_replaces_the_exclusion(self, page: Page, base_url: str):
        """Checking a row that is currently excluded must drop the exclusion. Sending
        both would ask for events that are and are not this camera — always nothing."""
        queries = _mock(page)
        _open_events_pane(page, base_url)

        with _issues_a_search(page, queries):
            _pick_facet(page, "event-filter-source", "Litter cam", exclude=True)
        assert queries[-1].get("exclude_source_uuids") == "src-1", "precondition: exclusion applied"

        with _issues_a_search(page, queries):
            _pick_facet(page, "event-filter-source", "Litter cam")
        assert queries[-1].get("source_uuids") == "src-1"
        assert "exclude_source_uuids" not in queries[-1]

    def test_two_values_of_one_facet_are_both_sent(self, page: Page, base_url: str):
        """The facet filters became multi-select alongside the ≠ toggle; picking a
        second camera must widen the search rather than replace the first."""
        queries = _mock(
            page,
            facets=_facets(sources=[{"ref": "src-1", "name": "Litter cam"}, {"ref": "src-2", "name": "Door cam"}]),
        )
        _open_events_pane(page, base_url)

        with _issues_a_search(page, queries):
            _pick_facet(page, "event-filter-source", "Litter cam")
        with _issues_a_search(page, queries):
            _pick_facet(page, "event-filter-source", "Door cam")

        sent = _query_values(queries[-1], "source_uuids")
        assert sorted(sent) == ["src-1", "src-2"], f"both cameras should be sent; got {sent}"

    def test_filtering_by_date_range(self, page: Page, base_url: str):
        """The `to` bound must cover the whole day it names, or picking one day for both
        returns only events at exactly midnight."""
        queries = _mock(page)
        _open_events_pane(page, base_url)

        page.locator("[data-testid='event-filter-from']").fill("2026-08-18")
        page.locator("[data-testid='event-filter-to']").fill("2026-08-18")
        # Wait on both *values* at once. Testing only that the keys are present is
        # satisfied by the page-open request whenever a range preset is active, and
        # waiting on `start_from` alone settles on the search that the first `fill` fired
        # — which legitimately has no `start_to` yet. Only the query carrying both is the
        # one this test is about.
        wait_until(
            page,
            lambda: queries
            and (queries[-1].get("start_from") or "").startswith("2026-08-18T00:00:00")
            and (queries[-1].get("start_to") or "").startswith("2026-08-18T23:59:59"),
            lambda: f"no search carried both typed bounds: {queries[-1] if queries else None}",
        )

    def test_a_typed_date_is_sent_with_an_explicit_offset(self, page: Page, base_url: str):
        """A date input holds a *local* calendar day, and the backend reads an offset-less
        datetime as UTC (`UtcDateTime.process_bind_param`). Sent bare, "18 Aug" meant
        17:00 on the 17th for a viewer at UTC−7 — the window was wrong, and the timeline
        drew its day gridlines against that wrong start.

        Asserted as "an offset is present" rather than against a literal zone: the
        browser's timezone decides which offset, and pinning one would make this pass
        only wherever CI happens to run.
        """
        queries = _mock(page)
        _open_events_pane(page, base_url)

        page.locator("[data-testid='event-filter-from']").fill("2026-08-18")
        page.locator("[data-testid='event-filter-to']").fill("2026-08-18")
        wait_until(page, lambda: queries and (queries[-1].get("start_to") or "").startswith("2026-08-18T23:59:59"))

        for key in ("start_from", "start_to"):
            assert _HAS_OFFSET.search(queries[-1][key]), f"{key} was sent without a zone: {queries[-1][key]}"

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
            _pick_facet(page, "event-filter-rule", "cat using litter box")
        assert queries[-1].get("rule_uuids") == "rule-1", "precondition: the filter must be applied before clearing it"

        with _issues_a_search(page, queries):
            page.locator("[data-testid='event-filter-clear']").click()
        assert "rule_uuids" not in queries[-1]

    def test_clearing_removes_an_exclusion_too(self, page: Page, base_url: str):
        """Clear resets the state object, but the dropdowns are what hold the ≠ marks —
        a rebuild that restored them from stale state would leave the search narrowed
        with no filter visibly set."""
        queries = _mock(page)
        _open_events_pane(page, base_url)

        with _issues_a_search(page, queries):
            _pick_facet(page, "event-filter-rule", "cat using litter box", exclude=True)
        assert queries[-1].get("exclude_rule_uuids") == "rule-1", "precondition: the exclusion must be applied"

        with _issues_a_search(page, queries):
            page.locator("[data-testid='event-filter-clear']").click()
        assert "exclude_rule_uuids" not in queries[-1]

    def test_an_empty_result_explains_itself(self, page: Page, base_url: str):
        _mock(page, events=[])
        _open_events_pane(page, base_url)

        expect(page.locator("[data-testid='event-search-empty']")).to_be_visible()


class TestBrowseEventsFromLiveView:
    """The Live View toolbar's "Browse events" button — liveViewModal.js publishes
    `OPEN_NVR_EVENTS_FOR_SOURCE`, and `events/main.js`'s `openEventsForSource` is what
    turns that into an actual filtered search, not just a tab switch that happens to
    land on the same camera's data by coincidence."""

    def test_clicking_browse_events_filters_to_the_cameras_own_events(self, page: Page, base_url: str):
        queries = _mock(page)
        _mock_nvr(page)
        _open_live_view(page, base_url)

        with _issues_a_search(page, queries):
            page.locator("[data-testid='watch-live-browse-events']").click()

        # The live view does not linger on top of the tab it just switched to.
        expect(page.locator("[data-testid='watch-live-dialog']")).to_have_count(0)
        expect(page.locator("[data-testid='event-search']")).to_be_visible()
        assert queries[-1].get("source_uuids") == "src-1"


class TestTheDetailModal:
    def _open_modal(self, page: Page, base_url: str, **mock_kwargs):
        _mock(page, **mock_kwargs)
        _open_events_pane(page, base_url)
        page.locator("[data-testid='event-card']").first.click()
        page.wait_for_selector("[data-testid='event-detail-modal']")

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
        page.wait_for_selector("[data-testid='event-detail-modal']")

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


class TestSavingTheSameFrameTwice:
    """Save, then Annotate, without moving the playhead.

    That is the obvious way to do the obvious thing — keep this frame, now let me draw
    on it — and it used to produce *two* dataset images of one instant, the second being
    the one the canvas opened. The first was left in the annotation queue as a duplicate
    nobody asked for and nobody would ever reconcile.

    These are the only tests here that need the clip to really decode — `saveFrame`
    stops at `captureVideoFrame`, which reads `videoWidth`. Hence `playable=True`.
    """

    def _open_with_uploads(self, page: Page, base_url: str):
        """Open the modal on a decodable clip, capturing every image upload."""
        uploads = []
        _mock(page, overlay=_overlay(), playable=True)
        _route_labels(page)

        def _capture(route: Route):
            uploads.append(route.request.url)
            route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps(
                    {
                        "status": "success",
                        "results": [
                            {"ok": True, "path": "captures/frame_0001.jpg", "uuid": "img-0001", "renamedFrom": None}
                        ],
                    }
                ),
            )

        page.route("**/api/images/upload", _capture)
        _open_events_pane(page, base_url)
        page.locator("[data-testid='event-card']").first.click()
        page.wait_for_selector("[data-testid='event-detail-modal']")
        # The barrier the whole class rests on: until the browser has decoded a frame,
        # every Save click takes the "not loaded yet" path and uploads nothing, which
        # would make an assertion of "exactly one upload" pass for the wrong reason.
        wait_until(
            page,
            lambda: page.locator("[data-testid='event-detail-video']").evaluate("v => v.videoWidth > 0"),
            "the clip never decoded a frame",
        )
        return uploads

    def test_annotating_after_saving_reuses_the_image_just_saved(self, page: Page, base_url: str):
        uploads = self._open_with_uploads(page, base_url)

        page.locator("[data-testid='event-detail-save-frame']").click()
        wait_until(page, lambda: len(uploads) == 1, "the frame was never saved")

        page.locator("[data-testid='event-detail-annotate-frame']").click()

        # The event modal closes on its way to the canvas, which is the evidence that
        # Annotate went somewhere rather than silently doing nothing.
        expect(page.locator("[data-testid='event-detail-modal']")).to_have_count(0)
        assert uploads == uploads[:1], f"the same frame must not be uploaded twice, saw {len(uploads)}"

    def test_moving_the_playhead_makes_it_a_different_frame_again(self, page: Page, base_url: str):
        """The other half, and the reason the memo is keyed on the playhead rather than
        being a one-shot latch: a second Save after a seek is a genuinely different
        picture and must still be saved."""
        uploads = self._open_with_uploads(page, base_url)

        page.locator("[data-testid='event-detail-save-frame']").click()
        wait_until(page, lambda: len(uploads) == 1, "the frame was never saved")

        # Played rather than seeked: `route.fulfill` serves the clip without Range
        # support, so assigning `currentTime` is refused and the playhead never moves —
        # which would make this pass or fail for a reason that has nothing to do with
        # the memo. Playback advances it for real.
        video = page.locator("[data-testid='event-detail-video']")
        video.evaluate("v => v.play()")
        wait_until(page, lambda: video.evaluate("v => v.currentTime") > 0.2, "the clip never advanced")
        video.evaluate("v => v.pause()")
        page.locator("[data-testid='event-detail-save-frame']").click()

        wait_until(page, lambda: len(uploads) == 2, "a frame from a different moment must save on its own")


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
        page.wait_for_selector("[data-testid='event-detail-modal']")

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
        page.wait_for_selector("[data-testid='event-detail-modal']")

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

    def test_a_tracker_fault_is_not_listed_at_all(self, page: Page, base_url: str):
        """Harvesting a tracker fault pads the set with frames the model already gets
        right, and it carries no diagnostic detail a user could act on either — so
        rather than show an inert row explaining itself, the panel asks for
        `harvestable=true` and the finding never appears."""
        self._open_modal(
            page,
            base_url,
            blips=[
                _blip(),
                _blip(uuid="blip-uuid-0002", kind="tracker_dropout", fault="tracker", harvestable=False),
            ],
        )

        rows = page.locator("[data-testid='event-blip-row']")
        expect(rows).to_have_count(1)
        expect(page.locator("[data-testid='event-blips-title']")).to_contain_text("1")
        expect(rows.locator("[data-testid='event-blip-check']")).to_have_count(1)

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

    def _mock_harvest_completion(self, page: Page, result_data: dict, run_name: str = "Tracking blips"):
        """Answer the task stream with one finished harvest, and serve the run it made.

        `result_data` goes inside the `{status, data}` envelope every task result is
        wrapped in — reading through that envelope is the whole point of the tests
        below, so the mock must not flatten it.
        """
        page.route(
            "**/api/tasks/stream",
            lambda route: route.fulfill(
                status=200,
                content_type="text/event-stream",
                body="data: "
                + json.dumps(
                    {
                        "tasks": {
                            "task-harvest-1": {
                                "instance": {
                                    "uuid": "task-harvest-1",
                                    "status": "completed",
                                    "result": {"status": "completed", "data": result_data},
                                },
                            }
                        }
                    }
                )
                + "\n\n",
            ),
        )
        page.route(
            re.compile(r".*/api/preview-runs/[\w-]+$"),
            lambda route: route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps(
                    {
                        "status": "success",
                        "run": {
                            "uuid": "run-harvest-1",
                            "name": run_name,
                            "modelName": "",
                            "pinned": False,
                            "stats": {},
                        },
                    }
                ),
            ),
        )
        page.route(
            re.compile(r".*/api/preview-runs/[\w-]+/candidates/query$"),
            lambda route: route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps({"status": "success", "candidates": [], "totalCount": 0}),
            ),
        )

    def test_a_finished_harvest_opens_the_run_it_produced(self, page: Page, base_url: str):
        """The frames are three clicks deep in Saved Previews, so a harvest that files
        them and says nothing is a harvest the user reports as never having finished —
        which is exactly what happened. The uuid lives at `result.data.previewRunUuid`;
        reading `result.previewRunUuid` yields undefined for *every* task in the app and
        silently downgrades a successful run to "nothing to review"."""
        self._mock_launch(page, [])
        self._mock_harvest_completion(
            page,
            {"previewRunUuid": "run-harvest-1", "framesHarvested": 3},
            run_name="Harvested frames",
        )
        self._open_modal(page, base_url, blips=[_blip()])

        page.locator("[data-testid='event-blip-check']").check()
        page.locator("[data-testid='event-blips-harvest']").click()

        expect(page.locator("[data-testid='preview-run-title']")).to_have_text("Harvested frames")

    def test_a_harvest_that_produced_nothing_opens_no_run(self, page: Page, base_url: str):
        """The other half of the same branch. A run that proposed nothing has no preview
        run to open, and a modal over an empty triage grid would be worse than the toast
        — so the assertion above must be a consequence of the uuid, not of finishing."""
        self._mock_launch(page, [])
        self._mock_harvest_completion(page, {"previewRunUuid": None, "framesHarvested": 0})
        self._open_modal(page, base_url, blips=[_blip()])

        page.locator("[data-testid='event-blip-check']").check()
        page.locator("[data-testid='event-blips-harvest']").click()

        expect(page.locator("[data-testid='event-blips-title']")).to_be_visible()
        expect(page.locator("[data-testid='preview-run-title']")).to_have_count(0)

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

    def test_the_player_stays_visible_while_the_blip_list_scrolls(self, page: Page, base_url: str):
        """`.event-detail__media` and `.event-detail__side` share one scroll container
        (`.event-detail__body`). With enough findings the list runs taller than the
        player, and reaching a play button far down it must not require scrolling the
        video itself out of view — it's sticky to the top of that shared container."""
        blips = [_blip(uuid=f"blip-uuid-{i:04d}", startS=float(i), endS=float(i) + 1, trackId=i) for i in range(40)]
        self._open_modal(page, base_url, blips=blips)

        body = page.locator(".event-detail__body")
        video = page.locator(".event-detail__video, .event-detail__image").first
        before = video.bounding_box()

        body.evaluate("el => el.scrollTop = el.scrollHeight")

        after = video.bounding_box()
        assert after is not None, "the player must still be on screen after scrolling the blip list"
        assert abs(after["y"] - before["y"]) <= 2, "the player should stay pinned, not scroll away"

    def test_the_panel_follows_the_walk(self, page: Page, base_url: str):
        """The modal walks events in place. A panel that kept the previous event's
        findings would attribute one event's mistakes to another."""
        events = _events(2)
        blips = [_blip(eventUuid=events[0]["uuid"])]
        _mock(page, events=events, blips=blips)
        _open_events_pane(page, base_url)
        page.locator("[data-testid='event-card']").first.click()
        page.wait_for_selector("[data-testid='event-detail-modal']")
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
        _mock(page, events=[_event(blipCounts={"identity_flip": 2, "full_dropout": 1})])
        _open_events_pane(page, base_url)

        badge = page.locator("[data-testid='event-card-blip-badge']")
        expect(badge).to_be_visible()
        # The card sums whatever `blipCounts` it is handed — the server is what decides
        # which kinds count (harvestable and not yet resolved, see `_attach_event_blip_
        # counts`); this only tests that the card adds them up correctly.
        expect(badge).to_contain_text("3")

    def test_a_clean_card_is_not_badged(self, page: Page, base_url: str):
        _mock(page, events=[_event()])
        _open_events_pane(page, base_url)

        expect(page.locator("[data-testid='event-card']")).to_have_count(1)
        expect(page.locator("[data-testid='event-card-blip-badge']")).to_have_count(0)


def _two_identities() -> list:
    """Four events of one kind and two of another — a set the grouped view must reduce."""
    drinks = [
        _event(
            uuid=f"drink-{i:04d}",
            ruleUuid="rule-water",
            ruleName="cat drinking water",
            subjectRef="label-cat",
            subjectName="Tabby",
            targetRef="zone-bowl",
            targetName="water bowl",
            startedAt=f"2026-08-18T09:{10 + i:02d}:00",
            endedAt=f"2026-08-18T09:{10 + i:02d}:20",
            recordings=[],
        )
        for i in range(4)
    ]
    litter = [
        _event(
            uuid=f"litter-{i:04d}",
            ruleUuid="rule-litter",
            ruleName="cat using litter box",
            subjectRef="label-other",
            subjectName="Clover",
            targetRef="zone-box",
            targetName="litter box",
            startedAt=f"2026-08-18T11:{10 + i:02d}:00",
            endedAt=f"2026-08-18T11:{10 + i:02d}:20",
            recordings=[],
        )
        for i in range(2)
    ]
    return drinks + litter


def _opacity(page: Page, selector: str, index: int = 0) -> float:
    return float(page.locator(selector).nth(index).evaluate("el => getComputedStyle(el).opacity"))


class TestGroupingCollapsesRepeatedEvents:
    """The reason this view exists.

    The flat grid shows one tile per event, which for a rule that fires four hundred
    times a week is the wrong *unit* rather than too much detail. The assertions here are
    all comparisons between the two views of the same data — an existence check on a
    group card would pass just as happily against a view that grouped nothing.
    """

    def test_six_events_of_two_kinds_become_two_cards(self, page: Page, base_url: str):
        _mock(page, events=_two_identities())
        _open_events_pane(page, base_url, view="all")
        expect(page.locator("[data-testid='event-card']")).to_have_count(6)

        page.locator("[data-testid='event-view-groups']").click()
        expect(page.locator("[data-testid='event-group-card']")).to_have_count(2)

    def test_the_card_says_how_many_events_it_stands_for(self, page: Page, base_url: str):
        """The count is the headline fact of a group; a card that omitted it would send
        the user back to the flat grid to find out how much it was hiding."""
        _mock(page, events=_two_identities())
        _open_events_pane(page, base_url, view="groups")

        counts = page.locator("[data-testid='event-group-card-count']")
        expect(counts).to_have_count(2)
        assert [c.strip() for c in counts.all_text_contents()] == ["4", "2"], "biggest group leads"

    def test_clicking_a_group_narrows_the_flat_view_to_it(self, page: Page, base_url: str):
        """Drilling in must actually filter, not merely switch views.

        The mock returns every event whatever is asked, so the check is on the request:
        a drill-down that switched views without sending the identity would look right on
        screen and show the wrong events against a real server.
        """
        queries = _mock(page, events=_two_identities())
        _open_events_pane(page, base_url, view="groups")

        page.locator("[data-testid='event-group-card-drill']").first.click()
        expect(page.locator("[data-testid='event-view-all']")).to_have_class(re.compile("event-search__view--active"))
        wait_until(page, lambda: _query_values(queries[-1], "rule_uuids") == ["rule-water"])
        assert _query_values(queries[-1], "subject_refs") == ["label-cat"]
        assert _query_values(queries[-1], "target_refs") == ["zone-bowl"]

    def test_selecting_a_group_arms_a_delete_for_every_event_in_it(self, page: Page, base_url: str):
        """A group's checkbox stands for four rows, and the button must say four.

        This is the one gesture on the page that can destroy far more than it appears to,
        so "2 selected" must never quietly mean 665 events.
        """
        _mock(page, events=_two_identities())
        _open_events_pane(page, base_url, view="groups")

        page.locator("[data-testid='event-group-card-select']").first.click()
        expect(page.locator("[data-testid='event-search-delete-selected']")).to_have_text("Delete 4 events")


class TestHoveringOneEventHighlightsEveryOtherLikeIt:
    """The requested behaviour: pick out one kind of event among many at a glance.

    Keyed on identity rather than on the element, so it is the same mechanism in the flat
    grid, on the group cards and along the timeline's lanes.
    """

    def test_hovering_a_tile_lights_its_own_kind_and_dims_the_rest(self, page: Page, base_url: str):
        _mock(page, events=_two_identities())
        _open_events_pane(page, base_url, view="all")
        expect(page.locator("[data-testid='event-card']")).to_have_count(6)

        page.locator("[data-event-uuid='drink-0000']").hover()

        # Every event of the hovered kind, not just the one under the pointer — which is
        # the whole difference between this and a plain :hover rule.
        expect(page.locator("[data-event-uuid='drink-0003'].is-focused")).to_have_count(1)
        expect(page.locator("[data-event-uuid='litter-0000'].is-focused")).to_have_count(0)

    def test_the_dimming_is_actually_visible(self, page: Page, base_url: str):
        """Guards the CSS, not the class.

        The class flip is meaningless if the rule that acts on it is deleted or renamed,
        and every other assertion here would still pass.
        """
        _mock(page, events=_two_identities())
        _open_events_pane(page, base_url, view="all")
        expect(page.locator("[data-testid='event-card']")).to_have_count(6)
        assert _opacity(page, "[data-event-uuid='litter-0000']") == 1.0

        page.locator("[data-event-uuid='drink-0000']").hover()

        wait_until(page, lambda: _opacity(page, "[data-event-uuid='litter-0000']") < 1.0)
        assert _opacity(page, "[data-event-uuid='drink-0003']") == 1.0

    def test_moving_away_restores_every_card(self, page: Page, base_url: str):
        """A highlight that never released would leave the page permanently dimmed."""
        _mock(page, events=_two_identities())
        _open_events_pane(page, base_url, view="all")
        expect(page.locator("[data-testid='event-card']")).to_have_count(6)

        page.locator("[data-event-uuid='drink-0000']").hover()
        wait_until(page, lambda: _opacity(page, "[data-event-uuid='litter-0000']") < 1.0)

        page.locator("[data-testid='event-search-summary']").hover()
        wait_until(page, lambda: _opacity(page, "[data-event-uuid='litter-0000']") == 1.0)


class TestTheTimelineShowsWhenThingsHappened:
    def test_one_lane_per_identity(self, page: Page, base_url: str):
        _mock(page, events=_two_identities())
        _open_events_pane(page, base_url, view="timeline")

        expect(page.locator("[data-testid='event-timeline-lane']")).to_have_count(2)
        expect(page.locator("[data-testid='event-timeline-mark']").first).to_be_attached()

    def test_hovering_a_lane_dims_the_others(self, page: Page, base_url: str):
        _mock(page, events=_two_identities())
        _open_events_pane(page, base_url, view="timeline")
        expect(page.locator("[data-testid='event-timeline-lane']")).to_have_count(2)

        page.locator("[data-testid='event-timeline-lane']").first.hover()

        expect(page.locator("[data-testid='event-timeline-lane'].is-focused")).to_have_count(1)
        wait_until(page, lambda: _opacity(page, "[data-testid='event-timeline-lane']", 1) < 1.0)

    def test_clicking_a_mark_opens_its_events_without_leaving_the_timeline(self, page: Page, base_url: str):
        """A mark opens its bin over a timeline that is still there.

        Clicking used to rewrite the page's filters into a grid drill-down. That answered
        the objection "opening one arbitrary event of the fifty in an hour is a guess at
        which" by opening none of them — at the cost of a one-way door, since getting back
        meant noticing the window chip, clearing it, re-picking the view and re-finding
        the lane. Browsing the whole bin answers the objection properly.
        """
        _mock(page, events=_two_identities())
        _open_events_pane(page, base_url, view="timeline")
        expect(page.locator("[data-testid='event-timeline-mark']").first).to_be_attached()

        page.locator("[data-testid='event-timeline-mark']").first.click()

        expect(page.locator("[data-testid='bucket-browse-modal']")).to_be_visible()
        # The negative half, and the point of the change: the view underneath is
        # untouched, where the drill-down used to have replaced it with a filtered grid.
        expect(page.locator("[data-testid='event-view-timeline']")).to_have_class(
            re.compile("event-search__view--active")
        )
        expect(page.locator("[data-testid='event-timeline']")).to_be_attached()
        expect(page.locator("[data-testid='event-window-chip']")).to_be_hidden()

    def test_a_mark_hands_the_player_the_whole_bin_to_walk(self, page: Page, base_url: str):
        """Opening one event of the bin with no way to the other three would be exactly
        the guess the drill-down existed to avoid.

        Two halves, because the mock answers any event query with its whole fixture and
        so cannot prove the *size* of the walk: that the request was scoped to this lane
        and this hour, and that the player came up with its navigation live — which it
        hides for a lone event, so it is only there if siblings were passed through the
        browser.
        """
        queries = _mock(page, events=_two_identities())
        _open_events_pane(page, base_url, view="timeline")
        expect(page.locator("[data-testid='event-timeline-mark']").first).to_be_attached()

        page.locator("[data-testid='event-timeline-lane']").first.locator(
            "[data-testid='event-timeline-mark']"
        ).first.click()

        page.locator("[data-testid='bucket-browse-modal'] [data-testid='event-card']").first.click()

        expect(page.locator("[data-testid='event-detail-modal']")).to_be_visible()
        expect(page.locator("[data-testid='event-detail-nav']")).to_be_visible()
        wait_until(page, lambda: _query_values(queries[-1], "rule_uuids") == ["rule-water"])
        assert queries[-1].get("start_from"), "the clicked bin must scope the fetch"
        assert queries[-1].get("start_to")

    def test_a_bin_of_one_skips_the_browser_and_plays_it(self, page: Page, base_url: str):
        """The conditional half of the rule above. A grid of exactly one card is a click
        spent on a question with one answer, so the single-event bin must go straight to
        the player — which is also the behaviour the multi-event case must *not* have."""
        _mock(page, events=[_event(uuid="lone-0000")])
        _open_events_pane(page, base_url, view="timeline")
        expect(page.locator("[data-testid='event-timeline-mark']").first).to_be_attached()

        page.locator("[data-testid='event-timeline-mark']").first.click()

        expect(page.locator("[data-testid='event-detail-modal']")).to_be_visible()
        expect(page.locator("[data-testid='bucket-browse-modal']")).to_have_count(0)

    def test_the_hover_preview_shows_the_bin_before_it_is_opened(self, page: Page, base_url: str):
        """The preview is what makes a mark worth aiming at: a 3px block with a tooltip
        of counts gives no reason to prefer one over another."""
        _mock(page, events=_two_identities())
        _open_events_pane(page, base_url, view="timeline")
        mark = page.locator("[data-testid='event-timeline-mark']").first
        expect(mark).to_be_attached()
        # Absent until asked for — the state-dependent half. A preview that were merely
        # rendered hidden would pass a bare visibility check on the popup below.
        expect(page.locator("[data-testid='event-timeline-preview']")).to_have_count(0)

        mark.hover()

        preview = page.locator("[data-testid='event-timeline-preview']")
        expect(preview).to_be_visible()
        expect(preview).to_contain_text("4 events")
        expect(preview.locator("img")).to_be_visible()

    def test_a_bin_of_one_shows_the_events_own_time_not_the_bucket_hour(self, page: Page, base_url: str):
        """A bin holding exactly one event fetches that event for the preview image
        anyway, so the timestamp text has no excuse to show the hour it was binned
        into (09:00) instead of when it actually happened (09:15) — the whole point
        of a single-event bin is that the precise answer is already on hand."""
        _mock(page, events=[_event(uuid="lone-0000", startedAt="2026-08-18T09:15:00")])
        _open_events_pane(page, base_url, view="timeline")
        mark = page.locator("[data-testid='event-timeline-mark']").first
        expect(mark).to_be_attached()

        mark.hover()

        preview = page.locator("[data-testid='event-timeline-preview']")
        expect(preview).to_be_visible()
        expect(preview).to_contain_text("1 event")
        expect(preview).to_contain_text("09:15")
        expect(preview).not_to_contain_text("09:00")

    def test_shift_clicking_a_mark_still_narrows_to_that_identity_and_that_hour(self, page: Page, base_url: str):
        """Narrowing the page to one identity and one hour did not stop being useful —
        it stopped being what an unmodified click does, because it is a one-way door and
        acting on a whole hour at once is the rarer errand."""
        queries = _mock(page, events=_two_identities())
        _open_events_pane(page, base_url, view="timeline")
        mark = page.locator("[data-testid='event-timeline-mark']").first
        expect(mark).to_be_attached()

        mark.click(modifiers=["Shift"])

        # The modal is the plain-click path, so it must *not* be what shift produced.
        expect(page.locator("[data-testid='event-detail-modal']")).to_have_count(0)
        expect(page.locator("[data-testid='event-view-all']")).to_have_class(re.compile("event-search__view--active"))
        expect(page.locator("[data-testid='event-window-chip']")).to_be_visible()
        wait_until(page, lambda: _query_values(queries[-1], "rule_uuids") == ["rule-water"])
        assert queries[-1].get("start_from"), "the clicked hour must reach the server"
        assert queries[-1].get("start_to")


class TestTheViewSwitchKeepsTheFilters:
    def test_a_filter_set_in_one_view_still_applies_in_another(self, page: Page, base_url: str):
        """The three views are projections of one query. A switch that dropped the
        filters would quietly widen the result set at the moment the user changed how
        they were looking at it."""
        queries = _mock(page, events=_two_identities())
        _open_events_pane(page, base_url, view="all")

        with _issues_a_search(page, queries):
            _pick_facet(page, "event-filter-rule", "cat using litter box")
        assert _query_values(queries[-1], "rule_uuids") == ["rule-1"]

        page.locator("[data-testid='event-view-groups']").click()
        wait_until(page, lambda: _query_values(queries[-1], "rule_uuids") == ["rule-1"])


def _session(**overrides) -> dict:
    session = {
        "ruleUuid": "rule-water",
        "ruleName": "cat drinking water",
        "subjectRef": "label-cat",
        "subjectName": "Tabby",
        "targetRef": "zone-bowl",
        "targetName": "water bowl",
        "sourceUuid": "src-1",
        "sessionIx": 1,
        "count": 7,
        "startedAt": "2026-08-18T09:15:00",
        "endedAt": "2026-08-18T09:22:00",
        "representativeUuid": EVENT_UUID,
    }
    session.update(overrides)
    return session


class TestSessionGroupingFoldsRepeats:
    """One visit to the bowl re-fires its rule several times; the flat grid shows a tile
    for each. The grouping itself is a window function covered in tests/db_server/ — what
    needs a browser is that the control asks for it and the page renders what returns."""

    def test_choosing_a_gap_asks_the_server_to_group(self, page: Page, base_url: str):
        queries = _mock(page, events=_events(3), sessions=[_session()])
        _open_events_pane(page, base_url, view="all")

        page.locator("[data-testid='event-session-gap']").select_option("120")

        wait_until(page, lambda: queries[-1].get("gap_s") == "120")
        expect(page.locator("[data-testid='event-session-card']")).to_have_count(1)

    def test_a_folded_visit_says_how_many_events_it_hides(self, page: Page, base_url: str):
        """Without the count the card is indistinguishable from a single event, and the
        seven it stands for are invisible."""
        _mock(page, events=_events(3), sessions=[_session(count=7)])
        _open_events_pane(page, base_url, view="all")
        page.locator("[data-testid='event-session-gap']").select_option("120")

        expect(page.locator("[data-testid='event-session-repeat-badge']")).to_have_text("×7")

    def test_a_visit_of_one_event_carries_no_repeat_badge(self, page: Page, base_url: str):
        """The conditional half of the rule above: a badge on every card would stop
        meaning "several were folded here"."""
        _mock(page, events=_events(3), sessions=[_session(count=1)])
        _open_events_pane(page, base_url, view="all")
        page.locator("[data-testid='event-session-gap']").select_option("120")

        expect(page.locator("[data-testid='event-session-card']")).to_have_count(1)
        expect(page.locator("[data-testid='event-session-repeat-badge']")).to_have_count(0)

    def test_opening_a_visit_ungroups_it(self, page: Page, base_url: str):
        """Drilling in with the collapse still on would land on the very card just
        clicked — it is the only thing inside its own window."""
        queries = _mock(page, events=_events(3), sessions=[_session()])
        _open_events_pane(page, base_url, view="all")
        page.locator("[data-testid='event-session-gap']").select_option("120")
        expect(page.locator("[data-testid='event-session-card']")).to_have_count(1)

        page.locator("[data-testid='event-session-card']").click()

        expect(page.locator("[data-testid='event-card']")).to_have_count(3)
        assert page.locator("[data-testid='event-session-gap']").input_value() == "0"
        assert queries[-1].get("gap_s") is None, "the ungrouped page must not ask for sessions"


class TestReviewTriage:
    def test_marking_a_selection_reviewed_sends_the_state(self, page: Page, base_url: str):
        states = []
        _mock(page, events=_events(3), review_states=states)
        _open_events_pane(page, base_url, view="all")

        page.locator("[data-testid='event-card-select']").first.click()
        page.locator("[data-testid='event-search-review-confirmed']").click()

        wait_until(page, lambda: len(states) == 1)
        assert states[0][0] == "confirmed"
        assert states[0][1] == ["event-uuid-0000"]

    def test_marking_a_group_reviewed_covers_every_event_behind_it(self, page: Page, base_url: str):
        """The payoff of pairing triage with grouping: the thing you want to dismiss is
        all four of these, not one card standing for them."""
        states = []
        _mock(page, events=_two_identities(), review_states=states)
        _open_events_pane(page, base_url, view="groups")

        page.locator("[data-testid='event-group-card-select']").first.click()
        page.locator("[data-testid='event-search-review-confirmed']").click()

        wait_until(page, lambda: len(states) == 1)
        assert states[0][0] == "confirmed"
        assert len(states[0][1]) == 6, "the drill-down fetch returns the mock's whole set"


def _one_rule_two_actors() -> list:
    """One event type, two actors, two targets — so every grain gives a different answer.

    detail (actor & target) → 3 rows · actor → 2 rows · event type → 1 row.
    A fixture where the grains agreed would let a broken fold pass.
    """
    spec = [
        ("Tabby", "label-tabby", "bowl A", "zone-a"),
        ("Tabby", "label-tabby", "bowl A", "zone-a"),
        ("Tabby", "label-tabby", "bowl B", "zone-b"),
        ("Clover", "label-clover", "bowl A", "zone-a"),
    ]
    return [
        _event(
            uuid=f"drink-{i:04d}",
            ruleUuid="rule-water",
            ruleName="cat drinking water",
            subjectRef=subject_ref,
            subjectName=subject,
            targetRef=target_ref,
            targetName=target,
            startedAt=f"2026-08-18T09:{10 + i:02d}:00",
            endedAt=f"2026-08-18T09:{10 + i:02d}:20",
            recordings=[],
        )
        for i, (subject, subject_ref, target, target_ref) in enumerate(spec)
    ]


class TestTheGroupingGrainIsSelectable:
    """The finest grain is the unit an event happened at; it is the wrong unit for
    "how much is each cat drinking", where one rule splits across several vessels and
    nothing sums them. Every assertion here compares two grains of the same data — a
    count read at one grain alone would prove nothing about the fold."""

    def test_each_grain_gives_a_different_number_of_cards(self, page: Page, base_url: str):
        _mock(page, events=_one_rule_two_actors())
        _open_events_pane(page, base_url, view="groups")
        expect(page.locator("[data-testid='event-group-card']")).to_have_count(3)

        page.locator("[data-testid='event-group-by']").select_option("actor")
        expect(page.locator("[data-testid='event-group-card']")).to_have_count(2)

        page.locator("[data-testid='event-group-by']").select_option("event")
        expect(page.locator("[data-testid='event-group-card']")).to_have_count(1)

    def test_a_coarser_grain_sums_the_finer_rows_rather_than_dropping_them(self, page: Page, base_url: str):
        """The summary row must account for every event under it. A fold that picked one
        sub-row instead of summing would look plausible and undercount."""
        _mock(page, events=_one_rule_two_actors())
        _open_events_pane(page, base_url, view="groups")

        counts = page.locator("[data-testid='event-group-card-count']")

        page.locator("[data-testid='event-group-by']").select_option("event")
        # The count settles before the text is read: a multi-element locator makes
        # `to_have_text` a strict-mode error rather than something it retries, so reading
        # text first would report a race as a failure.
        expect(counts).to_have_count(1)
        expect(counts).to_have_text("4")

        page.locator("[data-testid='event-group-by']").select_option("actor")
        expect(counts).to_have_count(2)
        assert sorted(int(c) for c in counts.all_text_contents()) == [1, 3]

    def test_a_per_actor_row_is_not_captioned_with_one_of_its_targets(self, page: Page, base_url: str):
        """Tabby drank from two bowls. A row summing both must not claim to be "bowl A" —
        it would be a wrong answer about a row whose number includes bowl B.
        """
        _mock(page, events=_one_rule_two_actors())
        _open_events_pane(page, base_url, view="groups")
        # At the finest grain the target is genuinely part of what the row is.
        expect(page.locator("[data-testid='event-group-card-headline']").first).to_contain_text("bowl A")

        page.locator("[data-testid='event-group-by']").select_option("actor")

        headlines = page.locator("[data-testid='event-group-card-headline']")
        expect(headlines).to_have_count(2)
        assert sorted(h.strip() for h in headlines.all_text_contents()) == ["Clover", "Tabby"]
        assert not any("bowl" in h for h in headlines.all_text_contents())

    def test_drilling_from_a_coarse_row_does_not_filter_by_the_folded_dimension(self, page: Page, base_url: str):
        """A per-actor row that also sent a target would open onto a fraction of what it
        counted, and the card's number would visibly disagree with the grid."""
        queries = _mock(page, events=_one_rule_two_actors())
        _open_events_pane(page, base_url, view="groups")
        page.locator("[data-testid='event-group-by']").select_option("actor")
        expect(page.locator("[data-testid='event-group-card']")).to_have_count(2)

        page.locator("[data-testid='event-group-card-drill']").first.click()

        wait_until(page, lambda: _query_values(queries[-1], "subject_refs") == ["label-tabby"])
        assert _query_values(queries[-1], "rule_uuids") == ["rule-water"]
        assert _query_values(queries[-1], "target_refs") == [], "the folded dimension must not narrow the drill-down"


class TestTimelineLanesFollowTheGrain:
    def test_grouping_per_actor_gives_one_lane_per_cat(self, page: Page, base_url: str):
        """The side-by-side comparison: one lane per cat for a single event type, rather
        than a lane per actor/target pairing."""
        _mock(page, events=_one_rule_two_actors())
        _open_events_pane(page, base_url, view="timeline")
        expect(page.locator("[data-testid='event-timeline-lane']")).to_have_count(3)

        page.locator("[data-testid='event-group-by']").select_option("actor")

        expect(page.locator("[data-testid='event-timeline-lane']")).to_have_count(2)
        labels = page.locator("[data-testid='event-timeline-label']")
        assert sorted(x.strip() for x in labels.locator(".event-timeline__label-primary").all_text_contents()) == [
            "Clover",
            "Tabby",
        ]

    def test_a_lane_names_its_rule_when_the_actor_alone_would_be_ambiguous(self, page: Page, base_url: str):
        """Two rules acting on one cat would otherwise be two lanes both reading "Luna"
        — the duplicate-headline problem the camera split already avoids."""
        _mock(page, events=_one_rule_two_actors() + _two_identities())
        _open_events_pane(page, base_url, view="timeline")
        page.locator("[data-testid='event-group-by']").select_option("actor")

        expect(page.locator(".event-timeline__label-rule").first).to_be_visible()
        expect(page.locator("[data-testid='event-timeline-label']").first).to_contain_text("drinking water")

    def test_splitting_by_camera_names_the_camera_on_every_lane(self, page: Page, base_url: str):
        """The split subdivides one identity into several lanes that are otherwise
        captioned identically — same actor, same rule, same count column. Without the
        camera on the lane the toggle looks like it duplicated the row rather than split
        it, and there is no way to tell which lane is which camera.
        """
        one_identity_two_cameras = [
            _event(
                uuid=f"drink-{i:04d}",
                ruleUuid="rule-water",
                ruleName="cat drinking water",
                subjectRef="label-cat",
                subjectName="Tabby",
                targetRef="zone-bowl",
                targetName="water bowl",
                sourceUuid=source,
                startedAt=f"2026-08-18T09:{10 + i:02d}:00",
                endedAt=f"2026-08-18T09:{10 + i:02d}:20",
                recordings=[],
            )
            for i, source in enumerate(["src-1", "src-1", "src-2", "src-2"])
        ]
        _mock(page, events=one_identity_two_cameras)
        # Registered after _mock's own, and so matched first: two cameras, because a
        # single-camera fixture cannot show a split at all.
        page.route(
            re.compile(r"/api/sources/get"),
            lambda r: r.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps(
                    {
                        "status": "success",
                        "sources": {
                            "src-1": {"uuid": "src-1", "name": "Litter cam", "typename": "F", "params": {}},
                            "src-2": {"uuid": "src-2", "name": "Door cam", "typename": "F", "params": {}},
                        },
                    }
                ),
            ),
        )
        _open_events_pane(page, base_url, view="timeline")
        lanes = page.locator("[data-testid='event-timeline-lane']")
        expect(lanes).to_have_count(1)

        page.locator("[data-testid='event-split-by-camera']").check()

        expect(lanes).to_have_count(2)
        cameras = sorted(
            x.strip()
            for x in page.locator(
                "[data-testid='event-timeline-label'] .event-timeline__label-camera"
            ).all_text_contents()
        )
        assert cameras == ["Door cam", "Litter cam"], cameras

    def test_a_lane_names_no_camera_until_the_split_asks_for_one(self, page: Page, base_url: str):
        """The conditional half: unsplit lanes cover every camera, so naming one would
        be picking whichever sub-group happened to be folded first."""
        _mock(page, events=_two_identities())
        _open_events_pane(page, base_url, view="timeline")
        expect(page.locator("[data-testid='event-timeline-lane']")).to_have_count(2)

        expect(page.locator(".event-timeline__label-camera")).to_have_count(0)

    def test_the_highlight_still_follows_the_coarser_row(self, page: Page, base_url: str):
        """The focus key is built from the same dimensions, so changing the grain must
        not leave the hover keyed on a row shape that no longer exists."""
        _mock(page, events=_one_rule_two_actors())
        _open_events_pane(page, base_url, view="timeline")
        page.locator("[data-testid='event-group-by']").select_option("actor")
        expect(page.locator("[data-testid='event-timeline-lane']")).to_have_count(2)

        page.locator("[data-testid='event-timeline-lane']").first.hover()

        expect(page.locator("[data-testid='event-timeline-lane'].is-focused")).to_have_count(1)
        wait_until(page, lambda: _opacity(page, "[data-testid='event-timeline-lane']", 1) < 1.0)


class TestTheDefaultRangeAndItsSteppers:
    """The page used to open on all of history, which grows without bound. It now opens
    on a window, which is a filter nobody asked for — so the control naming it is always
    visible, and every assertion here is about the range being *honest* on screen."""

    def test_the_page_opens_on_a_bounded_range_and_says_which(self, page: Page, base_url: str):
        queries = _mock(page)
        page.goto(base_url)
        page.wait_for_load_state("networkidle")
        page.locator(".main-tab-button[data-tab='tab-nvr']").click()
        page.locator(".workspace-tab-button[data-tab='tab-nvr-events']").click()
        page.wait_for_selector("[data-testid='event-search']")

        wait_until(page, lambda: bool(queries))
        assert queries[-1].get("start_from"), "the first search must already be bounded"
        assert page.locator("[data-testid='event-filter-range-select']").input_value() == "7"
        # The dates are the visible truth about what is filtered; a preset that narrowed
        # the search without filling them in would be an invisible filter.
        assert page.locator("[data-testid='event-filter-from']").input_value()
        assert page.locator("[data-testid='event-filter-to']").input_value()

    def test_all_time_removes_the_bound_entirely(self, page: Page, base_url: str):
        queries = _mock(page)
        _open_events_pane(page, base_url, date_range="7")

        with _issues_a_search(page, queries):
            page.locator("[data-testid='event-filter-range-select']").select_option("")

        assert "start_from" not in queries[-1]
        assert page.locator("[data-testid='event-filter-from']").input_value() == ""

    def test_stepping_back_moves_the_window_by_its_own_length(self, page: Page, base_url: str):
        """Abutting windows with no gap and no overlap is the whole point of the step —
        a fixed step would double-count or skip days whenever the range was not that size.
        """
        queries = _mock(page)
        _open_events_pane(page, base_url, date_range="7")
        before_from = page.locator("[data-testid='event-filter-from']").input_value()
        before_to = page.locator("[data-testid='event-filter-to']").input_value()

        with _issues_a_search(page, queries):
            page.locator("[data-testid='event-filter-range-prev']").click()

        after_from = page.locator("[data-testid='event-filter-from']").input_value()
        after_to = page.locator("[data-testid='event-filter-to']").input_value()
        assert (date.fromisoformat(before_from) - date.fromisoformat(after_from)).days == 7
        assert (date.fromisoformat(before_to) - date.fromisoformat(after_to)).days == 7
        # The new window ends the day before the old one starts: contiguous, not overlapping.
        assert (date.fromisoformat(before_from) - date.fromisoformat(after_to)).days == 1

    def test_stepping_back_then_forward_returns_to_where_it_started(self, page: Page, base_url: str):
        queries = _mock(page)
        _open_events_pane(page, base_url, date_range="7")
        original = page.locator("[data-testid='event-filter-from']").input_value()

        with _issues_a_search(page, queries):
            page.locator("[data-testid='event-filter-range-prev']").click()
        with _issues_a_search(page, queries):
            page.locator("[data-testid='event-filter-range-next']").click()

        assert page.locator("[data-testid='event-filter-from']").input_value() == original

    def test_stepping_off_a_preset_stops_calling_it_that(self, page: Page, base_url: str):
        """ "Last 7 days" names a window ending today. The week before last is not that,
        and a label still claiming it would misreport what is on screen."""
        queries = _mock(page)
        _open_events_pane(page, base_url, date_range="7")

        with _issues_a_search(page, queries):
            page.locator("[data-testid='event-filter-range-prev']").click()

        assert page.locator("[data-testid='event-filter-range-select']").input_value() == "custom"

    def test_there_is_no_stepping_into_the_future(self, page: Page, base_url: str):
        """A window that has not happened yet can only ever be empty."""
        _mock(page)
        _open_events_pane(page, base_url, date_range="7")

        expect(page.locator("[data-testid='event-filter-range-next']")).to_be_disabled()

    def test_the_steppers_are_dead_with_no_window_to_move(self, page: Page, base_url: str):
        _mock(page)
        _open_events_pane(page, base_url, date_range="")

        expect(page.locator("[data-testid='event-filter-range-prev']")).to_be_disabled()
        expect(page.locator("[data-testid='event-filter-range-next']")).to_be_disabled()

    def test_typing_a_date_by_hand_drops_the_preset(self, page: Page, base_url: str):
        queries = _mock(page)
        _open_events_pane(page, base_url, date_range="7")

        with _issues_a_search(page, queries):
            page.locator("[data-testid='event-filter-from']").fill("2026-08-01")

        assert page.locator("[data-testid='event-filter-range-select']").input_value() == "custom"

    def test_clearing_returns_to_the_range_the_page_opens_on(self, page: Page, base_url: str):
        """Not to all of history: Clear should leave the page as a reload would, and
        silently widening to every event ever recorded is a bigger change than the button
        appears to offer."""
        queries = _mock(page)
        _open_events_pane(page, base_url, date_range="")
        assert "start_from" not in queries[-1]

        with _issues_a_search(page, queries):
            page.locator("[data-testid='event-filter-clear']").click()

        assert queries[-1].get("start_from")
        assert page.locator("[data-testid='event-filter-range-select']").input_value() == "7"


def _named_groups() -> list:
    """Four identities whose alphabetical order is the reverse of their size order.

    Deliberately adversarial: if the two orders agreed, a sort control that did nothing
    would pass every assertion below.
    """
    spec = [
        ("Zara", "label-zara", "rule-water", "cat drinking water", 4),
        ("Mittens", "label-mittens", "rule-water", "cat drinking water", 3),
        ("Bella", "label-bella", "rule-litter", "cat using litter box", 2),
        ("Alfie", "label-alfie", "rule-litter", "cat using litter box", 1),
    ]
    rows = []
    for i, (subject, subject_ref, rule_uuid, rule_name, count) in enumerate(spec):
        for n in range(count):
            rows.append(
                _event(
                    uuid=f"{subject_ref}-{n}",
                    ruleUuid=rule_uuid,
                    ruleName=rule_name,
                    subjectRef=subject_ref,
                    subjectName=subject,
                    targetRef=None,
                    targetName=None,
                    startedAt=f"2026-08-18T{9 + i:02d}:{10 + n:02d}:00",
                    endedAt=f"2026-08-18T{9 + i:02d}:{10 + n:02d}:20",
                    recordings=[],
                )
            )
    return rows


def _headlines(page: Page) -> list:
    return [h.strip() for h in page.locator("[data-testid='event-group-card-headline']").all_text_contents()]


class TestSortingTheGroupedViews:
    """Biggest-first answers "what dominates my day"; it is useless for finding a
    particular cat among twenty. Each assertion compares two orders of the same rows."""

    def test_the_default_is_biggest_first(self, page: Page, base_url: str):
        _mock(page, events=_named_groups())
        _open_events_pane(page, base_url, view="groups")

        expect(page.locator("[data-testid='event-group-card']")).to_have_count(4)
        assert _headlines(page) == ["Zara", "Mittens", "Bella", "Alfie"]

    def test_sorting_by_name_is_alphabetical_not_by_size(self, page: Page, base_url: str):
        _mock(page, events=_named_groups())
        _open_events_pane(page, base_url, view="groups")

        page.locator("[data-testid='event-sort-by']").select_option("name")

        wait_until(page, lambda: _headlines(page) == ["Alfie", "Bella", "Mittens", "Zara"])

    def test_sorting_by_event_type_groups_the_rules_together(self, page: Page, base_url: str):
        """Rules contiguous and A–Z, and *within* a rule alphabetical by name too — the
        `rule` comparator falls back to `byName` so the order is total, which is what
        makes it stable between two renders rather than dependent on arrival order.

        ⚠️ The expected order must differ from the biggest-first default, asserted below.
        This test previously expected the default order, so `wait_until` was satisfied by
        the pre-sort DOM on its first poll and the test passed no matter what the sort
        did — it only surfaced once the re-fold became synchronous and beat the poll.
        """
        _mock(page, events=_named_groups())
        _open_events_pane(page, base_url, view="groups")
        default_order = ["Zara", "Mittens", "Bella", "Alfie"]
        expect(page.locator("[data-testid='event-group-card']")).to_have_count(4)
        assert _headlines(page) == default_order

        page.locator("[data-testid='event-sort-by']").select_option("rule")

        by_rule = ["Mittens", "Zara", "Alfie", "Bella"]
        assert by_rule != default_order, "the fixture must make this sort observable"
        wait_until(page, lambda: _headlines(page) == by_rule)
        rules = [r.strip() for r in page.locator(".event-group-card__rule").all_text_contents()]
        assert rules == sorted(rules), "rows of one event type must be contiguous and ordered"

    def test_the_sort_survives_a_change_of_grain(self, page: Page, base_url: str):
        """The two controls are independent; re-folding must not silently revert the
        order to the fold's own deterministic base."""
        _mock(page, events=_named_groups())
        _open_events_pane(page, base_url, view="groups")
        page.locator("[data-testid='event-sort-by']").select_option("name")
        wait_until(page, lambda: _headlines(page) == ["Alfie", "Bella", "Mittens", "Zara"])

        page.locator("[data-testid='event-group-by']").select_option("event")

        wait_until(page, lambda: _headlines(page) == ["cat drinking water", "cat using litter box"])

    def test_the_timeline_lanes_follow_the_same_order(self, page: Page, base_url: str):
        """The lanes and the cards are the same rows; two views of one list disagreeing
        about its order is the bug this shares a code path to avoid."""
        _mock(page, events=_named_groups())
        _open_events_pane(page, base_url, view="timeline")
        page.locator("[data-testid='event-sort-by']").select_option("name")

        wait_until(
            page,
            lambda: [
                x.strip()
                for x in page.locator(
                    "[data-testid='event-timeline-label'] .event-timeline__label-primary"
                ).all_text_contents()
            ]
            == ["Alfie", "Bella", "Mittens", "Zara"],
        )

    def test_sorting_is_not_offered_where_it_would_mean_nothing(self, page: Page, base_url: str):
        """The chronological view is ordered by the server, newest first — the only order
        a log of events has."""
        _mock(page, events=_named_groups())
        _open_events_pane(page, base_url, view="all")

        expect(page.locator("[data-testid='event-sort-by']")).to_be_hidden()


class TestTheGroupedViewDoesNotDim:
    """Dimming earns its place only where one identity has several elements among others.

    The grouped view draws exactly one card per identity, so dimming would grey out every
    other card to point at the one already under the pointer. These assertions are the
    reason that stays true through a future CSS tidy-up.
    """

    def test_hovering_a_group_card_leaves_the_others_fully_visible(self, page: Page, base_url: str):
        _mock(page, events=_two_identities())
        _open_events_pane(page, base_url, view="groups")
        expect(page.locator("[data-testid='event-group-card']")).to_have_count(2)

        page.locator("[data-testid='event-group-card']").first.hover()

        # The focus really is engaged — without this the test would pass just as well
        # against a page where hovering did nothing at all.
        expect(page.locator("[data-testid='event-group-card'].is-focused")).to_have_count(1)
        assert _opacity(page, "[data-testid='event-group-card']", 1) == 1.0

    def test_the_flat_grid_still_dims(self, page: Page, base_url: str):
        """The other half of the rule: removing the dim from group cards must not remove
        it from the view whose whole point is picking one kind out of a mixed page."""
        _mock(page, events=_two_identities())
        _open_events_pane(page, base_url, view="all")
        expect(page.locator("[data-testid='event-card']")).to_have_count(6)

        page.locator("[data-event-uuid='drink-0000']").hover()

        wait_until(page, lambda: _opacity(page, "[data-event-uuid='litter-0000']") < 1.0)


def _two_label_overlay() -> dict:
    """A sidecar whose frames hold two different labels.

    The multi-object case is the whole reason the correction is per detected label: a
    frame with a misidentified cat *and* a correctly detected bowl must not have both
    rewritten by one choice.
    """
    return _overlay(
        frames=[
            {
                "t": 0.0,
                "boxes": [
                    {"x": 0.1, "y": 0.1, "w": 0.2, "h": 0.2, "label": "cat", "labelUuid": "lbl-1", "conf": 0.8},
                    {"x": 0.5, "y": 0.5, "w": 0.1, "h": 0.1, "label": "cat", "labelUuid": "lbl-1", "conf": 0.7},
                    {"x": 0.7, "y": 0.2, "w": 0.1, "h": 0.1, "label": "bowl", "labelUuid": "lbl-2", "conf": 0.9},
                ],
                "tracks": [],
            },
        ],
    )


def _route_labels(page: Page):
    """Answer `/api/labels`, which the correction dialog needs for its dropdowns."""
    page.route(
        re.compile(r"/api/labels$"),
        lambda r: r.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps(
                {
                    "status": "success",
                    "labels": [
                        {"uuid": "lbl-1", "name": "Tabby", "color": "#fff", "parentUuid": None},
                        {"uuid": "lbl-2", "name": "Water bowl", "color": "#fff", "parentUuid": None},
                        {"uuid": "lbl-3", "name": "Clover", "color": "#fff", "parentUuid": None},
                    ],
                }
            ),
        ),
    )


class TestCorrectingAWrongLabel:
    """ "Wrong label, it was actually this" — the correction half of the training loop.

    The frame is already on screen with the model's boxes drawn on it, so the moment you
    notice the label is wrong is the cheapest possible moment to capture the correction.

    ⚠️ The save itself is not reachable from here. `captureVideoFrame` needs a decoded
    frame (`videoWidth`), and the mocked clip is an empty body that Chromium never
    decodes — which is why every existing save-frame test stops at the button too. What
    these cover is the dialog's behaviour and the remap logic. Everything downstream of
    a confirmed correction — the upload, the flag, and the "review now?" offer that
    follows it — is not covered by any test, here or elsewhere. Covering it needs a real
    decodable fixture (WebM; Chromium has no H.264), not a better assertion.
    """

    def _open_modal(self, page: Page, base_url: str, **mock_kwargs):
        _mock(page, **mock_kwargs)
        _route_labels(page)
        _open_events_pane(page, base_url)
        page.locator("[data-testid='event-card']").first.click()
        page.wait_for_selector("[data-testid='event-detail-modal']")

    def test_a_row_per_detected_label_not_per_box(self, page: Page, base_url: str):
        """Three boxes of two kinds is two decisions, not three: the same label detected
        twice is one thing to correct, and a row per box would ask the same question
        twice and allow two contradictory answers to it."""
        self._open_modal(page, base_url, overlay=_two_label_overlay())

        page.locator("[data-testid='event-detail-relabel-event']").click()

        rows = page.locator("[data-testid='label-correction-row']")
        expect(rows).to_have_count(2)
        # Counted, and biggest first — the cat has two boxes, the bowl one. Both counts
        # are shown because they answer different questions: how insistent the model was,
        # and how much training data correcting it would produce.
        expect(rows.first).to_contain_text("Tabby — 2 boxes in 1 frame")
        expect(rows.nth(1)).to_contain_text("Water bowl — 1 box in 1 frame")

    def test_confirming_is_refused_until_something_is_actually_corrected(self, page: Page, base_url: str):
        """A correction with nothing corrected would flag the event and save a duplicate
        frame for no reason, so the state of the button has to follow the choices."""
        self._open_modal(page, base_url, overlay=_two_label_overlay())
        page.locator("[data-testid='event-detail-relabel-event']").click()
        confirm = page.locator("[data-testid='label-correction-confirm']")
        expect(confirm).to_be_disabled()

        page.locator("[data-testid='label-correction-select']").first.select_option("lbl-3")

        expect(confirm).to_be_enabled()

    def test_a_label_is_not_offered_as_its_own_correction(self, page: Page, base_url: str):
        """ "Tabby should have been Tabby" is not a correction, and offering it invites a
        confirm that does nothing but flag the event."""
        self._open_modal(page, base_url, overlay=_two_label_overlay())

        page.locator("[data-testid='event-detail-relabel-event']").click()

        options = page.locator("[data-testid='label-correction-select']").first.locator("option")
        assert "Tabby" not in options.all_text_contents()
        assert "Clover" in options.all_text_contents()

    def test_a_clip_with_no_sidecar_offers_no_relabel_at_all(self, page: Page, base_url: str):
        """Without per-frame boxes there is nothing to relabel: the event's own subject
        label describes one analysed instant and cannot say where in the clip the box
        was. A button that explained that *after* being pressed is worse than absent."""
        self._open_modal(page, base_url, overlay=None)

        expect(page.locator("[data-testid='event-detail-relabel-event']")).to_have_count(0)
        # The frame tools are still there — this is the event action that is missing,
        # not the whole toolbar.
        expect(page.locator("[data-testid='event-detail-save-frame']")).to_be_visible()

    def test_the_event_fix_is_offered_only_when_it_would_change_the_event(self, page: Page, base_url: str):
        """Correcting the boxes and leaving the row saying "Tabby drinks water" is half a
        correction — the row is what every list, group and filter answers with.

        The conditional is the test: the offer appears when the corrected label *is* the
        event's own subject, and stays away when it is the bowl standing beside her,
        where "also fix this event" would be an offer to change nothing.
        """
        self._open_modal(
            page, base_url, events=[_event(subjectRef="lbl-1", subjectName="Tabby")], overlay=_two_label_overlay()
        )
        page.locator("[data-testid='event-detail-relabel-event']").click()

        fix_row = page.locator("[data-testid='label-correction-fix-event']")
        expect(fix_row).to_be_hidden()

        # The bowl is not this event's subject, so correcting it changes no slot.
        page.locator("[data-testid='label-correction-select']").nth(1).select_option("lbl-3")
        expect(fix_row).to_be_hidden()

        # Tabby is.
        page.locator("[data-testid='label-correction-select']").first.select_option("lbl-3")
        expect(fix_row).to_be_visible()
        expect(fix_row).to_contain_text("subject Tabby → Clover")

    def test_confirming_writes_the_correction_to_the_event_row(self, page: Page, base_url: str):
        """The database fix itself, which is the point of the offer above.

        It is asserted *before* the frame harvest deliberately: the harvest cannot run
        here (the mocked clip never decodes — see the class docstring), and the row
        correction must not be conditional on it. That ordering is the behaviour, not a
        convenience for the test.
        """
        self._open_modal(
            page, base_url, events=[_event(subjectRef="lbl-1", subjectName="Tabby")], overlay=_two_label_overlay()
        )

        patched = []
        page.route(
            re.compile(r"/api/watch-events/[\w-]+$"),
            lambda route: (
                patched.append(json.loads(route.request.post_data or "{}")) if route.request.method == "PATCH" else None
            )
            or route.fulfill(status=200, content_type="application/json", body=json.dumps({"status": "success"})),
        )

        page.locator("[data-testid='event-detail-relabel-event']").click()
        page.locator("[data-testid='label-correction-select']").first.select_option("lbl-3")
        page.locator("[data-testid='label-correction-confirm']").click()

        wait_until(page, lambda: bool(patched), "the correction never reached the event row")
        assert patched[0]["subjectRef"] == "lbl-3"
        # The name travels with the ref: a row that filters as one label and reads as
        # another is worse than either half being wrong on its own.
        assert patched[0]["subjectName"] == "Clover"

    def test_unchecking_the_offer_leaves_the_event_row_alone(self, page: Page, base_url: str):
        """The other half — it is an opt-*out*, so opting out has to work. Someone
        correcting the training data for a run of events they do not want re-grouped
        must be able to say so."""
        self._open_modal(
            page, base_url, events=[_event(subjectRef="lbl-1", subjectName="Tabby")], overlay=_two_label_overlay()
        )

        patched = []
        page.route(
            re.compile(r"/api/watch-events/[\w-]+$"),
            lambda route: (patched.append(route.request.method) if route.request.method == "PATCH" else None)
            or route.fulfill(status=200, content_type="application/json", body=json.dumps({"status": "success"})),
        )

        page.locator("[data-testid='event-detail-relabel-event']").click()
        page.locator("[data-testid='label-correction-select']").first.select_option("lbl-3")
        page.locator("[data-testid='label-correction-fix-event-box']").uncheck()
        page.locator("[data-testid='label-correction-confirm']").click()

        # Barrier on the dialog closing, which the confirm handler does before any of the
        # work — so this waits for the click to have been *processed*, not merely sent.
        expect(page.locator("[data-testid='label-correction-modal']")).to_have_count(0)
        assert patched == [], f"the event row must be left alone, saw {patched}"

    def test_the_correction_touches_only_the_slots_the_remap_names(self, page: Page, base_url: str):
        """Correcting the cat must not rewrite the litter box she was standing in.

        A zone-kind rule's target is a *zone* uuid, so it can never be a key in a remap
        of clip labels — the patch has to come out subject-only. And a remap that names
        neither slot returns null rather than an empty patch, which is what keeps the
        offer hidden instead of showing an offer to change nothing.
        """
        self._open_modal(page, base_url, overlay=_two_label_overlay())

        result = page.evaluate("""async () => {
                const { eventCorrectionFor } =
                  await import('/app-static/js/app/nvr/events/labelCorrectionModal.js');
                const labels = new Map([['lbl-3', { uuid: 'lbl-3', name: 'Clover' }]]);
                const event = { subjectRef: 'lbl-1', subjectName: 'Tabby', targetRef: 'zone-box', targetName: 'litter box' };
                return {
                  patch: eventCorrectionFor(event, new Map([['lbl-1', 'lbl-3']]), labels).patch,
                  untouched: eventCorrectionFor(event, new Map([['lbl-2', 'lbl-3']]), labels),
                };
            }""")

        assert result["patch"] == {"subjectRef": "lbl-3", "subjectName": "Clover"}
        assert result["untouched"] is None

    def test_the_remap_rewrites_only_the_labels_it_was_given(self, page: Page, base_url: str):
        """The multi-object case the per-label form exists for, asserted on the function
        that actually decides it — correcting the cat must leave the bowl alone."""
        self._open_modal(page, base_url, overlay=_two_label_overlay())

        result = page.evaluate("""async () => {
                const { applyLabelRemap } =
                  await import('/app-static/js/app/nvr/events/labelCorrectionModal.js');
                const boxes = [
                  { labelUuid: 'lbl-1', bbox: [0, 0, 1, 1] },
                  { labelUuid: 'lbl-1', bbox: [1, 1, 2, 2] },
                  { labelUuid: 'lbl-2', bbox: [2, 2, 3, 3] },
                ];
                return {
                  remapped: applyLabelRemap(boxes, new Map([['lbl-1', 'lbl-3']])).map(b => b.labelUuid),
                  untouched: boxes.map(b => b.labelUuid),
                };
            }""")

        assert result["remapped"] == ["lbl-3", "lbl-3", "lbl-2"], "only the remapped label may change"
        # The overlay's boxes belong to a running redraw loop; rewriting them in place
        # would change what is painted over the video.
        assert result["untouched"] == ["lbl-1", "lbl-1", "lbl-2"], "the caller's boxes must not be mutated"

    def test_the_tally_counts_boxes_and_frames_separately(self, page: Page, base_url: str):
        """Two boxes in one frame and two boxes across a minute are the same box count
        and very different situations — one yields one training image, the other two."""
        self._open_modal(page, base_url, overlay=_two_label_overlay())

        result = page.evaluate("""async () => {
                const { labelTallyAcrossClip } =
                  await import('/app-static/js/app/nvr/events/labelCorrectionModal.js');
                const doc = { frames: [
                  { t: 0, boxes: [{ labelUuid: 'a' }, { labelUuid: 'a' }, { labelUuid: 'b' }] },
                  { t: 5, boxes: [{ labelUuid: 'a' }] },
                ] };
                const labels = new Map([['a', { uuid: 'a', name: 'Cat' }]]);
                return labelTallyAcrossClip(doc, labels).map(r => [r.name, r.boxes, r.frames]);
            }""")

        assert result[0] == ["Cat", 3, 2], "three boxes, but only two frames to harvest"
        # A label the DB has never heard of still gets a row: it is exactly the kind of
        # stale detection someone wants to correct.
        assert result[1] == ["b", 1, 1]

    def test_the_harvest_is_a_spread_not_the_first_few_frames(self, page: Page, base_url: str):
        """The sampling policy, which is the whole difference between "some subset" and
        "the first two seconds of every event".

        Consecutive frames of a tracked animal are near-duplicates: a hundred of them is
        not a hundred times the signal, it is one pose weighted a hundredfold. So close
        frames are thinned first, and only then is the cap spent — evenly, across the
        whole span, keeping the first and last where the animal enters and leaves.
        """
        # Loaded only for an origin to import the module from; this asserts a pure
        # function, so it needs no mocks and no modal.
        page.goto(base_url)
        page.wait_for_load_state("networkidle")

        result = page.evaluate("""async () => {
                const { pickTrainingFrames } =
                  await import('/app-static/js/app/nvr/events/labelCorrectionModal.js');
                // 0.2s apart: a 5 fps burst, all of it one pose.
                const dense = Array.from({ length: 50 }, (_, i) => +(i * 0.2).toFixed(1));
                return {
                  thinned: pickTrainingFrames([0, 0.1, 0.2, 5], { maxFrames: 8, minGapS: 0.75 }),
                  capped: pickTrainingFrames(dense, { maxFrames: 4, minGapS: 0.75 }),
                  short: pickTrainingFrames([0, 2, 4], { maxFrames: 8, minGapS: 0.75 }),
                };
            }""")

        assert result["thinned"] == [0, 5], "frames inside the gap are the same pose twice"
        capped = result["capped"]
        assert len(capped) == 4, capped
        assert capped[0] == 0, "the start of the visit is where the animal enters"
        assert capped[-1] == 9.8, "and the end is where it leaves"
        assert capped == sorted(capped)
        # Not bunched at the front, which is what taking the first N would give.
        assert capped[1] > 2, capped
        # A short event yields what it has rather than being padded to the cap.
        assert result["short"] == [0, 2, 4]

    def test_the_flagged_events_can_be_found_again(self, page: Page, base_url: str):
        """Flagging is only worth doing if the flagged ones come back. The filter sends
        the review state to the server rather than hiding rows client-side, because the
        grid holds a page and the flagged event is usually not on it."""
        queries = _mock(page, events=_two_identities())
        _route_labels(page)
        _open_events_pane(page, base_url, view="all")

        with _issues_a_search(page, queries):
            page.locator("[data-testid='event-filter-review']").select_option("flagged")

        assert queries[-1].get("review_state") == "flagged"

    def test_any_review_state_is_not_a_filter(self, page: Page, base_url: str):
        """The negative half: "Any review state" must clear the parameter rather than
        sending an empty one, which the backend validates and rejects."""
        queries = _mock(page, events=_two_identities())
        _route_labels(page)
        _open_events_pane(page, base_url, view="all")
        with _issues_a_search(page, queries):
            page.locator("[data-testid='event-filter-review']").select_option("flagged")

        with _issues_a_search(page, queries):
            page.locator("[data-testid='event-filter-review']").select_option("")

        assert queries[-1].get("review_state") is None


class TestTheFoldControlsDoNotRefetch:
    """Grain, sort and split are folds over rows already on the client.

    Each of their handlers says so in a comment — "no request, no round trip" — and each
    of them called `search()`, which re-runs *two* queries. The assertions here are
    paired on purpose: that the view actually changed, and that nothing was fetched to
    change it. Either alone would pass against the bug.
    """

    def test_changing_the_grain_refolds_without_asking_the_server(self, page: Page, base_url: str):
        queries = _mock(page, events=_one_rule_two_actors())
        _open_events_pane(page, base_url, view="groups")
        expect(page.locator("[data-testid='event-group-card']")).to_have_count(3)
        before = len(queries)

        page.locator("[data-testid='event-group-by']").select_option("event")

        expect(page.locator("[data-testid='event-group-card']")).to_have_count(1)
        assert len(queries) == before, queries[before:]

    def test_changing_the_sort_reorders_without_asking_the_server(self, page: Page, base_url: str):
        queries = _mock(page, events=_two_identities())
        _open_events_pane(page, base_url, view="groups")
        counts = page.locator("[data-testid='event-group-card-count']")
        expect(counts).to_have_text(["4", "2"])
        before = len(queries)

        # Clover (2) sorts before Tabby (4), so an alphabetical order is visibly not the
        # count order it replaces — a sort that silently did nothing cannot pass this.
        page.locator("[data-testid='event-sort-by']").select_option("name")

        expect(counts).to_have_text(["2", "4"])
        assert len(queries) == before, queries[before:]


class TestALaneLabelDrillsDownWithoutPinning:
    def test_clicking_a_lane_label_does_not_leave_the_page_dimmed(self, page: Page, base_url: str):
        """The lane label carried two click handlers that undid each other: the drill-down
        cleared the focus and the pin, registered second, immediately restored it. The
        user landed in the flat grid with every non-matching tile dimmed and no control
        left on screen to unpin — the pin's only trigger was the lane they had just
        navigated away from.
        """
        _mock(page, events=_two_identities())
        _open_events_pane(page, base_url, view="timeline")
        expect(page.locator("[data-testid='event-timeline-label']")).to_have_count(2)

        page.locator("[data-testid='event-timeline-label']").first.click()

        expect(page.locator("[data-testid='event-view-all']")).to_have_class(re.compile("event-search__view--active"))
        expect(page.locator("[data-testid='event-search']")).not_to_have_class(re.compile("event-focus-pinned"))

    def test_the_lane_strip_still_pins(self, page: Page, base_url: str):
        """The pin did not merely get deleted — it moved to the element that can hold it.
        Touch has no hover, so without a sticky focus the lanes cannot be read at all."""
        _mock(page, events=_two_identities())
        _open_events_pane(page, base_url, view="timeline")
        expect(page.locator("[data-testid='event-timeline-lane']")).to_have_count(2)

        # The strip *between* the marks, not a mark: marks stop propagation and open their
        # own bin. The busiest lane leads, and its events are all in the earliest bucket,
        # so its mark sits at the left and the right-hand end of the track is bare.
        track = page.locator(".event-timeline__track").first
        box = track.bounding_box()
        track.click(position={"x": box["width"] * 0.9, "y": box["height"] / 2})

        expect(page.locator("[data-testid='event-search']")).to_have_class(re.compile("event-focus-pinned"))
        # Still on the timeline — pinning is a highlight, not a navigation.
        expect(page.locator("[data-testid='event-timeline']")).to_be_attached()


class TestAFacetFailureDoesNotClearTheFilters:
    def test_a_failed_facet_refresh_keeps_the_selection(self, page: Page, base_url: str):
        """`refreshFacets` runs fire-and-forget after every delete. Blanking the facet
        lists on failure looks conservative and is not: the controls prune the selection
        against the option list, so four empty lists clear all four filters — turning a
        drill-down showing a handful of events into an unfiltered grid of everything,
        with only a console line to say why.
        """
        _mock(
            page,
            events=_two_identities(),
            deletes=[],
            facets=_facets(
                rules=[
                    {"ref": "rule-water", "name": "cat drinking water"},
                    {"ref": "rule-litter", "name": "cat using litter box"},
                ]
            ),
        )
        _open_events_pane(page, base_url, view="all")
        _pick_facet(page, "event-filter-rule", "cat drinking water")
        trigger = page.locator("[data-testid='event-filter-rule'] .ms-dropdown__trigger")
        expect(trigger).to_contain_text("cat drinking water")

        # Registered after _mock's, and so matched first: the facet refresh that follows
        # the delete now fails.
        page.route(re.compile(r"/api/watch-events/facets"), lambda route: route.abort())

        page.locator("[data-testid='event-card-select']").nth(0).check()
        page.locator("[data-testid='event-search-delete-selected']").click()
        page.get_by_role("button", name="Delete", exact=True).click()
        expect(page.locator("[data-testid='event-card']")).to_have_count(5)

        expect(trigger).to_contain_text("cat drinking water")
        expect(trigger).not_to_contain_text("Any event type")
