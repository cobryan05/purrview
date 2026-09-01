"""
Annotated thumbnails on the NVR → Live cards (app/nvr/cardPreview.js).

The Live pane answers "is the house being watched?", and text alone cannot: a card
reading "running · 3 rules · 1 fps" looks identical whether the camera is watching the
litter box or has been pointing at a wall since Tuesday. So what is tested here is that
the card carries a *frame*, that its caption reports what the watcher currently sees,
and that the strip appears only when there is something to show.

The other property under test is cost. These thumbnails are an overview of every watcher
at once, so each card polls on its own timer at a rate its dropdown controls, and a card
that has been re-rendered away must not keep polling — a leaked timer is invisible until
the pane has been opened ten times.

Run with:
    pytest playwright_tests/test_nvr_card_preview.py -p no:asyncio -v
"""

import json

from playwright.sync_api import Page, expect

from playwright_tests.test_nvr_live_view import PROFILE_UUID, TASK_UUID, _preview, _profile, _task
from playwright_tests.waits import wait_until

SECOND_SOURCE = {"sourceUuid": "src-2", "sourceName": "Garden cam"}

# One page holds MAX_TILES (9) tiles, so anything testing paging needs more than that.
TILES_PER_PAGE = 9


def _cams(count: int) -> list:
    return [{"sourceUuid": f"src-{i}", "sourceName": f"Cam {i}"} for i in range(1, count + 1)]


def _multi_source_profile() -> dict:
    """A profile whose second source is enabled — the wide-card case."""
    profile = _profile()
    second = dict(profile["sources"][0])
    second.update({"uuid": "ps-2", "sourceUuid": "src-2", "order": 1})
    profile["sources"] = [profile["sources"][0], second]
    return profile


def _status_data(sources) -> dict:
    """The watcher's status payload — where the card gets its source *names*.

    Deliberately not the live-preview endpoint: that one encodes a JPEG to answer, so
    using it for names alone would cost a frame per card on every render.
    """
    return {
        "status": "success",
        "statusData": {
            TASK_UUID: {
                "stats": {},
                "sources": [
                    {
                        "source": s["sourceName"],
                        "sourceUuid": s["sourceUuid"],
                        "tracks": 0,
                        # The card's counter reads this for *every* source, including the
                        # ones past MAX_TILES that never get a thumbnail.
                        "health": s.get("health", "ok"),
                    }
                    for s in sources
                ],
                "activeTracksTable": [],
                "openEventsTable": [],
            }
        },
    }


def _mock(page: Page, *, profiles=None, preview=None, task_status="running", count_previews=False, sources=None):
    """Wire the pane's endpoints. Returns a list that collects preview URLs."""
    calls = []
    sources = sources if sources is not None else [{"sourceUuid": "src-1", "sourceName": "Litter cam"}]

    page.route(
        lambda url: url.split("?")[0].rstrip("/").endswith("/api/tasks/instances/status-data"),
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps(_status_data(sources)),
        ),
    )
    page.route(
        lambda url: url.split("?")[0].rstrip("/").endswith("/api/watch-profiles"),
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({"status": "success", "profiles": profiles or [_profile()]}),
        ),
    )
    tasks = {} if task_status is None else {TASK_UUID: _task(task_status)}
    page.route(
        lambda url: url.split("?")[0].rstrip("/").endswith("/api/tasks/instances/get"),
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({"status": "success", "tasks": tasks}),
        ),
    )

    def _fulfil_preview(route):
        if count_previews:
            calls.append(route.request.url)
        route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({"status": "success", "preview": preview if preview is not None else _preview()}),
        )

    page.route(lambda url: "/live-preview" in url, _fulfil_preview)
    return calls


def _open_live_pane(page: Page, base_url: str):
    page.goto(base_url)
    page.wait_for_load_state("networkidle")
    page.locator(".main-tab-button[data-tab='tab-nvr']").click()
    page.wait_for_selector("#nvr-live-box .nvr-profile-card")


class TestCardPreview:
    def test_a_running_watcher_card_shows_a_frame(self, page: Page, base_url: str):
        _mock(page)
        _open_live_pane(page, base_url)

        tile = page.locator("[data-testid='nvr-preview-tile']").first
        expect(tile).to_be_visible()
        expect(tile.locator("canvas")).to_be_visible()

    def test_the_tile_is_captioned_with_the_source_name(self, page: Page, base_url: str):
        """A tile captioned with a uuid is the problem this feature exists to fix."""
        _mock(page)
        _open_live_pane(page, base_url)

        expect(page.locator("[data-testid='nvr-preview-tile']").first).to_contain_text("Litter cam")

    def test_the_caption_reports_what_the_watcher_currently_sees(self, page: Page, base_url: str):
        _mock(page)
        _open_live_pane(page, base_url)

        expect(page.locator("[data-testid='nvr-preview-stat']").first).to_contain_text("1 track")

    def test_an_open_event_is_called_out_on_the_card(self, page: Page, base_url: str):
        """The card should say something is happening without being clicked into."""
        preview = _preview(
            openEvents=[{"ruleUuid": "r1", "ruleName": "Cat in litter box", "state": "active", "trackIds": [3]}],
            tracks=[{**_preview()["tracks"][0], "eventRules": ["Cat in litter box"]}],
        )
        _mock(page, preview=preview)
        _open_live_pane(page, base_url)

        expect(page.locator("[data-testid='nvr-preview-stat']").first).to_contain_text("1 event")
        # The border colour is the at-a-glance signal; the class carries it.
        expect(page.locator(".nvr-preview-tile--event")).to_have_count(1)

    def test_a_stale_frame_is_reported_rather_than_shown_as_normal(self, page: Page, base_url: str):
        """A still picture cannot show that it is frozen — the age has to be spelled out."""
        _mock(page, preview=_preview(frameAgeS=120.0))
        _open_live_pane(page, base_url)

        expect(page.locator("[data-testid='nvr-preview-stat']").first).to_contain_text("no signal")
        expect(page.locator("[data-testid='nvr-preview-stat']").first).to_contain_text("2m")

    def test_a_stopped_watcher_has_no_preview_strip(self, page: Page, base_url: str):
        """A stopped watcher has no frames; a strip of placeholders would look broken."""
        _mock(page, task_status="completed")
        _open_live_pane(page, base_url)

        expect(page.locator(".nvr-profile-card")).to_have_count(1)
        expect(page.locator("[data-testid='nvr-preview-strip']")).to_have_count(0)


class TestADeadSourceIsImpossibleToMiss:
    """A camera that stopped an hour ago must not present as a working one.

    Everything on the card kept saying "fine": the pill said running, the thumbnail
    showed a frame (the last one that ever arrived, with its boxes still drawn on it),
    and the only contradiction was a caption that read "source error" — no duration, no
    colour, smaller than the source name next to it. A source went dark for most of a
    run without anything drawing the eye to it.
    """

    def test_a_dead_source_marks_its_tile(self, page: Page, base_url: str):
        _mock(page, preview=_preview(frameAgeS=3600.0, sourceState="failed", downtimeS=3600.0))
        _open_live_pane(page, base_url)

        expect(page.locator(".nvr-preview-tile--stale")).to_have_count(1)

    def test_the_duration_survives_a_last_error(self, page: Page, base_url: str):
        """The regression. `lastError` used to short-circuit the caption to a bare
        "source error", so a camera down for an hour read exactly like one that
        missed a single frame — and the duration is the entire difference."""
        _mock(
            page,
            preview=_preview(
                frameAgeS=3600.0,
                downtimeS=3600.0,
                sourceState="failed",
                lastError="source returned no frame",
            ),
        )
        _open_live_pane(page, base_url)

        stat = page.locator("[data-testid='nvr-preview-stat']").first
        expect(stat).to_contain_text("1h")
        expect(stat).to_contain_text("no signal")

    def test_a_dead_source_does_not_keep_drawing_event_boxes(self, page: Page, base_url: str):
        """The tracks belong to the frozen frame. Highlighting them asserts the watcher
        is reacting to something it stopped seeing an hour ago."""
        preview = _preview(
            frameAgeS=3600.0,
            sourceState="failed",
            downtimeS=3600.0,
            openEvents=[{"ruleUuid": "r1", "ruleName": "Cat in litter box", "state": "active", "trackIds": [3]}],
        )
        _mock(page, preview=preview)
        _open_live_pane(page, base_url)

        expect(page.locator(".nvr-preview-tile--stale")).to_have_count(1)
        expect(page.locator(".nvr-preview-tile--event")).to_have_count(0)

    def test_the_card_names_the_offline_camera(self, page: Page, base_url: str):
        """A pill on the card, next to the green "running" it contradicts — the strip
        can be scrolled past, and "1 source offline" would send you hunting for which."""
        _mock(page, preview=_preview(frameAgeS=3600.0, sourceState="failed", downtimeS=3600.0))
        _open_live_pane(page, base_url)

        pill = page.locator("[data-testid='nvr-offline-pill']")
        expect(pill).to_be_visible()
        expect(pill).to_contain_text("Litter cam")

    def test_a_healthy_card_carries_no_alarm(self, page: Page, base_url: str):
        _mock(page)
        _open_live_pane(page, base_url)

        expect(page.locator("[data-testid='nvr-preview-tile']").first).to_be_visible()
        expect(page.locator("[data-testid='nvr-offline-pill']")).to_be_hidden()
        expect(page.locator(".nvr-preview-tile--stale")).to_have_count(0)

    def test_a_slow_camera_within_its_own_cadence_is_not_reported_as_dead(self, page: Page, base_url: str):
        """The Ring regression.

        A 30 s-old frame is a fault on an RTSP stream and a normal Tuesday on a snapshot
        camera, and the card judged both against one flat 15 s threshold — so a healthy
        Ring camera showed NO SIGNAL across its picture for most of every cycle, which is
        precisely how a card teaches its owner to ignore a real outage. `frameDueInS` is
        the provider's own deadline and it settles the question; the drill-down's status
        line had this right first.
        """
        _mock(page, preview=_preview(frameAgeS=30.0, frameDueInS=60.0))
        _open_live_pane(page, base_url)

        expect(page.locator("[data-testid='nvr-preview-stat']").first).to_contain_text("1 track")
        expect(page.locator(".nvr-preview-tile--stale")).to_have_count(0)
        expect(page.locator("[data-testid='nvr-offline-pill']")).to_be_hidden()

    def test_a_slow_camera_past_its_own_deadline_still_is(self, page: Page, base_url: str):
        """The other half — otherwise the fix would simply have switched the alarm off.
        The same age, and the only difference is that the provider's deadline has run
        out."""
        _mock(page, preview=_preview(frameAgeS=30.0, frameDueInS=0.0))
        _open_live_pane(page, base_url)

        expect(page.locator("[data-testid='nvr-preview-stat']").first).to_contain_text("no signal")
        expect(page.locator(".nvr-preview-tile--stale")).to_have_count(1)

    def test_a_source_still_connecting_is_not_reported_as_dead(self, page: Page, base_url: str):
        """Coming up is expected and bounded. Shouting about it is how a health signal
        teaches people to ignore it."""
        _mock(page, preview=_preview(frame=None, frameAgeS=None, sourceState="connecting", downtimeS=40.0))
        _open_live_pane(page, base_url)

        expect(page.locator("[data-testid='nvr-preview-stat']").first).to_contain_text("connecting")
        expect(page.locator(".nvr-preview-tile--stale")).to_have_count(0)
        expect(page.locator(".nvr-preview-tile--connecting")).to_have_count(1)


class TestTheCardCountsItsCameras:
    """Only one page of sources gets thumbnails, so on a bigger profile "no red tiles" stops
    meaning "no problems" — the strip is showing a subset and nothing says so. The
    counter is over every source the watcher has, tiled or not, and it reads its health
    from the status payload the pane already fetches."""

    def test_a_healthy_card_counts_all_its_cameras(self, page: Page, base_url: str):
        _mock(page)
        _open_live_pane(page, base_url)

        expect(page.locator("[data-testid='nvr-source-count']")).to_have_text("1/1 live")

    def test_a_dead_camera_is_missing_from_the_live_count(self, page: Page, base_url: str):
        _mock(page, preview=_preview(frameAgeS=3600.0, sourceState="failed", downtimeS=3600.0))
        _open_live_pane(page, base_url)

        expect(page.locator("[data-testid='nvr-source-count']")).to_have_text("0/1 live")

    def test_a_source_with_no_tile_still_reaches_the_count(self, page: Page, base_url: str):
        """The gap this closes: the last camera is on page two, so before the counter
        existed its being dark was not represented anywhere on the visible card."""
        sources = _cams(12)
        sources[11]["health"] = "failed — no frames for 1h 2m, still retrying"
        _mock(page, profiles=[_multi_source_profile()], sources=sources)
        _open_live_pane(page, base_url)

        expect(page.locator("[data-testid='nvr-source-count']")).to_have_text("11/12 live")
        expect(page.locator("[data-testid='nvr-offline-pill']")).to_contain_text("Cam 12")
        # And it is genuinely one of the untiled ones.
        expect(page.locator("[data-testid='nvr-preview-tile']")).to_have_count(TILES_PER_PAGE)

    def test_a_stopped_watcher_counts_nothing(self, page: Page, base_url: str):
        _mock(page, task_status="completed")
        _open_live_pane(page, base_url)

        expect(page.locator("[data-testid='nvr-source-count']")).to_be_hidden()


class TestTheCardChoosesItsOwnRefreshRate:
    """Per-card, because the right rate is per-camera: the doorway wants a picture
    every couple of seconds and the shed does not, and one global setting makes the
    whole pane pay for the fastest camera on it."""

    def test_turning_the_rate_up_speeds_the_polling(self, page: Page, base_url: str):
        """The dropdown has to reach the running timers. A version that only stored the
        choice would look completely correct and change nothing.

        Deliberately measured *upwards*: the default is 0.2 fps and the slowest option
        0.1 fps, so over any window short enough to be worth waiting for, an ignored
        dropdown and an honoured one are indistinguishable in that direction.
        """
        calls = _mock(page, count_previews=True)
        _open_live_pane(page, base_url)
        page.wait_for_selector("[data-testid='nvr-preview-tile']")

        page.locator("[data-testid='nvr-refresh-rate']").select_option("500")
        before = len(calls)

        # A rate test needs a *window*, so the wait stays time-bounded — but polling for
        # the third call rather than sleeping the whole window means the fast path
        # returns immediately and the slow path still fails honestly. The window is loose
        # on purpose: timer scheduling under a loaded browser is not exact, and it is the
        # order of magnitude that separates 2 fps from the 0.2 fps default (which would
        # need ~15s to manage three).
        wait_until(
            page,
            lambda: len(calls) - before >= 3,
            lambda: f"the fast rate was ignored: {len(calls) - before} polls in 3s",
            timeout=3000,
        )

    def test_the_chosen_rate_survives_a_re_render(self, page: Page, base_url: str):
        """The pane rebuilds every card whenever any watcher starts or stops, so a rate
        held only in the card's closure would silently snap back to the default while
        the user was looking at something else."""
        _mock(page)
        _open_live_pane(page, base_url)
        page.wait_for_selector("[data-testid='nvr-preview-tile']")

        page.locator("[data-testid='nvr-refresh-rate']").select_option("1000")

        page.locator(".main-tab-button[data-tab='tab-runs']").click()
        page.locator(".main-tab-button[data-tab='tab-nvr']").click()
        page.wait_for_selector("[data-testid='nvr-preview-tile']")

        expect(page.locator("[data-testid='nvr-refresh-rate']")).to_have_value("1000")

    def test_a_stopped_watcher_offers_no_rate(self, page: Page, base_url: str):
        """There is no strip to refresh — a dropdown over nothing is a control that
        cannot do anything."""
        _mock(page, task_status="completed")
        _open_live_pane(page, base_url)

        expect(page.locator("[data-testid='nvr-refresh-rate']")).to_have_count(0)


class TestMultiSourceCards:
    def test_each_source_gets_its_own_tile(self, page: Page, base_url: str):
        _mock(
            page,
            profiles=[_multi_source_profile()],
            sources=[{"sourceUuid": "src-1", "sourceName": "Litter cam"}, SECOND_SOURCE],
        )
        _open_live_pane(page, base_url)

        expect(page.locator("[data-testid='nvr-preview-tile']")).to_have_count(2)
        expect(page.locator("[data-testid='nvr-preview-tile']").last).to_contain_text("Garden cam")

    def test_a_multi_source_profile_gets_the_wide_card(self, page: Page, base_url: str):
        """The only thing that justifies the extra width: thumbnails side by side."""
        _mock(
            page,
            profiles=[_multi_source_profile()],
            sources=[{"sourceUuid": "src-1", "sourceName": "Litter cam"}, SECOND_SOURCE],
        )
        _open_live_pane(page, base_url)

        expect(page.locator(".nvr-profile-card--multi-source")).to_have_count(1)

    def test_a_single_source_profile_does_not(self, page: Page, base_url: str):
        _mock(page)
        _open_live_pane(page, base_url)

        expect(page.locator(".nvr-profile-card")).to_have_count(1)
        expect(page.locator(".nvr-profile-card--multi-source")).to_have_count(0)

    def test_clicking_a_tile_opens_that_camera_in_the_live_view(self, page: Page, base_url: str):
        """The second tile, not the first — opening on the watcher's first source is
        exactly what the card-level button already did, and is indistinguishable from
        a click handler that ignores which picture was pressed."""
        _mock(
            page,
            profiles=[_multi_source_profile()],
            sources=[{"sourceUuid": "src-1", "sourceName": "Litter cam"}, SECOND_SOURCE],
            # The modal builds its source list from the preview payload, so both
            # cameras have to be in it or there is nothing to select.
            preview=_preview(sources=[{"sourceUuid": "src-1", "sourceName": "Litter cam"}, SECOND_SOURCE]),
        )
        _open_live_pane(page, base_url)

        page.locator("[data-testid='nvr-preview-tile'][data-source-uuid='src-2']").click()

        expect(page.locator("[data-testid='watch-live-dialog']")).to_be_visible()
        expect(page.locator("[data-testid='watch-live-source']")).to_have_value("src-2")

    def test_only_a_page_of_many_sources_is_shown(self, page: Page, base_url: str):
        """A dozen postage stamps answers nothing; the rest are one page away."""
        _mock(page, profiles=[_multi_source_profile()], sources=_cams(11))
        _open_live_pane(page, base_url)

        expect(page.locator("[data-testid='nvr-preview-tile']")).to_have_count(TILES_PER_PAGE)
        expect(page.locator("[data-testid='nvr-preview-page']")).to_have_text("1–9 of 11")
        # The first page is the start of the list, so there is nowhere back to go.
        expect(page.locator("[data-testid='nvr-preview-prev']")).to_be_disabled()

    def test_the_pager_reaches_the_sources_the_first_page_hides(self, page: Page, base_url: str):
        """The whole point of paging over counting: the hidden cameras are reachable."""
        _mock(page, profiles=[_multi_source_profile()], sources=_cams(11))
        _open_live_pane(page, base_url)

        expect(page.locator("[data-testid='nvr-preview-tile'][data-source-uuid='src-10']")).to_have_count(0)
        page.locator("[data-testid='nvr-preview-next']").click()

        expect(page.locator("[data-testid='nvr-preview-tile']")).to_have_count(2)
        expect(page.locator("[data-testid='nvr-preview-tile'][data-source-uuid='src-10']")).to_have_count(1)
        expect(page.locator("[data-testid='nvr-preview-page']")).to_have_text("10–11 of 11")
        # Last page: forward is exhausted, back is not.
        expect(page.locator("[data-testid='nvr-preview-next']")).to_be_disabled()
        expect(page.locator("[data-testid='nvr-preview-prev']")).to_be_enabled()

    def test_a_single_page_profile_gets_no_pager(self, page: Page, base_url: str):
        """Chrome for a page that cannot exist is worse than no chrome."""
        _mock(
            page,
            profiles=[_multi_source_profile()],
            sources=[{"sourceUuid": "src-1", "sourceName": "Litter cam"}, SECOND_SOURCE],
        )
        _open_live_pane(page, base_url)

        expect(page.locator("[data-testid='nvr-preview-tile']")).to_have_count(2)
        expect(page.locator("[data-testid='nvr-preview-next']")).to_have_count(0)
        # Arranging two cameras is still meaningful, so that control stays.
        expect(page.locator("[data-testid='nvr-preview-arrange']")).to_have_count(1)


class TestSourceArranging:
    """Which cameras land on page one is the user's to decide, and must stick.

    The pane rebuilds every card whenever any watcher starts or stops, so an order held
    only in the DOM would be undone by an unrelated watcher — which is the same failure
    the refresh-rate store exists to prevent.
    """

    SOURCES = _cams(11)

    def _promote_from_page_two(self, page: Page):
        """Move Cam 10 (first tile of page 2) one place earlier, into page 1."""
        page.locator("[data-testid='nvr-preview-arrange']").click()
        page.locator("[data-testid='nvr-preview-next']").click()
        page.locator(
            "[data-testid='nvr-preview-tile'][data-source-uuid='src-10'] " "[data-testid='nvr-preview-move-earlier']"
        ).click()

    def test_moving_a_camera_earlier_follows_it_onto_the_previous_page(self, page: Page, base_url: str):
        """A pager that stayed put would make the moved tile appear to vanish."""
        _mock(page, profiles=[_multi_source_profile()], sources=self.SOURCES)
        _open_live_pane(page, base_url)

        self._promote_from_page_two(page)

        expect(page.locator("[data-testid='nvr-preview-page']")).to_have_text("1–9 of 11")
        expect(page.locator("[data-testid='nvr-preview-tile'][data-source-uuid='src-10']")).to_have_count(1)
        # It took the place of the camera it passed, which is now on page two.
        expect(page.locator("[data-testid='nvr-preview-tile'][data-source-uuid='src-9']")).to_have_count(0)

    def test_the_arranged_order_survives_a_reload(self, page: Page, base_url: str):
        _mock(page, profiles=[_multi_source_profile()], sources=self.SOURCES)
        _open_live_pane(page, base_url)
        self._promote_from_page_two(page)

        _open_live_pane(page, base_url)

        # Page one on a fresh load, without arrange mode or the pager being touched.
        expect(page.locator("[data-testid='nvr-preview-tile'][data-source-uuid='src-10']")).to_have_count(1)
        expect(page.locator("[data-testid='nvr-preview-tile'][data-source-uuid='src-9']")).to_have_count(0)

    def test_the_move_buttons_are_hidden_until_arranging(self, page: Page, base_url: str):
        """The card is a wall of pictures first; the buttons sit on top of the frames."""
        _mock(page, profiles=[_multi_source_profile()], sources=self.SOURCES)
        _open_live_pane(page, base_url)

        expect(page.locator("[data-testid='nvr-preview-move-earlier']")).to_have_count(0)
        page.locator("[data-testid='nvr-preview-arrange']").click()
        expect(page.locator("[data-testid='nvr-preview-move-earlier']")).to_have_count(TILES_PER_PAGE)
        # The first source cannot go earlier — a button that no-ops must say so.
        expect(
            page.locator(
                "[data-testid='nvr-preview-tile'][data-source-uuid='src-1'] " "[data-testid='nvr-preview-move-earlier']"
            )
        ).to_be_disabled()


class TestPreviewCost:
    def test_one_frame_request_per_source_on_open(self, page: Page, base_url: str):
        """Source *names* must not cost a frame.

        The live-preview endpoint encodes a JPEG to answer, so asking it for the source
        list — as an earlier version did — meant a wasted encode per card on every
        render. The names come from the status payload, which encodes nothing.
        """
        calls = _mock(
            page,
            profiles=[_multi_source_profile()],
            sources=[{"sourceUuid": "src-1", "sourceName": "Litter cam"}, SECOND_SOURCE],
            count_previews=True,
        )
        _open_live_pane(page, base_url)
        page.wait_for_selector("[data-testid='nvr-preview-tile']", timeout=5000)
        expect(page.locator("[data-testid='nvr-preview-stat']").last).not_to_have_text("…")

        # The stat filling in proves the *status* payload arrived; it says nothing about
        # the frame requests, which are issued separately per tile. Poll for those.
        wait_until(
            page,
            lambda: len(calls) == 2,
            lambda: f"expected one frame per source, got {len(calls)}: {calls}",
        )
        # Every request names its source: a source-less call is the discovery call that
        # is not supposed to exist any more.
        assert all("source_uuid=" in url for url in calls), calls

    def test_leaving_and_returning_does_not_leave_pollers_behind(self, page: Page, base_url: str):
        """Each render must stop the previous render's timers.

        A leaked timer is invisible: the pane looks right and simply polls twice as
        often for every time it has been opened.
        """
        calls = _mock(page, count_previews=True)
        _open_live_pane(page, base_url)
        page.wait_for_selector("[data-testid='nvr-preview-tile']")

        for _ in range(3):
            page.locator(".main-tab-button[data-tab='tab-runs']").click()
            page.locator(".main-tab-button[data-tab='tab-nvr']").click()
            page.wait_for_selector("[data-testid='nvr-preview-tile']")

        # One first frame per render, and no refresh has come due yet (they are 10s
        # apart). More than that means an earlier render's loop is still running.
        assert len(calls) <= 4, f"preview polls piling up across renders: {len(calls)}"


class TestPollScheduling:
    """One poll chain per tile, and it survives leaving the tab.

    Both halves were wrong in opposite directions: schedule() never cleared the
    pending timer, so every visibilitychange forked another chain that polled
    forever alongside the first; and leaving the NVR tab stopped the previews
    without anything restarting them on the way back.
    """

    def test_visibility_changes_do_not_fork_extra_poll_chains(self, page: Page, base_url: str):
        """Each fork costs a server-side JPEG encode per interval, permanently."""
        calls = _mock(page, count_previews=True)
        page.goto(base_url)
        page.wait_for_load_state("networkidle")

        polls = page.evaluate(
            """async (taskUuid) => {
            const m = await import('/app-static/js/app/nvr/cardPreview.js');
            const host = document.createElement('div');
            document.body.appendChild(host);
            const { stop } = m.attachCardPreviews(host, {
              taskUuid,
              sources: [{ sourceUuid: 'src-1', sourceName: 'Litter cam' }],
              refreshMs: 50,
            });
            // Each of these used to start an independent chain on top of the running one.
            for (let i = 0; i < 5; i++) document.dispatchEvent(new Event('visibilitychange'));
            await new Promise(r => setTimeout(r, 600));
            stop();
            host.remove();
            return true;
        }""",
            TASK_UUID,
        )
        assert polls

        # ~12 at one poll per 50ms; six concurrent chains would be ~70. The bound is
        # loose because timer scheduling under a loaded browser is not exact — it is
        # the order of magnitude that distinguishes one chain from six.
        assert len(calls) < 30, f"poll chains multiplied across visibility changes: {len(calls)}"

    def test_returning_to_the_tab_starts_the_thumbnails_again(self, page: Page, base_url: str):
        """Leaving the tab stops every preview; nothing else re-renders the pane on
        the way back, so the tiles used to stay frozen on their last frame until a
        watcher happened to start or stop."""
        calls = _mock(page, count_previews=True)
        _open_live_pane(page, base_url)
        page.wait_for_selector("[data-testid='nvr-preview-tile']")
        before = len(calls)

        page.locator(".main-tab-button[data-tab='tab-runs']").click()
        page.locator(".main-tab-button[data-tab='tab-nvr']").click()
        page.wait_for_selector("[data-testid='nvr-preview-tile']")

        # The `|| true` wait_for_function this replaces resolved on its first evaluation
        # by construction — it was never a barrier — leaving the 500ms sleep as the only
        # thing gating the assertion.
        wait_until(
            page,
            lambda: len(calls) > before,
            "no frame was fetched after returning to the NVR tab",
        )
