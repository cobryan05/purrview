"""
Annotated thumbnails on the NVR → Live cards (app/nvr/cardPreview.js).

The Live pane answers "is the house being watched?", and text alone cannot: a card
reading "running · 3 rules · 1 fps" looks identical whether the camera is watching the
litter box or has been pointing at a wall since Tuesday. So what is tested here is that
the card carries a *frame*, that its caption reports what the watcher currently sees,
and that the strip appears only when there is something to show.

The other property under test is cost. These thumbnails are an overview of every watcher
at once, so they refresh on a slow timer, and a card that has been re-rendered away must
not keep polling — a leaked timer is invisible until the pane has been opened ten times.

Run with:
    pytest playwright_tests/test_nvr_card_preview.py -p no:asyncio -v
"""

import json

from playwright.sync_api import Page, expect

from playwright_tests.test_nvr_live_view import PROFILE_UUID, TASK_UUID, _preview, _profile, _task

SECOND_SOURCE = {"sourceUuid": "src-2", "sourceName": "Garden cam"}


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
    page.wait_for_selector("#nvr-live-box .nvr-profile-card", timeout=5000)


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

    def test_a_source_still_connecting_is_not_reported_as_dead(self, page: Page, base_url: str):
        """Coming up is expected and bounded. Shouting about it is how a health signal
        teaches people to ignore it."""
        _mock(page, preview=_preview(frame=None, frameAgeS=None, sourceState="connecting", downtimeS=40.0))
        _open_live_pane(page, base_url)

        expect(page.locator("[data-testid='nvr-preview-stat']").first).to_contain_text("connecting")
        expect(page.locator(".nvr-preview-tile--stale")).to_have_count(0)
        expect(page.locator(".nvr-preview-tile--connecting")).to_have_count(1)


class TestTheCardCountsItsCameras:
    """Only four sources get thumbnails, so on a bigger profile "no red tiles" stops
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
        """The gap this closes: source six has no thumbnail, so before the counter
        existed its being dark was not represented anywhere on the card."""
        sources = [{"sourceUuid": f"src-{i}", "sourceName": f"Cam {i}"} for i in range(1, 7)]
        sources[5]["health"] = "failed — no frames for 1h 2m, still retrying"
        _mock(page, profiles=[_multi_source_profile()], sources=sources)
        _open_live_pane(page, base_url)

        expect(page.locator("[data-testid='nvr-source-count']")).to_have_text("5/6 live")
        expect(page.locator("[data-testid='nvr-offline-pill']")).to_contain_text("Cam 6")
        # And it is genuinely one of the untiled ones.
        expect(page.locator("[data-testid='nvr-preview-tile']")).to_have_count(4)

    def test_a_stopped_watcher_counts_nothing(self, page: Page, base_url: str):
        _mock(page, task_status="completed")
        _open_live_pane(page, base_url)

        expect(page.locator("[data-testid='nvr-source-count']")).to_be_hidden()


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

    def test_only_a_subset_of_many_sources_is_shown(self, page: Page, base_url: str):
        """A dozen postage stamps answers nothing; the rest are counted instead."""
        sources = [{"sourceUuid": f"src-{i}", "sourceName": f"Cam {i}"} for i in range(1, 7)]
        _mock(page, profiles=[_multi_source_profile()], sources=sources)
        _open_live_pane(page, base_url)

        expect(page.locator("[data-testid='nvr-preview-tile']")).to_have_count(4)
        expect(page.locator(".nvr-preview-more")).to_contain_text("+2 more")


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

        assert len(calls) == 2, f"expected one frame per source, got {len(calls)}: {calls}"
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
        page.wait_for_selector("[data-testid='nvr-preview-tile']", timeout=5000)

        for _ in range(3):
            page.locator(".main-tab-button[data-tab='tab-runs']").click()
            page.locator(".main-tab-button[data-tab='tab-nvr']").click()
            page.wait_for_selector("[data-testid='nvr-preview-tile']", timeout=5000)

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
            const stop = m.attachCardPreviews(host, {
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
        page.wait_for_selector("[data-testid='nvr-preview-tile']", timeout=5000)
        before = len(calls)

        page.locator(".main-tab-button[data-tab='tab-runs']").click()
        page.locator(".main-tab-button[data-tab='tab-nvr']").click()
        page.wait_for_selector("[data-testid='nvr-preview-tile']", timeout=5000)
        page.wait_for_function("count => window.__nvrPreviewCalls === undefined || true", arg=before, timeout=1000)
        page.wait_for_timeout(500)

        assert len(calls) > before, "no frame was fetched after returning to the NVR tab"
