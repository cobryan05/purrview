"""Service Status lives in Runs, not Settings.

Decision register #8 moves service health off Settings; UI_REDESIGN §12 calls
its presence there the original miscategorisation (it persists nothing and is
pure live monitoring). It rode along with Activity when Operations dissolved
into Configuration + Runs, which is the surface it belongs on: health sits
beside what is running.
"""

import json

from playwright.sync_api import Page, expect

from playwright_tests.waits import wait_until

SERVICES = [
    {"service": "db_server", "status": "ok", "hash_index": {"image_valid": 42, "needs_rebuild": False}},
    {"service": "inference_server", "status": "ok", "models": {"loaded": 1, "available": 3}},
    {"service": "tasks_server", "status": "ok", "tasks": {"running": 2}},
    {"service": "training_server", "status": "ok"},
    {"service": "stream_server", "status": "unreachable", "message": "connection refused"},
]


def _mock_service_status(page: Page, services=None):
    page.route(
        "**/api/service-status",
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({"status": "success", "services": services if services is not None else SERVICES}),
        ),
    )


def _navigate_to_runs(page: Page, base_url: str):
    page.goto(base_url)
    page.wait_for_load_state("networkidle")
    page.locator(".main-tab-button[data-tab='tab-runs']").click()
    page.wait_for_load_state("networkidle")


def _expand_service_status(page: Page):
    """Open the service-status disclosure.

    The panel is a <details> collapsed by default -- expanded it is five rows
    above every Runs pane, and the answer is almost always "all green".
    The per-service rows only exist to be looked at on demand, so tests that
    assert on them have to open it first.
    """
    panel = page.locator("[data-testid='service-status-panel']")
    expect(panel).to_be_visible()
    if not panel.evaluate("el => el.open"):
        page.locator(".service-status-panel__header").click()
    page.wait_for_selector(".service-status-row", state="visible", timeout=4000)
    # `<details>`'s `toggle` event is fired *asynchronously* (queued as a task),
    # so the pref write it drives has not necessarily run by the time `open` is
    # already true and the rows are already visible. Navigating inside that
    # window drops the write. Wait for the value to actually land.
    wait_until(
        page,
        lambda: page.evaluate("localStorage.getItem('ww.operations.serviceStatusExpanded')") == "true",
        "Service-status expanded state was never persisted",
    )


class TestServiceStatusInRuns:
    def test_panel_renders_a_row_per_service(self, page: Page, base_url: str):
        _mock_service_status(page)
        _navigate_to_runs(page, base_url)

        _expand_service_status(page)
        panel = page.locator("[data-testid='service-status-panel']")
        expect(panel.locator(".service-status-row")).to_have_count(len(SERVICES))

    def test_row_detail_reflects_service_payload(self, page: Page, base_url: str):
        _mock_service_status(page)
        _navigate_to_runs(page, base_url)

        _expand_service_status(page)
        expect(page.locator("[data-testid='service-status-row-tasks_server']")).to_contain_text("running: 2")
        expect(page.locator("[data-testid='service-status-row-inference_server']")).to_contain_text("loaded: 1/3")
        # Unreachable services surface their message rather than a detail summary.
        expect(page.locator("[data-testid='service-status-row-stream_server']")).to_contain_text("connection refused")

    def test_refresh_button_refetches(self, page: Page, base_url: str):
        calls: list[str] = []
        page.on("request", lambda req: calls.append(req.url) if "/api/service-status" in req.url else None)

        _mock_service_status(page)
        _navigate_to_runs(page, base_url)
        expect(page.locator("[data-testid='service-status-panel']")).to_be_visible()
        before = len(calls)

        page.locator("[data-testid='service-status-refresh']").click()
        # Poll the capture list, not `networkidle`: the latter is not a barrier on the
        # thing under test. It resolves the moment the network happens to be quiet —
        # which under load is *before* the click's fetch has been issued — so the
        # assertion read a stale `calls` and the test failed for reasons that had
        # nothing to do with the refresh button. Observed failing exactly once in a
        # 16-way parallel run, and passing on every sequential re-run.
        wait_until(
            page,
            lambda: len(calls) > before,
            "Refresh did not re-request /api/service-status",
        )

    def test_training_server_row_opens_the_job_modal(self, page: Page, base_url: str):
        """The training-server row is the one row that opens something, and its
        modal holds a per-job DELETE — the piece most easily lost when this
        panel is reworked. The `›` button is its trigger: the row body itself
        now belongs to the details disclosure."""
        _mock_service_status(page)
        page.route(
            "**/api/training/jobs",
            lambda route: route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps({"jobs": []}),
            ),
        )
        _navigate_to_runs(page, base_url)

        _expand_service_status(page)
        page.locator("[data-testid='service-status-open-training_server']").click()
        expect(page.locator("[data-testid='modal-shell-dialog']")).to_be_visible()


DETAILED_SERVICES = [
    {
        "service": "inference_server",
        "status": "ok",
        "models": {"loaded": 1, "available": 3, "pins": 2},
        "summary": "loaded: 1/3  ·  2 pins",
        "details": [
            {
                "title": "Loaded models (1)",
                "rows": [{"label": "yolo_cats", "value": "2 pins  ·  next expiry in 4m 10s"}],
                "empty": "No models loaded",
            }
        ],
    },
    {
        "service": "tasks_server",
        "status": "ok",
        "tasks": {"running": 0, "paused": 0},
        "summary": "running: 0",
        "details": [{"title": "Running tasks (0)", "rows": [], "empty": "No tasks running"}],
    },
    {
        "service": "training_server",
        "status": "ok",
        "summary": "training: 1",
        "details": [
            {
                "title": "Training now (1)",
                "rows": [{"label": "cats-v3", "value": "epoch 4/10  ·  40%"}],
                "empty": "No training running",
            }
        ],
    },
    # Down services must not render a detail body: the payload is either absent
    # or stale, and either way it would read as live.
    {
        "service": "stream_server",
        "status": "unreachable",
        "message": "connection refused",
        "details": [{"title": "Streams (9)", "rows": [{"label": "ghost", "value": "ready"}]}],
    },
]


class TestServiceStatusDetails:
    """Services may return `details` sections; those rows expand in place."""

    def test_service_summary_string_is_used_verbatim(self, page: Page, base_url: str):
        _mock_service_status(page, DETAILED_SERVICES)
        _navigate_to_runs(page, base_url)
        _expand_service_status(page)

        expect(page.locator("[data-testid='service-status-row-inference_server']")).to_contain_text(
            "loaded: 1/3  ·  2 pins"
        )

    def test_detail_rows_are_hidden_until_the_row_is_expanded(self, page: Page, base_url: str):
        _mock_service_status(page, DETAILED_SERVICES)
        _navigate_to_runs(page, base_url)
        _expand_service_status(page)

        details = page.locator("[data-testid='service-status-details-inference_server']")
        expect(details).not_to_be_visible()

        page.locator("[data-testid='service-status-row-inference_server'] .service-status-row__line").click()
        expect(details).to_be_visible()
        expect(details).to_contain_text("yolo_cats")
        expect(details).to_contain_text("next expiry in 4m 10s")
        expect(details).to_contain_text("Loaded models (1)")

    def test_a_section_with_no_rows_shows_its_empty_text(self, page: Page, base_url: str):
        _mock_service_status(page, DETAILED_SERVICES)
        _navigate_to_runs(page, base_url)
        _expand_service_status(page)

        page.locator("[data-testid='service-status-row-tasks_server'] .service-status-row__line").click()
        expect(page.locator("[data-testid='service-status-details-tasks_server']")).to_contain_text("No tasks running")

    def test_a_service_without_details_is_not_expandable(self, page: Page, base_url: str):
        _mock_service_status(page)  # the plain SERVICES payload
        _navigate_to_runs(page, base_url)
        _expand_service_status(page)

        expect(page.locator("[data-testid='service-status-details-db_server']")).to_have_count(0)
        expect(page.locator("[data-testid='service-status-twisty-db_server']")).to_be_hidden()

    def test_an_unreachable_service_does_not_render_stale_details(self, page: Page, base_url: str):
        _mock_service_status(page, DETAILED_SERVICES)
        _navigate_to_runs(page, base_url)
        _expand_service_status(page)

        row = page.locator("[data-testid='service-status-row-stream_server']")
        expect(row).to_contain_text("connection refused")
        expect(page.locator("[data-testid='service-status-details-stream_server']")).to_have_count(0)

    def test_the_open_button_does_not_toggle_the_disclosure(self, page: Page, base_url: str):
        """Training Server both expands and opens a modal; the `›` must do only
        the latter, or opening the modal silently expands the row behind it."""
        _mock_service_status(page, DETAILED_SERVICES)
        page.route(
            "**/api/training/jobs",
            lambda route: route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps({"jobs": []}),
            ),
        )
        _navigate_to_runs(page, base_url)
        _expand_service_status(page)

        row = page.locator("[data-testid='service-status-row-training_server']")
        page.locator("[data-testid='service-status-open-training_server']").click()

        expect(page.locator("[data-testid='modal-shell-dialog']")).to_be_visible()
        assert row.evaluate("el => el.open") is False

    def test_an_expanded_row_survives_a_refresh(self, page: Page, base_url: str):
        """Refresh rebuilds every row, so which ones are open has to live
        outside the DOM — otherwise refreshing collapses what you opened."""
        _mock_service_status(page, DETAILED_SERVICES)
        _navigate_to_runs(page, base_url)
        _expand_service_status(page)

        page.locator("[data-testid='service-status-row-inference_server'] .service-status-row__line").click()
        details = page.locator("[data-testid='service-status-details-inference_server']")
        expect(details).to_be_visible()
        wait_until(
            page,
            lambda: "inference_server"
            in (page.evaluate("localStorage.getItem('ww.operations.serviceStatusOpenRows')") or ""),
            "Open-row state was never persisted",
        )

        page.locator("[data-testid='service-status-refresh']").click()
        expect(page.locator("[data-testid='service-status-details-inference_server']")).to_be_visible()

    def test_an_expanded_row_survives_a_reload(self, page: Page, base_url: str):
        _mock_service_status(page, DETAILED_SERVICES)
        _navigate_to_runs(page, base_url)
        _expand_service_status(page)
        page.locator("[data-testid='service-status-row-inference_server'] .service-status-row__line").click()
        wait_until(
            page,
            lambda: "inference_server"
            in (page.evaluate("localStorage.getItem('ww.operations.serviceStatusOpenRows')") or ""),
            "Open-row state was never persisted",
        )

        _navigate_to_runs(page, base_url)

        expect(page.locator("[data-testid='service-status-details-inference_server']")).to_be_visible()


class TestServiceStatusCollapsedSummary:
    def test_panel_is_collapsed_by_default(self, page: Page, base_url: str):
        """Collapsed is the default: the strip must not eat a third of the pane
        above every Runs pane just to say everything is fine."""
        _mock_service_status(page)
        _navigate_to_runs(page, base_url)

        panel = page.locator("[data-testid='service-status-panel']")
        expect(panel).to_be_visible()
        assert panel.evaluate("el => el.open") is False
        expect(page.locator(".service-status-row").first).not_to_be_visible()

    def test_summary_names_the_unreachable_service(self, page: Page, base_url: str):
        """The collapsed line must say *which* service is unhealthy — a bare
        count would send you into the list anyway, defeating the collapse."""
        _mock_service_status(page)
        _navigate_to_runs(page, base_url)

        summary = page.locator("[data-testid='service-status-summary']")
        expect(summary).to_be_visible()
        expect(summary).to_contain_text("Stream Server")
        expect(summary).to_have_attribute("data-tone", "warn")

    def test_expanded_state_survives_a_reload(self, page: Page, base_url: str):
        """Whether the panel is open is a user preference, not per-visit state."""
        _mock_service_status(page)
        _navigate_to_runs(page, base_url)
        _expand_service_status(page)

        _navigate_to_runs(page, base_url)
        panel = page.locator("[data-testid='service-status-panel']")
        expect(panel).to_be_visible()
        assert panel.evaluate("el => el.open") is True


class TestServiceStatusNotInSettings:
    def test_settings_no_longer_shows_service_status(self, page: Page, base_url: str):
        _mock_service_status(page)
        page.goto(base_url)
        page.wait_for_load_state("networkidle")
        page.locator(".main-tab-button[data-tab='tab-settings']").click()
        page.wait_for_selector("#tab-settings .settings-wrap", timeout=4000)

        expect(page.locator("#tab-settings [data-testid='service-status-panel']")).to_have_count(0)


class TestServiceStatusWhileProbing:
    """A service whose /status hasn't answered yet.

    The backend answers immediately with a provisional `checking`/`waiting`
    status and keeps probing in the background (manager.py: PROBE_INLINE_WAIT /
    PROBE_SOFT_TIMEOUT), so the real answer only arrives on a *later* request.
    """

    @staticmethod
    def _mock_settling(page: Page, provisional_status="checking"):
        """First response has db_server still being probed; every later one has it up.

        Returns the call counter so a test can prove the panel asked again on its
        own rather than being nudged.
        """
        calls = {"n": 0}

        def handle(route):
            calls["n"] += 1
            first = calls["n"] == 1
            services = [
                {
                    "service": "db_server",
                    "status": provisional_status if first else "ok",
                    **({} if first else {"hash_index": {"image_valid": 7, "needs_rebuild": False}}),
                },
                {"service": "tasks_server", "status": "ok", "tasks": {"running": 1}},
            ]
            route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps({"status": "success", "services": services}),
            )

        page.route("**/api/service-status", handle)
        return calls

    def test_a_probe_in_flight_is_not_reported_as_down(self, page: Page, base_url: str):
        """`checking` fell through to the error branch: red ✕, and the collapsed
        summary said "DB Server down" for a service that was merely slow."""
        page.route(
            "**/api/service-status",
            lambda route: route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps(
                    {
                        "status": "success",
                        "services": [
                            {"service": "db_server", "status": "checking"},
                            {"service": "tasks_server", "status": "ok"},
                        ],
                    }
                ),
            ),
        )
        _navigate_to_runs(page, base_url)
        summary = page.locator("[data-testid='service-status-summary']")
        expect(summary).to_be_visible()
        expect(summary).not_to_contain_text("down")
        expect(summary).to_contain_text("Checking")
        assert summary.get_attribute("data-tone") != "bad"

    def test_the_panel_asks_again_until_the_probe_settles(self, page: Page, base_url: str):
        """The actual complaint: it said "checking…" forever. The panel refreshed
        exactly once, so a provisional status was never revisited and only a
        manual Refresh could move it."""
        calls = self._mock_settling(page)
        _navigate_to_runs(page, base_url)
        _expand_service_status(page)

        row = page.locator("[data-testid='service-status-row-db_server']")
        expect(row).to_contain_text("checking")
        # No refresh click anywhere in this test.
        wait_until(
            page,
            lambda: "checking" not in (row.text_content() or ""),
            "Service stayed at 'checking' — the panel never asked again",
            timeout=10000,
        )
        expect(row).to_contain_text("index: 7")
        assert calls["n"] >= 2, f"expected a follow-up request, saw {calls['n']}"

    def test_polling_stops_once_everything_is_settled(self, page: Page, base_url: str):
        """The retry is for probes in flight only — a settled panel must not sit
        there re-requesting forever."""
        calls = self._mock_settling(page)
        _navigate_to_runs(page, base_url)
        row = page.locator("[data-testid='service-status-row-db_server']")
        _expand_service_status(page)
        wait_until(
            page,
            lambda: "checking" not in (row.text_content() or ""),
            "Service never settled",
            timeout=10000,
        )
        settled = calls["n"]
        page.wait_for_timeout(3000)
        assert calls["n"] == settled, f"panel kept polling after settling ({settled} → {calls['n']})"

    def test_a_genuinely_broken_service_still_reads_as_down(self, page: Page, base_url: str):
        """Guard on the summary rewrite: 'checking' is not a blanket amnesty."""
        page.route(
            "**/api/service-status",
            lambda route: route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps(
                    {
                        "status": "success",
                        "services": [
                            {"service": "db_server", "status": "checking"},
                            {"service": "tasks_server", "status": "error", "message": "boom"},
                        ],
                    }
                ),
            ),
        )
        _navigate_to_runs(page, base_url)
        summary = page.locator("[data-testid='service-status-summary']")
        expect(summary).to_contain_text("Tasks Server down")
        assert summary.get_attribute("data-tone") == "bad"


class TestDisclosureAffordanceIsFindable:
    """Both carets were easy to miss — the row's sat hard against the panel's
    right edge, a full row away from the text it discloses."""

    def test_row_twisty_leads_the_row_instead_of_trailing_it(self, page: Page, base_url: str):
        _mock_service_status(
            page,
            [
                {
                    "service": "inference_server",
                    "status": "ok",
                    "summary": "2 loaded",
                    "details": [{"title": "Models", "rows": [{"label": "a.onnx", "value": "pinned"}]}],
                }
            ],
        )
        _navigate_to_runs(page, base_url)
        _expand_service_status(page)

        line = page.locator("[data-testid='service-status-row-inference_server'] .service-status-row__line")
        twisty = page.locator("[data-testid='service-status-twisty-inference_server']")
        expect(twisty).to_be_visible()

        line_box = line.bounding_box()
        twisty_box = twisty.bounding_box()
        offset = twisty_box["x"] - line_box["x"]
        assert offset < 24, f"twisty is {offset:.0f}px from the row's left edge — it used to be at the far right"
        # And it leads the status dot rather than trailing the summary text.
        assert twisty_box["x"] < line_box["x"] + line_box["width"] / 2

    def test_row_twisty_shares_a_column_with_the_panel_caret(self, page: Page, base_url: str):
        """They read as the same affordance at two levels, so they line up."""
        _mock_service_status(
            page,
            [
                {
                    "service": "inference_server",
                    "status": "ok",
                    "details": [{"title": "Models", "rows": [{"label": "a.onnx", "value": "pinned"}]}],
                }
            ],
        )
        _navigate_to_runs(page, base_url)
        _expand_service_status(page)

        caret_x = page.locator(".service-status-panel__header").evaluate(
            "el => el.getBoundingClientRect().x + parseFloat(getComputedStyle(el, '::before').width) / 2"
        )
        twisty_box = page.locator("[data-testid='service-status-twisty-inference_server']").bounding_box()
        twisty_centre = twisty_box["x"] + twisty_box["width"] / 2
        assert (
            abs(twisty_centre - caret_x) < 40
        ), f"row caret at {twisty_centre:.0f} is not in the same column as the panel caret at {caret_x:.0f}"

    def test_carets_are_legible_not_a_smudge(self, page: Page, base_url: str):
        """They were --text-2xs/--text-faint: present, but nobody saw them."""
        _mock_service_status(
            page,
            [
                {
                    "service": "inference_server",
                    "status": "ok",
                    "details": [{"title": "Models", "rows": [{"label": "a.onnx", "value": "pinned"}]}],
                }
            ],
        )
        _navigate_to_runs(page, base_url)
        _expand_service_status(page)

        size = page.locator("[data-testid='service-status-twisty-inference_server']").evaluate(
            "el => parseFloat(getComputedStyle(el).fontSize)"
        )
        assert size >= 12, f"row caret renders at {size}px — too small to notice"
        header_size = page.locator(".service-status-panel__header").evaluate(
            "el => parseFloat(getComputedStyle(el, '::before').fontSize)"
        )
        assert header_size >= 12, f"panel caret renders at {header_size}px — too small to notice"
