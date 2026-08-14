"""
Tests for the Phase 3 task status schema framework (LIVE_COLLECTION_PLAN.md).

Exercises the statusWidgets registry (ui/utils/statusWidgets/) and the
schema-driven ActiveTaskField.renderView() (ui/utils/fields/ActiveTaskField.js):
  - status-data fetch discipline: only while a card is expanded
  - stat_grid / annotated_image widgets rendering from a stubbed status-data payload
  - unknown widget types degrade gracefully instead of breaking the card
  - the default (no-override) schema reproduces the pre-Phase-3 card exactly
  - a `task_command` status action posts to the command endpoint

Run with:
    pytest playwright_tests/test_status_widgets.py -p no:xdist -p no:asyncio -v
"""

import base64
import json
import pathlib
import re

import cv2
import numpy as np
import pytest
from playwright.sync_api import Page, Route, expect

from playwright_tests.waits import wait_until

TASK_TYPE = "CollectTrainingImagesTask"

# A 1x1 transparent PNG, base64-encoded — small valid image for canvas loadImage().
TINY_PNG_B64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk" "+A8AAQUBAScY42YAAAAASUVORK5CYII="


def _solid_color_png_b64(size: int = 64, color_bgr: tuple = (0, 0, 255)) -> str:
    """A solid-color square PNG, base64-encoded — big enough (unlike TINY_PNG_B64)
    to tell apart "scaled to fill the container" from "collapsed to a 0x0 region
    at the origin" by sampling a pixel away from the top-left corner."""
    img = np.full((size, size, 3), color_bgr, dtype=np.uint8)
    success, buffer = cv2.imencode(".png", img)
    assert success
    return base64.b64encode(buffer).decode("utf-8")


def _make_task(uuid, name="Live Collector", **overrides):
    task = {
        "instance": {
            "uuid": uuid,
            "status": "running",
            "startedAt": "2024-01-01T00:00:00",
            "endedAt": None,
        },
        "config": {"name": name, "uuid": "cfg-" + uuid, "typename": TASK_TYPE},
        "progress": None,
        "isPausedInPlace": False,
        "logLines": ["frame 1 saved", "frame 2 saved"],
        "statusMessage": "Watching camera…",
        "statusActions": [],
    }
    task.update(overrides)
    return task


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _navigate_to_tasks(page: Page, base_url: str):
    page.goto(base_url)
    page.wait_for_load_state("networkidle")
    page.locator(".main-tab-button[data-tab='tab-runs']").click()
    page.wait_for_load_state("networkidle")


def _mock_task_types(page: Page):
    page.route(
        "**/api/tasks/types/list",
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({"status": "success", "types": [TASK_TYPE]}),
        ),
    )


def _mock_task_configs(page: Page):
    page.route(
        "**/api/tasks/configs/get",
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({"status": "success", "configs": {}}),
        ),
    )


def _mock_active_tasks(page: Page, tasks: dict):
    page.route(
        "**/api/tasks/instances/get",
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({"status": "success", "tasks": tasks}),
        ),
    )


def _mock_status_schema(page: Page, schema: list):
    page.route(
        "**/api/tasks/types/status-schema",
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({"status": "success", "schemas": {TASK_TYPE: schema}}),
        ),
    )


def _mock_status_data(page: Page, data_by_uuid: dict):
    """Route /api/tasks/instances/status-data, recording every call body."""
    calls: list[dict] = []

    def handle(route: Route):
        body = json.loads(route.request.post_data)
        calls.append(body)
        requested = body.get("taskUuids", [])
        matched = {u: v for u, v in data_by_uuid.items() if u in requested}
        route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({"status": "success", "statusData": matched}),
        )

    page.route("**/api/tasks/instances/status-data", handle)
    return calls


def _setup_base_mocks(page: Page, tasks: dict):
    _mock_task_types(page)
    _mock_task_configs(page)
    _mock_active_tasks(page, tasks)


def _mock_image_file(page: Page, image_b64: str = TINY_PNG_B64):
    page.route(
        "**/api/images/*/file",
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps(
                {
                    "status": "success",
                    "imageBase64": image_b64,
                    "mimeType": "image/png",
                    "image": {"uuid": "img-1", "filename": "folder/frame.jpg", "bboxes": []},
                }
            ),
        ),
    )


# ---------------------------------------------------------------------------
# Status-data fetch discipline: only while expanded
# ---------------------------------------------------------------------------


class TestStatusDataFetchGating:
    def test_expanded_card_fetches_status_data(self, page: Page, base_url: str):
        """Opening a card whose schema has a status-source widget triggers a
        POST to /api/tasks/instances/status-data for that task's uuid."""
        task_id = "task-expand-1"
        schema = [{"id": "phase", "type": "text", "source": "status"}]
        _mock_status_schema(page, schema)
        calls = _mock_status_data(page, {task_id: {"phase": "capturing"}})
        _setup_base_mocks(page, {task_id: _make_task(task_id)})

        _navigate_to_tasks(page, base_url)
        page.wait_for_selector(f"[data-task-uuid='{task_id}'] summary", timeout=5000)
        page.locator(f"[data-task-uuid='{task_id}'] summary").click()
        wait_until(page, lambda: calls, "Expected a POST to /api/tasks/instances/status-data after expanding the card")
        assert task_id in calls[0].get("taskUuids", [])

    def test_collapsed_card_does_not_fetch_status_data(self, page: Page, base_url: str):
        """A card that is never expanded must not call /api/tasks/instances/status-data,
        even though its schema declares a status-source widget — this is the whole
        point of splitting status-data out of the cheap all-tasks summary poll."""
        task_id = "task-collapsed-1"
        schema = [{"id": "phase", "type": "text", "source": "status"}]
        _mock_status_schema(page, schema)
        calls = _mock_status_data(page, {task_id: {"phase": "capturing"}})
        _setup_base_mocks(page, {task_id: _make_task(task_id)})

        _navigate_to_tasks(page, base_url)
        page.wait_for_selector(f"[data-task-uuid='{task_id}'] summary", timeout=5000)
        page.wait_for_timeout(800)

        assert not calls, f"Collapsed card must not fetch status-data; got calls: {calls}"

    def test_collapsing_again_stops_further_fetches(self, page: Page, base_url: str):
        """Closing an expanded card stops its status-data polling."""
        task_id = "task-close-1"
        schema = [{"id": "phase", "type": "text", "source": "status"}]
        _mock_status_schema(page, schema)
        calls = _mock_status_data(page, {task_id: {"phase": "capturing"}})
        _setup_base_mocks(page, {task_id: _make_task(task_id)})

        _navigate_to_tasks(page, base_url)
        page.wait_for_selector(f"[data-task-uuid='{task_id}'] summary", timeout=5000)
        summary = page.locator(f"[data-task-uuid='{task_id}'] summary")
        summary.click()  # open
        wait_until(page, lambda: calls, "Expected at least one fetch while expanded")

        summary.click()  # close
        calls.clear()
        page.wait_for_timeout(6000)  # longer than the ~5s in-card poll interval
        assert not calls, f"Expected no further fetches after collapsing; got: {calls}"


# ---------------------------------------------------------------------------
# stat_grid / annotated_image widget rendering
# ---------------------------------------------------------------------------


class TestStatGridWidget:
    def test_stat_grid_renders_from_stubbed_payload(self, page: Page, base_url: str):
        task_id = "task-statgrid-1"
        schema = [
            {
                "id": "stats",
                "type": "stat_grid",
                "source": "status",
                "fields": {"saved": "Saved", "processed": "Processed"},
            }
        ]
        _mock_status_schema(page, schema)
        _mock_status_data(page, {task_id: {"stats": {"saved": 12, "processed": 34}}})
        _setup_base_mocks(page, {task_id: _make_task(task_id)})

        _navigate_to_tasks(page, base_url)
        page.wait_for_selector(f"[data-task-uuid='{task_id}'] summary", timeout=5000)
        page.locator(f"[data-task-uuid='{task_id}'] summary").click()
        page.wait_for_timeout(800)

        content = page.locator(f"[data-task-uuid='{task_id}']").inner_text()
        assert "12" in content and "Saved" in content, f"stat_grid values not found in: {content!r}"
        assert "34" in content and "Processed" in content, f"stat_grid values not found in: {content!r}"


class TestAnnotatedImageWidget:
    def test_annotated_image_renders_from_stubbed_payload(self, page: Page, base_url: str):
        task_id = "task-annotimg-1"
        schema = [
            {
                "id": "lastFrame",
                "type": "annotated_image",
                "source": "status",
                "label": "Last capture",
            }
        ]
        _mock_status_schema(page, schema)
        _mock_status_data(page, {task_id: {"lastFrame": {"path": "folder/frame.jpg", "bboxes": []}}})
        _mock_image_file(page)
        _setup_base_mocks(page, {task_id: _make_task(task_id)})

        _navigate_to_tasks(page, base_url)
        page.wait_for_selector(f"[data-task-uuid='{task_id}'] summary", timeout=5000)
        page.locator(f"[data-task-uuid='{task_id}'] summary").click()

        # The widget wraps createCanvasInstance() — a Konva stage renders a <canvas>.
        canvas = page.locator(f"[data-task-uuid='{task_id}'] canvas").first
        expect(canvas).to_be_visible(timeout=5000)

        content = page.locator(f"[data-task-uuid='{task_id}']").inner_text()
        assert "Last capture" in content

    def test_image_is_scaled_to_fill_container_not_collapsed_to_corner(self, page: Page, base_url: str):
        """Regression test: annotated_image used to create the Konva canvas and
        load the image (createCanvasInstance -> loadImage, which internally
        computes the fit-to-view scale from the container's clientWidth/
        clientHeight) before the widget's own element was attached to the
        document, so clientWidth/clientHeight read 0 and the scale collapsed to
        0 — the background image and every bbox rendered at (0,0) with 0 size.

        A later ResizeObserver (canvas/main.js) does correct the Konva stage's
        own width/height once the container is actually attached, so checking
        the <canvas> element's bounding box alone is NOT enough to catch this —
        handleResize() (canvas/manager.js) reuses the already-computed (stale,
        zero-scale) viewport rather than recomputing it, so the image/bboxes
        stay frozen at the origin even after the stage itself resizes. This
        test instead samples an actual rendered pixel away from the corner to
        confirm the image was really scaled to fill the container.

        mount() must only run after the caller has appended the element to the
        live DOM, so the fit-to-view scale is computed against the real
        container size from the start."""
        task_id = "task-canvas-scale-1"
        schema = [
            {
                "id": "lastFrame",
                "type": "annotated_image",
                "source": "status",
                "label": "Last capture",
                "height": "16em",
            }
        ]
        _mock_status_schema(page, schema)
        _mock_status_data(page, {task_id: {"lastFrame": {"path": "folder/frame.jpg", "bboxes": []}}})
        _mock_image_file(page, _solid_color_png_b64(size=64, color_bgr=(0, 0, 255)))  # solid red
        _setup_base_mocks(page, {task_id: _make_task(task_id)})

        _navigate_to_tasks(page, base_url)
        page.wait_for_selector(f"[data-task-uuid='{task_id}'] summary", timeout=5000)
        page.locator(f"[data-task-uuid='{task_id}'] summary").click()

        canvas = page.locator(f"[data-task-uuid='{task_id}'] canvas").first
        expect(canvas).to_be_visible(timeout=5000)
        page.wait_for_timeout(500)  # let the image decode + first draw settle

        center_rgba = canvas.evaluate("""(el) => {
                const ctx = el.getContext('2d');
                const x = Math.floor(el.width / 2);
                const y = Math.floor(el.height / 2);
                return Array.from(ctx.getImageData(x, y, 1, 1).data);
            }""")
        r, g, b, a = center_rgba
        assert a > 0 and r > 150 and g < 100 and b < 100, (
            f"Center pixel is not the test image's solid red fill (got rgba={center_rgba!r}); "
            "the image likely collapsed to a 0x0 region at the origin instead of scaling to fill "
            "the container"
        )


class TestSummarySourceWidgetMount:
    def test_annotated_image_with_summary_source_calls_mount(self, page: Page, base_url: str):
        """Regression test: the source:'summary' render path used to append each
        widget's `el` without ever calling `mount()`. annotated_image's real setup
        (creating the Konva canvas via createCanvasInstance) happens entirely inside
        mount() (Factories.js) — so a summary-source annotated_image widget rendered
        an empty container forever, with no <canvas> ever created."""
        task_id = "task-summary-annotimg-1"
        schema = [
            {
                "id": "lastFrame",
                "type": "annotated_image",
                "source": "summary",
                "label": "Last capture",
            }
        ]
        _mock_status_schema(page, schema)
        _mock_image_file(page)
        _setup_base_mocks(
            page,
            {task_id: _make_task(task_id, lastFrame={"path": "folder/frame.jpg", "bboxes": []})},
        )

        _navigate_to_tasks(page, base_url)
        page.wait_for_selector(f"[data-task-uuid='{task_id}'] summary", timeout=5000)
        page.wait_for_timeout(500)

        canvas = page.locator(f"[data-task-uuid='{task_id}'] canvas")
        assert canvas.count() > 0, "Expected mount() to create a <canvas> for a summary-source annotated_image widget"


class TestStatusWidgetPollDiffing:
    def test_unchanged_status_data_does_not_rebuild_widget_across_polls(self, page: Page, base_url: str):
        """Regression test: a poll tick used to tear down and rebuild every
        status widget's DOM unconditionally, every ~5s, even when the
        underlying data was byte-for-byte identical to the previous poll —
        for annotated_image this meant destroying and recreating the Konva
        canvas (and re-fetching/re-decoding the image) for no reason, which
        is what made the panel visibly flash/flicker while open."""
        task_id = "task-diffing-1"
        schema = [
            {
                "id": "lastFrame",
                "type": "annotated_image",
                "source": "status",
                "label": "Last capture",
            }
        ]
        _mock_status_schema(page, schema)
        _mock_status_data(page, {task_id: {"lastFrame": {"path": "folder/frame.jpg", "bboxes": []}}})
        _mock_image_file(page)
        _setup_base_mocks(page, {task_id: _make_task(task_id)})

        _navigate_to_tasks(page, base_url)
        page.wait_for_selector(f"[data-task-uuid='{task_id}'] summary", timeout=5000)
        page.locator(f"[data-task-uuid='{task_id}'] summary").click()

        canvas = page.locator(f"[data-task-uuid='{task_id}'] canvas").first
        expect(canvas).to_be_visible(timeout=5000)
        canvas.evaluate("el => el.dataset.testMarker = 'stable'")

        page.wait_for_timeout(6000)  # longer than the ~5s in-card poll interval; data never changes

        canvas_after = page.locator(f"[data-task-uuid='{task_id}'] canvas").first
        marker = canvas_after.evaluate("el => el.dataset.testMarker")
        assert marker == "stable", "Widget was torn down and rebuilt despite unchanged status-data"


# ---------------------------------------------------------------------------
# Unknown widget type degrades gracefully
# ---------------------------------------------------------------------------


class TestUnknownWidgetType:
    def test_unknown_widget_type_does_not_break_siblings(self, page: Page, base_url: str):
        """A schema entry with a `type` the frontend doesn't know about must not
        throw, and must not prevent its sibling widgets from rendering."""
        task_id = "task-unknown-1"
        schema = [
            {"id": "statusMessage", "type": "text", "source": "summary"},
            {"id": "mystery", "type": "some_future_widget_type", "source": "status"},
            {"id": "logLines", "type": "log_tail", "source": "summary"},
        ]
        _mock_status_schema(page, schema)
        _mock_status_data(page, {})
        _setup_base_mocks(page, {task_id: _make_task(task_id)})

        errors: list[str] = []
        page.on("console", lambda msg: errors.append(msg.text) if msg.type == "error" else None)

        _navigate_to_tasks(page, base_url)
        page.wait_for_selector(f"[data-task-uuid='{task_id}'] summary", timeout=5000)
        page.locator(f"[data-task-uuid='{task_id}'] summary").click()
        page.wait_for_timeout(800)

        content = page.locator(f"[data-task-uuid='{task_id}']").inner_text()
        assert "Watching camera…" in content, f"Sibling summary widget missing: {content!r}"
        assert "frame 1 saved" in content, f"Sibling log widget missing: {content!r}"

        uncaught = [e for e in errors if "TypeError" in e or "Uncaught" in e]
        assert uncaught == [], f"Uncaught JS errors from unknown widget type: {uncaught}"


# ---------------------------------------------------------------------------
# Default schema reproduces the pre-Phase-3 card (regression guard)
# ---------------------------------------------------------------------------


class TestDefaultSchemaRegression:
    def test_default_schema_matches_pre_phase3_card(self, page: Page, base_url: str):
        """With no status-schema override for the typename (the conftest default
        returns {schemas: {}}), the card must still show the indeterminate
        progress bar, the status message, and the log tail — exactly like
        before the schema-driven rewrite."""
        task_id = "task-default-1"
        _setup_base_mocks(page, {task_id: _make_task(task_id, progress=None)})

        _navigate_to_tasks(page, base_url)
        page.wait_for_selector(f"[data-task-uuid='{task_id}'] summary", timeout=5000)
        page.locator(f"[data-task-uuid='{task_id}'] summary").click()

        bar = page.locator(f"[data-task-uuid='{task_id}'] progress")
        expect(bar).to_be_visible()
        expect(bar).to_have_class(re.compile(r"\bprogress-indeterminate\b"))

        content = page.locator(f"[data-task-uuid='{task_id}']").inner_text()
        assert "Watching camera…" in content
        assert "frame 1 saved" in content
        assert "frame 2 saved" in content


# ---------------------------------------------------------------------------
# task_command status action back-channel
# ---------------------------------------------------------------------------


class TestTaskCommandAction:
    def test_task_command_action_posts_expected_body(self, page: Page, base_url: str):
        task_id = "task-command-1"
        task = _make_task(
            task_id,
            statusActions=[
                {
                    "type": "task_command",
                    "label": "Capture now",
                    "command": "capture_now",
                    "params": {"source": "cam1"},
                },
            ],
        )
        _setup_base_mocks(page, {task_id: task})

        command_calls: list[dict] = []

        def handle_command(route: Route):
            command_calls.append(json.loads(route.request.post_data))
            route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps({"status": "success", "message": "queued"}),
            )

        page.route("**/api/tasks/instances/command", handle_command)

        _navigate_to_tasks(page, base_url)
        page.wait_for_selector(f"[data-task-uuid='{task_id}'] summary", timeout=5000)
        page.locator(f"[data-task-uuid='{task_id}'] summary").click()

        btn = page.locator(f"[data-task-uuid='{task_id}'] button", has_text="Capture now")
        expect(btn).to_be_visible(timeout=5000)
        btn.click()
        wait_until(page, lambda: command_calls, "Expected a POST to /api/tasks/instances/command")
        payload = command_calls[0]
        assert payload.get("taskUuid") == task_id
        assert payload.get("command") == "capture_now"
        assert payload.get("params") == {"source": "cam1"}


# ---------------------------------------------------------------------------
# histogram widget (dedup threshold calibration)
# ---------------------------------------------------------------------------


class TestHistogramWidget:
    """The histogram is how a user reads the duplicate-threshold distribution,
    so the bar lengths and the threshold marker have to be right — a chart that
    renders but misplaces the marker is worse than no chart."""

    _SCHEMA = [{"id": "dist", "type": "histogram", "source": "status", "label": "Nearest distance"}]

    def _open(self, page: Page, base_url: str, task_id: str, data: dict):
        _mock_status_schema(page, self._SCHEMA)
        _mock_status_data(page, {task_id: data})
        _setup_base_mocks(page, {task_id: _make_task(task_id)})
        _navigate_to_tasks(page, base_url)
        page.wait_for_selector(f"[data-task-uuid='{task_id}'] summary", timeout=5000)
        page.locator(f"[data-task-uuid='{task_id}'] summary").click()

    def test_renders_one_row_per_bin_with_counts(self, page: Page, base_url: str):
        task_id = "task-hist-1"
        bins = [
            {"label": "0", "count": 4, "marked": False},
            {"label": "1", "count": 8, "marked": True},
            {"label": "2", "count": 2, "marked": False},
        ]
        self._open(page, base_url, task_id, {"dist": {"bins": bins, "total": 14}})
        card = page.locator(f"[data-task-uuid='{task_id}']")
        expect(card).to_contain_text("Nearest distance")
        # Every bin's count is shown as text.
        for label, count in (("0", "4"), ("1", "8"), ("2", "2")):
            expect(card).to_contain_text(label)
            expect(card).to_contain_text(count)
        expect(card).to_contain_text("14 sample(s)")

    def test_marks_the_current_threshold_in_text_not_colour_alone(self, page: Page, base_url: str):
        """Colour alone would be inaccessible; the marked bin carries a glyph
        and the caption names the threshold."""
        task_id = "task-hist-2"
        bins = [
            {"label": "0", "count": 1, "marked": False},
            {"label": "5", "count": 3, "marked": True},
        ]
        self._open(page, base_url, task_id, {"dist": {"bins": bins, "total": 4}})
        card = page.locator(f"[data-task-uuid='{task_id}']")
        expect(card).to_contain_text("5 ◀")
        expect(card).to_contain_text("current threshold (5)")

    def test_bar_widths_are_proportional_to_counts(self, page: Page, base_url: str):
        task_id = "task-hist-3"
        bins = [
            {"label": "0", "count": 10, "marked": False},
            {"label": "1", "count": 5, "marked": False},
        ]
        self._open(page, base_url, task_id, {"dist": {"bins": bins, "total": 15}})
        card = page.locator(f"[data-task-uuid='{task_id}']")
        expect(card).to_contain_text("10")
        widths = card.locator(".status-histogram__fill")
        wait_until(page, lambda: widths.count() >= 2, "Expected two histogram bars")
        full = widths.nth(0).evaluate("el => el.style.width")
        half = widths.nth(1).evaluate("el => el.style.width")
        assert full == "100%", f"largest bin should fill the track, got {full}"
        assert half == "50%", f"half-count bin should be half as long, got {half}"

    def test_empty_measurements_show_an_empty_state(self, page: Page, base_url: str):
        task_id = "task-hist-4"
        self._open(page, base_url, task_id, {"dist": {"bins": [], "total": 0}})
        expect(page.locator(f"[data-task-uuid='{task_id}']")).to_contain_text("No measurements yet.")

    def test_missing_data_does_not_break_the_card(self, page: Page, base_url: str):
        """Before the first status-data fetch lands the widget gets undefined."""
        task_id = "task-hist-5"
        self._open(page, base_url, task_id, {})
        expect(page.locator(f"[data-task-uuid='{task_id}']")).to_contain_text("No measurements yet.")


class TestStatusWidgetContrast:
    """These widgets predate the dark-only theme and hardcoded light surfaces
    (#f5f5f5 cells, #f0f0f0 tracks) while their text inherits the themed --text,
    which is near-white in dark — stat_grid values rendered white on white and
    were unreadable. Every colour must come from a token so the two sides move
    together."""

    def test_factories_declare_no_hardcoded_hex_colours(self):
        """A unit-style guard: cheaper and more complete than probing each widget."""
        src = (
            pathlib.Path(__file__).resolve().parents[1]
            / "apps/image_tagging_server/static/js/ui/utils/statusWidgets/Factories.js"
        ).read_text()
        leftover = re.findall(r"#(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{6})\b", src)
        assert leftover == [], f"hardcoded colours must use tokens instead: {leftover}"

    def test_stat_grid_value_contrasts_with_its_cell(self, page: Page, base_url: str):
        """The actual regression: value text must not match its own background."""
        task_id = "task-contrast-1"
        schema = [{"id": "stats", "type": "stat_grid", "source": "status", "fields": {"saved": "Saved"}}]
        _mock_status_schema(page, schema)
        _mock_status_data(page, {task_id: {"stats": {"saved": 42}}})
        _setup_base_mocks(page, {task_id: _make_task(task_id)})
        _navigate_to_tasks(page, base_url)
        page.wait_for_selector(f"[data-task-uuid='{task_id}'] summary", timeout=5000)
        page.locator(f"[data-task-uuid='{task_id}'] summary").click()

        card = page.locator(f"[data-task-uuid='{task_id}']")
        expect(card).to_contain_text("42")
        value = card.get_by_text("42", exact=True).first
        # Measure real contrast, not mere inequality: the broken version rendered
        # near-white #f2f4f8 text on a near-white #f5f5f5 cell, which is unreadable
        # but not *equal*, so an equality check sails straight past the bug.
        ratio = value.evaluate("""el => {
                const parse = c => c.match(/[\\d.]+/g).slice(0, 3).map(Number);
                const lum = ([r, g, b]) => {
                    const f = v => {
                        v /= 255;
                        return v <= 0.03928 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4);
                    };
                    return 0.2126 * f(r) + 0.7152 * f(g) + 0.0722 * f(b);
                };
                const fg = getComputedStyle(el).color;
                let n = el, bg = 'rgba(0, 0, 0, 0)';
                while (n && (bg === 'rgba(0, 0, 0, 0)' || bg === 'transparent')) {
                    bg = getComputedStyle(n).backgroundColor;
                    n = n.parentElement;
                }
                const [a, b] = [lum(parse(fg)), lum(parse(bg))].sort((x, y) => y - x);
                return (a + 0.05) / (b + 0.05);
            }""")
        assert ratio >= 3.0, f"stat_grid value is unreadable against its cell (contrast {ratio:.2f}:1)"


class TestScrollPreservedAcrossRerender:
    """A running collection task appends log lines, which changes the render
    fingerprint and rebuilds every card. The rebuilt card starts with empty
    status-widget slots and only fills them once its own status-data fetch
    returns, so an expanded card briefly collapsed to a fraction of its height;
    the browser clamped the page scroll to the shorter document and the user was
    thrown to the top mid-read — which made the dedup calibration tables at the
    bottom of a card unreadable while the task was running."""

    _SCHEMA = [
        {
            "id": "stats",
            "type": "stat_grid",
            "source": "status",
            "label": "Collection stats",
            "fields": {f"k{i}": f"Field {i}" for i in range(8)},
        },
        {
            "id": "sweep",
            "type": "table",
            "source": "status",
            "label": "Sweep",
            "columns": [{"key": "a", "label": "T"}, {"key": "b", "label": "M"}],
        },
        # Needed to observe the rebuild at all: the pushed update differs only in
        # its log line, so without a log_tail widget it is invisible to the test.
        {"id": "logLines", "type": "log_tail", "source": "summary"},
    ]

    @staticmethod
    def _payload(log_suffix: str):
        task = _make_task("task-scroll-0", name="Collector")
        task["logLines"] = [f"frame {log_suffix} saved"]
        return {"task-scroll-0": task}

    def test_expanded_card_rebuild_keeps_the_page_scrolled(self, page: Page, base_url: str):
        # Short viewport so the one expanded card overflows and the page scrolls.
        page.set_viewport_size({"width": 900, "height": 500})
        _mock_status_schema(page, self._SCHEMA)
        _mock_status_data(
            page,
            {
                "task-scroll-0": {
                    "stats": {f"k{i}": i for i in range(8)},
                    "sweep": [{"a": str(i), "b": str(i * 2)} for i in range(12)],
                }
            },
        )
        _mock_task_types(page)
        _mock_task_configs(page)
        _mock_active_tasks(page, self._payload("one"))
        # Drive the update through the app's own EventSource handler: fulfilling
        # the SSE route instead means waiting on reconnect timing, which is slow
        # and flaky for no added coverage.
        page.add_init_script("""
            const Orig = window.EventSource;
            window.__streams = [];
            window.EventSource = function (url, cfg) {
                const es = new Orig(url, cfg);
                window.__streams.push({ url: String(url), es });
                return es;
            };
            window.EventSource.prototype = Orig.prototype;
            """)

        _navigate_to_tasks(page, base_url)
        page.wait_for_selector("[data-task-uuid='task-scroll-0'] summary", timeout=5000)
        page.locator("[data-task-uuid='task-scroll-0'] summary").click()
        # Let the first status-data fetch land so the card reaches full height.
        wait_until(
            page,
            lambda: page.evaluate("document.scrollingElement.scrollHeight") > page.evaluate("window.innerHeight"),
            "Expected the expanded card to make the page scrollable",
        )

        page.evaluate("() => window.scrollTo(0, 300)")
        before = page.evaluate("window.scrollY")
        assert before > 0, "test setup failed to scroll the page"

        delivered = page.evaluate(
            """payload => {
                const s = (window.__streams || []).find(x => x.url.includes('/api/tasks/stream'));
                if (!s || typeof s.es.onmessage !== 'function') return false;
                s.es.onmessage({ data: JSON.stringify(payload) });
                return true;
            }""",
            {"tasks": self._payload("two")},
        )
        assert delivered, "app did not open a task stream to push an update through"

        wait_until(
            page,
            # text_content(), not inner_text(): the log tail lives inside a
            # collapsed <details>, so it is in the DOM but not always rendered.
            lambda: "frame two saved" in (page.locator("#active-tasks-list").text_content() or ""),
            "Expected the pushed payload to trigger a re-render",
            timeout=8000,
        )
        page.wait_for_timeout(400)  # allow any late clamp to show up

        after = page.evaluate("window.scrollY")
        assert after == before, f"re-render jumped the page to {after} (was {before})"

    def test_rebuilt_card_keeps_showing_its_status_data(self, page: Page, base_url: str):
        """The seeding must carry real values across, not just reserve height.

        Status-data requests fail from the moment the rebuild is pushed. A failed
        fetch leaves existing widget DOM untouched, so the value can only still
        be on screen if the new card was seeded from the old one — otherwise the
        widgets sit empty, which is exactly the flash of blank tables being fixed
        here. (Returning *empty* data instead would prove nothing: the app would
        rightly render that as empty.)
        """
        page.set_viewport_size({"width": 900, "height": 500})
        _mock_status_schema(page, self._SCHEMA)
        _mock_task_types(page)
        _mock_task_configs(page)
        _mock_active_tasks(page, self._payload("one"))

        blocked = {"on": False}

        def status_data(route: Route):
            if blocked["on"]:
                route.abort()
                return
            route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps(
                    {"status": "success", "statusData": {"task-scroll-0": {"stats": {"k0": 4242}, "sweep": []}}}
                ),
            )

        page.route("**/api/tasks/instances/status-data", status_data)
        page.add_init_script("""
            const Orig = window.EventSource;
            window.__streams = [];
            window.EventSource = function (url, cfg) {
                const es = new Orig(url, cfg);
                window.__streams.push({ url: String(url), es });
                return es;
            };
            window.EventSource.prototype = Orig.prototype;
            """)

        _navigate_to_tasks(page, base_url)
        page.wait_for_selector("[data-task-uuid='task-scroll-0'] summary", timeout=5000)
        page.locator("[data-task-uuid='task-scroll-0'] summary").click()
        card = page.locator("[data-task-uuid='task-scroll-0']")
        expect(card).to_contain_text("4242")

        blocked["on"] = True
        page.evaluate(
            """payload => {
                const s = (window.__streams || []).find(x => x.url.includes('/api/tasks/stream'));
                s.es.onmessage({ data: JSON.stringify(payload) });
            }""",
            {"tasks": self._payload("two")},
        )
        wait_until(
            page,
            lambda: "frame two saved" in (page.locator("#active-tasks-list").text_content() or ""),
            "Expected the pushed payload to trigger a re-render",
            timeout=8000,
        )

        expect(card).to_contain_text("4242")


# ---------------------------------------------------------------------------
# `group` — nestable collapsible sections inside a task card
# ---------------------------------------------------------------------------


class TestStatusWidgetGroups:
    """The dedup calibration block (two histograms + two sweep tables) made a
    running card several screens tall, so scrolling past it in the Activity list
    was a chore. `group` folds it to one line — and, being lazy, a closed group
    renders nothing at all until it is opened."""

    _SCHEMA = [
        {"id": "phase", "type": "text", "source": "status", "label": "Phase"},
        {
            "id": "calibration",
            "type": "group",
            "label": "Duplicate threshold calibration",
            "collapsed": True,
            "widgets": [
                {"id": "hint", "type": "text", "source": "status"},
                {
                    "id": "image_group",
                    "type": "group",
                    "label": "Image distance",
                    "widgets": [
                        {"id": "dist", "type": "histogram", "source": "status", "label": "Nearest distance"},
                    ],
                },
            ],
        },
    ]

    _DATA = {
        "phase": "capturing",
        "hint": "Measuring only — nothing discarded.",
        "dist": {"bins": [{"label": "0", "count": 3, "marked": True}], "total": 3},
    }

    def _open_card(self, page: Page, base_url: str, task_id: str):
        _mock_status_schema(page, self._SCHEMA)
        calls = _mock_status_data(page, {task_id: self._DATA})
        _setup_base_mocks(page, {task_id: _make_task(task_id)})
        _navigate_to_tasks(page, base_url)
        page.wait_for_selector(f"[data-task-uuid='{task_id}'] summary", timeout=5000)
        page.locator(f"[data-task-uuid='{task_id}'] summary").first.click()
        return calls

    def _group(self, page: Page, task_id: str, group_id: str):
        return page.locator(f"[data-task-uuid='{task_id}'] [data-status-group-id='{group_id}']")

    def test_collapsed_group_hides_its_contents_but_shows_its_label(self, page: Page, base_url: str):
        task_id = "task-group-1"
        self._open_card(page, base_url, task_id)
        card = page.locator(f"[data-task-uuid='{task_id}']")
        # Sibling widgets outside the group are unaffected.
        expect(card).to_contain_text("capturing")
        expect(card).to_contain_text("Duplicate threshold calibration")
        expect(card).not_to_contain_text("Measuring only")

    def test_closed_group_renders_no_children_at_all(self, page: Page, base_url: str):
        """Lazy, not merely hidden — nothing inside a closed group is built."""
        task_id = "task-group-2"
        self._open_card(page, base_url, task_id)
        body = self._group(page, task_id, "calibration").locator(".status-group__body")
        wait_until(page, lambda: body.count() == 1, "Expected the group to render")
        assert body.evaluate("el => el.children.length") == 0

    def test_opening_the_group_reveals_nested_content(self, page: Page, base_url: str):
        task_id = "task-group-3"
        self._open_card(page, base_url, task_id)
        self._group(page, task_id, "calibration").locator("> summary").click()
        card = page.locator(f"[data-task-uuid='{task_id}']")
        expect(card).to_contain_text("Measuring only")
        # The nested group is present and, not being marked collapsed, open —
        # so its histogram is on screen without a second click.
        expect(self._group(page, task_id, "image_group")).to_be_visible()
        expect(card).to_contain_text("Nearest distance")
        expect(card).to_contain_text("3 sample(s)")

    def test_nested_group_can_be_closed_independently(self, page: Page, base_url: str):
        task_id = "task-group-4"
        self._open_card(page, base_url, task_id)
        self._group(page, task_id, "calibration").locator("> summary").click()
        inner_body = self._group(page, task_id, "image_group").locator("> .status-group__body")
        expect(inner_body).to_be_visible()
        expect(inner_body).to_contain_text("Nearest distance")

        # Assert on visibility, not on the card's text: once built, a group's
        # children stay in the DOM when it is closed (only the *first* open pays
        # to build them), so textContent still carries them.
        self._group(page, task_id, "image_group").locator("> summary").click()
        expect(inner_body).to_be_hidden()
        # Closing the inner one leaves the outer one — and its own text — alone.
        expect(self._group(page, task_id, "calibration").locator("> .status-group__body")).to_be_visible()
        expect(page.locator(f"[data-task-uuid='{task_id}']")).to_contain_text("Measuring only")

    def test_group_open_state_survives_a_card_rebuild(self, page: Page, base_url: str):
        """A running task rebuilds its card on every poll tick; a group the user
        opened must not snap shut under them mid-read."""
        task_id = "task-group-5"
        # A log_tail is what makes the rebuild observable at all: the pushed
        # update differs only in its log line, and the schema's other widgets
        # would render identically either side of the swap.
        _mock_status_schema(page, [*self._SCHEMA, {"id": "logLines", "type": "log_tail", "source": "summary"}])
        _mock_status_data(page, {task_id: self._DATA})
        _mock_task_types(page)
        _mock_task_configs(page)

        def _payload(log_suffix: str):
            task = _make_task(task_id)
            task["logLines"] = [f"frame {log_suffix} saved"]
            return {task_id: task}

        _mock_active_tasks(page, _payload("one"))
        page.add_init_script("""
            const Orig = window.EventSource;
            window.__streams = [];
            window.EventSource = function (url, cfg) {
                const es = new Orig(url, cfg);
                window.__streams.push({ url: String(url), es });
                return es;
            };
            window.EventSource.prototype = Orig.prototype;
            """)

        _navigate_to_tasks(page, base_url)
        page.wait_for_selector(f"[data-task-uuid='{task_id}'] summary", timeout=5000)
        page.locator(f"[data-task-uuid='{task_id}'] summary").first.click()
        self._group(page, task_id, "calibration").locator("> summary").click()
        card = page.locator(f"[data-task-uuid='{task_id}']")
        expect(card).to_contain_text("Measuring only")

        delivered = page.evaluate(
            """payload => {
                const s = (window.__streams || []).find(x => x.url.includes('/api/tasks/stream'));
                if (!s || typeof s.es.onmessage !== 'function') return false;
                s.es.onmessage({ data: JSON.stringify(payload) });
                return true;
            }""",
            {"tasks": _payload("two")},
        )
        assert delivered, "app did not open a task stream to push an update through"

        wait_until(
            page,
            lambda: "frame two saved" in (page.locator("#active-tasks-list").text_content() or ""),
            "Expected the pushed payload to trigger a re-render",
            timeout=8000,
        )
        # The rebuilt card must come back with the group still open.
        expect(self._group(page, task_id, "calibration")).to_have_attribute("open", "")
        expect(card).to_contain_text("Measuring only")

    def test_group_widgets_still_poll_when_all_are_grouped(self, page: Page, base_url: str):
        """Every status widget behind a collapsed group would leave the flat slot
        list empty at wire-up time — the poller has to be started off the schema,
        not off that list, or opening the group shows permanently blank widgets."""
        task_id = "task-group-6"
        schema = [
            {
                "id": "only_group",
                "type": "group",
                "label": "Everything",
                "collapsed": True,
                "widgets": [{"id": "phase", "type": "text", "source": "status", "label": "Phase"}],
            }
        ]
        _mock_status_schema(page, schema)
        calls = _mock_status_data(page, {task_id: {"phase": "capturing"}})
        _setup_base_mocks(page, {task_id: _make_task(task_id)})
        _navigate_to_tasks(page, base_url)
        page.wait_for_selector(f"[data-task-uuid='{task_id}'] summary", timeout=5000)
        page.locator(f"[data-task-uuid='{task_id}'] summary").first.click()
        wait_until(page, lambda: calls, "Expected status-data polling even with every widget grouped")

        self._group(page, task_id, "only_group").locator("> summary").click()
        expect(page.locator(f"[data-task-uuid='{task_id}']")).to_contain_text("capturing")


class TestExpandedCardFitsItsBox:
    """A running task's card carries action buttons either side of its content
    (pause/suspend left, stop right). The content column is a flex item, and a
    flex item defaults to `min-width: auto` — it refuses to shrink below its own
    content — so once the buttons had taken their share the widgets pushed the
    row wider than the card, and the histogram/tables spilled past the border."""

    _SCHEMA = [
        {
            "id": "stats",
            "type": "stat_grid",
            "source": "status",
            "label": "Collection stats",
            "fields": {f"k{i}": f"Field number {i}" for i in range(6)},
        },
        {
            "id": "calibration",
            "type": "group",
            "label": "Duplicate threshold calibration",
            "widgets": [
                {
                    "id": "image_group",
                    "type": "group",
                    "label": "Image distance",
                    "widgets": [
                        {
                            "id": "dist",
                            "type": "histogram",
                            "source": "status",
                            "label": "Distance to nearest stored image",
                        },
                        {
                            "id": "sweep",
                            "type": "table",
                            "source": "status",
                            "label": "Image threshold sweep (what each value would have done)",
                            "columns": [
                                {"key": "threshold", "label": "Image threshold"},
                                {"key": "matched", "label": "Frames matching"},
                                {"key": "discarded", "label": "Would discard"},
                                {"key": "kept", "label": "Would keep"},
                            ],
                        },
                    ],
                },
            ],
        },
    ]

    _DATA = {
        "stats": {f"k{i}": i * 1000 for i in range(6)},
        "dist": {"bins": [{"label": str(i), "count": i, "marked": i == 2} for i in range(6)], "total": 15},
        "sweep": [{"threshold": str(i), "matched": "120", "discarded": "34", "kept": "86"} for i in range(6)],
    }

    def test_expanded_card_does_not_overflow_its_border(self, page: Page, base_url: str):
        task_id = "task-fit-1"
        # Narrow enough that the grid lays out a single ~340px column — the width
        # the card actually gets in the Activity list on a normal window.
        page.set_viewport_size({"width": 760, "height": 900})
        _mock_status_schema(page, self._SCHEMA)
        _mock_status_data(page, {task_id: self._DATA})
        # Running, so the card gets pause + suspend on the left and stop on the right.
        _setup_base_mocks(page, {task_id: _make_task(task_id)})

        _navigate_to_tasks(page, base_url)
        page.wait_for_selector(f"[data-task-uuid='{task_id}'] summary", timeout=5000)
        page.locator(f"[data-task-uuid='{task_id}'] summary").first.click()
        card = page.locator(f"[data-task-uuid='{task_id}']")
        expect(card).to_contain_text("sample(s)")

        overflow = card.evaluate("el => el.scrollWidth - el.clientWidth")
        assert overflow <= 1, f"expanded card overflows its box by {overflow}px"

    def test_widget_content_stays_inside_the_card(self, page: Page, base_url: str):
        """Not just the card's own scrollWidth: the widgets' painted boxes must
        sit inside the card's border, which is what the user actually sees."""
        task_id = "task-fit-2"
        page.set_viewport_size({"width": 760, "height": 900})
        _mock_status_schema(page, self._SCHEMA)
        _mock_status_data(page, {task_id: self._DATA})
        _setup_base_mocks(page, {task_id: _make_task(task_id)})

        _navigate_to_tasks(page, base_url)
        page.wait_for_selector(f"[data-task-uuid='{task_id}'] summary", timeout=5000)
        page.locator(f"[data-task-uuid='{task_id}'] summary").first.click()
        card = page.locator(f"[data-task-uuid='{task_id}']")
        expect(card).to_contain_text("sample(s)")

        spill = card.evaluate("""el => {
                const right = el.getBoundingClientRect().right;
                let worst = 0;
                for (const node of el.querySelectorAll('.status-group, table, .status-histogram__track')) {
                    worst = Math.max(worst, node.getBoundingClientRect().right - right);
                }
                return Math.round(worst);
            }""")
        assert spill <= 1, f"widget content spills {spill}px past the card's right edge"


class TestLineChartWidget:
    """The training curves. A line chart that renders but misplaces a point is
    worse than no chart, so these assert on the geometry the SVG actually
    produced — not merely that an <svg> appeared."""

    _SCHEMA = [
        {
            "id": "loss_curve",
            "type": "line_chart",
            "source": "status",
            "label": "Loss by epoch",
            "x": "epoch",
            "xLabel": "Epoch",
            "yMin": 0,
            "series": [
                {"key": "train_loss", "label": "Train loss"},
                {"key": "val_loss", "label": "Val loss"},
            ],
        }
    ]

    def _open(self, page: Page, base_url: str, task_id: str, data: dict, schema=None):
        _mock_status_schema(page, schema if schema is not None else self._SCHEMA)
        _mock_status_data(page, {task_id: data})
        _setup_base_mocks(page, {task_id: _make_task(task_id)})
        _navigate_to_tasks(page, base_url)
        page.wait_for_selector(f"[data-task-uuid='{task_id}'] summary", timeout=5000)
        page.locator(f"[data-task-uuid='{task_id}'] summary").click()
        return page.locator(f"[data-task-uuid='{task_id}']")

    @staticmethod
    def _points(card, index=0):
        """The nth polyline's points, as [(x, y), ...] in viewBox units."""
        raw = card.locator("svg polyline").nth(index).get_attribute("points")
        return [tuple(float(v) for v in pair.split(",")) for pair in raw.split()]

    def test_draws_one_polyline_per_series(self, page: Page, base_url: str):
        task_id = "task-chart-1"
        card = self._open(
            page,
            base_url,
            task_id,
            {
                "loss_curve": {
                    "points": [
                        {"epoch": 1, "train_loss": 1.0, "val_loss": 0.9},
                        {"epoch": 2, "train_loss": 0.5, "val_loss": 0.6},
                        {"epoch": 3, "train_loss": 0.0, "val_loss": 0.3},
                    ]
                }
            },
        )
        expect(card).to_contain_text("Loss by epoch")
        wait_until(page, lambda: card.locator("svg polyline").count() == 2, "Expected two series lines")

    def test_y_positions_are_proportional_to_the_values(self, page: Page, base_url: str):
        """The real risk: an off-by-one in the scale that still "looks like" a curve.
        With yMin=0 pinned and a max of 1.0, the 0.5 sample must land exactly
        halfway between the 1.0 and 0.0 samples."""
        task_id = "task-chart-2"
        card = self._open(
            page,
            base_url,
            task_id,
            {
                "loss_curve": {
                    "points": [
                        {"epoch": 1, "train_loss": 1.0, "val_loss": 0.5},
                        {"epoch": 2, "train_loss": 0.5, "val_loss": 0.5},
                        {"epoch": 3, "train_loss": 0.0, "val_loss": 0.5},
                    ]
                }
            },
        )
        wait_until(page, lambda: card.locator("svg polyline").count() == 2, "Expected two series lines")
        pts = self._points(card, 0)
        assert len(pts) == 3
        top, mid, bottom = (p[1] for p in pts)
        assert bottom > mid > top, f"y must increase downward as loss falls: {pts}"
        assert mid == pytest.approx((top + bottom) / 2, abs=0.51), f"0.5 is not halfway: {pts}"
        # x is evenly spaced across three equally-spaced epochs.
        xs = [p[0] for p in pts]
        assert xs[1] == pytest.approx((xs[0] + xs[2]) / 2, abs=0.51), f"x spacing is uneven: {xs}"

    def test_a_missing_metric_breaks_the_line_instead_of_plotting_zero(self, page: Page, base_url: str):
        """The detection trainers report no val_loss. Drawing the gap as 0 would
        invent a measurement — and a very flattering one, for a loss curve."""
        task_id = "task-chart-3"
        card = self._open(
            page,
            base_url,
            task_id,
            {
                "loss_curve": {
                    "points": [
                        {"epoch": 1, "train_loss": 1.0},
                        {"epoch": 2, "train_loss": 0.8},
                        {"epoch": 3, "train_loss": 0.6},
                    ]
                }
            },
        )
        # Only train_loss has data, so only one line is drawn — no flat zero line,
        # and no legend entry promising a series that isn't there.
        wait_until(page, lambda: card.locator("svg polyline").count() == 1, "Expected only the train_loss line")
        expect(card).not_to_contain_text("Val loss")

    def test_series_are_named_in_text_not_colour_alone(self, page: Page, base_url: str):
        task_id = "task-chart-4"
        card = self._open(
            page,
            base_url,
            task_id,
            {
                "loss_curve": {
                    "points": [
                        {"epoch": 1, "train_loss": 1.0, "val_loss": 0.9},
                        {"epoch": 2, "train_loss": 0.5, "val_loss": 0.6},
                    ]
                }
            },
        )
        expect(card).to_contain_text("Train loss")
        expect(card).to_contain_text("Val loss")
        # The latest value rides in the legend, so the headline number is
        # readable without hovering anything.
        expect(card).to_contain_text("0.5")

    def test_percent_format_renders_fractions_as_percentages(self, page: Page, base_url: str):
        task_id = "task-chart-5"
        schema = [
            {
                "id": "acc",
                "type": "line_chart",
                "source": "status",
                "label": "Validation accuracy by epoch",
                "x": "epoch",
                "xLabel": "Epoch",
                "yMin": 0,
                "yMax": 1,
                "valueFormat": "percent",
                "series": [{"key": "val_accuracy", "label": "Val accuracy"}],
            }
        ]
        card = self._open(
            page,
            base_url,
            task_id,
            {"acc": {"points": [{"epoch": 1, "val_accuracy": 0.5}, {"epoch": 2, "val_accuracy": 0.875}]}},
            schema=schema,
        )
        expect(card).to_contain_text("87.5%")

    def test_data_table_carries_the_same_numbers(self, page: Page, base_url: str):
        """Colour-free access to the data — the chart is never the only way to read it."""
        task_id = "task-chart-6"
        card = self._open(
            page,
            base_url,
            task_id,
            {
                "loss_curve": {
                    "points": [
                        {"epoch": 1, "train_loss": 1.25, "val_loss": 0.9},
                        {"epoch": 2, "train_loss": 0.75, "val_loss": 0.6},
                    ]
                }
            },
        )
        table_toggle = card.locator("summary", has_text="Data table")
        expect(table_toggle).to_contain_text("2 points")
        table_toggle.click()
        expect(card.locator("table")).to_contain_text("1.25")
        expect(card.locator("table")).to_contain_text("0.75")

    def test_hovering_the_plot_shows_the_sample_under_the_cursor(self, page: Page, base_url: str):
        task_id = "task-chart-7"
        card = self._open(
            page,
            base_url,
            task_id,
            {
                "loss_curve": {
                    "points": [
                        {"epoch": 1, "train_loss": 1.0, "val_loss": 0.9},
                        {"epoch": 2, "train_loss": 0.5, "val_loss": 0.6},
                        {"epoch": 3, "train_loss": 0.25, "val_loss": 0.3},
                    ]
                }
            },
        )
        svg = card.locator("svg").first
        wait_until(page, lambda: svg.count() == 1, "Expected the chart svg")
        box = svg.bounding_box()
        # Hover hard right — the nearest sample is the last epoch.
        svg.hover(position={"x": box["width"] - 4, "y": box["height"] / 2})
        tooltip = card.locator("div", has_text="Epoch 3").last
        wait_until(page, lambda: "0.25" in (tooltip.text_content() or ""), "Tooltip did not report epoch 3")

    def test_empty_points_show_an_empty_state(self, page: Page, base_url: str):
        task_id = "task-chart-8"
        card = self._open(page, base_url, task_id, {"loss_curve": {"points": []}})
        expect(card).to_contain_text("No epoch data yet.")

    def test_missing_data_does_not_break_the_card(self, page: Page, base_url: str):
        """Before the first status-data fetch lands the widget gets undefined."""
        task_id = "task-chart-9"
        card = self._open(page, base_url, task_id, {})
        expect(card).to_contain_text("No epoch data yet.")

    def test_series_colours_come_from_theme_tokens(self, page: Page, base_url: str):
        """Same rule as every other widget: no hardcoded hex, so marks and the
        chart surface move together across themes."""
        task_id = "task-chart-10"
        card = self._open(
            page,
            base_url,
            task_id,
            {
                "loss_curve": {
                    "points": [
                        {"epoch": 1, "train_loss": 1.0, "val_loss": 0.9},
                        {"epoch": 2, "train_loss": 0.5, "val_loss": 0.6},
                    ]
                }
            },
        )
        wait_until(page, lambda: card.locator("svg polyline").count() == 2, "Expected two series lines")
        strokes = [card.locator("svg polyline").nth(i).evaluate("el => getComputedStyle(el).stroke") for i in range(2)]
        assert all(s.startswith("rgb") for s in strokes), f"series stroke did not resolve: {strokes}"
        assert strokes[0] != strokes[1], "the two series resolved to the same colour"


class TestLineChartDualAxisAndHold:
    """Loss and a 0–1 accuracy on one plot: two y-scales, colour-coded, with the
    sparse series carrying its last measurement forward.

    The detection trainers validate on an interval (libreyolo's default is every
    10th epoch), so val_accuracy arrives on a fraction of the epochs train_loss
    does. Left alone that draws a line that spends most of its length at nothing.
    """

    _SCHEMA = [
        {
            "id": "loss_curve",
            "type": "line_chart",
            "source": "status",
            "label": "Training curves by epoch",
            "x": "epoch",
            "xLabel": "Epoch",
            "yMin": 0,
            "yRight": {"min": 0, "max": 1, "valueFormat": "percent"},
            "series": [
                {"key": "train_loss", "label": "Train loss"},
                {"key": "val_accuracy", "label": "Val accuracy", "axis": "right", "hold": True},
            ],
        }
    ]

    # Deliberately not a subclass of TestLineChartWidget: inheriting would re-run
    # every single-axis test against this schema, which has no val_loss series.
    _open = TestLineChartWidget._open
    _points = staticmethod(TestLineChartWidget._points)

    # Epochs 1-5, validated on 1 and 4 only.
    _SPARSE = {
        "loss_curve": {
            "points": [
                {"epoch": 1, "train_loss": 2.0, "val_accuracy": 0.2},
                {"epoch": 2, "train_loss": 1.6},
                {"epoch": 3, "train_loss": 1.4},
                {"epoch": 4, "train_loss": 1.1, "val_accuracy": 0.8},
                {"epoch": 5, "train_loss": 1.0},
            ]
        }
    }

    def test_the_held_series_spans_the_whole_run(self, page: Page, base_url: str):
        """Without the hold it would be two lone dots; with it, one step line
        that reaches the last epoch."""
        card = self._open(page, base_url, "task-chart-dual-1", self._SPARSE)
        wait_until(page, lambda: card.locator("svg polyline").count() == 2, "Expected loss + held accuracy lines")

        held = self._points(card, 1)
        xs = [p[0] for p in held]
        loss_xs = [p[0] for p in self._points(card, 0)]
        assert min(xs) == pytest.approx(min(loss_xs), abs=0.51), "held line does not start at the first sample"
        assert max(xs) == pytest.approx(max(loss_xs), abs=0.51), "held line stops short of the last epoch"

    def test_the_hold_is_a_step_not_a_ramp(self, page: Page, base_url: str):
        """Carrying a value forward must look like carrying it forward: flat
        until the next measurement, then a jump. A diagonal would read as
        gradual improvement nobody measured."""
        card = self._open(page, base_url, "task-chart-dual-2", self._SPARSE)
        wait_until(page, lambda: card.locator("svg polyline").count() == 2, "Expected loss + held accuracy lines")

        held = self._points(card, 1)
        ys = [p[1] for p in held]
        # Only two distinct levels were ever measured, so the whole line sits at
        # one of exactly two heights.
        assert len(set(round(y, 3) for y in ys)) == 2, f"held line has intermediate levels: {held}"

    def test_the_held_run_is_dashed_and_the_measurements_are_dotted(self, page: Page, base_url: str):
        card = self._open(page, base_url, "task-chart-dual-3", self._SPARSE)
        wait_until(page, lambda: card.locator("svg polyline").count() == 2, "Expected loss + held accuracy lines")

        dash = card.locator("svg polyline").nth(1).get_attribute("stroke-dasharray")
        assert dash, "the inferred (held) run is not visually distinguished from measured data"
        assert not card.locator("svg polyline").nth(0).get_attribute("stroke-dasharray")

    def test_the_right_axis_has_its_own_scale(self, page: Page, base_url: str):
        """0.8 accuracy must sit at 80% of the right axis, not be squashed
        against the floor of a loss axis that runs to 2.0."""
        card = self._open(page, base_url, "task-chart-dual-4", self._SPARSE)
        wait_until(page, lambda: card.locator("svg polyline").count() == 2, "Expected loss + held accuracy lines")

        held = self._points(card, 1)
        top = min(p[1] for p in held)
        bottom = max(p[1] for p in held)
        loss = self._points(card, 0)
        plot_top = min(p[1] for p in loss)  # the 2.0 loss sample, at the top of a 0–2 axis
        plot_bottom = max(p[1] for p in loss)
        plot_h = plot_bottom - plot_top
        # 0.8 on a 0–1 axis sits far above 0.8/2.0 on the loss axis would.
        assert (plot_bottom - top) > plot_h * 0.5, f"held series appears squashed onto the left scale: {held}"
        assert bottom > top, "the two measurements did not separate vertically"

    def test_percent_ticks_appear_for_the_right_axis(self, page: Page, base_url: str):
        card = self._open(page, base_url, "task-chart-dual-5", self._SPARSE)
        wait_until(page, lambda: card.locator("svg text").count() > 0, "Expected axis ticks")
        ticks = card.locator("svg text").all_text_contents()
        assert any(t.strip().endswith("%") for t in ticks), f"no percent ticks on the right axis: {ticks}"
        # The left axis keeps plain loss numbers.
        assert any(t.strip() in {"0", "2", "1"} for t in ticks), f"left axis ticks missing: {ticks}"

    def test_each_axis_is_named_in_the_legend_not_colour_alone(self, page: Page, base_url: str):
        card = self._open(page, base_url, "task-chart-dual-6", self._SPARSE)
        expect(card).to_contain_text("Train loss (left)")
        expect(card).to_contain_text("Val accuracy (right)")
        # And the accuracy legend value reads as a percentage, from the right
        # axis' own format rather than the chart-wide one.
        expect(card).to_contain_text("80.0%")

    def test_the_data_table_shows_gaps_not_held_values(self, page: Page, base_url: str):
        """The chart may infer; the table stays the record of what was measured."""
        card = self._open(page, base_url, "task-chart-dual-7", self._SPARSE)
        table_toggle = card.locator("summary", has_text="Data table")
        table_toggle.click()
        rows = card.locator("table tbody tr")
        wait_until(page, lambda: rows.count() == 5, "Expected one row per epoch")
        # Epoch 2 was not validated.
        assert "—" in (rows.nth(1).text_content() or ""), rows.nth(1).text_content()
        assert "20.0%" in (rows.nth(0).text_content() or ""), rows.nth(0).text_content()
