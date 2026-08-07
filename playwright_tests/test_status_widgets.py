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
    page.locator(".main-tab-button[data-tab='tab-operations']").click()
    page.wait_for_load_state("networkidle")
    # Task *instances* live on the Activity sub-tab; Tasks holds the config editor.
    page.locator("#tab-operations .workspace-tab-button[data-tab='tab-task-activity']").click()
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
