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
import re

import cv2
import numpy as np
import pytest
from playwright.sync_api import Page, Route, expect

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
    page.locator(".main-tab-button[data-tab='tab-configuration']").click()
    page.wait_for_load_state("networkidle")
    page.locator("#tab-configuration .workspace-tab-button[data-tab='tab-task-management']").click()
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
        page.wait_for_timeout(800)

        assert calls, "Expected a POST to /api/tasks/instances/status-data after expanding the card"
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
        page.wait_for_timeout(500)
        assert calls, "Expected at least one fetch while expanded"

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
        page.wait_for_timeout(600)

        assert command_calls, "Expected a POST to /api/tasks/instances/command"
        payload = command_calls[0]
        assert payload.get("taskUuid") == task_id
        assert payload.get("command") == "capture_now"
        assert payload.get("params") == {"source": "cam1"}
