"""
The watcher live view (app/nvr/liveViewModal.js) and the card button that opens it.

The view exists to answer "is this watcher actually seeing anything?", so what is
tested is the *diagnostic* content, not the decoration:

  - detections and tracks are reported separately, because a full detection list next
    to an empty track list is the tracker's fault and the opposite reading is the
    model's;
  - a stopped watcher does not offer a live view (and no longer claims to be running,
    which the pane got wrong: finished instances are retained and were counted);
  - "save frame for annotation" sends the frame the watcher analysed, with its
    detections, to the ordinary upload + metadata endpoints.

Run with:
    pytest playwright_tests/test_nvr_live_view.py -p no:asyncio -v
"""

import json

from playwright.sync_api import Page, expect

PROFILE_UUID = "99999999-eeee-0000-0000-000000000001"
TASK_UUID = "task-uuid-live-1"

# A real 64×36 JPEG, so the page decodes an actual image rather than exercising the
# error path a bogus payload would take.
FRAME_B64 = (
    "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAA0JCgsKCA0LCgsODg0PEyAVExISEyccHhcgLikxMC4pLSwzOko+MzZGNywt"
    "QFdBRkxOUlNSMj5aYVpQYEpRUk//2wBDAQ4ODhMREyYVFSZPNS01T09PT09PT09PT09PT09PT09PT09PT09PT09PT09P"
    "T09PT09PT09PT09PT09PT09PT0//wAARCAAkAEADASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcI"
    "CQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRol"
    "JicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ip"
    "qrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAA"
    "AAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLR"
    "ChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaX"
    "mJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEA"
    "PwDn6KKK8Y/SQooooA5CiiivZPzYKKKKAOvooorxj9JCiiigDkKKKK9k/NgooooA6+iiivGP0kKKKKAOQooor2T82Cii"
    "igD/2Q=="
)


def _profile(uuid: str = PROFILE_UUID, name: str = "House cams") -> dict:
    return {
        "uuid": uuid,
        "name": name,
        "description": "",
        "enabled": True,
        "trackerTypename": "ByteTracker",
        "trackerParamsJson": {},
        "defaultModelGroupUuid": None,
        "defaultModelName": None,
        "defaultAnalysisFps": 2.0,
        "defaultAnalysisWidth": 640,
        "sources": [
            {
                "uuid": "ps-1",
                "profileUuid": uuid,
                "sourceUuid": "src-1",
                "order": 0,
                "enabled": True,
                "modelGroupUuid": None,
                "modelName": None,
                "analysisFps": 2.0,
                "analysisWidth": 640,
                "recordRetentionS": 45.0,
                "trackerParamsJson": None,
                "bufferMode": "auto",
                "ringSeconds": 20.0,
                "ringMaxMb": 32.0,
                "zones": [],
            }
        ],
        "rules": [],
        "actions": [],
        "labelLifetimes": [],
    }


def _task(status: str = "running") -> dict:
    return {
        "config": {
            "uuid": "cfg-1",
            "name": "watch-house",
            "typename": "StreamWatcherTask",
            "params": {"watch_profile_uuid": PROFILE_UUID},
            "markedForDelete": False,
            "description": None,
        },
        "instance": {
            "uuid": TASK_UUID,
            "configUuid": "cfg-1",
            "status": status,
            "resumeData": None,
            "result": None,
            "errorMessage": None,
        },
        "progress": None,
        "statusMessage": "Watching",
        "statusActions": [],
        "logLines": [],
        "isPausedInPlace": False,
        "continuous": True,
    }


def _preview(**overrides) -> dict:
    preview = {
        "profileUuid": PROFILE_UUID,
        "profileName": "House cams",
        "dryRun": False,
        "sources": [{"sourceUuid": "src-1", "sourceName": "Litter cam"}],
        "sourceUuid": "src-1",
        "sourceName": "Litter cam",
        "frame": FRAME_B64,
        "frameSeq": 7,
        "frameWidth": 64,
        "frameHeight": 36,
        "frameAgeS": 0.2,
        "analysisFps": 2.0,
        "analysisWidth": 640,
        "modelNames": ["yolov8n"],
        "framesAnalysed": 41,
        "healthy": True,
        "lastError": None,
        "buffer": "● 45s rewind (native)",
        "detections": [
            {"bbox": [4.0, 4.0, 30.0, 30.0], "labelUuid": "label-cat", "labelName": "Cat", "confidence": 0.81}
        ],
        "tracks": [
            {
                "trackId": 3,
                "bbox": [4.0, 4.0, 30.0, 30.0],
                "labelUuid": "label-cat",
                "labelName": "Cat",
                "confidence": 0.81,
                "firstSeen": 1.0,
                "lastSeen": 2.0,
                "hits": 5,
                "isAnchor": False,
                "ageS": 0.1,
            }
        ],
        "zones": [],
        "openEvents": [],
    }
    preview.update(overrides)
    return preview


def _mock_nvr(page: Page, *, task_status: str | None = "running", preview: dict | None = None):
    """Wire the three endpoints the pane and the view read.

    `task_status=None` means no watcher task exists at all.
    """
    page.route(
        lambda url: url.split("?")[0].rstrip("/").endswith("/api/watch-profiles"),
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({"status": "success", "profiles": [_profile()]}),
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
    page.route(
        lambda url: "/live-preview" in url,
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({"status": "success", "preview": preview if preview is not None else _preview()}),
        ),
    )


def _open_live_view(page: Page, base_url: str):
    page.goto(base_url)
    page.wait_for_load_state("networkidle")
    page.locator(".main-tab-button[data-tab='tab-nvr']").click()
    page.wait_for_selector("#nvr-live-box .nvr-profile-card", timeout=5000)
    page.locator("[data-testid='nvr-live-view']").first.click()
    page.wait_for_selector("[data-testid='watch-live-dialog']", timeout=5000)


class TestLiveViewButton:
    def test_a_running_watcher_offers_a_live_view(self, page: Page, base_url: str):
        _mock_nvr(page)
        page.goto(base_url)
        page.wait_for_load_state("networkidle")
        page.locator(".main-tab-button[data-tab='tab-nvr']").click()
        page.wait_for_selector("#nvr-live-box .nvr-profile-card", timeout=5000)

        expect(page.locator("[data-testid='nvr-live-view']")).to_be_visible()
        expect(page.locator(".nvr-state-pill").first).to_have_text("running")

    def test_a_finished_instance_is_not_a_running_watcher(self, page: Page, base_url: str):
        """Finished instances are retained; counting them claimed a stopped watcher was live."""
        _mock_nvr(page, task_status="completed")
        page.goto(base_url)
        page.wait_for_load_state("networkidle")
        page.locator(".main-tab-button[data-tab='tab-nvr']").click()
        page.wait_for_selector("#nvr-live-box .nvr-profile-card", timeout=5000)

        expect(page.locator(".nvr-state-pill").first).to_have_text("stopped")
        expect(page.locator("[data-testid='nvr-live-view']")).to_have_count(0)


class TestLiveViewContent:
    def test_shows_the_frame_and_its_source(self, page: Page, base_url: str):
        _mock_nvr(page)
        _open_live_view(page, base_url)

        expect(page.locator("[data-testid='watch-live-canvas']")).to_be_visible()
        expect(page.locator("[data-testid='watch-live-source']")).to_contain_text("Litter cam")
        expect(page.locator("[data-testid='watch-live-stats']")).to_contain_text("yolov8n")

    def test_detections_and_tracks_are_counted_separately(self, page: Page, base_url: str):
        """The core diagnosis: detections present, tracks empty ⇒ the tracker."""
        _mock_nvr(page, preview=_preview(tracks=[]))
        _open_live_view(page, base_url)

        stats = page.locator("[data-testid='watch-live-stats']")
        expect(stats).to_contain_text("Detections (this frame)")
        expect(page.locator("[data-testid='watch-live-lists']")).to_contain_text("the tracker is discarding them")

    def test_boxes_are_drawn_in_their_label_colour(self, page: Page, base_url: str):
        """Every box used to be the same green, which cannot say *what* was detected.

        Asserted on the pixels rather than on a call to the overlay: the bug was that the
        colour the watcher already sends was never reaching the canvas at all.
        """
        preview = _preview(
            detections=[
                {
                    "bbox": [4.0, 4.0, 30.0, 30.0],
                    "labelUuid": "label-cat",
                    "labelName": "Cat",
                    "labelColor": "#ff00ff",
                    "confidence": 0.81,
                }
            ],
            tracks=[],
        )
        _mock_nvr(page, preview=preview)
        _open_live_view(page, base_url)

        page.wait_for_function(
            "() => document.querySelector(\"[data-testid='watch-live-canvas']\").width === 64",
            timeout=5000,
        )
        # The stroke is centred on the box edge, so the corner pixel is stroke, not frame.
        rgba = page.evaluate("""() => {
                const canvas = document.querySelector("[data-testid='watch-live-canvas']");
                return Array.from(canvas.getContext('2d').getImageData(4, 4, 1, 1).data);
            }""")
        red, green, blue, _alpha = rgba
        # Magenta *hue*, not full intensity: the dashed stroke is antialiased over the
        # frame, so what identifies the colour is red ≈ blue with green well below both.
        assert red > 40 and blue > 40, rgba
        assert green * 2 < min(red, blue), rgba

    def test_a_source_error_is_surfaced_not_buried(self, page: Page, base_url: str):
        _mock_nvr(page, preview=_preview(healthy=False, lastError="connection refused"))
        _open_live_view(page, base_url)

        expect(page.locator("[data-testid='watch-live-status']")).to_contain_text("connection refused")

    def test_says_when_a_source_has_no_models(self, page: Page, base_url: str):
        """The boring explanation for "nothing is detected", worth ruling out first."""
        _mock_nvr(page, preview=_preview(modelNames=[], detections=[], tracks=[]))
        _open_live_view(page, base_url)

        expect(page.locator("[data-testid='watch-live-stats']")).to_contain_text("none configured")

    def test_a_stopped_watcher_stops_the_poll_and_says_so(self, page: Page, base_url: str):
        """The view is open, the watcher is stopped elsewhere — a normal ending."""
        _mock_nvr(page)
        _open_live_view(page, base_url)

        page.route(
            lambda url: "/live-preview" in url,
            lambda route: route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps({"status": "failure", "message": "Task is not currently running"}),
            ),
        )

        expect(page.locator("[data-testid='watch-live-status']")).to_contain_text("no longer running", timeout=5000)
        expect(page.locator("[data-testid='watch-live-pause']")).to_have_text("Resume")


class TestSaveFrameForAnnotation:
    def _mock_save_endpoints(self, page: Page, captured: dict):
        def _upload(route):
            # post_data would raise: the multipart body carries raw JPEG bytes, which
            # are not valid UTF-8. Only the form fields are being asserted on.
            body = route.request.post_data_buffer or b""
            captured["upload"] = body.decode("utf-8", errors="replace")
            route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps(
                    {
                        "status": "success",
                        "results": [
                            {
                                "filename": "frame.jpg",
                                "ok": True,
                                "path": "watch_captures/frame.jpg",
                                "uuid": "img-1",
                                "duplicateOf": None,
                                "duplicateDistance": None,
                                "renamedFrom": None,
                            }
                        ],
                    }
                ),
            )

        def _metadata(route):
            captured["metadata"] = json.loads(route.request.post_data or "{}")
            route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps({"status": "success"}),
            )

        page.route(lambda url: url.split("?")[0].endswith("/api/images/upload"), _upload)
        page.route(lambda url: url.split("?")[0].endswith("/api/images/metadata/update"), _metadata)

    def test_saves_the_frame_with_its_detections(self, page: Page, base_url: str):
        captured: dict = {}
        _mock_nvr(page)
        self._mock_save_endpoints(page, captured)
        _open_live_view(page, base_url)

        page.locator("[data-testid='watch-live-save']").click()
        expect(page.locator("[data-testid='watch-live-saved-hint']")).to_contain_text(
            "watch_captures/frame.jpg", timeout=5000
        )

        assert "watch_captures" in captured["upload"]
        boxes = captured["metadata"]["boxes"]
        assert len(boxes) == 1
        # Normalised against the *frame* size, not the canvas's displayed size.
        assert 0 < boxes[0]["x"] < 1 and 0 < boxes[0]["width"] <= 1
        assert boxes[0]["labelUuid"] == "label-cat"
        # Unverified: a model's guess seeds annotation, it is never ground truth.
        assert "00000000-0000-0000-0000-000000000001" in boxes[0]["tagUuids"]

    def test_can_save_a_bare_frame_without_boxes(self, page: Page, base_url: str):
        captured: dict = {}
        _mock_nvr(page)
        self._mock_save_endpoints(page, captured)
        _open_live_view(page, base_url)

        page.locator("[data-testid='watch-live-save-boxes']").uncheck()
        page.locator("[data-testid='watch-live-save']").click()
        expect(page.locator("[data-testid='watch-live-saved-hint']")).to_contain_text(
            "watch_captures/frame.jpg", timeout=5000
        )

        # No boxes means the metadata call is not made at all, rather than being made
        # with an empty list — which on this endpoint means "delete every box".
        assert "metadata" not in captured
