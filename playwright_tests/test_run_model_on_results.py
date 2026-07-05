"""
Tests for app/review/runModelOnResults.js — the "Run Model on Results" button,
scope resolution, run dialog, and task launch/poll/cancel plumbing (Chunk B).

Chunk B's job ends at "task completed, candidates fetched" — it hands the
result off to `openRelabelConfirmationModal`, a placeholder stubbed to toast
"${candidates.length} candidates" until Chunk C replaces it. These tests
assert up to that handoff.

Because the Playwright `live_server` proxies are static mocks with no DB, the
task endpoints (configs/create, instances/start, instances/get, instances/result)
are mocked directly via page.route() with a fake task that "completes" after a
couple of status polls.

Run with:
    docker compose --profile dev run --rm --no-deps -w /project whiskerwatcher-dev \
        python3.11 -m pytest playwright_tests/test_run_model_on_results.py -q
"""
import json

import pytest
from playwright.sync_api import Page, Route, expect

UNVERIFIED_TAG_UUID = "00000000-0000-0000-0000-000000000001"

LABEL_CAT = {"uuid": "lbl-cat", "name": "Cat", "color": "#ff0000", "parentUuid": None}
LABEL_DOG = {"uuid": "lbl-dog", "name": "Dog", "color": "#00ff00", "parentUuid": None}

UNVERIFIED_TAG = {"uuid": UNVERIFIED_TAG_UUID, "name": "Unverified", "color": "#888", "kind": "system", "icon": None}

BBOX_1 = {
    "uuid": "bbox-1", "x": 0.1, "y": 0.1, "width": 0.2, "height": 0.2,
    "label": {"uuid": "lbl-cat", "name": "Cat", "color": "#ff0000"},
    "labelUuid": "lbl-cat",
    "tags": [UNVERIFIED_TAG],
}
BBOX_2 = {
    "uuid": "bbox-2", "x": 0.3, "y": 0.3, "width": 0.1, "height": 0.1,
    "label": {"uuid": "lbl-dog", "name": "Dog", "color": "#00ff00"},
    "labelUuid": "lbl-dog",
    "tags": [UNVERIFIED_TAG],
}

IMAGE_1 = {"uuid": "img-1", "filename": "folder/image1.jpg", "bboxes": [BBOX_1]}
IMAGE_2 = {"uuid": "img-2", "filename": "folder/image2.jpg", "bboxes": [BBOX_2]}

PAGE_1_IMAGES = [IMAGE_1, IMAGE_2]


# ---------------------------------------------------------------------------
# Mocking helpers
# ---------------------------------------------------------------------------

def _mock_review_queue(page: Page, images: list):
    page.route(
        "**/api/images/review",
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({"status": "success", "images": images, "hasMore": False, "totalCount": len(images)}),
        ),
    )


def _mock_labels(page: Page, labels: list):
    page.route(
        lambda url: url.split("?")[0].endswith("/api/labels") and "api/labels/" not in url,
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({"status": "success", "labels": labels}),
        ),
    )


def _mock_tags(page: Page, tags: list):
    page.route(
        "**/api/tags/list",
        lambda route: route.fulfill(status=200, content_type="application/json", body=json.dumps({"tags": tags})),
    )


def _mock_models(page: Page, detector_names: list, classifier_names: list = None):
    classifier_names = classifier_names or []
    all_names = detector_names + classifier_names

    def handle_list(route: Route):
        route.fulfill(
            status=200, content_type="application/json",
            body=json.dumps({"status": "success", "models": all_names}),
        )
    page.route("**/api/models", handle_list)

    def handle_meta(route: Route):
        name = route.request.url.rstrip("/").split("/")[-2]
        kind = "detector" if name in detector_names else "image_classifier"
        route.fulfill(
            status=200, content_type="application/json",
            body=json.dumps({"status": "success", "modelKind": kind}),
        )
    page.route("**/api/models/*/meta", handle_meta)


class _FakeTask:
    """Tracks poll count so the second `instances/get` call reports COMPLETED."""

    def __init__(self, config_uuid="cfg-1", task_uuid="task-1"):
        self.config_uuid = config_uuid
        self.task_uuid = task_uuid
        self.create_bodies = []
        self.poll_count = 0

    def install(self, page: Page, candidates: list):
        def handle_create(route: Route):
            body = json.loads(route.request.post_data or "{}")
            self.create_bodies.append(body)
            route.fulfill(
                status=200, content_type="application/json",
                body=json.dumps({"status": "success", "configUuid": self.config_uuid}),
            )
        page.route("**/api/tasks/configs/create", handle_create)

        def handle_start(route: Route):
            route.fulfill(
                status=200, content_type="application/json",
                body=json.dumps({"status": "success", "taskUuid": self.task_uuid}),
            )
        page.route("**/api/tasks/instances/start", handle_start)

        def handle_get(route: Route):
            self.poll_count += 1
            status = "running" if self.poll_count < 2 else "completed"
            progress = 50.0 if status == "running" else 100.0
            route.fulfill(
                status=200, content_type="application/json",
                body=json.dumps({
                    "status": "success",
                    "tasks": {
                        self.task_uuid: {
                            "config": {"uuid": self.config_uuid, "typename": "PreviewModelRelabelTask", "name": "x", "params": {}},
                            "instance": {"uuid": self.task_uuid, "status": status},
                            "progress": progress,
                            "statusMessage": "",
                            "statusActions": [],
                            "logLines": [],
                            "isPausedInPlace": False,
                        }
                    },
                }),
            )
        page.route("**/api/tasks/instances/get", handle_get)

        def handle_result(route: Route):
            route.fulfill(
                status=200, content_type="application/json",
                body=json.dumps({
                    "status": "success",
                    "results": {
                        self.task_uuid: {
                            "data": {
                                "images_processed": 2,
                                "bbox_count": len(candidates),
                                "candidate_count": len(candidates),
                                "candidates": candidates,
                            }
                        }
                    },
                }),
            )
        page.route("**/api/tasks/instances/result", handle_result)

        def handle_delete(route: Route):
            route.fulfill(status=200, content_type="application/json", body=json.dumps({"status": "success"}))
        page.route("**/api/tasks/instances/delete", handle_delete)
        page.route("**/api/tasks/instances/cancel", handle_delete)


def _navigate_to_review(page: Page, base_url: str):
    page.goto(base_url)
    page.wait_for_load_state("networkidle")
    page.locator(".main-tab-button[data-tab='tab-review']").click()
    page.wait_for_load_state("networkidle")
    page.wait_for_selector("#review-root .review-grid", timeout=4000)


def _run_model_btn(page: Page):
    return page.locator("#review-root .review-action-bar button", has_text="Run Model on Results")


CANDIDATES = [
    {
        "bboxUuid": "bbox-1", "imageUuid": "img-1", "imagePath": "folder/image1.jpg",
        "currentLabelUuid": "lbl-cat", "currentLabelName": "Cat",
        "proposedLabelUuid": "lbl-dog", "proposedLabelName": "Dog", "confidence": 0.81,
    },
    {
        "bboxUuid": "bbox-2", "imageUuid": "img-2", "imagePath": "folder/image2.jpg",
        "currentLabelUuid": "lbl-dog", "currentLabelName": "Dog",
        "proposedLabelUuid": "lbl-cat", "proposedLabelName": "Cat", "confidence": 0.77,
    },
]


# ---------------------------------------------------------------------------
# Button + dialog basics
# ---------------------------------------------------------------------------

class TestRunModelButton:
    def test_button_visible(self, page: Page, base_url: str):
        _mock_review_queue(page, PAGE_1_IMAGES)
        _mock_labels(page, [LABEL_CAT, LABEL_DOG])
        _mock_tags(page, [UNVERIFIED_TAG])
        _navigate_to_review(page, base_url)

        expect(_run_model_btn(page)).to_be_visible()

    def test_click_opens_dialog_with_candidate_count(self, page: Page, base_url: str):
        _mock_review_queue(page, PAGE_1_IMAGES)
        _mock_labels(page, [LABEL_CAT, LABEL_DOG])
        _mock_tags(page, [UNVERIFIED_TAG])
        _mock_models(page, ["yolo9-detector"])
        fake_task = _FakeTask()
        fake_task.install(page, CANDIDATES)

        _navigate_to_review(page, base_url)
        _run_model_btn(page).click()

        dialog = page.locator(".run-model-on-results-dialog")
        expect(dialog).to_be_visible()
        # Both bboxes (bbox-1, bbox-2) are candidates in the default (Unverified) filter.
        expect(dialog).to_contain_text("2 candidate bbox")
        expect(dialog).to_contain_text("2 image")

    def test_no_candidates_toasts_and_skips_dialog(self, page: Page, base_url: str):
        _mock_review_queue(page, [{"uuid": "img-9", "filename": "x.jpg", "bboxes": []}])
        _mock_labels(page, [LABEL_CAT, LABEL_DOG])
        _mock_tags(page, [UNVERIFIED_TAG])
        _navigate_to_review(page, base_url)

        _run_model_btn(page).click()
        page.wait_for_timeout(200)

        expect(page.locator(".run-model-on-results-dialog")).to_have_count(0)
        expect(page.locator("#toast-container")).to_contain_text("No candidate bboxes")


# ---------------------------------------------------------------------------
# Full run: launch -> poll -> result -> handoff to the Chunk-C placeholder
# ---------------------------------------------------------------------------

class TestRunModelFullFlow:
    def test_run_sends_expected_task_params_and_completes(self, page: Page, base_url: str):
        _mock_review_queue(page, PAGE_1_IMAGES)
        _mock_labels(page, [LABEL_CAT, LABEL_DOG])
        _mock_tags(page, [UNVERIFIED_TAG])
        _mock_models(page, ["yolo9-detector"])
        fake_task = _FakeTask()
        fake_task.install(page, CANDIDATES)

        _navigate_to_review(page, base_url)
        _run_model_btn(page).click()

        dialog = page.locator(".run-model-on-results-dialog")
        expect(dialog).to_be_visible()
        dialog.locator("select").select_option("yolo9-detector")
        dialog.locator("button", has_text="Run").click()

        # Poll loop runs on a ~700ms interval; give it a couple of rounds.
        expect(page.locator("#toast-container")).to_contain_text("2 candidates", timeout=5000)

        assert len(fake_task.create_bodies) == 1
        body = fake_task.create_bodies[0]
        assert body["typename"] == "PreviewModelRelabelTask"
        assert body["persistent"] is False
        assert body["isTemporary"] is True
        params = body["params"]
        assert params["model_name"] == "yolo9-detector"
        assert params["conf_thresh"] == 0.5
        assert params["iou_thresh"] == 0.5
        assert sorted(params["bbox_uuids"]) == ["bbox-1", "bbox-2"]
        assert sorted(params["image_uuids"]) == ["img-1", "img-2"]

        # Dialog closed after handoff.
        expect(dialog).to_have_count(0)

    def test_cancel_stops_polling_and_closes_dialog(self, page: Page, base_url: str):
        _mock_review_queue(page, PAGE_1_IMAGES)
        _mock_labels(page, [LABEL_CAT, LABEL_DOG])
        _mock_tags(page, [UNVERIFIED_TAG])
        _mock_models(page, ["yolo9-detector"])

        # A task that never completes, so the only way out is Cancel.
        cancel_called = {"n": 0}

        def handle_create(route: Route):
            route.fulfill(status=200, content_type="application/json", body=json.dumps({"status": "success", "configUuid": "cfg-1"}))
        page.route("**/api/tasks/configs/create", handle_create)

        def handle_start(route: Route):
            route.fulfill(status=200, content_type="application/json", body=json.dumps({"status": "success", "taskUuid": "task-1"}))
        page.route("**/api/tasks/instances/start", handle_start)

        def handle_get(route: Route):
            route.fulfill(
                status=200, content_type="application/json",
                body=json.dumps({
                    "status": "success",
                    "tasks": {"task-1": {
                        "config": {"uuid": "cfg-1", "typename": "PreviewModelRelabelTask", "name": "x", "params": {}},
                        "instance": {"uuid": "task-1", "status": "running"},
                        "progress": 10.0, "statusMessage": "", "statusActions": [], "logLines": [], "isPausedInPlace": False,
                    }},
                }),
            )
        page.route("**/api/tasks/instances/get", handle_get)

        def handle_cancel(route: Route):
            cancel_called["n"] += 1
            route.fulfill(status=200, content_type="application/json", body=json.dumps({"status": "success"}))
        page.route("**/api/tasks/instances/cancel", handle_cancel)
        page.route("**/api/tasks/instances/delete", handle_cancel)

        _navigate_to_review(page, base_url)
        _run_model_btn(page).click()

        dialog = page.locator(".run-model-on-results-dialog")
        expect(dialog).to_be_visible()
        dialog.locator("select").select_option("yolo9-detector")
        dialog.locator("button", has_text="Run").click()

        cancel_btn = dialog.locator("button", has_text="Cancel").first
        expect(cancel_btn).to_be_visible(timeout=5000)
        cancel_btn.click()

        expect(dialog).to_have_count(0, timeout=3000)
        assert cancel_called["n"] >= 1


# ---------------------------------------------------------------------------
# Caps — hard refuse (>5000) and soft warn+confirm (>1000)
# ---------------------------------------------------------------------------

class TestRunModelCaps:
    def _setup(self, page: Page, base_url: str):
        _mock_review_queue(page, PAGE_1_IMAGES)
        _mock_labels(page, [LABEL_CAT, LABEL_DOG])
        _mock_tags(page, [UNVERIFIED_TAG])
        _mock_models(page, ["yolo9-detector"])
        _navigate_to_review(page, base_url)

    def test_hard_cap_refuses_without_opening_dialog(self, page: Page, base_url: str):
        self._setup(page, base_url)

        result = page.evaluate("""async () => {
            const { runModelOnResultsFlow } = await import('/app-static/js/app/review/runModelOnResults.js');
            const bboxes = Array.from({length: 5001}, (_, i) => ({ uuid: 'b' + i, labelUuid: 'lbl-cat', tags: [] }));
            const visibleEntries = bboxes.map(b => ({ kind: 'bbox', bbox: b, img: { uuid: 'img-x' } }));
            await runModelOnResultsFlow({
                fetchImages: null,
                filterPanel: { getState: () => ({ showAllBboxes: true, tagUuids: [], labelUuids: [] }) },
                viewFilter: 'bboxes',
                selectionKind: 'bbox',
                selected: new Set(bboxes.map(b => b.uuid)),
                visibleEntries,
                bboxFilterFn: null,
                labels: new Map(),
                sortBy: 'created_at',
                sortOrder: 'desc',
                refresh: () => {},
            });
            return true;
        }""")
        assert result is True

        expect(page.locator(".run-model-on-results-dialog")).to_have_count(0)
        expect(page.locator("#toast-container")).to_contain_text("Too many candidates")

    def test_warn_cap_requires_confirm_checkbox(self, page: Page, base_url: str):
        self._setup(page, base_url)

        page.evaluate("""async () => {
            const { runModelOnResultsFlow } = await import('/app-static/js/app/review/runModelOnResults.js');
            const bboxes = Array.from({length: 1200}, (_, i) => ({ uuid: 'b' + i, labelUuid: 'lbl-cat', tags: [] }));
            const visibleEntries = bboxes.map(b => ({ kind: 'bbox', bbox: b, img: { uuid: 'img-' + b.uuid } }));
            window.__runFlow = runModelOnResultsFlow({
                fetchImages: null,
                filterPanel: { getState: () => ({ showAllBboxes: true, tagUuids: [], labelUuids: [] }) },
                viewFilter: 'bboxes',
                selectionKind: 'bbox',
                selected: new Set(bboxes.map(b => b.uuid)),
                visibleEntries,
                bboxFilterFn: null,
                labels: new Map(),
                sortBy: 'created_at',
                sortOrder: 'desc',
                refresh: () => {},
            });
        }""")

        dialog = page.locator(".run-model-on-results-dialog")
        expect(dialog).to_be_visible(timeout=3000)
        expect(dialog).to_contain_text("1,200 candidate bbox")
        expect(dialog).to_contain_text("Yes, run anyway")

        # Selecting a model and clicking Run without checking the confirm box warns and does not launch.
        dialog.locator("select").select_option("yolo9-detector")
        dialog.locator("button", has_text="Run").click()
        page.wait_for_timeout(200)
        expect(page.locator("#toast-container")).to_contain_text("confirm")
        expect(dialog).to_be_visible()
