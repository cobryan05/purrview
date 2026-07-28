"""
Tests for app/review/runModelOnResults.js — the "Run Model on Results" button,
scope resolution, run dialog, task launch/poll/cancel plumbing (Chunk B), and
the relabel-confirmation modal + apply (Chunk C).

Because the Playwright `live_server` proxies are static mocks with no DB, the
task endpoints (configs/create, instances/start, instances/get, instances/result)
are mocked directly via page.route() with a fake task that "completes" after a
couple of status polls. Bbox thumbnails, bulk-apply-by-uuids, and the thumbnail
invalidation endpoint are likewise mocked (no real DB/detector behind them).

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
    "uuid": "bbox-1",
    "x": 0.1,
    "y": 0.1,
    "width": 0.2,
    "height": 0.2,
    "label": {"uuid": "lbl-cat", "name": "Cat", "color": "#ff0000"},
    "labelUuid": "lbl-cat",
    "tags": [UNVERIFIED_TAG],
}
BBOX_2 = {
    "uuid": "bbox-2",
    "x": 0.3,
    "y": 0.3,
    "width": 0.1,
    "height": 0.1,
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
            status=200,
            content_type="application/json",
            body=json.dumps({"status": "success", "models": all_names}),
        )

    page.route("**/api/models", handle_list)

    def handle_meta(route: Route):
        name = route.request.url.rstrip("/").split("/")[-2]
        kind = "detector" if name in detector_names else "image_classifier"
        route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({"status": "success", "modelKind": kind}),
        )

    page.route("**/api/models/*/meta", handle_meta)


_TINY_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00"
    b"\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
)


def _mock_thumbnails(page: Page):
    page.route(
        "**/bbox/*/thumbnail*",
        lambda route: route.fulfill(status=200, content_type="image/png", body=_TINY_PNG),
    )


def _mock_image_thumbnails(page: Page):
    """Mocks the whole-image thumbnail endpoint used for `isNew` candidates
    (getImageThumbnailUrl), distinct from the bbox-crop thumbnail above."""
    page.route(
        lambda url: "/thumbnail" in url and "/bbox/" not in url,
        lambda route: route.fulfill(status=200, content_type="image/png", body=_TINY_PNG),
    )


def _mock_bulk_apply_by_uuids(page: Page, modified_count=None, fail_count=0):
    calls = []

    def handle(route: Route):
        body = json.loads(route.request.post_data or "{}")
        calls.append(body)
        mc = modified_count if modified_count is not None else len(body.get("bboxUuids", []))
        route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({"status": "success", "modifiedCount": mc, "failCount": fail_count}),
        )

    page.route("**/api/images/bbox/bulk-apply-by-uuids", handle)
    return calls


def _mock_invalidate_thumbnails(page: Page):
    calls = []

    def handle(route: Route):
        calls.append(json.loads(route.request.post_data or "{}"))
        route.fulfill(status=200, content_type="application/json", body=json.dumps({"status": "success"}))

    page.route("**/api/images/bbox/thumbnails/invalidate", handle)
    return calls


class _FakeTask:
    """Tracks poll count so the second `instances/get` call reports COMPLETED."""

    def __init__(self, config_uuid="cfg-1", task_uuid="task-1"):
        self.config_uuid = config_uuid
        self.task_uuid = task_uuid
        self.create_bodies = []
        self.poll_count = 0

    def install(self, page: Page, candidates: list, stats: dict = None):
        def handle_create(route: Route):
            body = json.loads(route.request.post_data or "{}")
            self.create_bodies.append(body)
            route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps({"status": "success", "configUuid": self.config_uuid}),
            )

        page.route("**/api/tasks/configs/create", handle_create)

        def handle_start(route: Route):
            route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps({"status": "success", "taskUuid": self.task_uuid}),
            )

        page.route("**/api/tasks/instances/start", handle_start)

        def handle_get(route: Route):
            self.poll_count += 1
            status = "running" if self.poll_count < 2 else "completed"
            progress = 50.0 if status == "running" else 100.0
            route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps(
                    {
                        "status": "success",
                        "tasks": {
                            self.task_uuid: {
                                "config": {
                                    "uuid": self.config_uuid,
                                    "typename": "PreviewModelRelabelTask",
                                    "name": "x",
                                    "params": {},
                                },
                                "instance": {"uuid": self.task_uuid, "status": status},
                                "progress": progress,
                                "statusMessage": "",
                                "statusActions": [],
                                "logLines": [],
                                "isPausedInPlace": False,
                            }
                        },
                    }
                ),
            )

        page.route("**/api/tasks/instances/get", handle_get)

        def handle_result(route: Route):
            data = {
                "images_processed": 2,
                "bbox_count": len(candidates),
                "candidate_count": len(candidates),
                "candidates": candidates,
                "images_failed": 0,
                "sample_errors": [],
                "no_detection_count": 0,
                "no_op_count": 0,
            }
            data.update(stats or {})
            route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps(
                    {
                        "status": "success",
                        "results": {self.task_uuid: {"data": data}},
                    }
                ),
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
        "bboxUuid": "bbox-1",
        "imageUuid": "img-1",
        "imagePath": "folder/image1.jpg",
        "currentLabelUuid": "lbl-cat",
        "currentLabelName": "Cat",
        "proposedLabelUuid": "lbl-dog",
        "proposedLabelName": "Dog",
        "confidence": 0.81,
    },
    {
        "bboxUuid": "bbox-2",
        "imageUuid": "img-2",
        "imagePath": "folder/image2.jpg",
        "currentLabelUuid": "lbl-dog",
        "currentLabelName": "Dog",
        "proposedLabelUuid": "lbl-cat",
        "proposedLabelName": "Cat",
        "confidence": 0.77,
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
        _mock_thumbnails(page)
        fake_task = _FakeTask()
        fake_task.install(page, CANDIDATES)

        _navigate_to_review(page, base_url)
        _run_model_btn(page).click()

        dialog = page.locator(".run-model-on-results-dialog")
        expect(dialog).to_be_visible()
        dialog.locator(".run-model-on-results-model-select").select_option("yolo9-detector")
        dialog.locator("button", has_text="Run").click()

        # Poll loop runs on a ~700ms interval; give it a couple of rounds.
        # Handoff lands in the Chunk-C confirmation modal.
        confirm_modal = page.locator(".relabel-confirm-modal")
        expect(confirm_modal).to_be_visible(timeout=5000)
        expect(confirm_modal).to_contain_text("2 candidates from yolo9-detector")

        assert len(fake_task.create_bodies) == 1
        body = fake_task.create_bodies[0]
        assert body["typename"] == "PreviewModelRelabelTask"
        assert body["persistent"] is False
        assert body["isTemporary"] is True
        params = body["params"]
        assert params["model_name"] == "yolo9-detector"
        assert params["crop_mode"] == "image"
        assert params["conf_thresh"] == 0.5
        assert params["iou_thresh"] == 0.5
        assert sorted(params["bbox_uuids"]) == ["bbox-1", "bbox-2"]
        assert sorted(params["image_uuids"]) == ["img-1", "img-2"]

        # Run dialog closed after handoff (confirmation modal took over).
        expect(dialog).to_have_count(0)

    def test_bbox_crop_mode_sent_and_hides_iou_slider(self, page: Page, base_url: str):
        _mock_review_queue(page, PAGE_1_IMAGES)
        _mock_labels(page, [LABEL_CAT, LABEL_DOG])
        _mock_tags(page, [UNVERIFIED_TAG])
        _mock_models(page, ["yolo9-detector"])
        _mock_thumbnails(page)
        fake_task = _FakeTask()
        fake_task.install(page, CANDIDATES)

        _navigate_to_review(page, base_url)
        _run_model_btn(page).click()

        dialog = page.locator(".run-model-on-results-dialog")
        expect(dialog).to_be_visible()
        dialog.locator(".run-model-on-results-model-select").select_option("yolo9-detector")
        dialog.locator(".run-model-on-results-crop-mode-select").select_option("bbox")

        iou_row = dialog.locator("text=Min IoU:").locator("..")
        expect(iou_row).to_be_hidden()

        dialog.locator("button", has_text="Run").click()

        confirm_modal = page.locator(".relabel-confirm-modal")
        expect(confirm_modal).to_be_visible(timeout=5000)

        assert len(fake_task.create_bodies) == 1
        params = fake_task.create_bodies[0]["params"]
        assert params["crop_mode"] == "bbox"

    def test_cancel_stops_polling_and_closes_dialog(self, page: Page, base_url: str):
        _mock_review_queue(page, PAGE_1_IMAGES)
        _mock_labels(page, [LABEL_CAT, LABEL_DOG])
        _mock_tags(page, [UNVERIFIED_TAG])
        _mock_models(page, ["yolo9-detector"])

        # A task that never completes, so the only way out is Cancel.
        cancel_called = {"n": 0}

        def handle_create(route: Route):
            route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps({"status": "success", "configUuid": "cfg-1"}),
            )

        page.route("**/api/tasks/configs/create", handle_create)

        def handle_start(route: Route):
            route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps({"status": "success", "taskUuid": "task-1"}),
            )

        page.route("**/api/tasks/instances/start", handle_start)

        def handle_get(route: Route):
            route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps(
                    {
                        "status": "success",
                        "tasks": {
                            "task-1": {
                                "config": {
                                    "uuid": "cfg-1",
                                    "typename": "PreviewModelRelabelTask",
                                    "name": "x",
                                    "params": {},
                                },
                                "instance": {"uuid": "task-1", "status": "running"},
                                "progress": 10.0,
                                "statusMessage": "",
                                "statusActions": [],
                                "logLines": [],
                                "isPausedInPlace": False,
                            }
                        },
                    }
                ),
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
        dialog.locator(".run-model-on-results-model-select").select_option("yolo9-detector")
        dialog.locator("button", has_text="Run").click()

        cancel_btn = dialog.locator("button", has_text="Cancel").first
        expect(cancel_btn).to_be_visible(timeout=5000)
        cancel_btn.click()

        expect(dialog).to_have_count(0, timeout=3000)
        assert cancel_called["n"] >= 1


# ---------------------------------------------------------------------------
# Large-run warning — no hard ceiling. Below WARN_THRESHOLD (1000) the run
# dialog opens directly; above it, a warning modal lets the user cap how many
# candidates to process (blank = all) or cancel.
# ---------------------------------------------------------------------------


class TestRunModelCaps:
    def _setup(self, page: Page, base_url: str):
        _mock_review_queue(page, PAGE_1_IMAGES)
        _mock_labels(page, [LABEL_CAT, LABEL_DOG])
        _mock_tags(page, [UNVERIFIED_TAG])
        _mock_models(page, ["yolo9-detector"])
        _navigate_to_review(page, base_url)

    def _start_flow(self, page: Page, count: int):
        page.evaluate(
            """async (count) => {
            const { runModelOnResultsFlow } = await import('/app-static/js/app/review/runModelOnResults.js');
            const bboxes = Array.from({length: count}, (_, i) => ({ uuid: 'b' + i, labelUuid: 'lbl-cat', tags: [] }));
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
        }""",
            count,
        )

    def test_below_warn_threshold_opens_dialog_directly(self, page: Page, base_url: str):
        self._setup(page, base_url)
        self._start_flow(page, 500)

        # No warning modal for small runs — straight to the run dialog.
        expect(page.locator(".run-model-large-warning-dialog")).to_have_count(0)
        dialog = page.locator(".run-model-on-results-dialog")
        expect(dialog).to_be_visible(timeout=3000)
        expect(dialog).to_contain_text("500 candidate bbox")

    def test_above_warn_threshold_process_all(self, page: Page, base_url: str):
        self._setup(page, base_url)
        self._start_flow(page, 1200)

        warning = page.locator(".run-model-large-warning-dialog")
        expect(warning).to_be_visible(timeout=3000)
        expect(warning).to_contain_text("1,200 candidate bbox")

        # Leave the count blank → process all; the run dialog shows the full count.
        warning.locator("button", has_text="Continue").click()
        dialog = page.locator(".run-model-on-results-dialog")
        expect(dialog).to_be_visible(timeout=3000)
        expect(dialog).to_contain_text("1,200 candidate bbox")

    def test_above_warn_threshold_caps_count(self, page: Page, base_url: str):
        self._setup(page, base_url)
        self._start_flow(page, 1200)

        warning = page.locator(".run-model-large-warning-dialog")
        expect(warning).to_be_visible(timeout=3000)

        # Cap to the first 300 → the run dialog reflects the truncated scope.
        warning.locator(".run-model-large-warning-limit").fill("300")
        warning.locator("button", has_text="Continue").click()
        dialog = page.locator(".run-model-on-results-dialog")
        expect(dialog).to_be_visible(timeout=3000)
        expect(dialog).to_contain_text("300 candidate bbox")
        expect(dialog).to_contain_text("300 image")

    def test_warning_cancel_aborts_run(self, page: Page, base_url: str):
        self._setup(page, base_url)
        self._start_flow(page, 1200)

        warning = page.locator(".run-model-large-warning-dialog")
        expect(warning).to_be_visible(timeout=3000)
        warning.locator("button", has_text="Cancel").click()

        expect(warning).to_have_count(0)
        expect(page.locator(".run-model-on-results-dialog")).to_have_count(0)


# ---------------------------------------------------------------------------
# Chunk C: the relabel-confirmation modal + apply
# ---------------------------------------------------------------------------

# Three candidates across two images, deliberately *not* pre-sorted by
# confidence, with two candidates sharing a proposed label (lbl-dog) so
# grouping-by-proposed-label is exercised.
CANDIDATES_MIXED = [
    {
        "bboxUuid": "bbox-1",
        "imageUuid": "img-1",
        "imagePath": "folder/image1.jpg",
        "currentLabelUuid": "lbl-cat",
        "currentLabelName": "Cat",
        "proposedLabelUuid": "lbl-dog",
        "proposedLabelName": "Dog",
        "confidence": 0.55,
    },
    {
        "bboxUuid": "bbox-2",
        "imageUuid": "img-2",
        "imagePath": "folder/image2.jpg",
        "currentLabelUuid": "lbl-dog",
        "currentLabelName": "Dog",
        "proposedLabelUuid": "lbl-cat",
        "proposedLabelName": "Cat",
        "confidence": 0.91,
    },
    {
        "bboxUuid": "bbox-3",
        "imageUuid": "img-1",
        "imagePath": "folder/image1.jpg",
        "currentLabelUuid": "lbl-cat",
        "currentLabelName": "Cat",
        "proposedLabelUuid": "lbl-dog",
        "proposedLabelName": "Dog",
        "confidence": 0.70,
    },
]


def _open_confirm_modal(page: Page, candidates: list, model_name: str = "yolo9-detector", stats: dict = None):
    """Drives straight into `openRelabelConfirmationModal`, bypassing the run
    dialog/task-poll plumbing that Chunk B already covers. Stashes a
    `window.__refreshCount` counter incremented by `ctx.refresh`."""
    page.evaluate(
        """async ([candidates, modelName, stats]) => {
            const { openRelabelConfirmationModal } = await import('/app-static/js/app/review/runModelOnResults.js');
            window.__refreshCount = 0;
            const labels = new Map([
                ['lbl-cat', { uuid: 'lbl-cat', name: 'Cat', color: '#ff0000' }],
                ['lbl-dog', { uuid: 'lbl-dog', name: 'Dog', color: '#00ff00' }],
            ]);
            window.__confirmModalPromise = openRelabelConfirmationModal(candidates, {
                modelName,
                confThresh: 0.5,
                iouThresh: 0.5,
                refresh: () => { window.__refreshCount++; },
                labels,
                stats,
            });
        }""",
        [candidates, model_name, stats],
    )


class TestRelabelConfirmationModal:
    def _setup(self, page: Page, base_url: str):
        _mock_review_queue(page, PAGE_1_IMAGES)
        _mock_labels(page, [LABEL_CAT, LABEL_DOG])
        _mock_tags(page, [UNVERIFIED_TAG])
        _mock_thumbnails(page)
        _navigate_to_review(page, base_url)

    def test_cards_rendered_sorted_by_confidence_desc(self, page: Page, base_url: str):
        self._setup(page, base_url)
        _open_confirm_modal(page, CANDIDATES_MIXED)

        modal = page.locator(".relabel-confirm-modal")
        expect(modal).to_be_visible()
        expect(modal).to_contain_text("3 candidates from yolo9-detector")

        cards = modal.locator(".relabel-modal-card")
        expect(cards).to_have_count(3)
        # Sorted desc by confidence: bbox-2 (91%), bbox-3 (70%), bbox-1 (55%).
        expect(cards.nth(0)).to_contain_text("91%")
        expect(cards.nth(1)).to_contain_text("70%")
        expect(cards.nth(2)).to_contain_text("55%")

    def test_uncheck_row_reduces_apply_count(self, page: Page, base_url: str):
        self._setup(page, base_url)
        _open_confirm_modal(page, CANDIDATES_MIXED)

        modal = page.locator(".relabel-confirm-modal")
        apply_btn = modal.locator("button", has_text="Apply to")
        expect(apply_btn).to_contain_text("Apply to 3 checked")

        modal.locator(".relabel-modal-card").nth(0).locator("input[type=checkbox]").uncheck()
        expect(apply_btn).to_contain_text("Apply to 2 checked")

    def test_select_none_disables_apply(self, page: Page, base_url: str):
        self._setup(page, base_url)
        _open_confirm_modal(page, CANDIDATES_MIXED)

        modal = page.locator(".relabel-confirm-modal")
        modal.locator("button", has_text="Select none").click()
        apply_btn = modal.locator("button", has_text="Apply to")
        expect(apply_btn).to_contain_text("Apply to 0 checked")
        expect(apply_btn).to_be_disabled()

    def test_delete_mode_greys_proposed_column_and_applies_delete(self, page: Page, base_url: str):
        self._setup(page, base_url)
        calls = _mock_bulk_apply_by_uuids(page)
        invalidate_calls = _mock_invalidate_thumbnails(page)
        _open_confirm_modal(page, CANDIDATES_MIXED)

        modal = page.locator(".relabel-confirm-modal")
        modal.locator("input[type=radio][value=delete]").check()
        proposed_chip = modal.locator(".relabel-modal-proposed-chip").first
        expect(proposed_chip).to_have_css("opacity", "0.35")

        modal.locator("button", has_text="Apply to").click()
        expect(modal).to_have_count(0, timeout=3000)

        assert len(calls) == 1
        assert calls[0]["action"] == "delete"
        assert sorted(calls[0]["bboxUuids"]) == ["bbox-1", "bbox-2", "bbox-3"]
        assert len(invalidate_calls) == 1
        assert sorted(invalidate_calls[0]["bboxUuids"]) == ["bbox-1", "bbox-2", "bbox-3"]
        assert page.evaluate("window.__refreshCount") == 1

    def test_relabel_groups_by_proposed_label(self, page: Page, base_url: str):
        self._setup(page, base_url)
        calls = _mock_bulk_apply_by_uuids(page)
        _mock_invalidate_thumbnails(page)
        _open_confirm_modal(page, CANDIDATES_MIXED)

        modal = page.locator(".relabel-confirm-modal")
        modal.locator("button", has_text="Apply to").click()
        expect(modal).to_have_count(0, timeout=3000)

        # Two distinct proposed labels among the three checked candidates:
        # lbl-dog (bbox-1, bbox-3) and lbl-cat (bbox-2) -> one call each.
        assert len(calls) == 2
        by_label = {c["labelUuid"]: sorted(c["bboxUuids"]) for c in calls}
        assert by_label == {
            "lbl-dog": ["bbox-1", "bbox-3"],
            "lbl-cat": ["bbox-2"],
        }
        for c in calls:
            assert c["action"] == "relabel"
        assert page.evaluate("window.__refreshCount") == 1

    def test_failcount_is_surfaced_without_crashing(self, page: Page, base_url: str):
        self._setup(page, base_url)
        _mock_bulk_apply_by_uuids(page, modified_count=1, fail_count=2)
        _mock_invalidate_thumbnails(page)
        _open_confirm_modal(page, CANDIDATES_MIXED)

        modal = page.locator(".relabel-confirm-modal")
        modal.locator("input[type=radio][value=delete]").check()
        modal.locator("button", has_text="Apply to").click()

        expect(modal).to_have_count(0, timeout=3000)
        expect(page.locator("#toast-container")).to_contain_text("2 failed")
        assert page.evaluate("window.__refreshCount") == 1

    def test_empty_candidates_shows_empty_state(self, page: Page, base_url: str):
        self._setup(page, base_url)
        _open_confirm_modal(page, [])

        modal = page.locator(".relabel-confirm-modal")
        expect(modal).to_be_visible()
        expect(modal).to_contain_text("No candidates above threshold")
        expect(modal.locator(".relabel-modal-card")).to_have_count(0)

    def test_empty_candidates_with_stats_explains_why(self, page: Page, base_url: str):
        self._setup(page, base_url)
        _open_confirm_modal(
            page,
            [],
            stats={
                "imagesFailed": 3,
                "noDetectionCount": 5,
                "noOpCount": 7,
                "sampleErrors": ["img-1: FileNotFoundError: /data/img-1.jpg"],
            },
        )

        modal = page.locator(".relabel-confirm-modal")
        expect(modal).to_be_visible()
        expect(modal).to_contain_text("7 bbox(es) already matched the model's prediction")
        expect(modal).to_contain_text("5 bbox(es) had no matching detection")
        expect(modal).to_contain_text("3 image(s) failed to load/process")
        expect(modal).to_contain_text("FileNotFoundError")

    def test_new_box_candidate_draws_geometry_overlay(self, page: Page, base_url: str):
        """New-box candidates have no bboxUuid, so the preview falls back to a
        whole-image thumbnail — it must still draw the proposed geometry so the
        user can see where the box will land, not just that a box exists."""
        self._setup(page, base_url)
        _mock_image_thumbnails(page)
        candidate = {
            "bboxUuid": None,
            "imageUuid": "img-1",
            "imagePath": "folder/image1.jpg",
            "currentLabelUuid": None,
            "currentLabelName": None,
            "proposedLabelUuid": "lbl-dog",
            "proposedLabelName": "Dog",
            "confidence": 0.8,
            "geometry": {"x": 0.2, "y": 0.3, "width": 0.4, "height": 0.1},
            "isNew": True,
        }
        _open_confirm_modal(page, [candidate])

        modal = page.locator(".relabel-confirm-modal")
        expect(modal).to_be_visible()
        card = modal.locator(".relabel-modal-card").first
        expect(card).to_contain_text("＋ NEW")

        overlay = card.locator("canvas.thumb-overlay")
        expect(overlay).to_have_count(1)
        # The overlay only gets sized once the thumb image has loaded and
        # laid out — wait for that instead of asserting immediately.
        overlay.page.wait_for_function("el => el.width > 0", arg=overlay.element_handle())
        assert overlay.evaluate("el => el.height") > 0
