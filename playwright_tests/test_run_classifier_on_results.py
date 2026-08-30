"""Tests for the classifier branch of "Run Model on Results"
(app/review/runModelOnResults.js's classifier path + runClassifierOnResults.js's
`openClassifierTagConfirmationModal`).

Mirrors test_run_model_on_results.py's mocking approach: task endpoints
(configs/create, instances/start, instances/get, instances/result) are mocked
directly via page.route() with a fake task that "completes" after a couple of
status polls; image thumbnails, tags, and the image-level bulk-apply-by-uuids
endpoint are likewise mocked (no real DB/model behind them).

Run with:
    docker compose --profile dev run --rm --no-deps -w /project purrview-dev \
        python3.11 -m pytest playwright_tests/test_run_classifier_on_results.py -q
"""

import json

from playwright.sync_api import Page, Route, expect

LABEL_CAT = {"uuid": "lbl-cat", "name": "Cat", "color": "#ff0000", "parentUuid": None}
LABEL_CORRUPT = {"uuid": "lbl-corrupt", "name": "Corrupt", "color": "#888888", "parentUuid": None}

UNVERIFIED_TAG_UUID = "00000000-0000-0000-0000-000000000001"
UNVERIFIED_TAG = {"uuid": UNVERIFIED_TAG_UUID, "name": "Unverified", "color": "#888", "kind": "system", "icon": None}
CORRUPT_TAG = {"uuid": "tag-corrupt", "name": "Corrupt", "color": "#cc0000", "kind": "user", "icon": None}

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
IMAGE_1 = {"uuid": "img-1", "filename": "folder/image1.jpg", "bboxes": [BBOX_1]}
IMAGE_2 = {"uuid": "img-2", "filename": "folder/image2.jpg", "bboxes": [BBOX_1 | {"uuid": "bbox-2"}]}

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


def _mock_image_thumbnails(page: Page):
    page.route(
        lambda url: "/thumbnail" in url and "/bbox/" not in url,
        lambda route: route.fulfill(status=200, content_type="image/png", body=_TINY_PNG),
    )


def _mock_image_bulk_apply(page: Page, modified_count=None, fail_count=0):
    calls = []

    def handle(route: Route):
        body = json.loads(route.request.post_data or "{}")
        calls.append(body)
        mc = modified_count if modified_count is not None else len(body.get("imageUuids", []))
        route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({"status": "success", "modifiedCount": mc, "failCount": fail_count}),
        )

    page.route("**/api/images/bulk-apply-by-uuids", handle)
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
                                    "typename": "PreviewClassifierMatchTask",
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
                "candidate_count": len(candidates),
                "candidates": candidates,
                "images_failed": 0,
                "sample_errors": [],
                "no_match_count": 0,
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
    page.locator(".main-tab-button[data-tab='tab-image-tagging']").click()
    page.locator("#tab-image-tagging .section-tab-button[data-tab='tab-review']").click()
    page.wait_for_load_state("networkidle")
    page.wait_for_selector("#review-root .review-grid")


def _run_model_btn(page: Page):
    return page.get_by_test_id("review-run-model-on-results")


def _check_model_in_dialog(dialog, name: str):
    """Checks the checkbox for `name` in the (multi-select) model/classifier/group list."""
    item = dialog.locator(".run-model-on-results-model-list .inference-model-item", has_text=name)
    item.locator("input[type=checkbox]").check()


CLASSIFIER_CANDIDATES = [
    {
        "imageUuid": "img-1",
        "imagePath": "folder/image1.jpg",
        "predictedLabelUuid": "lbl-corrupt",
        "predictedLabelName": "Corrupt",
        "confidence": 0.91,
    },
    {
        "imageUuid": "img-2",
        "imagePath": "folder/image2.jpg",
        "predictedLabelUuid": "lbl-corrupt",
        "predictedLabelName": "Corrupt",
        "confidence": 0.68,
    },
]


class TestModelPickerKindAwareness:
    def test_classifier_selection_hides_detector_only_controls(self, page: Page, base_url: str):
        _mock_review_queue(page, PAGE_1_IMAGES)
        _mock_labels(page, [LABEL_CAT, LABEL_CORRUPT])
        _mock_tags(page, [UNVERIFIED_TAG, CORRUPT_TAG])
        _mock_models(page, ["yolo9-detector"], ["corrupt-classifier"])

        _navigate_to_review(page, base_url)
        _run_model_btn(page).click()

        dialog = page.locator(".run-model-on-results-dialog")
        expect(dialog).to_be_visible()
        _check_model_in_dialog(dialog, "corrupt-classifier")

        expect(dialog.locator(".run-model-on-results-crop-mode-select")).to_be_hidden()
        expect(dialog.locator("text=Min IoU:")).to_be_hidden()
        expect(dialog.locator(".run-model-on-results-new-box-mode")).to_be_hidden()

        _check_model_in_dialog(dialog, "yolo9-detector")
        expect(dialog.locator(".run-model-on-results-crop-mode-select")).to_be_visible()
        expect(dialog.locator(".run-model-on-results-new-box-mode")).to_be_visible()


class TestClassifierRunFlow:
    def test_run_launches_classifier_task_with_expected_params(self, page: Page, base_url: str):
        _mock_review_queue(page, PAGE_1_IMAGES)
        _mock_labels(page, [LABEL_CAT, LABEL_CORRUPT])
        _mock_tags(page, [UNVERIFIED_TAG, CORRUPT_TAG])
        _mock_models(page, ["yolo9-detector"], ["corrupt-classifier"])
        fake_task = _FakeTask()
        fake_task.install(page, CLASSIFIER_CANDIDATES)

        _navigate_to_review(page, base_url)
        # Registered after navigation — "/thumbnail" is a substring of the
        # thumbnailHoverPreview.js module URL too, so mocking it before the
        # page's own module graph has loaded breaks that import.
        _mock_image_thumbnails(page)
        _run_model_btn(page).click()

        dialog = page.locator(".run-model-on-results-dialog")
        expect(dialog).to_be_visible()
        _check_model_in_dialog(dialog, "corrupt-classifier")
        dialog.locator("button", has_text="Run").click()

        expect(dialog).to_have_count(0, timeout=5000)
        confirm_dialog = page.locator(".classifier-match-confirm-modal")
        expect(confirm_dialog).to_be_visible()
        expect(confirm_dialog).to_contain_text("2 matches from corrupt-classifier")

        assert len(fake_task.create_bodies) == 1
        body = fake_task.create_bodies[0]
        assert body["typename"] == "PreviewClassifierMatchTask"
        params = body["params"]
        assert params["model_name"] == "corrupt-classifier"
        assert set(params["image_uuids"]) == {"img-1", "img-2"}
        assert "bbox_uuids" not in params
        assert "iou_thresh" not in params
        assert "crop_mode" not in params

    def test_apply_opens_bulk_modal_and_adds_tag(self, page: Page, base_url: str):
        _mock_review_queue(page, PAGE_1_IMAGES)
        _mock_labels(page, [LABEL_CAT, LABEL_CORRUPT])
        _mock_tags(page, [UNVERIFIED_TAG, CORRUPT_TAG])
        _mock_models(page, ["yolo9-detector"], ["corrupt-classifier"])
        fake_task = _FakeTask()
        fake_task.install(page, CLASSIFIER_CANDIDATES)
        calls = _mock_image_bulk_apply(page)

        _navigate_to_review(page, base_url)
        _mock_image_thumbnails(page)
        _run_model_btn(page).click()

        dialog = page.locator(".run-model-on-results-dialog")
        _check_model_in_dialog(dialog, "corrupt-classifier")
        dialog.locator("button", has_text="Run").click()

        confirm_dialog = page.locator(".classifier-match-confirm-modal")
        expect(confirm_dialog).to_be_visible(timeout=5000)
        confirm_dialog.locator("button", has_text="Apply to 2 checked").click()

        # Generic image bulk-action modal opens on top; pick "Add Tags" + Corrupt.
        bulk_modal = page.locator("text=Apply to 2 selected images").locator("..")
        expect(bulk_modal).to_be_visible(timeout=3000)
        bulk_modal.locator("label", has_text="Add Tags").locator("input[type=radio]").check()
        # The tag picker renders inline (no trigger to open), and is scoped
        # explicitly: the label picker in the same modal also has a "Corrupt" row.
        bulk_modal.locator(".bulk-modal-picker--tags .ms-item", has_text="Corrupt").locator(
            "input[type=checkbox]"
        ).check()
        bulk_modal.locator("button", has_text="Apply").click()

        expect(page.locator("#toast-container")).to_contain_text("Applied to 2 image(s)", timeout=3000)
        assert len(calls) == 1
        assert calls[0]["action"] == "addTags"
        assert calls[0]["tagUuids"] == ["tag-corrupt"]
        assert set(calls[0]["imageUuids"]) == {"img-1", "img-2"}

    def test_empty_candidates_shows_empty_state(self, page: Page, base_url: str):
        _mock_review_queue(page, PAGE_1_IMAGES)
        _mock_labels(page, [LABEL_CAT, LABEL_CORRUPT])
        _mock_tags(page, [UNVERIFIED_TAG, CORRUPT_TAG])
        _mock_models(page, ["yolo9-detector"], ["corrupt-classifier"])
        fake_task = _FakeTask()
        fake_task.install(page, [], stats={"no_match_count": 2})

        _navigate_to_review(page, base_url)
        _run_model_btn(page).click()

        dialog = page.locator(".run-model-on-results-dialog")
        _check_model_in_dialog(dialog, "corrupt-classifier")
        dialog.locator("button", has_text="Run").click()

        confirm_dialog = page.locator(".classifier-match-confirm-modal")
        expect(confirm_dialog).to_be_visible(timeout=5000)
        expect(confirm_dialog).to_contain_text("No matches from corrupt-classifier")
        expect(confirm_dialog).to_contain_text("2 image(s) below threshold")
