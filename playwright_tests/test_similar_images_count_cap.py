"""
Test for M3 of the "unify Review filters" plan — the top-N Count control
added to openSimilarImagesModal (similarSearch.js). Unlike the dissimilar/
diversity searches, this threshold-based BK-tree query had no cap of its own;
a generous threshold could return most of the dataset, which is unbounded
input to a whole-set operation (Apply-to-All / "search with current
filters"). Mirrors dissimilar's UX: a `buildCountControl` (default 100) that
top-N-slices the distance-sorted match list.

Run with:
    docker compose exec -T mono-dev bash -lc \
      'cd /project/playwright_tests && python -m pytest test_similar_images_count_cap.py \
       -o addopts="" -p no:xdist -q'
"""

import json

from playwright.sync_api import Page, Route, expect

LABEL_CAT = {"uuid": "lbl-cat", "name": "Cat", "color": "#ff0000", "parentUuid": None}

N_MATCHES = 20
MATCHES = [{"imageUuid": f"img-{i:03d}", "distance": i} for i in range(N_MATCHES)]
IMAGES_BY_UUID = {
    m["imageUuid"]: {
        "uuid": m["imageUuid"],
        "filename": f"folder/{m['imageUuid']}.jpg",
        "bboxes": [
            {
                "uuid": f"bbox-{m['imageUuid']}",
                "x": 0.1,
                "y": 0.1,
                "width": 0.2,
                "height": 0.2,
                "label": {"uuid": "lbl-cat", "name": "Cat", "color": "#ff0000"},
                "labelUuid": "lbl-cat",
                "tags": [],
            }
        ],
    }
    for m in MATCHES
}


def _mock_settings_404(page: Page):
    page.route("**/api/settings/*", lambda route: route.fulfill(status=404, content_type="application/json", body="{}"))


def _mock_find_similar(page: Page):
    page.route(
        "**/api/images/find-similar-by-image-uuid",
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({"status": "success", "matches": MATCHES}),
        ),
    )


def _mock_batch_by_uuids(page: Page):
    def _handle(route: Route):
        req = json.loads(route.request.post_data or "{}")
        uuids = req.get("imageUuids", [])
        images = [IMAGES_BY_UUID[u] for u in uuids if u in IMAGES_BY_UUID]
        route.fulfill(
            status=200, content_type="application/json", body=json.dumps({"status": "success", "images": images})
        )

    page.route("**/api/images/batch-by-uuids", _handle)


def _mock_labels(page: Page):
    page.route(
        lambda url: url.split("?")[0].endswith("/api/labels") and "api/labels/" not in url,
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({"status": "success", "labels": [LABEL_CAT]}),
        ),
    )


def _mock_tags(page: Page):
    page.route(
        "**/api/tags/list",
        lambda route: route.fulfill(status=200, content_type="application/json", body=json.dumps({"tags": []})),
    )


def _mock_image_labels_batch(page: Page):
    page.route(
        "**/api/images/labels/batch*",
        lambda route: route.fulfill(
            status=200, content_type="application/json", body=json.dumps({"status": "success", "imageLabels": {}})
        ),
    )


def _open_similar_images_modal(page: Page, base_url: str):
    page.goto(base_url)
    page.wait_for_load_state("networkidle")
    page.evaluate("""async () => {
            const { openSimilarImagesModal } = await import('/app-static/js/ui/similarSearch.js');
            await openSimilarImagesModal({ imageUuids: ['anchor-1'] });
        }""")
    page.wait_for_selector(".review-grid", timeout=4000)


class TestSimilarImagesCountControl:
    def test_count_control_present_with_default_100(self, page: Page, base_url: str):
        _mock_settings_404(page)
        _mock_find_similar(page)
        _mock_batch_by_uuids(page)
        _mock_labels(page)
        _mock_tags(page)
        _mock_image_labels_batch(page)

        _open_similar_images_modal(page, base_url)

        count_input = page.locator("input.modal-count-input")
        expect(count_input).to_be_visible()
        expect(count_input).to_have_value("100")

        # Below the default cap (100) — all 20 matches should be visible.
        expect(page.locator("[data-image-uuid]")).to_have_count(N_MATCHES)

    def test_lowering_count_caps_visible_results(self, page: Page, base_url: str):
        _mock_settings_404(page)
        _mock_find_similar(page)
        _mock_batch_by_uuids(page)
        _mock_labels(page)
        _mock_tags(page)
        _mock_image_labels_batch(page)

        _open_similar_images_modal(page, base_url)
        expect(page.locator("[data-image-uuid]")).to_have_count(N_MATCHES)

        # buildCountControl clamps to a minimum of 10 (searchUtils.js default).
        count_input = page.locator("input.modal-count-input")
        count_input.fill("10")
        count_input.dispatch_event("change")

        expect(page.locator("[data-image-uuid]")).to_have_count(10)
