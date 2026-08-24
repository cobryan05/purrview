"""
Test for PR 0.3 of the UI redesign — the top-N Count control added to
openSimilarBboxesModal (similarSearch.js). Like the similar-images modal
(see test_similar_images_count_cap.py, which this file mirrors), this
threshold-based BK-tree query has no server-side result cap of its own; a
generous threshold can return most of the dataset, which is unbounded input
to a whole-set operation (Apply-to-All / "search with current filters").

This is a sibling file rather than a class added to
test_similar_images_count_cap.py: that file is named/scoped for the images
modal specifically (module docstring, mock helper names, fixture data all
say "images"), and there is no existing "bboxes" counterpart to append to —
so a parallel file matches the codebase's existing images/bboxes module
split (similarSearch.js itself exports two top-level functions, and
dissimilarSearch.js is similarly split) better than overloading one file
with two unrelated fixture sets.

Unlike the image modal, the cap here slices the *bbox* match list (the
anchor and results are both bboxes) before `_ensureMatches` groups matches
by image — see the comment in `openSimilarBboxesModal`'s `_runQuery` in
similarSearch.js. To keep that distinction observable and the fixture
simple, each mocked match lives on its own image (one bbox per image), so
counting `[data-bbox-uuid]` cards in the bboxes-view grid directly reflects
how many bboxes survived the cap.

Run with:
    docker compose exec -T mono-dev bash -lc \
      'cd /project/playwright_tests && python -m pytest test_similar_bboxes_count_cap.py \
       -o addopts="" -p no:xdist -q'
"""

import json

from playwright.sync_api import Page, Route, expect

LABEL_CAT = {"uuid": "lbl-cat", "name": "Cat", "color": "#ff0000", "parentUuid": None}

N_MATCHES = 30
MATCHES = [
    {"bboxUuid": f"bbox-{i:03d}", "imageUuid": f"img-{i:03d}", "labelUuid": "lbl-cat", "distance": i}
    for i in range(N_MATCHES)
]
IMAGES_BY_UUID = {
    m["imageUuid"]: {
        "uuid": m["imageUuid"],
        "filename": f"folder/{m['imageUuid']}.jpg",
        "bboxes": [
            {
                "uuid": m["bboxUuid"],
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


def _mock_find_similar_bboxes(page: Page):
    page.route(
        "**/api/images/bbox/find-similar-by-bbox-uuid",
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


def _open_similar_bboxes_modal(page: Page, base_url: str):
    page.goto(base_url)
    page.wait_for_load_state("networkidle")
    page.evaluate("""async () => {
            const { openSimilarBboxesModal } = await import('/app-static/js/ui/similarSearch.js');
            await openSimilarBboxesModal({ bboxes: [{ bboxUuid: 'anchor-bbox' }] });
        }""")
    page.wait_for_selector(".review-grid")


class TestSimilarBboxesCountControl:
    def test_count_control_present_with_default_100(self, page: Page, base_url: str):
        _mock_settings_404(page)
        _mock_find_similar_bboxes(page)
        _mock_batch_by_uuids(page)
        _mock_labels(page)
        _mock_tags(page)
        _mock_image_labels_batch(page)

        _open_similar_bboxes_modal(page, base_url)

        count_input = page.locator("input.modal-count-input")
        expect(count_input).to_be_visible()
        expect(count_input).to_have_value("100")

        # Below the default cap (100) — all 30 matches should be visible.
        expect(page.locator("[data-bbox-uuid]")).to_have_count(N_MATCHES)

    def test_lowering_count_caps_visible_results(self, page: Page, base_url: str):
        _mock_settings_404(page)
        _mock_find_similar_bboxes(page)
        _mock_batch_by_uuids(page)
        _mock_labels(page)
        _mock_tags(page)
        _mock_image_labels_batch(page)

        _open_similar_bboxes_modal(page, base_url)
        expect(page.locator("[data-bbox-uuid]")).to_have_count(N_MATCHES)

        # buildCountControl clamps to a minimum of 10 (searchUtils.js default).
        count_input = page.locator("input.modal-count-input")
        count_input.fill("10")
        count_input.dispatch_event("change")

        expect(page.locator("[data-bbox-uuid]")).to_have_count(10)
