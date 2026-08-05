"""
Tests for Staging step 5 of the preview-card-sharing refactor: the
Run-Model-on-Results preview candidate cards behaving like real review cards.

Asserts that a candidate preview card can be:
  (a) click-selected (gets the `.selected` class via the shared cardGrid
      controller, independent of the "will Apply" checkbox),
  (b) right-clicked to change its proposed label (the proposed chip updates in
      place, mutating only the in-memory candidate — no DB write),
  (c) double-clicked to open the shared canvas geometry editor, and — after
      editing the box's geometry — have the change land back on the in-memory
      candidate so the existing Apply path persists the *edited* geometry
      (bboxGeometry) rather than the original.

The preview candidates are transient in-memory objects; nothing touches the DB
until Apply (the existing batch contract). We drive straight into
`openRelabelConfirmationModal` (bypassing the run-dialog/task-poll plumbing that
test_run_model_on_results.py already covers) and mock the small set of API
endpoints the cards + canvas editor + Apply need.

Run with:
    docker compose --profile dev run --rm --no-deps -w /project whiskerwatcher-dev \
        python3.11 -m pytest playwright_tests/test_run_model_preview_cards.py -q
"""

import base64
import json
import re

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
IMAGE_1 = {"uuid": "img-1", "filename": "folder/image1.jpg", "bboxes": [BBOX_1]}
PAGE_1_IMAGES = [IMAGE_1]

# One existing-box candidate proposing a relabel Cat -> Dog on bbox-1.
CANDIDATE = {
    "bboxUuid": "bbox-1",
    "imageUuid": "img-1",
    "imagePath": "folder/image1.jpg",
    "currentLabelUuid": "lbl-cat",
    "currentLabelName": "Cat",
    "proposedLabelUuid": "lbl-dog",
    "proposedLabelName": "Dog",
    "confidence": 0.81,
    "geometry": {"x": 0.1, "y": 0.1, "width": 0.2, "height": 0.2},
}

_TINY_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00"
    b"\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
)


def _make_png(width: int, height: int) -> str:
    """Build a solid PNG of the given size, base64-encoded (no external deps)."""
    import struct
    import zlib

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)  # 8-bit RGB
    raw = bytearray()
    for _ in range(height):
        raw.append(0)  # filter type 0
        raw.extend((200, 200, 200) * width)
    idat = zlib.compress(bytes(raw), 9)
    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", idat) + chunk(b"IEND", b"")
    return base64.b64encode(png).decode()


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


def _mock_thumbnails(page: Page):
    page.route(
        "**/bbox/*/thumbnail*",
        lambda route: route.fulfill(status=200, content_type="image/png", body=_TINY_PNG),
    )


def _mock_image_file(page: Page):
    """The canvas editor loads the real image + its bboxes via GET .../file;
    hand it back a real-sized image with bbox-1 so the editor has a box to edit."""
    png_b64 = _make_png(100, 100)

    def handle(route: Route):
        route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps(
                {
                    "status": "success",
                    "mimeType": "image/png",
                    "imageBase64": png_b64,
                    "image": {"uuid": "img-1", "filename": "folder/image1.jpg", "bboxes": [BBOX_1]},
                }
            ),
        )

    page.route("**/api/images/*/file", handle)


def _mock_bulk_apply_by_uuids(page: Page):
    calls = []

    def handle(route: Route):
        body = json.loads(route.request.post_data or "{}")
        calls.append(body)
        route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({"status": "success", "modifiedCount": len(body.get("bboxUuids", [])), "failCount": 0}),
        )

    page.route("**/api/images/bbox/bulk-apply-by-uuids", handle)
    return calls


def _mock_invalidate_thumbnails(page: Page):
    page.route(
        "**/api/images/bbox/thumbnails/invalidate",
        lambda route: route.fulfill(
            status=200, content_type="application/json", body=json.dumps({"status": "success"})
        ),
    )


def _navigate_to_review(page: Page, base_url: str):
    page.goto(base_url)
    page.wait_for_load_state("networkidle")
    page.locator(".main-tab-button[data-tab='tab-review']").click()
    page.wait_for_load_state("networkidle")
    page.wait_for_selector("#review-root .review-grid", timeout=4000)


def _open_confirm_modal(page: Page, candidates: list):
    page.evaluate(
        """async (candidates) => {
            const { openRelabelConfirmationModal } = await import('/app-static/js/app/review/runModelOnResults.js');
            const labels = new Map([
                ['lbl-cat', { uuid: 'lbl-cat', name: 'Cat', color: '#ff0000', scope: 'object' }],
                ['lbl-dog', { uuid: 'lbl-dog', name: 'Dog', color: '#00ff00', scope: 'object' }],
            ]);
            window.__refreshCount = 0;
            openRelabelConfirmationModal(candidates, {
                modelName: 'yolo9-detector',
                refresh: () => { window.__refreshCount++; },
                labels,
            });
        }""",
        candidates,
    )


class TestPreviewCardBehavior:
    def _setup(self, page: Page, base_url: str):
        _mock_review_queue(page, PAGE_1_IMAGES)
        _mock_labels(page, [LABEL_CAT, LABEL_DOG])
        _mock_tags(page, [UNVERIFIED_TAG])
        _mock_thumbnails(page)
        _navigate_to_review(page, base_url)

    def test_card_click_selects(self, page: Page, base_url: str):
        """Clicking a candidate card body toggles the shared-controller
        selection (the `.selected` class), independent of the Apply checkbox."""
        self._setup(page, base_url)
        _open_confirm_modal(page, [CANDIDATE])

        card = page.locator(".relabel-confirm-modal .relabel-modal-card").first
        expect(card).to_be_visible()
        expect(card).not_to_have_class(re.compile(r"\bselected\b"))

        # Click the card body (not the checkbox). Selection is applied on a
        # 220ms debounce, so wait for the class to appear.
        card.locator(".card-info").click()
        expect(card).to_have_class(re.compile(r"\bselected\b"), timeout=2000)

        # The Apply checkbox stays checked (its "will Apply" state is unchanged
        # by a selection click).
        expect(card.locator("input[type=checkbox]")).to_be_checked()

    def test_right_click_relabel_updates_proposed_chip(self, page: Page, base_url: str):
        """Right-click -> Apply label -> pick Cat updates the proposed chip in
        place, mutating only the in-memory candidate (no DB call)."""
        self._setup(page, base_url)
        _open_confirm_modal(page, [CANDIDATE])

        modal = page.locator(".relabel-confirm-modal")
        chip = modal.locator(".relabel-modal-proposed-chip").first
        expect(chip).to_have_text("Dog")

        card = modal.locator(".relabel-modal-card").first
        card.locator(".card-info").click(button="right")

        # The candidateBbox menu shows only Relabel / Edit-geometry / Similar /
        # Remove — no verification or tag editing (scope decision 3).
        menu = page.locator(".ctx-menu").first
        expect(menu).to_be_visible(timeout=2000)
        item_text = menu.inner_text()
        assert "Apply label" in item_text
        assert "Edit geometry" in item_text
        assert "Remove from preview" in item_text
        assert "Verification" not in item_text
        assert "Tags" not in item_text

        # Hover "Apply label" to open the label-picker flyout, type to filter,
        # and click the "Cat" row.
        menu.locator(".ctx-menu-item", has_text="Apply label").hover()
        picker_input = page.locator(".ctx-menu input").first
        expect(picker_input).to_be_visible(timeout=2000)
        picker_input.fill("Cat")
        page.locator(".ctx-menu").last.get_by_text("Cat", exact=False).first.click()

        expect(chip).to_have_text("Cat", timeout=2000)

    def test_double_click_opens_canvas_editor(self, page: Page, base_url: str):
        """Double-clicking a candidate card opens the shared canvas geometry
        editor (a `.canvas-modal-dialog`)."""
        self._setup(page, base_url)
        _mock_image_file(page)
        _open_confirm_modal(page, [CANDIDATE])

        card = page.locator(".relabel-confirm-modal .relabel-modal-card").first
        card.locator(".card-info").dblclick()

        expect(page.get_by_test_id("canvas-edit-modal-dialog")).to_be_visible(timeout=5000)

    def test_geometry_edit_lands_on_candidate_and_apply_persists_it(self, page: Page, base_url: str):
        """Full contract: open the editor, move bbox-1 to a new geometry (via a
        real Konva dragend on the modal's stage, which both updates the uiBBox
        and marks the runtime dirty), Save & Close (running the preview's
        in-memory persist — no DB), then Apply. The bulk-apply call must carry
        the *edited* bboxGeometry, proving the edit landed on the candidate and
        the existing Apply path persists it."""
        self._setup(page, base_url)
        _mock_image_file(page)
        apply_calls = _mock_bulk_apply_by_uuids(page)
        _mock_invalidate_thumbnails(page)
        _open_confirm_modal(page, [CANDIDATE])

        card = page.locator(".relabel-confirm-modal .relabel-modal-card").first
        card.locator(".card-info").dblclick()
        dialog = page.get_by_test_id("canvas-edit-modal-dialog")
        expect(dialog).to_be_visible(timeout=5000)

        # The dialog becomes visible synchronously (main.js appends it to the DOM)
        # before the modal's image finishes loading and its BboxView Konva nodes
        # are created (both happen inside an awaited loadImage() call afterwards).
        # Wait for bbox-1's node to actually exist on the modal's stage before
        # trying to drag it, or this races and intermittently fails under load.
        page.wait_for_function(
            """() => {
                const stages = (window.Konva && window.Konva.stages) || [];
                for (const stage of stages) {
                    if (!stage.container()?.closest('.canvas-modal-canvas')) continue;
                    if (stage.findOne('#bbox-1')) return true;
                }
                return false;
            }""",
            timeout=5000,
        )

        # Move bbox-1 to normalized (0.5, 0.6, 0.25, 0.15) by repositioning its
        # Konva node using the view's own viewport transform, then firing the
        # 'dragend' event exactly as a real drag would. The BboxView 'dragend'
        # handler recomputes uiBBox from the node position; the stage-level
        # 'dragend' listener runs pushState() (marking the runtime dirty).
        moved = page.evaluate("""() => {
                const stages = (window.Konva && window.Konva.stages) || [];
                for (const stage of stages) {
                    if (!stage.container()?.closest('.canvas-modal-canvas')) continue;
                    const node = stage.findOne('#bbox-1');
                    if (!node) continue;
                    const vp = node._viewport;
                    const sw = vp.imageWidth * vp.scale;
                    const sh = vp.imageHeight * vp.scale;
                    // Target normalized geometry -> pixel node attrs.
                    node.x(0.5 * sw + vp.offsetX);
                    node.y(0.6 * sh + vp.offsetY);
                    node.width(0.25 * sw);
                    node.height(0.15 * sh);
                    node.fire('dragend', { target: node }, true);
                    return { x: node.uiBBox.x, y: node.uiBBox.y, w: node.uiBBox.width, h: node.uiBBox.height };
                }
                return null;
            }""")
        assert moved is not None, "modal canvas stage / bbox-1 node not found"

        # Save & Close persists (in-memory) and closes the editor.
        dialog.locator("button", has_text="Save & Close").click()
        expect(dialog).to_have_count(0, timeout=5000)

        # Apply the (single) candidate — relabel mode by default.
        modal = page.locator(".relabel-confirm-modal")
        modal.locator("button", has_text="Apply to").click()
        expect(modal).to_have_count(0, timeout=5000)

        assert len(apply_calls) == 1
        call = apply_calls[0]
        assert call["action"] == "relabel"
        assert call["bboxUuids"] == ["bbox-1"]
        # The edited geometry must have landed on the candidate and been sent.
        geom = call["bboxGeometry"]["bbox-1"]
        assert abs(geom["x"] - 0.5) < 1e-3
        assert abs(geom["y"] - 0.6) < 1e-3
        assert abs(geom["width"] - 0.25) < 1e-3
        assert abs(geom["height"] - 0.15) < 1e-3
