"""The select-mode label list must show what the ACTIVE TRAINING RUN still needs.

With a training run active, every label that run has no verdict on for the
current image is a "gap". The panel already shows ✓ / ✗ / ? per label, but a
reviewer scanning a long list had no way to tell which of those `?`s the run
actually cares about. Gaps now mark the row and the status button that resolves
them, and the sticky header carries a running count.

Contract asserted here:
  - server-reported gaps highlight their row + status button
  - labels outside the run's gap list are never highlighted
  - resolving a gap locally (cycling the status) clears the highlight without a
    refetch — the panel must not wait for a save to stop nagging
  - an image-scope gap is settled by EITHER verdict; an object-scope one is not
    settled by Present alone (it needs a verified box or an explicit Absent),
    mirroring the server's own accounting rule

Run with:
    pytest playwright_tests/test_label_list_training_gaps.py -p no:asyncio -v
"""

import json

from playwright.sync_api import Page, expect

CAT_UUID = "11111111-0000-0000-0000-00000000cafe"
CORRUPT_UUID = "22222222-0000-0000-0000-00000000beef"
DOG_UUID = "33333333-0000-0000-0000-0000000000dd"

LABELS_PAYLOAD = {
    "status": "success",
    "labels": [
        {
            "uuid": CAT_UUID,
            "name": "Cat",
            "color": "#ff0000",
            "labelType": "standard",
            "scope": "object",
            "parentUuid": None,
        },
        {
            "uuid": DOG_UUID,
            "name": "Dog",
            "color": "#00ff00",
            "labelType": "standard",
            "scope": "object",
            "parentUuid": None,
        },
        {
            "uuid": CORRUPT_UUID,
            "name": "Corrupt",
            "color": "#ffaa00",
            "labelType": "standard",
            "scope": "image",
            "parentUuid": None,
        },
    ],
}

TRAINING_CONFIG = {
    "uuid": "tc-1",
    "name": "Nightly",
    "modelConfigNodeUuid": "node-root",
    "datasetUuids": [],
    "hyperparamsJson": {},
}

MODEL_CONFIGS = {
    "status": "success",
    "configs": [
        {
            "uuid": "mc-1",
            "name": "Shape",
            "nodes": [
                {"uuid": "node-root", "labelUuid": None},
                {"uuid": "node-cat", "labelUuid": CAT_UUID},
                {"uuid": "node-corrupt", "labelUuid": CORRUPT_UUID},
            ],
        }
    ],
}


def _route_common(page: Page, gaps: list):
    """Mock everything the panel touches; `gaps` is the image-gaps list for img1."""
    page.route(
        "**/api/labels",
        lambda route: route.fulfill(status=200, content_type="application/json", body=json.dumps(LABELS_PAYLOAD)),
    )
    page.route(
        "**/api/images/**/labels",
        lambda route: route.fulfill(
            status=200, content_type="application/json", body=json.dumps({"status": "success", "labels": []})
        ),
    )
    page.route(
        "**/api/training-configs",
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({"status": "success", "configs": [TRAINING_CONFIG]}),
        ),
    )
    page.route(
        "**/api/model-configs",
        lambda route: route.fulfill(status=200, content_type="application/json", body=json.dumps(MODEL_CONFIGS)),
    )
    # Stats + readiness are fetched by activeTrainingConfig.setActive and, when
    # they land, publish ACTIVE_TRAINING_CONFIG_CHANGED — which remounts the
    # panel. Served immediately so that remount happens before the test clicks,
    # instead of racing it and wiping the unsaved presence state mid-test.
    page.route(
        "**/api/training-configs/tc-1/label-stats",
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({"status": "success", "labelStats": [], "totalImages": 0}),
        ),
    )
    page.route(
        "**/api/training-configs/tc-1/readiness",
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps(
                {
                    "status": "success",
                    "readyCount": 0,
                    "needsReviewCount": 0,
                    "notStartedCount": 0,
                    "datasetUuids": [],
                    "labelUuids": [],
                    "rollUpActive": False,
                }
            ),
        ),
    )
    page.route(
        "**/api/training-configs/tc-1/image-gaps",
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({"status": "success", "gaps": {"img1": gaps} if gaps else {}}),
        ),
    )


# The panel is mounted directly (same harness as test_label_list_conflict_toast.py)
# rather than driven through the canvas modal: the gap logic lives entirely in
# labelList.js, and this keeps the test about that.
_MOUNT_JS = """
async (opts) => {
    const { createLabelList } = await import('/app-static/js/ui/labelList.js');
    const { createImageTaggingRuntime } = await import('/app-static/js/app/image-tagging/runtime.js');
    const { events, EventTypes } = await import('/app-static/js/shared/events/index.js');
    const { clearLabelsCache } = await import('/app-static/js/shared/api/labels.js');
    const { activeTrainingConfig } = await import('/app-static/js/shared/activeTrainingConfig.js');
    const { clearTrainingConfigsCache } = await import('/app-static/js/shared/api/trainingConfigs.js');

    clearLabelsCache();
    clearTrainingConfigsCache();

    const runtime = createImageTaggingRuntime();
    runtime.state.image = { uuid: 'img1', image_path: 'test.png', bboxes: new Map() };

    await activeTrainingConfig.setActive({
        uuid: 'tc-1', name: 'Nightly', modelConfigNodeUuid: 'node-root',
    });

    const div = document.createElement('div');
    div.id = 'gap-test-host';
    document.body.appendChild(div);
    const list = await createLabelList(div, { mode: 'select', runtime });

    events.publish(EventTypes.IMAGE_LOADED_ONTO_CANVAS, { path: 'test.png', runtimeId: runtime.id });
    // Long enough for the stats fetch to land and remount the panel, so the
    // clicks below act on the settled DOM rather than racing a rebuild.
    await new Promise(r => setTimeout(r, 600));

    // Cycle the named labels' status buttons the requested number of times
    // (uncertain -> absent -> present -> uncertain).
    for (const [uuid, clicks] of Object.entries(opts.cycle ?? {})) {
        const row = div.querySelector(`.label-panel-row[data-uuid="${uuid}"]`);
        for (let i = 0; i < clicks; i++) row.querySelector('.label-status-btn').click();
    }
    await new Promise(r => setTimeout(r, 100));

    function readRow(uuid) {
        const row = div.querySelector(`.label-panel-row[data-uuid="${uuid}"]`);
        if (!row) return null;
        const btn = row.querySelector('.label-status-btn');
        return {
            rowGap: row.classList.contains('label-panel-row--gap'),
            btnGap: btn.classList.contains('label-status-btn--gap'),
            status: btn.textContent,
        };
    }

    const summary = div.querySelector('[data-testid="label-gap-summary"]');
    const out = {
        cat: readRow(opts.catUuid),
        dog: readRow(opts.dogUuid),
        corrupt: readRow(opts.corruptUuid),
        summaryHidden: summary.hidden,
        summaryText: summary.textContent,
    };

    list.destroy();
    div.remove();
    await activeTrainingConfig.setActive(null);
    return out;
}
"""


def _mount(page: Page, base_url: str, gaps: list, cycle: dict | None = None):
    _route_common(page, gaps)
    page.goto(base_url)
    page.wait_for_load_state("networkidle")
    return page.evaluate(
        _MOUNT_JS,
        {
            "catUuid": CAT_UUID,
            "dogUuid": DOG_UUID,
            "corruptUuid": CORRUPT_UUID,
            "cycle": cycle or {},
        },
    )


def _gap(uuid: str, name: str, scope: str) -> dict:
    return {"labelUuid": uuid, "labelName": name, "labelScope": scope}


class TestTrainingGapHighlight:
    def test_gap_labels_are_marked_and_others_are_not(self, page: Page, base_url: str):
        res = _mount(
            page,
            base_url,
            gaps=[_gap(CAT_UUID, "Cat", "object"), _gap(CORRUPT_UUID, "Corrupt", "image")],
        )

        assert res["cat"]["rowGap"] is True, res
        assert res["cat"]["btnGap"] is True, res
        assert res["corrupt"]["rowGap"] is True, res
        # Dog is in the vocabulary but not a gap for this run/image.
        assert res["dog"]["rowGap"] is False, res
        assert res["dog"]["btnGap"] is False, res

        assert res["summaryHidden"] is False, res
        assert "2 labels needed" in res["summaryText"], res
        assert "Nightly" in res["summaryText"], res

    def test_no_gaps_leaves_every_row_plain(self, page: Page, base_url: str):
        res = _mount(page, base_url, gaps=[])

        assert res["cat"]["rowGap"] is False, res
        assert res["corrupt"]["rowGap"] is False, res
        assert res["summaryHidden"] is True, res

    def test_marking_an_image_scope_gap_present_clears_it(self, page: Page, base_url: str):
        """An image-scope label is settled by either verdict, so two clicks
        (uncertain -> absent -> present) must clear the highlight."""
        res = _mount(
            page,
            base_url,
            gaps=[_gap(CORRUPT_UUID, "Corrupt", "image")],
            cycle={CORRUPT_UUID: 2},
        )

        assert res["corrupt"]["status"] == "✓", res
        assert res["corrupt"]["rowGap"] is False, res
        assert res["summaryHidden"] is True, res

    def test_object_scope_gap_needs_absent_not_present(self, page: Page, base_url: str):
        """Present alone is not evidence for an object label — the run wants a
        verified box or an explicit Absent, so the highlight must survive it."""
        present = _mount(
            page,
            base_url,
            gaps=[_gap(CAT_UUID, "Cat", "object")],
            cycle={CAT_UUID: 2},  # uncertain -> absent -> present
        )
        assert present["cat"]["status"] == "✓", present
        assert present["cat"]["rowGap"] is True, present

        absent = _mount(
            page,
            base_url,
            gaps=[_gap(CAT_UUID, "Cat", "object")],
            cycle={CAT_UUID: 1},  # uncertain -> absent
        )
        assert absent["cat"]["status"] == "✗", absent
        assert absent["cat"]["rowGap"] is False, absent
        assert absent["summaryHidden"] is True, absent


# ---------------------------------------------------------------------------
# The same panel, reached through the review tab's canvas modal
# ---------------------------------------------------------------------------

TINY_PNG_B64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk" "+A8AAQUBAScY42YAAAAASUVORK5CYII="

MODAL_IMAGE = {"uuid": "img1", "filename": "folder/image1.jpg", "bboxes": [], "tags": []}


class TestGapsInTheCanvasModalSidebar:
    """The canvas modal builds its Labels pane through the very same
    createSidebarPanels → createLabelList(mode:'select') path the image-tagging
    tab uses, so gap highlighting must appear there too. Asserted end-to-end
    (not by mounting the component directly) precisely because "same component"
    is the claim under test."""

    def test_gap_highlight_shows_in_the_modal_label_pane(self, page: Page, base_url: str):
        import base64

        _route_common(page, gaps=[_gap(CAT_UUID, "Cat", "object")])
        page.route(
            "**/api/images/review",
            lambda route: route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps({"status": "success", "images": [MODAL_IMAGE], "hasMore": False, "totalCount": 1}),
            ),
        )
        png_bytes = base64.b64decode(TINY_PNG_B64)
        page.route(
            "**/thumbnail?*",
            lambda route: route.fulfill(status=200, content_type="image/png", body=png_bytes),
        )
        page.route(
            "**/api/images/**/file",
            lambda route: route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps(
                    {
                        "status": "success",
                        "imageBase64": TINY_PNG_B64,
                        "mimeType": "image/png",
                        "image": MODAL_IMAGE,
                    }
                ),
            ),
        )
        # Active before boot, so activeTrainingConfig.init() picks the run up.
        page.add_init_script("sessionStorage.setItem('wtch_active_tc_uuid', 'tc-1');")

        page.goto(base_url)
        page.wait_for_load_state("networkidle")
        page.locator(".main-tab-button[data-tab='tab-review']").click()
        page.wait_for_selector("#review-root .review-grid", timeout=5000)
        page.locator("#review-root .view-btn[data-view-key='images']").click()

        card = page.locator("#review-root [data-image-uuid='img1'][data-kind='image']")
        card.locator("img.thumb").dblclick()

        modal = page.get_by_test_id("canvas-edit-modal")
        expect(modal).to_be_visible(timeout=8000)

        gap_row = modal.locator(f".label-panel-row[data-uuid='{CAT_UUID}'].label-panel-row--gap")
        expect(gap_row).to_have_count(1, timeout=8000)
        expect(modal.locator(".label-status-btn--gap")).to_have_count(1)
        expect(modal.get_by_test_id("label-gap-summary")).to_contain_text("1 label needed")
