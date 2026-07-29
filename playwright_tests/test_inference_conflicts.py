"""
Tests for the inference-result conflict resolution in canvas/image.js:
addInferenceResults() de-duplicates a new detection against an existing bbox
when they significantly overlap (same IOU-matched box), optionally replacing
it when labels differ and `replaceExisting` is set, and flashes a temporary
glow to flag the outcome either way.

Run with:
    pytest playwright_tests/test_inference_conflicts.py -p no:xdist --override-ini="addopts=" -v
"""
import pytest
from playwright.sync_api import Page

UNVERIFIED_TAG_UUID = "00000000-0000-0000-0000-000000000001"

# Creates a runtime + canvas + blank 400x300 image (scale=2, offset=0,0).
_SETUP = """
    const div = document.createElement('div');
    div.style.cssText = 'width:800px;height:600px;position:fixed;top:0;left:0;';
    document.body.appendChild(div);

    const runtime = createImageTaggingRuntime();
    runtime.setCanvasContainer(div);

    const imgCanvas = document.createElement('canvas');
    imgCanvas.width = 400; imgCanvas.height = 300;
    const img = new Image();
    img.src = imgCanvas.toDataURL();
    await new Promise(r => { img.onload = r; });

    runtime.setImage({
        uuid: 'img', filename: 'test.png', image_path: '/test.png',
        img, bboxes: new Map(), dirty: false,
    });
"""


def _detection(uuid, x, y, width, height, label_uuid=None, class_str=None, confidence=0.9):
    return {
        "confidence": confidence,
        "bbox": {
            "uuid": uuid,
            "x": x, "y": y, "width": width, "height": height,
            "labelUuid": label_uuid,
            "classStr": class_str,
            "tagUuids": [],
        },
    }


# ---------------------------------------------------------------------------
# computeIou / labelKey — pure logic
# ---------------------------------------------------------------------------

class TestComputeIou:
    def test_identical_rects_have_iou_one(self, shared_page: Page):
        page = shared_page
        result = page.evaluate("""async () => {
            const { computeIou } = await import('/app-static/js/app/image-tagging/canvas/image.js');
            const a = { x: 0.1, y: 0.1, width: 0.2, height: 0.2 };
            return computeIou(a, { ...a });
        }""")
        assert result == pytest.approx(1.0)

    def test_disjoint_rects_have_iou_zero(self, shared_page: Page):
        page = shared_page
        result = page.evaluate("""async () => {
            const { computeIou } = await import('/app-static/js/app/image-tagging/canvas/image.js');
            const a = { x: 0, y: 0, width: 0.1, height: 0.1 };
            const b = { x: 0.5, y: 0.5, width: 0.1, height: 0.1 };
            return computeIou(a, b);
        }""")
        assert result == 0

    def test_partial_overlap(self, shared_page: Page):
        page = shared_page
        result = page.evaluate("""async () => {
            const { computeIou } = await import('/app-static/js/app/image-tagging/canvas/image.js');
            // a: (0,0)-(0.2,0.2) area 0.04; b: (0.1,0.1)-(0.3,0.3) area 0.04
            // intersection: (0.1,0.1)-(0.2,0.2) area 0.01; union = 0.04+0.04-0.01=0.07
            const a = { x: 0, y: 0, width: 0.2, height: 0.2 };
            const b = { x: 0.1, y: 0.1, width: 0.2, height: 0.2 };
            return computeIou(a, b);
        }""")
        assert result == pytest.approx(0.01 / 0.07)


class TestLabelKey:
    def test_same_uuid_matches_regardless_of_text(self, shared_page: Page):
        page = shared_page
        result = page.evaluate("""async () => {
            const { labelKey } = await import('/app-static/js/app/image-tagging/canvas/image.js');
            return labelKey({ uuid: 'cat-uuid', text: 'Cat (91%)' }) ===
                   labelKey({ uuid: 'cat-uuid', text: 'Cat (55%)' });
        }""")
        assert result is True

    def test_unlabeled_text_strips_confidence_suffix(self, shared_page: Page):
        page = shared_page
        result = page.evaluate("""async () => {
            const { labelKey } = await import('/app-static/js/app/image-tagging/canvas/image.js');
            return labelKey({ text: 'dog (91%)' }) === labelKey({ text: 'dog (55%)' });
        }""")
        assert result is True

    def test_no_label_returns_null(self, shared_page: Page):
        page = shared_page
        result = page.evaluate("""async () => {
            const { labelKey } = await import('/app-static/js/app/image-tagging/canvas/image.js');
            return labelKey(undefined);
        }""")
        assert result is None


# ---------------------------------------------------------------------------
# addInferenceResults — conflict resolution behaviour
# ---------------------------------------------------------------------------

class TestAddInferenceResultsConflicts:
    def test_duplicate_same_label_is_not_added(self, shared_page: Page):
        """A detection that matches an existing box's geometry and label is skipped."""
        page = shared_page
        result = page.evaluate("""async () => {
            const { createImageTaggingRuntime } = await import('/app-static/js/app/image-tagging/runtime.js');
            const { createUIBBbox } = await import('/app-static/js/shared/domain/image/mapper.js');
            const { rerenderBboxViews } = await import('/app-static/js/app/image-tagging/canvas/manager.js');
            const { addInferenceResults } = await import('/app-static/js/app/image-tagging/canvas/image.js');
            const { labelsFetcher } = await import('/app-static/js/shared/api/labels.js');
            const { tagsFetcher } = await import('/app-static/js/shared/api/tags.js');

            """ + _SETUP + """

            labelsFetcher.set('labelsMap', new Map([
                ['cat-uuid', { uuid: 'cat-uuid', name: 'Cat', color: '#ff0000' }],
            ]));
            tagsFetcher.set('tagsMap', new Map());

            const existing = createUIBBbox({
                uuid: 'existing', x: 0.1, y: 0.1, width: 0.2, height: 0.2,
                label: { uuid: 'cat-uuid', text: 'Cat', color: '#ff0000' },
            });
            runtime.state.image.bboxes.set('existing', existing);
            rerenderBboxViews(runtime.canvas, runtime.state.image.bboxes, runtime.canvas.viewport);

            const results = {
                detections: [{
                    confidence: 0.9,
                    bbox: {
                        uuid: 'new-detection', x: 0.1, y: 0.1, width: 0.2, height: 0.2,
                        labelUuid: 'cat-uuid', classStr: null, tagUuids: [],
                    },
                }],
            };
            await addInferenceResults(runtime, results);

            return {
                size: runtime.state.image.bboxes.size,
                hasExisting: runtime.state.image.bboxes.has('existing'),
                hasNew: runtime.state.image.bboxes.has('new-detection'),
            };
        }""")
        assert result["size"] == 1
        assert result["hasExisting"] is True
        assert result["hasNew"] is False

    def test_conflicting_label_kept_when_not_replacing(self, shared_page: Page):
        """A detection overlapping an existing box with a *different* label is
        left alone (and not added) unless replaceExisting is set."""
        page = shared_page
        result = page.evaluate("""async () => {
            const { createImageTaggingRuntime } = await import('/app-static/js/app/image-tagging/runtime.js');
            const { createUIBBbox } = await import('/app-static/js/shared/domain/image/mapper.js');
            const { rerenderBboxViews } = await import('/app-static/js/app/image-tagging/canvas/manager.js');
            const { addInferenceResults } = await import('/app-static/js/app/image-tagging/canvas/image.js');
            const { labelsFetcher } = await import('/app-static/js/shared/api/labels.js');
            const { tagsFetcher } = await import('/app-static/js/shared/api/tags.js');

            """ + _SETUP + """

            labelsFetcher.set('labelsMap', new Map([
                ['cat-uuid', { uuid: 'cat-uuid', name: 'Cat', color: '#ff0000' }],
                ['dog-uuid', { uuid: 'dog-uuid', name: 'Dog', color: '#00ff00' }],
            ]));
            tagsFetcher.set('tagsMap', new Map());

            const existing = createUIBBbox({
                uuid: 'existing', x: 0.1, y: 0.1, width: 0.2, height: 0.2,
                label: { uuid: 'cat-uuid', text: 'Cat', color: '#ff0000' },
            });
            runtime.state.image.bboxes.set('existing', existing);
            rerenderBboxViews(runtime.canvas, runtime.state.image.bboxes, runtime.canvas.viewport);

            const results = {
                detections: [{
                    confidence: 0.9,
                    bbox: {
                        uuid: 'new-detection', x: 0.1, y: 0.1, width: 0.2, height: 0.2,
                        labelUuid: 'dog-uuid', classStr: null, tagUuids: [],
                    },
                }],
            };
            await addInferenceResults(runtime, results, { replaceExisting: false });

            return {
                size: runtime.state.image.bboxes.size,
                existingLabelUuid: runtime.state.image.bboxes.get('existing')?.label?.uuid,
                hasNew: runtime.state.image.bboxes.has('new-detection'),
            };
        }""")
        assert result["size"] == 1
        assert result["existingLabelUuid"] == "cat-uuid"
        assert result["hasNew"] is False

    def test_conflicting_label_replaces_when_replace_existing(self, shared_page: Page):
        """With replaceExisting set, an overlapping different-label detection
        removes the old box and adds the new one in its place."""
        page = shared_page
        result = page.evaluate("""async () => {
            const { createImageTaggingRuntime } = await import('/app-static/js/app/image-tagging/runtime.js');
            const { createUIBBbox } = await import('/app-static/js/shared/domain/image/mapper.js');
            const { rerenderBboxViews } = await import('/app-static/js/app/image-tagging/canvas/manager.js');
            const { addInferenceResults } = await import('/app-static/js/app/image-tagging/canvas/image.js');
            const { labelsFetcher } = await import('/app-static/js/shared/api/labels.js');
            const { tagsFetcher } = await import('/app-static/js/shared/api/tags.js');

            """ + _SETUP + """

            labelsFetcher.set('labelsMap', new Map([
                ['cat-uuid', { uuid: 'cat-uuid', name: 'Cat', color: '#ff0000' }],
                ['dog-uuid', { uuid: 'dog-uuid', name: 'Dog', color: '#00ff00' }],
            ]));
            tagsFetcher.set('tagsMap', new Map());

            const existing = createUIBBbox({
                uuid: 'existing', x: 0.1, y: 0.1, width: 0.2, height: 0.2,
                label: { uuid: 'cat-uuid', text: 'Cat', color: '#ff0000' },
            });
            runtime.state.image.bboxes.set('existing', existing);
            rerenderBboxViews(runtime.canvas, runtime.state.image.bboxes, runtime.canvas.viewport);

            const results = {
                detections: [{
                    confidence: 0.9,
                    bbox: {
                        uuid: 'new-detection', x: 0.1, y: 0.1, width: 0.2, height: 0.2,
                        labelUuid: 'dog-uuid', classStr: null, tagUuids: [],
                    },
                }],
            };
            await addInferenceResults(runtime, results, { replaceExisting: true });

            return {
                size: runtime.state.image.bboxes.size,
                hasExisting: runtime.state.image.bboxes.has('existing'),
                newLabelUuid: runtime.state.image.bboxes.get('new-detection')?.label?.uuid,
            };
        }""")
        assert result["size"] == 1
        assert result["hasExisting"] is False
        assert result["newLabelUuid"] == "dog-uuid"

    def test_non_overlapping_detection_is_added_alongside_existing(self, shared_page: Page):
        """Detections with no significant overlap are added normally, existing box untouched."""
        page = shared_page
        result = page.evaluate("""async () => {
            const { createImageTaggingRuntime } = await import('/app-static/js/app/image-tagging/runtime.js');
            const { createUIBBbox } = await import('/app-static/js/shared/domain/image/mapper.js');
            const { rerenderBboxViews } = await import('/app-static/js/app/image-tagging/canvas/manager.js');
            const { addInferenceResults } = await import('/app-static/js/app/image-tagging/canvas/image.js');
            const { labelsFetcher } = await import('/app-static/js/shared/api/labels.js');
            const { tagsFetcher } = await import('/app-static/js/shared/api/tags.js');

            """ + _SETUP + """

            labelsFetcher.set('labelsMap', new Map([
                ['cat-uuid', { uuid: 'cat-uuid', name: 'Cat', color: '#ff0000' }],
            ]));
            tagsFetcher.set('tagsMap', new Map());

            const existing = createUIBBbox({
                uuid: 'existing', x: 0.1, y: 0.1, width: 0.2, height: 0.2,
                label: { uuid: 'cat-uuid', text: 'Cat', color: '#ff0000' },
            });
            runtime.state.image.bboxes.set('existing', existing);
            rerenderBboxViews(runtime.canvas, runtime.state.image.bboxes, runtime.canvas.viewport);

            const results = {
                detections: [{
                    confidence: 0.9,
                    bbox: {
                        uuid: 'new-detection', x: 0.6, y: 0.6, width: 0.2, height: 0.2,
                        labelUuid: 'cat-uuid', classStr: null, tagUuids: [],
                    },
                }],
            };
            await addInferenceResults(runtime, results);

            return {
                size: runtime.state.image.bboxes.size,
                hasExisting: runtime.state.image.bboxes.has('existing'),
                hasNew: runtime.state.image.bboxes.has('new-detection'),
            };
        }""")
        assert result["size"] == 2
        assert result["hasExisting"] is True
        assert result["hasNew"] is True
