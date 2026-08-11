"""E2E: inference -> save -> reload keeps the confidence on the label."""

import json

from playwright.sync_api import Page


def test_confidence_survives_save_and_reload(page: Page, base_url: str):
    saved = {}

    def handle_update(route):
        body = json.loads(route.request.post_data or "{}")
        saved["boxes"] = body["boxes"]
        route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({"status": "success", "image": {"filename": "x.jpg", "bboxes": []}}),
        )

    page.route("**/api/images/metadata/update", handle_update)
    page.goto(base_url)
    page.wait_for_load_state("networkidle")

    # 1. A live detection becomes a UIBbox, then is serialized by updateImage.
    sent = page.evaluate("""async () => {
        const { createUIBBbox } = await import('/app-static/js/shared/domain/image/mapper.js');
        const { updateImage } = await import('/app-static/js/shared/api/images.js');
        const { CONFIDENCE_KEY } = await import('/app-static/js/shared/domain/image/confidence.js');
        const bbox = createUIBBbox({
            uuid: 'b1', label: { uuid: 'l1', text: 'Cat', color: '#f00' },
            x: 0.1, y: 0.1, width: 0.2, height: 0.2,
            confidence: 0.82,
            metadataJson: { [CONFIDENCE_KEY]: String(0.82) },
        });
        await updateImage({ image_path: 'x.jpg', bboxes: new Map([['b1', bbox]]) });
        return true;
    }""")
    assert sent is True
    assert saved["boxes"][0]["extra"] == {"confidence": "0.82"}, saved["boxes"][0]

    # 2. That same payload, coming back from the API, rehydrates to a label with the pct.
    label = page.evaluate("""async () => {
        const { bboxReadToUI } = await import('/app-static/js/shared/domain/image/mapper.js');
        const { BboxView } = await import('/app-static/js/app/image-tagging/canvas/views/bboxView.js');
        const ui = bboxReadToUI({
            uuid: 'b1', x: 0.1, y: 0.1, width: 0.2, height: 0.2,
            label: { uuid: 'l1', name: 'Cat', color: '#f00' },
            tags: [{ uuid: '00000000-0000-0000-0000-000000000001', icon: null }],
            metadataJson: { confidence: '0.82' },
        });
        const div = document.createElement('div');
        document.body.appendChild(div);
        const stage = new Konva.Stage({ container: div, width: 400, height: 300 });
        const layer = new Konva.Layer(); stage.add(layer);
        const view = new BboxView(ui, { offsetX: 0, offsetY: 0, scale: 1, imageWidth: 400, imageHeight: 300 });
        layer.add(view);
        return view.findOne('Text').text();
    }""")
    assert label == "Cat (82%)", label


def test_collect_training_images_shaped_bbox_shows_its_confidence(page: Page, base_url: str):
    """CollectTrainingImagesTask writes `extra={"confidence": str(conf)}` — it is
    the highest-volume producer of confidences in this app, and pointing the
    reader at `source_confidence` alone silently blanked every box it made.
    """
    page.goto(base_url)
    page.wait_for_load_state("networkidle")

    label = page.evaluate("""async () => {
        const { bboxReadToUI } = await import('/app-static/js/shared/domain/image/mapper.js');
        const { BboxView } = await import('/app-static/js/app/image-tagging/canvas/views/bboxView.js');
        const ui = bboxReadToUI({
            uuid: 'b1', x: 0.1, y: 0.1, width: 0.2, height: 0.2,
            label: { uuid: 'l1', name: 'Cat', color: '#f00' },
            tags: [{ uuid: '00000000-0000-0000-0000-000000000001', icon: null }],
            // Exactly the shape CollectTrainingImagesTask posts.
            metadataJson: { confidence: '0.6231' },
        });
        const div = document.createElement('div');
        document.body.appendChild(div);
        const stage = new Konva.Stage({ container: div, width: 400, height: 300 });
        const layer = new Konva.Layer(); stage.add(layer);
        const view = new BboxView(ui, { offsetX: 0, offsetY: 0, scale: 1, imageWidth: 400, imageHeight: 300 });
        layer.add(view);
        return view.findOne('Text').text();
    }""")
    assert label == "Cat (62%)", label
