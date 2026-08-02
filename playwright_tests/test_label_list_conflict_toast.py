"""Regression test for the "N label state(s) failed to save" false-alarm bug.

ui/labelList.js's CANVAS_ANNOTATIONS_SAVED flush used to treat a deliberate
verified-bbox-conflict rejection from POST /api/images/labels/set exactly like
a real failure: it threw client-side, was caught by Promise.allSettled, got
re-queued into pendingFlush, and popped a red error toast — every single save,
forever, since the conflict can never be resolved by retrying. This asserts
the fixed contract end-to-end at the DOM/event level:
  - a `skipped: true` response does not produce the red "failed to save" toast
  - it is not re-queued, so a second save doesn't re-issue the same request
"""

import json

from playwright.sync_api import Page


def test_verified_bbox_conflict_shows_info_not_error_and_is_not_requeued(page: Page, base_url: str):
    label_uuid = "22222222-3333-4444-5555-000000000099"
    labels_payload = {
        "status": "success",
        "labels": [
            {
                "uuid": label_uuid,
                "name": "conflict-label",
                "color": "#abcdef",
                "labelType": "standard",
                "scope": "object",
                "parentUuid": None,
            },
        ],
    }

    set_calls = {"n": 0}

    def _serve_labels(route):
        route.fulfill(status=200, content_type="application/json", body=json.dumps(labels_payload))

    def _serve_image_labels(route):
        route.fulfill(status=200, content_type="application/json", body=json.dumps({"status": "success", "labels": []}))

    def _serve_set_label(route):
        set_calls["n"] += 1
        route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps(
                {
                    "status": "success",
                    "skipped": True,
                    "message": "Cannot mark label absent: the image has a verified bbox with this label",
                }
            ),
        )

    page.goto(base_url)
    page.wait_for_load_state("networkidle")
    page.route("**/api/labels", _serve_labels)
    page.route("**/api/images/*/labels", _serve_image_labels)
    page.route("**/api/images/labels/set", _serve_set_label)

    result = page.evaluate("""async () => {
        const { createLabelList } = await import('/app-static/js/ui/labelList.js');
        const { createImageTaggingRuntime } = await import('/app-static/js/app/image-tagging/runtime.js');
        const { events, EventTypes } = await import('/app-static/js/shared/events/index.js');
        const { clearLabelsCache } = await import('/app-static/js/shared/api/labels.js');

        clearLabelsCache();
        const runtime = createImageTaggingRuntime();
        runtime.state.image = { uuid: 'img1', bboxes: new Map() };

        const div = document.createElement('div');
        document.body.appendChild(div);
        const list = await createLabelList(div, { mode: 'select', runtime });

        events.publish(EventTypes.IMAGE_LOADED_ONTO_CANVAS, { path: 'test.png', runtimeId: runtime.id });
        await new Promise(r => setTimeout(r, 300));

        const row = div.querySelector('.label-panel-row');
        row.querySelector('button').click(); // uncertain -> absent

        events.publish(EventTypes.CANVAS_ANNOTATIONS_SAVED, { runtimeId: runtime.id });
        await new Promise(r => setTimeout(r, 300));

        const toastContainer = document.getElementById('toast-container');
        const firstSaveToasts = toastContainer
          ? [...toastContainer.children].map(c => ({ text: c.textContent, bg: c.style.background }))
          : [];

        // A second save with nothing re-queued must not re-issue the request.
        events.publish(EventTypes.CANVAS_ANNOTATIONS_SAVED, { runtimeId: runtime.id });
        await new Promise(r => setTimeout(r, 300));

        list.destroy();
        div.remove();
        return { firstSaveToasts };
    }""")

    toasts = result["firstSaveToasts"]
    assert toasts, "expected a toast after the conflict response"
    assert not any(
        t["bg"] == "#c0392b" for t in toasts
    ), f"verified-bbox conflict must not show the red error toast, got: {toasts}"
    assert any(
        "left unchanged" in t["text"] for t in toasts
    ), f"expected an info toast explaining the skip, got: {toasts}"

    # Exactly one POST to the set-label endpoint: the conflict must not be
    # re-queued into pendingFlush and re-sent on the next save.
    assert set_calls["n"] == 1, f"conflict was re-sent on a later save: {set_calls['n']} calls"
