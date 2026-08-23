"""Regression tests: unsaved presence marks must survive async refreshes.

Two independent paths used to silently undo a "Mark absent" on screen while
still queuing it for the next save — so the panel showed the opposite of what
would be written:

1. `adoptImage()` assigned the GET /api/images/<path>/labels response straight
   into `serverStates`. That response is a snapshot from when the request was
   issued, so marks made while it was in flight were wiped when it landed. The
   window grows with database size (the read competes with the training-run
   stats/readiness aggregates), which is why it only showed up on busy installs.

2. ACTIVE_TRAINING_CONFIG_CHANGED re-mounted the whole panel, even when only the
   cached stats had been refreshed. The re-mount tears down the closure holding
   `pendingFlush`, so unsaved marks were destroyed outright. Every canvas save
   fires that stats refresh, which is the "I marked absent, then it undid it a
   second later" report.
"""

import json

from playwright.sync_api import Page

LABEL_UUID = "22222222-3333-4444-5555-0000000000aa"
LABELS_PAYLOAD = {
    "status": "success",
    "labels": [
        {
            "uuid": LABEL_UUID,
            "name": "slow-fetch-label",
            "color": "#abcdef",
            "labelType": "standard",
            "scope": "object",
            "parentUuid": None,
        },
    ],
}


def _route_labels(page: Page):
    page.route(
        "**/api/labels",
        lambda route: route.fulfill(status=200, content_type="application/json", body=json.dumps(LABELS_PAYLOAD)),
    )
    page.route(
        "**/api/images/*/labels",
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({"status": "success", "labels": []}),
        ),
    )


_MOUNT_JS = """
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
    const statusText = () => div.querySelector('.label-status-btn')?.textContent;
"""


def test_mark_absent_survives_an_in_flight_label_fetch(page: Page, base_url: str):
    """A label-state response issued before the click must not undo the click."""
    page.goto(base_url)
    page.wait_for_load_state("networkidle")
    _route_labels(page)

    result = page.evaluate("""async () => {
        %s
        // Publishing starts adoptImage()'s label-state fetch. Marking absent in
        // the same tick puts the click strictly inside the request window — the
        // real bug just has a much wider one (a loaded database, competing
        // stats queries), not a different mechanism.
        events.publish(EventTypes.IMAGE_LOADED_ONTO_CANVAS, { path: 'slow.png', runtimeId: runtime.id });
        div.querySelector('.label-status-btn').click(); // uncertain -> absent
        const immediately = statusText();

        // Let the in-flight response land on top of the mark.
        await new Promise(r => setTimeout(r, 800));
        const afterFetchLands = statusText();

        list.destroy();
        div.remove();
        return { immediately, afterFetchLands };
    }""" % _MOUNT_JS)

    assert result["immediately"] == "✗", f"click did not mark absent: {result}"
    assert result["afterFetchLands"] == "✗", f"a label fetch issued before the click undid it on screen: {result}"


def test_mark_absent_survives_a_stats_only_training_config_publish(page: Page, base_url: str):
    """A stats refresh repaints badges; it must not discard unsaved marks."""
    page.goto(base_url)
    page.wait_for_load_state("networkidle")
    _route_labels(page)

    result = page.evaluate("""async () => {
        %s
        events.publish(EventTypes.IMAGE_LOADED_ONTO_CANVAS, { path: 'test.png', runtimeId: runtime.id });
        await new Promise(r => setTimeout(r, 300));

        div.querySelector('.label-status-btn').click(); // uncertain -> absent
        const immediately = statusText();

        // What activeTrainingConfig publishes when its stats/readiness fetch
        // resolves — fired off by every canvas save.
        events.publish(EventTypes.ACTIVE_TRAINING_CONFIG_CHANGED, {
          config: null, visibilityPreset: 'all', targetPreset: 'all', statsUpdated: true,
        });
        await new Promise(r => setTimeout(r, 400));
        const afterStatsPublish = statusText();

        list.destroy();
        div.remove();
        return { immediately, afterStatsPublish };
    }""" % _MOUNT_JS)

    assert result["immediately"] == "✗", f"click did not mark absent: {result}"
    assert result["afterStatsPublish"] == "✗", f"a stats-only refresh discarded the unsaved mark: {result}"
