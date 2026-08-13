"""What "Mark absent" / "Reset" apply to must survive a new browser session.

The label panel's rightmost column (◉ / ⊘) picks which labels the sticky
header's bulk buttons hit. Hand-picking there used to be stored in
sessionStorage, so closing the tab silently reset the selection to "every
label" — and the next bulk op marked more than the user had lined up. The
selection (and the named target preset behind it) now lives in localStorage.

Contract asserted here:
  - a hand-picked ⊘ exclusion survives a session restart (sessionStorage
    cleared + reload) and still drives the "Apply to N targeted" count
  - a named preset chosen from the 🎯 menu survives the same restart
  - a selection written by an older build into sessionStorage is migrated to
    localStorage on the next boot rather than being dropped
  - the active training run survives the same restart (the 'training-run'
    preset means "all" without it), an explicit deactivation stays deactivated,
    and a pointer to a deleted run is dropped instead of retried forever

Run with:
    pytest playwright_tests/test_label_list_target_persistence.py -p no:asyncio -v
"""

import json

from playwright.sync_api import Page

CAT_UUID = "11111111-0000-0000-0000-00000000cafe"
DOG_UUID = "33333333-0000-0000-0000-0000000000dd"
CORRUPT_UUID = "22222222-0000-0000-0000-00000000beef"

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


def _route_common(page: Page) -> None:
    page.route(
        lambda url: url.split("?")[0].endswith("/api/labels") and "api/labels/" not in url,
        lambda route: route.fulfill(status=200, content_type="application/json", body=json.dumps(LABELS_PAYLOAD)),
    )
    page.route(
        "**/api/images/*/labels",
        lambda route: route.fulfill(
            status=200, content_type="application/json", body=json.dumps({"status": "success", "labels": []})
        ),
    )
    page.route(
        "**/api/training-configs",
        lambda route: route.fulfill(
            status=200, content_type="application/json", body=json.dumps({"status": "success", "configs": []})
        ),
    )


TRAINING_CONFIG = {
    "uuid": "tc-1",
    "name": "Nightly",
    "modelConfigNodeUuid": "node-root",
    "datasetUuids": [],
    "hyperparamsJson": {},
}

# The run covers Cat and Corrupt; Dog is deliberately outside it, so the
# 'training-run' target preset has something to exclude.
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


def _route_run(page: Page) -> None:
    """Serve a real training run on top of _route_common's empty list."""
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
            status=200, content_type="application/json", body=json.dumps({"status": "success", "gaps": {}})
        ),
    )


# The panel is mounted directly (same harness as test_label_list_training_gaps.py)
# rather than driven through the canvas modal: targeting lives entirely in
# labelList.js + activeTrainingConfig.js, and this keeps the test about that.
_MOUNT_JS = """
async (opts) => {
    const { createLabelList } = await import('/app-static/js/ui/labelList.js');
    const { createImageTaggingRuntime } = await import('/app-static/js/app/image-tagging/runtime.js');
    const { events, EventTypes } = await import('/app-static/js/shared/events/index.js');
    const { clearLabelsCache } = await import('/app-static/js/shared/api/labels.js');

    clearLabelsCache();

    const runtime = createImageTaggingRuntime();
    runtime.state.image = { uuid: 'img1', image_path: 'test.png', bboxes: new Map() };

    const div = document.createElement('div');
    div.id = 'target-test-host';
    document.body.appendChild(div);
    const list = await createLabelList(div, { mode: 'select', runtime });

    events.publish(EventTypes.IMAGE_LOADED_ONTO_CANVAS, { path: 'test.png', runtimeId: runtime.id });
    await new Promise(r => setTimeout(r, 300));

    for (const uuid of opts.toggle ?? []) {
        div.querySelector(`.label-panel-row[data-uuid="${uuid}"] .label-target-toggle`).click();
    }
    await new Promise(r => setTimeout(r, 100));

    const icon = (uuid) => {
        const el = div.querySelector(`.label-panel-row[data-uuid="${uuid}"] .label-target-toggle`);
        return el ? el.textContent : null;
    };
    const out = {
        caption: div.querySelector('.label-bulk-caption').textContent,
        cat: icon(opts.catUuid),
        dog: icon(opts.dogUuid),
        corrupt: icon(opts.corruptUuid),
    };

    list.destroy();
    div.remove();
    return out;
}
"""

_UUIDS = {"catUuid": CAT_UUID, "dogUuid": DOG_UUID, "corruptUuid": CORRUPT_UUID}


def _mount(page: Page, toggle: list | None = None) -> dict:
    return page.evaluate(_MOUNT_JS, {**_UUIDS, "toggle": toggle or []})


def _boot(page: Page, base_url: str) -> None:
    page.goto(base_url)
    page.wait_for_load_state("networkidle")


def _restart_session(page: Page, base_url: str) -> None:
    """Simulate closing the tab and opening a new one: sessionStorage is gone,
    localStorage is not."""
    page.evaluate("sessionStorage.clear()")
    _boot(page, base_url)


class TestTargetSelectionSurvivesASession:
    def test_hand_picked_exclusion_survives_a_restart(self, page: Page, base_url: str):
        _route_common(page)
        _boot(page, base_url)

        before = _mount(page, toggle=[DOG_UUID])
        assert before["dog"] == "⊘", before
        assert before["cat"] == "◉", before
        assert "Apply to 2 targeted" in before["caption"], before

        _restart_session(page, base_url)
        after = _mount(page)

        assert after["dog"] == "⊘", after
        assert after["cat"] == "◉", after
        assert "Apply to 2 targeted" in after["caption"], after

    def test_named_preset_survives_a_restart(self, page: Page, base_url: str):
        _route_common(page)
        _boot(page, base_url)
        page.evaluate("""async () => {
                const { activeTrainingConfig } = await import('/app-static/js/shared/activeTrainingConfig.js');
                activeTrainingConfig.setTargetPreset('image');
            }""")

        _restart_session(page, base_url)
        after = _mount(page)

        # 'image' targets image-scope labels only — Corrupt in, Cat and Dog out.
        assert after["corrupt"] == "◉", after
        assert after["cat"] == "⊘", after
        assert "Apply to 1 targeted" in after["caption"], after

    def test_a_selection_left_in_session_storage_by_an_older_build_is_migrated(self, page: Page, base_url: str):
        """Users mid-session when this shipped have their pick in sessionStorage;
        it must be adopted, not silently reset to 'every label'."""
        _route_common(page)
        page.add_init_script(
            "sessionStorage.setItem('wtch_tc_target_preset', 'custom');"
            f"sessionStorage.setItem('wtch_tc_custom_targeted', '{json.dumps([CAT_UUID])}');"
        )
        _boot(page, base_url)

        res = _mount(page)
        assert res["cat"] == "◉", res
        assert res["dog"] == "⊘", res
        assert "Apply to 1 targeted" in res["caption"], res

        migrated = page.evaluate("localStorage.getItem('wtch_tc_target_preset')")
        assert migrated == "custom", migrated


_ACTIVE_RUN_JS = """
async () => {
    const { activeTrainingConfig } = await import('/app-static/js/shared/activeTrainingConfig.js');
    const tc = activeTrainingConfig.getActive();
    return tc ? tc.uuid : null;
}
"""


class TestActiveRunSurvivesASession:
    """The 'training-run' target preset is only meaningful while its run is
    active — with the run gone, nothing is excluded and it silently means
    'all'. So the run itself has to outlive the tab too."""

    def test_the_active_run_and_its_target_preset_come_back_together(self, page: Page, base_url: str):
        _route_common(page)
        _route_run(page)
        _boot(page, base_url)
        page.evaluate("""async () => {
                const { activeTrainingConfig } = await import('/app-static/js/shared/activeTrainingConfig.js');
                await activeTrainingConfig.setActive({
                    uuid: 'tc-1', name: 'Nightly', modelConfigNodeUuid: 'node-root',
                });
                activeTrainingConfig.setTargetPreset('training-run');
            }""")

        _restart_session(page, base_url)

        assert page.evaluate(_ACTIVE_RUN_JS) == "tc-1"
        after = _mount(page)
        # Cat and Corrupt are in the run, Dog is not.
        assert after["cat"] == "◉", after
        assert after["dog"] == "⊘", after
        assert "Apply to 2 targeted" in after["caption"], after

    def test_deactivating_the_run_stays_deactivated(self, page: Page, base_url: str):
        _route_common(page)
        _route_run(page)
        _boot(page, base_url)
        page.evaluate("""async () => {
                const { activeTrainingConfig } = await import('/app-static/js/shared/activeTrainingConfig.js');
                await activeTrainingConfig.setActive({
                    uuid: 'tc-1', name: 'Nightly', modelConfigNodeUuid: 'node-root',
                });
                await activeTrainingConfig.setActive(null);
            }""")

        _restart_session(page, base_url)

        assert page.evaluate(_ACTIVE_RUN_JS) is None

    def test_a_run_left_in_session_storage_by_an_older_build_is_migrated(self, page: Page, base_url: str):
        _route_common(page)
        _route_run(page)
        page.add_init_script("sessionStorage.setItem('wtch_active_tc_uuid', 'tc-1');")
        _boot(page, base_url)

        assert page.evaluate(_ACTIVE_RUN_JS) == "tc-1"
        assert page.evaluate("localStorage.getItem('wtch_active_tc_uuid')") == "tc-1"

    def test_a_deleted_run_is_dropped_rather_than_retried(self, page: Page, base_url: str):
        """The stored pointer outlives the run it names — a config deleted from
        the Configuration tab must not resurrect as a boot-time 404 forever."""
        _route_common(page)  # training-configs list is empty here
        page.add_init_script("localStorage.setItem('wtch_active_tc_uuid', 'tc-gone');")
        _boot(page, base_url)

        assert page.evaluate(_ACTIVE_RUN_JS) is None
        assert page.evaluate("localStorage.getItem('wtch_active_tc_uuid')") is None
