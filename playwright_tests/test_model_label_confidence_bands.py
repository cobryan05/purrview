"""
Tests for confidence-band authoring in ModelLabelSelectField / LabelNumberField
(apps/image_tagging_server/static/js/ui/utils/fields/{ModelLabelSelectField,LabelNumberField}.js).

Phase 5C of LIVE_COLLECTION_PLAN.md: the backend (CollectTrainingImagesTask.py)
already accepts a `{min, max}` confidence band per label (or a legacy bare
scalar floor, read as `{min: scalar, max: 1.0}`). This file exercises the UI
side that authors those bands via the Task Management config editor:
  - a legacy scalar value loads into the widget as min=scalar, max=1.0
  - editing min/max writes a `{min, max}` band into the saved task config
  - a saved scalar of 0 ("label disabled" convention) survives as 0, not the
    `??` vs `||` DEFAULT_CONFIDENCE fallback bug

Run with:
    pytest playwright_tests/test_model_label_confidence_bands.py -p no:asyncio -v
"""

import json

import pytest
from playwright.sync_api import Page, Route, expect

TASK_TYPE = "FakeCollectorTask"
MODEL_NAME = "yolo-fake-model"
CAT_UUID = "11111111-0000-0000-0000-000000000001"

# A minimal schema with a single `model_label` field — exercises exactly the
# same fieldFactories.model_label -> ModelLabelSelectField -> LabelNumberField
# chain that CollectTrainingImagesTask.model_label_config (an array of these)
# uses, without needing to also drive ArrayField's add/remove-item UI.
MODEL_LABEL_SCHEMA = {
    "meta": {"order": ["model_label_config"]},
    "model_label_config": {
        "type": "model_label",
        "label": "Model & Labels",
        "description": "Model + per-label confidence band",
    },
}

FAKE_CONFIG_UUID = "eeeeeeee-0000-0000-0000-000000000099"


def _config(label_values: dict) -> dict:
    return {
        "uuid": FAKE_CONFIG_UUID,
        "name": "Band Test Config",
        "typename": TASK_TYPE,
        "params": {
            "model_label_config": {
                "modelName": MODEL_NAME,
                "labelValues": label_values,
            }
        },
    }


def _navigate_to_tasks(page: Page, base_url: str):
    page.goto(base_url)
    page.wait_for_load_state("networkidle")
    page.locator(".main-tab-button[data-tab='tab-operations']").click()
    page.wait_for_load_state("networkidle")
    page.locator("#tab-operations .workspace-tab-button[data-tab='tab-task-management']").click()
    page.wait_for_load_state("networkidle")


def _mock_apis(page: Page, configs: dict | None = None):
    configs = configs or {}

    page.route(
        "**/api/tasks/types/list",
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({"status": "success", "types": [TASK_TYPE]}),
        ),
    )
    page.route(
        "**/api/tasks/types/schema",
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({"schemas": {TASK_TYPE: MODEL_LABEL_SCHEMA}}),
        ),
    )
    page.route(
        "**/api/tasks/configs/get",
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({"status": "success", "configs": configs}),
        ),
    )
    page.route(
        "**/api/tasks/instances/get",
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({"status": "success", "tasks": {}}),
        ),
    )
    page.route(
        "**/api/models",
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({"status": "success", "models": [MODEL_NAME]}),
        ),
    )
    page.route(
        "**/api/models/classes/bulk",
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps(
                {
                    # Real shape: Dict[modelName, Dict[classIndex, labelUuid]].
                    # Single-label model keeps the rendered number-input count
                    # unambiguous (exactly one min + one max input) for these tests.
                    "status": "success",
                    "models": {MODEL_NAME: {"0": CAT_UUID}},
                }
            ),
        ),
    )
    page.route(
        "**/api/labels",
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps(
                {
                    "status": "success",
                    "labels": [
                        {"uuid": CAT_UUID, "name": "cat", "color": "#ff0000", "labelType": "standard"},
                    ],
                }
            ),
        ),
    )


class TestLegacyScalarLoadsAsBand:
    def test_legacy_scalar_loads_min_equals_scalar_max_equals_one(self, page: Page, base_url: str):
        """An existing config saved with a bare scalar (pre-Phase-5B) loads
        into the widget as min=scalar, max=1.0 — matching the backend's
        `_parse_confidence_band` convention."""
        _mock_apis(page, configs={FAKE_CONFIG_UUID: _config({CAT_UUID: 0.6})})
        _navigate_to_tasks(page, base_url)

        page.wait_for_selector("#task-config-list button[title='Edit']", timeout=5000)
        page.locator("#task-config-list button[title='Edit']").first.click()
        # Model is pre-selected from the saved config, so label rows render
        # immediately without further interaction.
        page.wait_for_selector("#task-config-list input[type='number']", timeout=5000)

        number_inputs = page.locator("#task-config-list input[type='number']")
        expect(number_inputs).to_have_count(2)
        assert number_inputs.nth(0).input_value() == "0.6"
        assert number_inputs.nth(1).input_value() == "1"

    def test_saved_zero_stays_zero_not_default_confidence(self, page: Page, base_url: str):
        """Regression test for the `||` vs `??` bug: a saved scalar of 0 (the
        'label disabled' convention) must load as min=0, not silently jump to
        DEFAULT_CONFIDENCE (0.25) and re-enable the label."""
        _mock_apis(page, configs={FAKE_CONFIG_UUID: _config({CAT_UUID: 0})})
        _navigate_to_tasks(page, base_url)

        page.wait_for_selector("#task-config-list button[title='Edit']", timeout=5000)
        page.locator("#task-config-list button[title='Edit']").first.click()
        page.wait_for_selector("#task-config-list input[type='number']", timeout=5000)

        number_inputs = page.locator("#task-config-list input[type='number']")
        expect(number_inputs).to_have_count(2)
        assert number_inputs.nth(0).input_value() == "0", (
            f"Expected min=0 to survive, got {number_inputs.nth(0).input_value()!r} "
            "(likely the ||-vs-?? DEFAULT_CONFIDENCE fallback bug)"
        )
        assert number_inputs.nth(1).input_value() == "1"


class TestEditingBandWritesToConfig:
    def test_editing_min_and_max_saves_band(self, page: Page, base_url: str):
        """Editing the min/max inputs and saving writes a {min, max} band into
        the updated task config's params."""
        update_calls: list[dict] = []

        def handle_update(route: Route):
            update_calls.append(json.loads(route.request.post_data))
            route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps({"status": "success"}),
            )

        _mock_apis(page, configs={FAKE_CONFIG_UUID: _config({CAT_UUID: 0.2})})
        page.route("**/api/tasks/configs/update", handle_update)

        _navigate_to_tasks(page, base_url)
        page.wait_for_selector("#task-config-list button[title='Edit']", timeout=5000)
        page.locator("#task-config-list button[title='Edit']").first.click()
        page.wait_for_selector("#task-config-list input[type='number']", timeout=5000)

        number_inputs = page.locator("#task-config-list input[type='number']")
        expect(number_inputs).to_have_count(2)
        number_inputs.nth(0).fill("0.3")
        number_inputs.nth(1).fill("0.9")

        page.locator("#task-config-list button[title='Save']").first.click()
        page.wait_for_timeout(600)

        assert update_calls, "Expected POST to /api/tasks/configs/update"
        payload = update_calls[0]
        band = payload["params"]["model_label_config"]["labelValues"][CAT_UUID]
        assert band == {"min": 0.3, "max": 0.9}, f"Unexpected band payload: {band}"

    def test_new_config_model_label_field_writes_band(self, page: Page, base_url: str):
        """Full create flow: select type → select model → edit a label's
        min/max → Add Config → payload carries a {min, max} band."""
        create_calls: list[dict] = []

        def handle_create(route: Route):
            create_calls.append(json.loads(route.request.post_data))
            route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps({"status": "success", "configUuid": FAKE_CONFIG_UUID}),
            )

        _mock_apis(page, configs={})
        page.route("**/api/tasks/configs/create", handle_create)

        _navigate_to_tasks(page, base_url)
        page.wait_for_selector("#task-config-list select", timeout=5000)

        page.locator("#task-config-list select").first.select_option(TASK_TYPE)
        page.wait_for_timeout(800)

        # Second <select> is the model picker inside ModelLabelSelectField.
        page.locator("#task-config-list select").nth(1).select_option(MODEL_NAME)
        page.wait_for_selector("#task-config-list input[type='number']", timeout=5000)

        number_inputs = page.locator("#task-config-list input[type='number']")
        expect(number_inputs).to_have_count(2)
        number_inputs.nth(0).fill("0.4")
        number_inputs.nth(1).fill("0.8")

        page.locator("#task-config-list input[placeholder='Enter new config name']").fill("Band Config")
        page.locator("#task-config-list button[title='Add Config']").click()
        page.wait_for_timeout(600)

        assert create_calls, "Expected POST to /api/tasks/configs/create"
        payload = create_calls[0]
        band = payload["params"]["model_label_config"]["labelValues"][CAT_UUID]
        assert band == {"min": 0.4, "max": 0.8}, f"Unexpected band payload: {band}"


class TestBandNormalizationMatchesBackend:
    """`normalizeBand()` must agree with the backend's `_parse_confidence_band`
    for every value either side can produce, or the config the user sees drifts
    from the band the collect task actually enforces (LIVE_COLLECTION_PLAN.md
    Phase 5C explicitly wants the two readings not to drift apart).

    The two divergences these cover both collapse a band to a single point that
    matches essentially no real detection, silently halting collection for that
    label.
    """

    def _edit_and_save(self, page: Page, base_url: str, saved_band, min_fill: str, max_fill: str) -> dict:
        update_calls: list[dict] = []

        def handle_update(route: Route):
            update_calls.append(json.loads(route.request.post_data))
            route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps({"status": "success"}),
            )

        _mock_apis(page, configs={FAKE_CONFIG_UUID: _config({CAT_UUID: saved_band})})
        page.route("**/api/tasks/configs/update", handle_update)

        _navigate_to_tasks(page, base_url)
        page.wait_for_selector("#task-config-list button[title='Edit']", timeout=5000)
        page.locator("#task-config-list button[title='Edit']").first.click()
        page.wait_for_selector("#task-config-list input[type='number']", timeout=5000)

        number_inputs = page.locator("#task-config-list input[type='number']")
        expect(number_inputs).to_have_count(2)
        number_inputs.nth(0).fill(min_fill)
        number_inputs.nth(1).fill(max_fill)

        page.locator("#task-config-list button[title='Save']").first.click()
        page.wait_for_timeout(600)

        assert update_calls, "Expected POST to /api/tasks/configs/update"
        return update_calls[0]["params"]["model_label_config"]["labelValues"][CAT_UUID]

    def test_blank_max_saves_as_unbounded_one(self, page: Page, base_url: str):
        """Clearing the max input means 'no upper cap' and must save as 1.0.

        `Number('')` is 0 — finite — so a naive `Number.isFinite(max) ? max : 1`
        guard never fires and writes `max: 0`. The backend then clamps
        `max < min` up to `min`, collapsing the band to [0.3, 0.3]. Python's own
        reading of an unparseable max is the opposite (1.0).
        """
        band = self._edit_and_save(page, base_url, 0.2, min_fill="0.3", max_fill="")
        assert band == {"min": 0.3, "max": 1}, (
            f"Blank max should mean unbounded (1.0), got {band} (Number('') === 0 slipping past the isFinite guard)"
        )

    def test_blank_min_saves_as_zero_disabled(self, page: Page, base_url: str):
        """Clearing min yields 0 — the backend's 'label disabled' convention,
        and what `_parse_confidence_band` returns for an unparseable min."""
        band = self._edit_and_save(page, base_url, 0.2, min_fill="", max_fill="0.8")
        assert band == {"min": 0, "max": 0.8}, f"Blank min should save as 0, got {band}"

    def test_inverted_band_is_normalized_on_save(self, page: Page, base_url: str):
        """An inverted band must be corrected client-side the same way the
        backend corrects it (`max = min`), so the saved value and the displayed
        value both match what is enforced."""
        band = self._edit_and_save(page, base_url, 0.2, min_fill="0.9", max_fill="0.1")
        assert band == {"min": 0.9, "max": 0.9}, (
            f"Inverted band should collapse to max=min like the backend does, got {band}"
        )

    def test_out_of_range_band_is_clamped_on_save(self, page: Page, base_url: str):
        """Values outside [0, 1] are clamped, mirroring the backend's clamps.
        HTML min/max attributes don't prevent a typed out-of-range value from
        reaching getValue()."""
        band = self._edit_and_save(page, base_url, 0.2, min_fill="5", max_fill="9")
        assert band == {"min": 1, "max": 1}, f"Out-of-range band should clamp into [0, 1], got {band}"

    def test_saved_inverted_band_displays_as_backend_reads_it(self, page: Page, base_url: str):
        """Load side: an already-saved inverted band renders as the backend
        reads it, instead of the UI showing 0.9/0.1 forever while the task
        enforces [0.9, 0.9]."""
        _mock_apis(page, configs={FAKE_CONFIG_UUID: _config({CAT_UUID: {"min": 0.9, "max": 0.1}})})
        _navigate_to_tasks(page, base_url)

        page.wait_for_selector("#task-config-list button[title='Edit']", timeout=5000)
        page.locator("#task-config-list button[title='Edit']").first.click()
        page.wait_for_selector("#task-config-list input[type='number']", timeout=5000)

        number_inputs = page.locator("#task-config-list input[type='number']")
        expect(number_inputs).to_have_count(2)
        assert number_inputs.nth(0).input_value() == "0.9"
        assert number_inputs.nth(1).input_value() == "0.9", (
            f"Expected inverted max to display as 0.9, got {number_inputs.nth(1).input_value()!r}"
        )

    def test_saved_null_max_displays_as_one(self, page: Page, base_url: str):
        """A hand-edited or legacy config with `max: null` must load as 1.0 —
        `float(None)` raises in Python and falls back to 1.0, but
        `Number(null)` is 0."""
        _mock_apis(page, configs={FAKE_CONFIG_UUID: _config({CAT_UUID: {"min": 0.3, "max": None}})})
        _navigate_to_tasks(page, base_url)

        page.wait_for_selector("#task-config-list button[title='Edit']", timeout=5000)
        page.locator("#task-config-list button[title='Edit']").first.click()
        page.wait_for_selector("#task-config-list input[type='number']", timeout=5000)

        number_inputs = page.locator("#task-config-list input[type='number']")
        expect(number_inputs).to_have_count(2)
        assert number_inputs.nth(0).input_value() == "0.3"
        assert number_inputs.nth(1).input_value() == "1", (
            f"Expected null max to load as 1.0, got {number_inputs.nth(1).input_value()!r} "
            "(Number(null) === 0)"
        )
