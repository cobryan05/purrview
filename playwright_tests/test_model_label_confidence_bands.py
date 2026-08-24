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

from playwright_tests.waits import wait_until

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

# The shape CollectTrainingImagesTask actually declares: `model_label` as the
# item type of an *array*. It matters for layout, not for band semantics —
# ArrayField lays each item out in a row flex, which is a shrink-to-fit context,
# whereas a bare field lands in a stretched column flex. The widget renders very
# differently in the two, so the layout tests must cover this one.
MODEL_LABEL_ARRAY_SCHEMA = {
    "meta": {"order": ["model_label_config"]},
    "model_label_config": {
        "type": "array",
        "label": "Model/Label Configuration",
        "allow_group_insert": True,
        "items": {
            "type": "model_label",
            "label": "Model/Label selection",
            "description": "Labels to collect",
        },
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


def _array_config(label_values: dict) -> dict:
    """Same config, but with the model/label block wrapped in a one-item array."""
    cfg = _config(label_values)
    cfg["params"]["model_label_config"] = [cfg["params"]["model_label_config"]]
    return cfg


def _navigate_to_tasks(page: Page, base_url: str):
    page.goto(base_url)
    page.wait_for_load_state("networkidle")
    page.locator(".main-tab-button[data-tab='tab-configuration']").click()
    page.wait_for_load_state("networkidle")
    page.locator("#tab-configuration .workspace-tab-button[data-tab='tab-task-management']").click()
    page.wait_for_load_state("networkidle")


def _editor_bar(page: Page):
    """The task-config editor's sticky header bar, which holds Save/Create/Delete."""
    return page.locator("#task-config-list .le-editor .panel-box__bar")


def _open_edit(page: Page):
    """Load the first task config into the editor.

    Task configs moved to a list-beside-editor screen: selecting the row *is*
    opening it for editing, so there is no longer a per-row Edit button.
    """
    page.wait_for_selector("#task-config-list .le-list-item")
    page.locator("#task-config-list .le-list-item").first.click()
    page.wait_for_timeout(700)


def _mock_apis(
    page: Page,
    configs: dict | None = None,
    labels: list[dict] | None = None,
    schema: dict | None = None,
):
    """Mock every API the widget touches.

    `labels` defaults to the single "cat" label, which keeps the rendered
    number-input count unambiguous (exactly one min + one max) for the band
    tests. Pass a longer list — each entry `{"uuid", "name", "color"}` — for the
    layout tests, which need several rows of differing name lengths.

    `schema` defaults to the bare `model_label` field; pass
    MODEL_LABEL_ARRAY_SCHEMA for the array-wrapped shape the real task uses.
    """
    configs = configs or {}
    labels = labels or [{"uuid": CAT_UUID, "name": "cat", "color": "#ff0000"}]
    schema = schema or MODEL_LABEL_SCHEMA

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
            body=json.dumps({"schemas": {TASK_TYPE: schema}}),
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
                    "status": "success",
                    "models": {MODEL_NAME: {str(i): lbl["uuid"] for i, lbl in enumerate(labels)}},
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
                    "labels": [{**lbl, "labelType": "standard"} for lbl in labels],
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

        _open_edit(page)
        # Model is pre-selected from the saved config, so label rows render
        # immediately without further interaction.
        page.wait_for_selector("#task-config-list input[type='number']")

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

        _open_edit(page)
        page.wait_for_selector("#task-config-list input[type='number']")

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
        _open_edit(page)
        page.wait_for_selector("#task-config-list input[type='number']")

        number_inputs = page.locator("#task-config-list input[type='number']")
        expect(number_inputs).to_have_count(2)
        number_inputs.nth(0).fill("0.3")
        number_inputs.nth(1).fill("0.9")

        _editor_bar(page).locator("button", has_text="Save").click()
        wait_until(page, lambda: update_calls, "Expected POST to /api/tasks/configs/update")
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
        page.wait_for_selector("[data-testid='task-type-select']")

        page.get_by_test_id("task-type-select").select_option(TASK_TYPE)
        page.wait_for_timeout(800)

        # The model picker inside ModelLabelSelectField is the first <select>
        # in the schema form (the task-type one sits above it, outside the form).
        page.locator("#task-config-list .schema-form select").first.select_option(MODEL_NAME)
        page.wait_for_selector("#task-config-list input[type='number']")

        number_inputs = page.locator("#task-config-list input[type='number']")
        expect(number_inputs).to_have_count(2)
        number_inputs.nth(0).fill("0.4")
        number_inputs.nth(1).fill("0.8")

        page.get_by_test_id("task-name-input").fill("Band Config")
        _editor_bar(page).locator("button", has_text="Create").click()
        wait_until(page, lambda: create_calls, "Expected POST to /api/tasks/configs/create")
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
        _open_edit(page)
        page.wait_for_selector("#task-config-list input[type='number']")

        number_inputs = page.locator("#task-config-list input[type='number']")
        expect(number_inputs).to_have_count(2)
        number_inputs.nth(0).fill(min_fill)
        number_inputs.nth(1).fill(max_fill)

        _editor_bar(page).locator("button", has_text="Save").click()
        wait_until(page, lambda: update_calls, "Expected POST to /api/tasks/configs/update")
        return update_calls[0]["params"]["model_label_config"]["labelValues"][CAT_UUID]

    def test_blank_max_saves_as_unbounded_one(self, page: Page, base_url: str):
        """Clearing the max input means 'no upper cap' and must save as 1.0.

        `Number('')` is 0 — finite — so a naive `Number.isFinite(max) ? max : 1`
        guard never fires and writes `max: 0`. The backend then clamps
        `max < min` up to `min`, collapsing the band to [0.3, 0.3]. Python's own
        reading of an unparseable max is the opposite (1.0).
        """
        band = self._edit_and_save(page, base_url, 0.2, min_fill="0.3", max_fill="")
        assert band == {
            "min": 0.3,
            "max": 1,
        }, f"Blank max should mean unbounded (1.0), got {band} (Number('') === 0 slipping past the isFinite guard)"

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
        assert band == {
            "min": 0.9,
            "max": 0.9,
        }, f"Inverted band should collapse to max=min like the backend does, got {band}"

    def test_out_of_range_band_is_clamped_on_save(self, page: Page, base_url: str):
        """Values outside [0, 1] are clamped, mirroring the backend's clamps.
        HTML min/max attributes don't prevent a typed out-of-range value from
        reaching getValue()."""
        band = self._edit_and_save(page, base_url, 0.2, min_fill="5", max_fill="9")
        assert band == {"min": 1, "max": 1}, f"Out-of-range band should clamp into [0, 1], got {band}"

    def test_max_zero_survives_save_as_review_disabled_not_clamped_to_min(self, page: Page, base_url: str):
        """Setting max to 0 with a nonzero min means 'tag this label, but never
        let it alone trigger a review' — a distinct signal from a genuine
        inversion, which must not be coerced to `max = min` here."""
        band = self._edit_and_save(page, base_url, 0.2, min_fill="0.6", max_fill="0")
        assert band == {
            "min": 0.6,
            "max": 0,
        }, f"max=0 should survive save as review-disabled, not collapse to max=min, got {band}"

    def test_saved_inverted_band_displays_as_backend_reads_it(self, page: Page, base_url: str):
        """Load side: an already-saved inverted band renders as the backend
        reads it, instead of the UI showing 0.9/0.1 forever while the task
        enforces [0.9, 0.9]."""
        _mock_apis(page, configs={FAKE_CONFIG_UUID: _config({CAT_UUID: {"min": 0.9, "max": 0.1}})})
        _navigate_to_tasks(page, base_url)

        _open_edit(page)
        page.wait_for_selector("#task-config-list input[type='number']")

        number_inputs = page.locator("#task-config-list input[type='number']")
        expect(number_inputs).to_have_count(2)
        assert number_inputs.nth(0).input_value() == "0.9"
        assert (
            number_inputs.nth(1).input_value() == "0.9"
        ), f"Expected inverted max to display as 0.9, got {number_inputs.nth(1).input_value()!r}"

    def test_saved_null_max_displays_as_one(self, page: Page, base_url: str):
        """A hand-edited or legacy config with `max: null` must load as 1.0 —
        `float(None)` raises in Python and falls back to 1.0, but
        `Number(null)` is 0."""
        _mock_apis(page, configs={FAKE_CONFIG_UUID: _config({CAT_UUID: {"min": 0.3, "max": None}})})
        _navigate_to_tasks(page, base_url)

        _open_edit(page)
        page.wait_for_selector("#task-config-list input[type='number']")

        number_inputs = page.locator("#task-config-list input[type='number']")
        expect(number_inputs).to_have_count(2)
        assert number_inputs.nth(0).input_value() == "0.3"
        assert number_inputs.nth(1).input_value() == "1", (
            f"Expected null max to load as 1.0, got {number_inputs.nth(1).input_value()!r} " "(Number(null) === 0)"
        )


# Narrow enough that the editor pane is under the widget's 46rem two-column
# breakpoint; wide enough that it is over it. Both are viewport sizes, but what
# the widget actually queries is the pane it sits in (`container-type:
# inline-size`), so these are chosen by measuring the pane, not the window.
NARROW_VIEWPORT = {"width": 900, "height": 900}
WIDE_VIEWPORT = {"width": 1600, "height": 900}


class _BandGridBase:
    LABELS = [
        {"uuid": CAT_UUID, "name": "cat", "color": "#ff0000"},
        {
            "uuid": "11111111-0000-0000-0000-000000000002",
            "name": "a-very-long-label-name-indeed",
            "color": "#00ff00",
        },
        {"uuid": "11111111-0000-0000-0000-000000000003", "name": "dog", "color": "#0000ff"},
        {"uuid": "11111111-0000-0000-0000-000000000004", "name": "raccoon", "color": "#00ffff"},
    ]

    def _open_multi_label_editor(self, page: Page, base_url: str, viewport: dict):
        page.set_viewport_size(viewport)
        _mock_apis(
            page,
            configs={FAKE_CONFIG_UUID: _config({lbl["uuid"]: 0.5 for lbl in self.LABELS})},
            labels=self.LABELS,
        )
        _navigate_to_tasks(page, base_url)
        _open_edit(page)
        page.wait_for_selector("#task-config-list .mlsf-band-row")
        expect(page.locator("#task-config-list .mlsf-band-row")).to_have_count(len(self.LABELS))

    def _rows(self, page: Page):
        """Per row: the name cell's box and both input boxes."""
        rows = page.locator("#task-config-list .mlsf-band-row")
        out = []
        for i in range(rows.count()):
            row = rows.nth(i)
            inputs = row.locator("input[type='number']")
            expect(inputs).to_have_count(2)
            out.append(
                {
                    "name": row.locator(".mlsf-band-row__name").bounding_box(),
                    "min": inputs.nth(0).bounding_box(),
                    "max": inputs.nth(1).bounding_box(),
                }
            )
        return out

    def _visible_heads(self, page: Page):
        heads = page.locator("#task-config-list .mlsf-bands__head:visible")
        return [heads.nth(i).inner_text().strip().lower() for i in range(heads.count())]


class TestBandGridAlignment(_BandGridBase):
    """The band table is a real three-column grid, not a stack of independent rows.

    Each row used to lay itself out with its own flex box — "name  min [ ]  max [ ]"
    — so the inputs began wherever that row's label name happened to end, and the
    words "min"/"max" were repeated on every line. Nothing about that arrangement
    put two rows' numbers in the same place. These tests pin the column behaviour
    in a narrow pane, where all rows share one set of columns.
    """

    def test_min_and_max_inputs_share_a_column(self, page: Page, base_url: str):
        """Every row's min input starts at the same x, and so does every max —
        even though one label name is ten times longer than another."""
        self._open_multi_label_editor(page, base_url, NARROW_VIEWPORT)
        rows = self._rows(page)

        for key in ("min", "max"):
            xs = [r[key]["x"] for r in rows]
            widths = [r[key]["width"] for r in rows]
            assert max(xs) - min(xs) < 1, f"{key} inputs are not column-aligned: xs={xs}"
            assert max(widths) - min(widths) < 1, f"{key} inputs differ in width: widths={widths}"

        # The two columns must be distinct, or "aligned" would be trivially true.
        assert rows[0]["max"]["x"] > rows[0]["min"]["x"] + rows[0]["min"]["width"] - 1, "min/max columns overlap"

    def test_columns_are_named_once_in_a_header(self, page: Page, base_url: str):
        """Min/max are named by one header row rather than by a tag beside each
        of the 2N inputs."""
        self._open_multi_label_editor(page, base_url, NARROW_VIEWPORT)
        assert self._visible_heads(page) == ["label", "min", "max"]

    def test_header_sits_above_its_own_column(self, page: Page, base_url: str):
        """The Min/Max headers share the horizontal span of the inputs they name
        — a header that drifted into the neighbouring column would mislabel the
        values underneath it."""
        self._open_multi_label_editor(page, base_url, NARROW_VIEWPORT)

        heads = page.locator("#task-config-list .mlsf-bands__head:visible")
        first = self._rows(page)[0]
        for head_idx, key in ((1, "min"), (2, "max")):
            head = heads.nth(head_idx).bounding_box()
            assert abs(head["x"] - first[key]["x"]) < 1, f"{key} header x={head['x']} vs input x={first[key]['x']}"
            assert abs(head["width"] - first[key]["width"]) < 1, f"{key} header width mismatch"


class TestBandGridTwoColumn(_BandGridBase):
    """A pane with room for two Label/Min/Max groups uses both.

    A tall single column wasted most of the editor's width on models with many
    mapped labels. The switch is a container query on the widget, not a viewport
    media query, because the editor pane is user-resizable — so these tests drive
    it by pane width and check the fallback still holds when the pane is narrow.
    """

    def test_wide_pane_pairs_rows_side_by_side(self, page: Page, base_url: str):
        """Rows 1 and 2 sit on one line, with row 2 entirely right of row 1."""
        self._open_multi_label_editor(page, base_url, WIDE_VIEWPORT)
        rows = self._rows(page)

        assert abs(rows[0]["min"]["y"] - rows[1]["min"]["y"]) < 1, "first two rows are not on the same line"
        assert rows[1]["name"]["x"] > rows[0]["max"]["x"], "second row is not in the right-hand group"
        # Third row wraps back to the left group, under the first.
        assert rows[2]["min"]["y"] > rows[0]["min"]["y"], "third row did not wrap to a new line"
        assert abs(rows[2]["min"]["x"] - rows[0]["min"]["x"]) < 1, "wrapped row is not back in the left group"

    def test_wide_pane_columns_stay_aligned_within_each_group(self, page: Page, base_url: str):
        """Two groups means two sets of columns, each internally straight —
        rows 0 and 2 share one min column, rows 1 and 3 share the other."""
        self._open_multi_label_editor(page, base_url, WIDE_VIEWPORT)
        rows = self._rows(page)

        for key in ("min", "max"):
            assert abs(rows[0][key]["x"] - rows[2][key]["x"]) < 1, f"left group {key} column is ragged"
            assert abs(rows[1][key]["x"] - rows[3][key]["x"]) < 1, f"right group {key} column is ragged"

    def test_wide_pane_shows_a_header_per_group(self, page: Page, base_url: str):
        """Each group is labelled; the second group's header is not orphaned."""
        self._open_multi_label_editor(page, base_url, WIDE_VIEWPORT)
        assert self._visible_heads(page) == ["label", "min", "max", "label", "min", "max"]

        heads = page.locator("#task-config-list .mlsf-bands__head:visible")
        rows = self._rows(page)
        # The second group's Label header sits over the second group's names.
        assert abs(heads.nth(3).bounding_box()["x"] - rows[1]["name"]["x"]) < 1

    def test_narrow_pane_falls_back_to_one_column(self, page: Page, base_url: str):
        """Below the breakpoint every row gets its own line — no truncated
        two-column layout squeezed into a narrow pane."""
        self._open_multi_label_editor(page, base_url, NARROW_VIEWPORT)
        ys = [r["min"]["y"] for r in self._rows(page)]
        assert len(set(round(y) for y in ys)) == len(ys), f"rows share lines in a narrow pane: {ys}"

    def test_values_stay_next_to_their_own_label(self, page: Page, base_url: str):
        """The table sizes to its content instead of stretching across the pane.

        The name track was first written as `minmax(4em, 1fr)`, which hands the
        whole surplus width of the pane to the label column. Each label then sat
        at the far left of a wide blank gap with its own Min/Max pushed to the
        right — landing hard against the *next* group's label, so the numbers
        read as belonging to the row beside them. A content-sized track keeps
        every value beside the label it belongs to.
        """
        self._open_multi_label_editor(page, base_url, WIDE_VIEWPORT)

        grid = page.locator("#task-config-list .mlsf-bands").first.bounding_box()
        mlsf = page.locator("#task-config-list .mlsf").first.bounding_box()
        assert grid["width"] < mlsf["width"] - 40, (
            f"band table ({grid['width']}px) is stretching to fill its container "
            f"({mlsf['width']}px) — the surplus belongs outside the table, not inside a label column"
        )

    def test_gutter_separates_the_two_groups(self, page: Page, base_url: str):
        """The slack sits in front of each label, so the widest gap on a line is
        the one between the groups rather than the one inside them. Only in
        two-column mode — a single column has no neighbour to be confused with."""
        pad = "el => getComputedStyle(el).paddingLeft"

        self._open_multi_label_editor(page, base_url, WIDE_VIEWPORT)
        wide = page.locator("#task-config-list .mlsf-band-row__name").first.evaluate(pad)
        assert float(wide.rstrip("px")) > 0, f"no gutter in front of labels in two-column mode ({wide})"

        self._open_multi_label_editor(page, base_url, NARROW_VIEWPORT)
        narrow = page.locator("#task-config-list .mlsf-band-row__name").first.evaluate(pad)
        assert float(narrow.rstrip("px")) == 0, f"single column should not be indented ({narrow})"


class TestBandGridInsideArrayField(_BandGridBase):
    """The widget must lay out correctly where the real task puts it: as an item
    of an ArrayField, not as a bare schema field.

    ArrayField wraps each item in a `display: flex; flex-direction: row` row —
    a shrink-to-fit context. `.mlsf` carries `container-type: inline-size` for
    the two-column switch, and inline-size containment computes an element's
    width *as if it had no content*, so in that row the widget collapsed to 18px
    and its grid spilled out across the rest of the form: names truncated to a
    few characters, help text one word wide, rows running under the footer. Only
    the bare-field shape was covered, and that one lands in a stretched column
    flex, so it never saw the collapse.
    """

    def _open_array_editor(self, page: Page, base_url: str, viewport: dict):
        page.set_viewport_size(viewport)
        _mock_apis(
            page,
            configs={FAKE_CONFIG_UUID: _array_config({lbl["uuid"]: 0.5 for lbl in self.LABELS})},
            labels=self.LABELS,
            schema=MODEL_LABEL_ARRAY_SCHEMA,
        )
        _navigate_to_tasks(page, base_url)
        _open_edit(page)
        page.wait_for_selector("#task-config-list .mlsf-band-row")
        expect(page.locator("#task-config-list .mlsf-band-row")).to_have_count(len(self.LABELS))

    def test_widget_fills_its_array_row_instead_of_collapsing(self, page: Page, base_url: str):
        """It takes most of the editor's width, rather than the ~18px that
        inline-size containment leaves an element sized by its own content."""
        self._open_array_editor(page, base_url, WIDE_VIEWPORT)

        mlsf = page.locator("#task-config-list .mlsf").first.bounding_box()
        editor = page.locator("#task-config-list .le-editor").bounding_box()
        assert mlsf["width"] > editor["width"] * 0.6, (
            f"widget is {mlsf['width']}px inside a {editor['width']}px editor — " "inline-size containment collapsed it"
        )

    def test_band_grid_stays_inside_the_widget(self, page: Page, base_url: str):
        """Nothing overflows the widget's box — the visible symptom of the
        collapse was the grid drawing outside its own border."""
        self._open_array_editor(page, base_url, WIDE_VIEWPORT)

        mlsf = page.locator("#task-config-list .mlsf").first.bounding_box()
        grid = page.locator("#task-config-list .mlsf-bands").first.bounding_box()
        assert grid["x"] >= mlsf["x"] - 1, "grid starts left of its container"
        assert (
            grid["x"] + grid["width"] <= mlsf["x"] + mlsf["width"] + 1
        ), f"grid overflows its container: grid right={grid['x'] + grid['width']}, mlsf right={mlsf['x'] + mlsf['width']}"

        for row in self._rows(page):
            right = row["max"]["x"] + row["max"]["width"]
            assert right <= mlsf["x"] + mlsf["width"] + 1, f"a max input overflows the widget at x={right}"

    def test_array_shape_still_pairs_rows_side_by_side(self, page: Page, base_url: str):
        """The two-column layout works here too — this is the shape that has to
        benefit from it, since it is the one users actually see."""
        self._open_array_editor(page, base_url, WIDE_VIEWPORT)
        rows = self._rows(page)

        assert abs(rows[0]["min"]["y"] - rows[1]["min"]["y"]) < 1, "first two rows are not on the same line"
        assert rows[1]["name"]["x"] > rows[0]["max"]["x"], "second row is not in the right-hand group"

    def test_array_shape_falls_back_to_one_column_when_narrow(self, page: Page, base_url: str):
        self._open_array_editor(page, base_url, NARROW_VIEWPORT)
        ys = [r["min"]["y"] for r in self._rows(page)]
        assert len(set(round(y) for y in ys)) == len(ys), f"rows share lines in a narrow pane: {ys}"
