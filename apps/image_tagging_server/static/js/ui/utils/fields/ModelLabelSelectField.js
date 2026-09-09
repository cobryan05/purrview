import { DropDownField } from './DropDownField.js';
import { Field } from './Field.js';
import { LabelNumberField } from './LabelNumberField.js';
import { fetchLabelByUuid } from '/app-static/js/shared/api/labels.js';
import { fetchModelsLabelMappings, fetchModelsList } from '/app-static/js/shared/api/models.js';
export class ModelLabelSelectField extends Field {
  static DEFAULT_CONFIDENCE = 0.25;

  /** Column headers and help copy when the schema doesn't override them. */
  static DEFAULT_BAND_COPY = {
    nameLabel: 'Label',
    minLabel: 'Min',
    maxLabel: 'Max',
    bandHelp:
      'Select a model, then set a min/max confidence band to enable labels (min > 0 enables; min = 0 disables)',
  };

  /**
   * The band's *meaning* is the caller's business, not this field's. A task that
   * uses the two bounds for something other than a plain floor/ceiling passes
   * its own `nameLabel`/`minLabel`/`maxLabel`/`bandHelp` through its
   * `params_schema()` entry (Factories.model_label spreads fieldMeta straight
   * in), and the wording follows. Defaults keep the neutral Min/Max reading, so
   * this stays a generic per-label range picker rather than one task's widget.
   */
  constructor({ nameLabel, minLabel, maxLabel, bandHelp, ...rest }) {
    super(rest);
    const d = ModelLabelSelectField.DEFAULT_BAND_COPY;
    this._nameLabel = nameLabel || d.nameLabel;
    this._minLabel = minLabel || d.minLabel;
    this._maxLabel = maxLabel || d.maxLabel;
    this._bandHelp = bandHelp || d.bandHelp;
  }

  static async create({ ...rest } = {}) {
    const { value } = rest;
    const { labelValues, modelName } = value || {};
    const instance = new ModelLabelSelectField({ ...rest });
    const models = await fetchModelsList();
    await instance._handleModelChange(modelName, labelValues);

    instance._dropDownField = new DropDownField({
      options: models,
      value: modelName,
      onChange: async (modelName) => {
        instance._labelValues = new Map();
        await instance._handleModelChange(modelName);
      }
    });

    return instance;
  }

  async _handleModelChange(modelName, values = {}) {
    this._labelUuidMapping = modelName ? (await fetchModelsLabelMappings({modelNames: [modelName]})).get(modelName) : new Map();
    this._fields = new Map();

    if (this._labelUuidMapping.size > 0) {
      const uniqueUuids = [...new Set(this._labelUuidMapping.values())];
      const labelInfos = (await Promise.all(uniqueUuids.map(fetchLabelByUuid))).filter((info) => info != null);

      for (const labelInfo of labelInfos) {
        const { color, name, uuid } = labelInfo;
        this._fields.set(uuid, new LabelNumberField({
          label: name,
          color: color,
          min: 0,
          max: 1,
          step: 0.01,
          // `??` (not `||`) so a legitimately-saved scalar `0` — the existing
          // "label disabled" convention — survives instead of silently
          // falling back to DEFAULT_CONFIDENCE and re-enabling the label.
          value: values[uuid] ?? ModelLabelSelectField.DEFAULT_CONFIDENCE,
        }));
      }
    }
    await this._renderLabelList();
  }

  /**
   * Fills the band table: one header row, then one `display: contents` row per
   * label. The header carries the bound wording (schema-overridable, "Min"/"Max"
   * by default) that used to be repeated as a tag beside every single input —
   * with the inputs actually in columns, naming them once is both shorter and
   * clearer.
   */
  async _renderLabelList() {
    if (!this._labelsContainer) {
      return;
    }

    this._labelsContainer.innerHTML = '';

    if (this._fields.size === 0) {
      this._labelsContainer.classList.add('mlsf-bands--empty');
      const empty = document.createElement('div');
      empty.className = 'mlsf-bands__empty';
      empty.textContent = 'No labels mapped for this model.';
      this._labelsContainer.appendChild(empty);
      return;
    }

    this._labelsContainer.classList.remove('mlsf-bands--empty');
    // The header is emitted twice. A wide enough pane lays the table out as two
    // side-by-side Label/Min/Max groups (see the `@container` rule in app.css),
    // and the second group needs its own header; a narrow one hides the copy.
    // Whether there is room is a pure layout question, so CSS decides it — JS
    // would have to duplicate the breakpoint and watch the pane for resizes.
    for (const pass of [0, 1]) {
      // `--value` marks the two numeric columns, which are right-aligned over
      // right-aligned digits. Positional selectors would do it, but they'd break
      // the moment the header count changed.
      for (const [text, kind] of [[this._nameLabel, 'name'], [this._minLabel, 'value'], [this._maxLabel, 'value']]) {
        const head = document.createElement('div');
        head.className = `mlsf-bands__head mlsf-bands__head--${kind}`;
        if (pass) head.classList.add('mlsf-bands__head--second');
        head.textContent = text;
        this._labelsContainer.appendChild(head);
      }
    }

    // Rendered sequentially rather than via Promise.all: the rows are grid
    // children in DOM order, so an out-of-order append would scramble the table.
    for (const field of this._fields.values()) {
      this._labelsContainer.appendChild(await field.renderView());
    }
  }

  async renderEdit() {
    const container = document.createElement('div');
    container.className = 'mlsf';

    // Row with model dropdown + help text
    const modelRow = document.createElement('div');
    modelRow.className = 'mlsf__model-row';

    const dropDown = await this._dropDownField.renderEdit();
    dropDown.classList.add('mlsf__model-select');
    modelRow.appendChild(dropDown);

    const helpText = document.createElement('div');
    helpText.className = 'mlsf__help';
    helpText.textContent = this._bandHelp;
    modelRow.appendChild(helpText);

    // Labels grid — three real columns, so every row's min/max inputs line up
    // no matter how wide the label names are.
    this._labelsContainer = document.createElement('div');
    this._labelsContainer.className = 'mlsf-bands';

    await this._renderLabelList();

    container.appendChild(modelRow);
    container.appendChild(this._labelsContainer);
    return container;
  }


  async renderView() {
    const container = document.createElement('div');
    container.style.display = 'flex';
    container.style.alignItems = 'center';
    container.style.gap = '0.5em';

    container.appendChild(await this._dropDownField.renderView());

    return container;
  }

  getValue() {
    return {
      modelName: this._dropDownField.getValue(),
      labelValues: Object.fromEntries([...this._fields.entries()].map(([labelName, field]) => [labelName, field.getValue()])),
    };
  }
}
