import { DropDownField } from './DropDownField.js';
import { Field } from './Field.js';
import { LabelNumberField } from './LabelNumberField.js'
import { fetchModelsList, fetchModelLabelMappings, fetchLabelByUuid } from '/app-static/js/ui/utils/index.js';
import { createGenericRow } from '/app-static/js/ui/utils/index.js';
export class ModelLabelSelectField extends Field {
  static DEFAULT_CONFIDENCE = 0.25;

  constructor({ ...rest }) {
    super(rest);
  }

  static async create({ modelName = '', ...rest } = {}) {
    const instance = new ModelLabelSelectField({ ...rest });
    const models = await fetchModelsList();
    await instance._handleModelChange(modelName);

    // Initialize DropDownField with onChange
    instance._dropDownField = new DropDownField({
      options: models,
      value: modelName,
      onChange: async (modelName) => {
        await instance._handleModelChange(modelName);
      }
    });

    return instance;
  }

  async _handleModelChange(modelName) {
    this._labelUuidMapping = modelName ? await fetchModelLabelMappings(modelName) : new Map();
    this._fields = new Map();

    if (this._labelUuidMapping.size > 0) {
      const uniqueUuids = [...new Set(this._labelUuidMapping.values())];
      const labelInfos = (await Promise.all(uniqueUuids.map(fetchLabelByUuid))).filter((info) => info != null);

      for (const labelInfo of labelInfos) {
        const labelName = labelInfo.metadata.name;
        this._fields.set(labelInfo.metadata.uuid, new LabelNumberField({
          label: labelName,
          color: labelInfo.metadata.color,
          min: 0,
          max: 1,
          step: 0.01,
          value: ModelLabelSelectField.DEFAULT_CONFIDENCE,
          onChange: (val) => {
            this._values.set(labelName, val);
          },
        }));
      }
    }
    await this._renderLabelList();
  }

  async _renderLabelList() {
    if (!this._labelsContainer) {
      return;
    }

    this._labelsContainer.innerHTML = '';
    for (const [labelName, field] of this._fields) {
      const row = createGenericRow({ field });
      this._labelsContainer.appendChild(row);
    }
  }

  async renderEdit() {
    const container = document.createElement('div');
    container.style.display = 'inline-flex';
    container.style.flexDirection = 'column';
    container.style.border = '1px dotted #fff8';
    container.style.padding = '0.5em';
    container.style.gap = '0.5em';
    container.style.alignItems = 'flex-start';

    // Row with model dropdown + help text
    const modelRow = document.createElement('div');
    modelRow.style.display = 'grid';
    modelRow.style.gridTemplateColumns = 'auto 1fr'; // dropdown on left, help takes remaining space
    modelRow.style.alignItems = 'center';
    modelRow.style.gap = '1em';

    // Left column: dropdown
    modelRow.appendChild(await this._dropDownField.renderEdit());

    // Right column: help text
    const helpText = document.createElement('div');
    helpText.textContent =
      'Select a model then set non-zero confidence thresholds to enable labels';
    helpText.style.fontSize = '0.85em';
    helpText.style.color = '#aaa';
    helpText.style.fontStyle = 'italic';
    helpText.style.whiteSpace = 'nowrap'; // keep it on one line
    helpText.style.justifySelf = 'start'; // align to left of its cell
    modelRow.appendChild(helpText);

    // Labels grid
    this._labelsContainer = document.createElement('div');
    this._labelsContainer.style.display = 'grid';
    this._labelsContainer.style.gridTemplateColumns = 'auto auto auto';
    this._labelsContainer.style.gap = '0.25em 1em';
    this._labelsContainer.style.alignItems = 'center';

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
