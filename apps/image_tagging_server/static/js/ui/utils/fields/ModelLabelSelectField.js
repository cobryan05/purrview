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

  static async create({ modelName = '', labelUuidMapping = null, ...rest } = {}) {
    const instance = new ModelLabelSelectField({ ...rest });
    const models = await fetchModelsList();

    // If no mapping passed but a model name was provided, fetch the labels now
    if (!labelUuidMapping && modelName) {
      labelUuidMapping = await fetchModelLabelMappings(modelName);
    }
    instance._labelUuidMapping = labelUuidMapping || new Map();
    instance._values = new Map();
    // Initialize DropDownField with onChange
    instance._dropDownField = new DropDownField({
      options: models,
      value: modelName,
      onChange: async (modelName) => {
        instance._labelUuidMapping = await fetchModelLabelMappings(modelName);
        instance._values = new Map();
        await instance._renderLabelList();
      }
    });
    return instance;
  }

  async _renderLabelList() {
    if (!this._labelsContainer) {
      return;
    }

    this._labelsContainer.innerHTML = '';

    const uniqueUuids = [...new Set(
      [...this._labelUuidMapping.values()].filter(uuid => uuid != null)
    )];

    const labelInfos = await Promise.all(uniqueUuids.map(fetchLabelByUuid));
    labelInfos.sort((a, b) =>
      a.metadata.name.localeCompare(b.metadata.name, undefined, { sensitivity: 'base' })
    );

    for (const labelInfo of labelInfos) {
      if( !this._values.has(labelInfo.metadata.name) ) {
        this._values.set(labelInfo.metadata.name, ModelLabelSelectField.DEFAULT_CONFIDENCE);
      }
      const row = createGenericRow({
        field: new LabelNumberField({
          label: labelInfo.metadata.name,
          color: labelInfo.metadata.color,
          min: 0,
          max: 1,
          step: 0.01,
          value: this._values.get(labelInfo.metadata.name),
          onChange: (val) => {
            this._values.set(labelInfo.metadata.name, val);
          },
        }),
      });

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
      typename: this._dropDownField.getValue().option ?? null,
      labelValues: Object.fromEntries(this._values),
    };
  }
}
