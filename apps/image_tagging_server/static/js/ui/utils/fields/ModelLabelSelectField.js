import { DropDownField } from './DropDownField.js';
import { Field } from './Field.js';
import { LabelNumberField } from './LabelNumberField.js'
import { fetchModelsList, fetchModelsClassMappings, fetchClassByUuid } from '/app-static/js/ui/utils/index.js';
import { createGenericRow } from '/app-static/js/ui/utils/index.js';
export class ModelLabelSelectField extends Field {
  static DEFAULT_CONFIDENCE = 0.25;

  constructor({ ...rest }) {
    super(rest);
  }

  static async create({ ...rest } = {}) {
    const { value } = rest;
    const { labelValues, modelName } = value || {};
    const instance = new ModelLabelSelectField({ ...rest });
    const models = await fetchModelsList();
    await instance._handleModelChange(modelName, labelValues);

    // Initialize DropDownField with onChange
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
    this._classUuidMapping = modelName ? (await fetchModelsClassMappings({modelNames: [modelName]})).get(modelName) : new Map();
    this._fields = new Map();

    if (this._classUuidMapping.size > 0) {
      const uniqueUuids = [...new Set(this._classUuidMapping.values())];
      const classInfos = (await Promise.all(uniqueUuids.map(fetchClassByUuid))).filter((info) => info != null);

      for (const classInfo of classInfos) {
        const { color, name, uuid } = classInfo.metadata;
        this._fields.set(uuid, new LabelNumberField({
          label: name,
          color: color,
          min: 0,
          max: 1,
          step: 0.01,
          value: values[uuid] || ModelLabelSelectField.DEFAULT_CONFIDENCE,
        }));
      }
    }
    await this._renderClassList();
  }

  async _renderClassList() {
    if (!this._classesContainer) {
      return;
    }

    this._classesContainer.innerHTML = '';
    for (const [labelName, field] of this._fields) {
      const row = createGenericRow({ field });
      this._classesContainer.appendChild(row);
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
    modelRow.style.gridTemplateColumns = 'auto 1fr';
    modelRow.style.alignItems = 'center';
    modelRow.style.gap = '1em';

    // Left column: dropdown
    const dropDown = await this._dropDownField.renderEdit();
    dropDown.style.minWidth = '10em'; // keep it readable
    modelRow.appendChild(dropDown);

    // Right column: help text
    const helpText = document.createElement('div');
    helpText.textContent =
      'Select a model, then set a non-zero confidence threshold to enable classes';
    helpText.style.fontSize = '0.85em';
    helpText.style.color = '#aaa';
    helpText.style.fontStyle = 'italic';
    helpText.style.whiteSpace = 'normal'; // allow wrapping
    helpText.style.lineHeight = '1.2em'; // make wrapped text look good
    helpText.style.alignSelf = 'start'; // align text nicely when multi-line
    modelRow.appendChild(helpText);

    // Labels grid
    this._classesContainer = document.createElement('div');
    this._classesContainer.style.display = 'grid';
    this._classesContainer.style.gridTemplateColumns = 'auto auto auto';
    this._classesContainer.style.gap = '0.25em 1em';
    this._classesContainer.style.alignItems = 'center';

    await this._renderClassList();

    container.appendChild(modelRow);
    container.appendChild(this._classesContainer);
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
      classesValues: Object.fromEntries([...this._fields.entries()].map(([className, field]) => [className, field.getValue()])),
    };
  }
}
