import { DropDownField } from './DropDownField.js';
import { Field } from './Field.js';
import { SchemaField } from './SchemaField.js';
import { TextField } from './TextField.js';
import { fetchTaskConfigs, fetchTaskTypeList, fetchTaskTypeSchemas } from '/app-static/js/shared/api/tasks.js';
import { Logger } from '/app-static/js/ui/utils/index.js';
import { pickExistingModal } from '/app-static/js/ui/utils/pickExistingModal.js';

const FROM_EXISTING_SENTINEL = '__from_existing__';

export class TaskConfigField extends Field {
  constructor({ ...rest }) {
    super(rest);
  }

  static async create({ name = '', typename = null, schema = null, uuid = null, ...rest } = {}) {
    const instance = new TaskConfigField({ ...rest });


    // If no schema passed but a typename is provided, fetch the schema now
    if (!schema && typename) {
      schema = await fetchTaskTypeSchemas([typename]).then(schemas => schemas.get(typename));
    }

    instance._uuid = uuid
    instance._textField = new TextField({ value: name, placeholder: 'Enter new config name' });
    instance._schemaField = await SchemaField.create({ schema: schema ?? {}, values: instance._value || {} });
    const taskTypes = await fetchTaskTypeList();
    const taskTypeOptions = [
      { key: FROM_EXISTING_SENTINEL, text: 'From Existing...' },
      ...taskTypes,
    ];

    instance._dropDownField = new DropDownField({
      options: taskTypeOptions,
      value: typename,
      onChange: async (newTypename) => {
        if (newTypename === FROM_EXISTING_SENTINEL) {
          try {
            const configs = await fetchTaskConfigs();
            const picked = await pickExistingModal({ items: configs, getLabel: c => c.name });
            if (picked) {
              await instance._prefillFrom(picked);
            } else {
              instance._dropDownField.setValue('');
            }
          } catch (err) {
            Logger.error('Failed to copy from existing task config:', err);
          }
          return;
        }
        try {
          const newSchema = await fetchTaskTypeSchemas([newTypename]).then(schemas => schemas.get(newTypename));
          instance._schemaField = await SchemaField.create({
            schema: newSchema,
            values: instance._schemaField.getValue()
          });
          await instance._rerenderSchema();
        } catch (err) {
          Logger.error('Failed to fetch schema:', err);
        }
      }
    });

    return instance;
  }

  async _prefillFrom({ typename, params_json, name }) {
    this._dropDownField.setValue(typename);
    this._textField.setValue(`Copy of ${name}`);
    try {
      const schema = await fetchTaskTypeSchemas([typename]).then(s => s.get(typename));
      this._schemaField = await SchemaField.create({ schema, values: params_json });
      await this._rerenderSchema();
    } catch (err) {
      Logger.error('Failed to load schema for copied task config:', err);
    }
  }

  // Helper to re-render the schemaField in the DOM
  async _rerenderSchema() {
    if (!this._schemaContainer) return;
    this._schemaContainer.innerHTML = '';
    this._schemaContainer.appendChild(await this._schemaField.renderEdit());
  }

  async renderEdit() {
    const container = document.createElement('div');
    container.style.display = 'flex';
    container.style.flexDirection = 'column';
    container.style.width = '100%';

    // Row 1: dropdown + text
    const topRow = document.createElement('div');
    topRow.style.display = 'flex';
    topRow.style.gap = '0.5em';
    topRow.appendChild(await this._dropDownField.renderEdit());
    topRow.appendChild(await this._textField.renderEdit());

    const uuidInfo = document.createElement('div');
    uuidInfo.textContent = `UUID: ${this._uuid || '(none)'}`;
    uuidInfo.style.fontSize = '0.75em';
    uuidInfo.style.color = '#888';
    uuidInfo.style.marginLeft = '0.25em';
    uuidInfo.style.userSelect = 'text';
    uuidInfo.style.fontFamily = 'monospace';
    uuidInfo.style.overflowWrap = 'anywhere'; // wrap long UUID if needed

    const schemaRow = document.createElement('div');
    this._schemaContainer = schemaRow; // remember container for re-render
    await this._rerenderSchema();

    container.appendChild(topRow);
    if (this._uuid) {
      container.appendChild(uuidInfo);
    }
    container.appendChild(schemaRow);
    return container;
  }

  async renderView() {
    const container = document.createElement('div');
    container.style.display = 'flex';
    container.style.alignItems = 'center';
    container.style.gap = '0.5em';

    container.appendChild(await this._dropDownField.renderView());
    container.appendChild(await this._textField.renderView());

    return container;
  }

  getValue() {
    return {
      name: this._textField.getValue(),
      typename: this._dropDownField.getValue(),
      params: this._schemaField.getValue()
    };
  }
}
