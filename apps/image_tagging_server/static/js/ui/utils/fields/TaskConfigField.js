import { DropDownField } from './DropDownField.js';
import { Field } from './Field.js';
import { SchemaField } from './SchemaField.js';
import { TextField } from './TextField.js';
import { error, fetchTaskTypeList, fetchTaskTypeSchema } from '/app-static/js/ui/utils/index.js';

export class TaskConfigField extends Field {
  constructor({ ...rest }) {
    super(rest);
  }

  static async create({ name = '', typename = null, schema = null, ...rest } = {}) {
    const instance = new TaskConfigField({ ...rest });


    // If no schema passed but a typename is provided, fetch the schema now
    if (!schema && typename) {
      schema = await fetchTaskTypeSchema(typename);
    }

    instance._textField = new TextField({ value: name, placeholder: 'Enter new task name' });
    instance._schemaField = new SchemaField({ schema: schema ?? {}, values: instance._value || {} });
    const taskTypes = await fetchTaskTypeList();

    // Initialize DropDownField with onChange
    instance._dropDownField = new DropDownField({
      options: taskTypes,
      value: typename,
      onChange: async (newTypename) => {
        try {
          const newSchema = await fetchTaskTypeSchema(newTypename);
          instance._schemaField = new SchemaField({
            schema: newSchema,
            values: instance._schemaField.getValue() // preserve current values
          });
          await instance._rerenderSchema();
        } catch (err) {
          error('Failed to fetch schema:', err);
        }
      }
    });

    return instance;
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
    const row1 = document.createElement('div');
    row1.style.display = 'flex';
    row1.style.gap = '0.5em';
    row1.appendChild(await this._dropDownField.renderEdit());
    row1.appendChild(await this._textField.renderEdit());

    // Row 2: schema field
    const row2 = document.createElement('div');
    this._schemaContainer = row2; // remember container for re-render
    row2.appendChild(await this._schemaField.renderEdit());

    container.appendChild(row1);
    container.appendChild(row2);
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
      ...this._textField.getValue(),
      typename: this._dropDownField.getValue().option ?? null,
      schema: this._schemaField.getValue()
    };
  }
}
