import { Field } from './Field.js';
import { fieldFactories } from './Factories.js';
import { TextField } from './TextField.js'

export class SchemaField extends Field {
  constructor({ ...rest }) {
    super(rest);
  }

  static async create({ schema = {}, values = {}, ...rest } = {}) {
    const instance = new SchemaField({ ...rest });
    instance._schema = schema; // The schema definition
    instance._init_values = values; // Current values for the fields
    instance._fields = {};

    for (const [fieldName, fieldMeta] of Object.entries(schema)) {
      if (fieldName === 'meta') continue;

      // If schema specified a default value then set it here.
      if (fieldMeta.hasOwnProperty('default') && !instance._init_values.hasOwnProperty(fieldName)) {
        instance._init_values[fieldName] = fieldMeta.default;
      }

      const inputField = await instance._createInputField(fieldName, fieldMeta);
      instance._fields[fieldName] = inputField;
    }

    return instance;
  }

  /**
   * Renders the form in edit mode based on the schema.
   * @returns {HTMLElement} The form container.
   */
  async renderEdit() {
    return this._renderFieldsWith((field) => field.renderEdit());
  }

  /**
   * Renders the form in view mode (read-only).
   * @returns {HTMLElement} The form container.
   */
  async renderView() {
    return this._renderFieldsWith((field) => field.renderView());
  }

  /**
 * Renders the form children using the callback for each item
 * @returns {HTMLElement} The form container.
 */
  async _renderFieldsWith(fieldRenderCallback) {
    const container = document.createElement('div');
    container.style.display = 'flex';
    container.style.flexDirection = 'column';
    container.style.gap = '0.5em';

    const order = this._schema.meta?.order;
    let fieldEntries;

    if (order && Array.isArray(order)) {
      const orderedFields = order
        .filter((field) => field in this._fields)
        .map((field) => [field, this._fields[field]]);

      const remainingFields = Object.entries(this._fields).filter(
        ([key]) => key !== 'meta' && !order.includes(key)
      );

      fieldEntries = [...orderedFields, ...remainingFields];
    } else {
      fieldEntries = Object.entries(this._fields).filter(([key]) => key !== 'meta');
    }

    for (const [fieldName, fieldInstance] of fieldEntries) {
      const fieldContainer = document.createElement('div');
      fieldContainer.style.display = 'flex';
      fieldContainer.style.flexDirection = 'column';

      const label = document.createElement('label');
      label.textContent = this._schema[fieldName].label || fieldName;
      label.style.fontWeight = 'bold';
      fieldContainer.appendChild(label);

      const node = await fieldRenderCallback(fieldInstance);
      fieldContainer.appendChild(node);
      container.appendChild(fieldContainer);
    }

    return container;
  }

  /**
   * Creates an input field based on the schema metadata.
   * @param {string} fieldName - The name of the field.
   * @param {object} fieldMeta - Metadata for the field (e.g., type, required).
   * @returns {Field} The input field
   */
  async _createInputField(fieldName, fieldMeta) {
    const factory = fieldFactories[fieldMeta.type];
    let field;

    if (factory) {
      field = await factory(fieldName, fieldMeta, this._init_values,
        (value) => {
          this._init_values[fieldName] = value;
          this._onChange?.(this._init_values);
        });
    } else {
      // Fallback generic input
      field = new TextField({
        name: fieldName,
        placeholder: `Unhandled type: ${fieldMeta.type}`,
        value: this._init_values[fieldName],
        onChange: (value) => {
          this._init_values[fieldName] = value;
          this._onChange?.(this._init_values);
        },
      });
    }
    field.name = fieldName;
    return field
  }


  /**
   * Returns the current values of the form.
   * @returns {object} The current values.
   */
  getValue() {
    const values = {};
    for (const [fieldName, fieldInstance] of Object.entries(this._fields)) {
      values[fieldName] = fieldInstance.getValue();
    }
    return values;
  }
}
