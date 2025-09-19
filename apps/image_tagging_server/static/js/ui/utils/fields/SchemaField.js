import { Field } from './Field.js';
import { fieldFactories } from './Factories.js';
import { TextField } from './TextField.js'

export class SchemaField extends Field {
  constructor({ schema = {}, values = {}, onChange = null } = {}) {
    super({ onChange });
    this._schema = schema; // The schema definition
    this._values = values; // Current values for the fields
  }

  /**
   * Renders the form in edit mode based on the schema.
   * @returns {HTMLElement} The form container.
   */
  async renderEdit() {
    const container = document.createElement('div');
    container.style.display = 'flex';
    container.style.flexDirection = 'column';
    container.style.gap = '0.5em';

    Object.entries(this._schema).forEach(async ([fieldName, fieldMeta]) => {
      const fieldContainer = document.createElement('div');
      fieldContainer.style.display = 'flex';
      fieldContainer.style.flexDirection = 'column';

      // Label
      const label = document.createElement('label');
      label.textContent = fieldMeta.label || fieldName;
      label.style.fontWeight = 'bold';
      fieldContainer.appendChild(label);

      // Input
      const inputField = await this._createInputField(fieldName, fieldMeta);
      inputField.onChange = () => {
        this._values[fieldName] = inputField.getValue(); //inputField.type === 'checkbox' ? inputField.checked : inputField.value;
        this._onChange?.(this._values);
      };
      fieldContainer.appendChild(await inputField.renderEdit());

      container.appendChild(fieldContainer);
    });

    return container;
  }

  /**
   * Renders the form in view mode (read-only).
   * @returns {HTMLElement} The form container.
   */
  async renderView() {
    const container = document.createElement('div');
    container.style.display = 'flex';
    container.style.flexDirection = 'column';
    container.style.gap = '0.5em';

    Object.entries(this._schema).forEach(([fieldName, fieldMeta]) => {
      const fieldContainer = document.createElement('div');
      fieldContainer.style.display = 'flex';
      fieldContainer.style.flexDirection = 'column';

      // Label
      const label = document.createElement('label');
      label.textContent = fieldMeta.label || fieldName;
      label.style.fontWeight = 'bold';
      fieldContainer.appendChild(label);

      // Value
      const value = document.createElement('span');
      value.textContent = this._values[fieldName] || '(none)';
      fieldContainer.appendChild(value);

      container.appendChild(fieldContainer);
    });

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
      field = await factory(fieldName, fieldMeta, this._values, this._onChange);
    } else {
      // Fallback generic input
      field = new TextField({
        name: fieldName,
        placeholder: `Unhandled type: ${fieldMeta.type}`,
        value: this._values[fieldName],
        onChange: (value) => {
          this._values[fieldName] = value;
          this._onChange?.(this._values);
        },
      });
    }
    field.name = fieldName;

    // If it’s a custom field class, render it
    return field
  }


  /**
   * Returns the current values of the form.
   * @returns {object} The current values.
   */
  getValue() {
    return this._values;
  }
}
