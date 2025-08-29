import { Field, TextField } from '/app-static/js/ui/utils/index.js';

export class SchemaField extends Field {
  constructor({ schema = {}, values = {}, onChange = null } = {}) {
    super({ onChange });
    this.schema = schema; // The schema definition
    this.values = values; // Current values for the fields
  }

  /**
   * Renders the form in edit mode based on the schema.
   * @returns {HTMLElement} The form container.
   */
  renderEdit() {
    const container = document.createElement('div');
    container.style.display = 'flex';
    container.style.flexDirection = 'column';
    container.style.gap = '0.5em';

    Object.entries(this.schema).forEach(([fieldName, fieldMeta]) => {
      const fieldContainer = document.createElement('div');
      fieldContainer.style.display = 'flex';
      fieldContainer.style.flexDirection = 'column';

      // Label
      const label = document.createElement('label');
      label.textContent = fieldMeta.label || fieldName;
      label.style.fontWeight = 'bold';
      fieldContainer.appendChild(label);

      // Input
      const input = this.createInputField(fieldName, fieldMeta);
      input.value = this.values[fieldName] || '';
      input.onchange = () => {
        this.values[fieldName] = input.type === 'checkbox' ? input.checked : input.value;
        this.onChange?.(this.values);
      };
      fieldContainer.appendChild(input);

      container.appendChild(fieldContainer);
    });

    return container;
  }

  /**
   * Renders the form in view mode (read-only).
   * @returns {HTMLElement} The form container.
   */
  renderView() {
    const container = document.createElement('div');
    container.style.display = 'flex';
    container.style.flexDirection = 'column';
    container.style.gap = '0.5em';

    Object.entries(this.schema).forEach(([fieldName, fieldMeta]) => {
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
      value.textContent = this.values[fieldName] || '(none)';
      fieldContainer.appendChild(value);

      container.appendChild(fieldContainer);
    });

    return container;
  }

  /**
   * Creates an input field based on the schema metadata.
   * @param {string} fieldName - The name of the field.
   * @param {object} fieldMeta - Metadata for the field (e.g., type, required).
   * @returns {HTMLElement} The input element.
   */
  createInputField(fieldName, fieldMeta) {
    let input;

    switch (fieldMeta.type) {
      case 'text':
        field = new TextField({
          placeholder: fieldMeta.placeholder || '',
          value: this.values[fieldName] || '',
          onChange: (value) => {
            this.values[fieldName] = value;
            this.onChange?.(this.values);
          }
        });
        input = field.renderView();
        break;

      case 'number':
        input = document.createElement('input');
        input.type = 'number';
        input.placeholder = fieldMeta.placeholder || '';
        break;

      case 'boolean':
        input = document.createElement('input');
        input.type = 'checkbox';
        input.checked = !!this.values[fieldName];
        break;

      case 'select':
        input = document.createElement('select');
        (fieldMeta.options || []).forEach(option => {
          const opt = document.createElement('option');
          opt.value = option.value || option;
          opt.textContent = option.label || option;
          input.appendChild(opt);
        });
        break;

      default:
        input = document.createElement('input');
        input.type = 'text';
        input.placeholder = fieldMeta.placeholder || '';
        break;
    }

    if (fieldMeta.required) {
      input.required = true;
    }

    input.name = fieldName;
    return input;
  }

  /**
   * Returns the current values of the form.
   * @returns {object} The current values.
   */
  getValue() {
    return this.values;
  }
}
