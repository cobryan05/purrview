import { Field } from './Field.js';

export class BooleanCheckboxField extends Field {
  constructor({ label = 'Enabled', ...rest  } = {}) {
    super(rest);
    this._label = label;
  }

  async renderEdit() {
    const container = document.createElement('div');

    const labelEl = document.createElement('label');
    const input = document.createElement('input');
    input.type = 'checkbox';
    input.checked = this._value;

    input.addEventListener('change', () => {
      this._value = input.checked;
      this._onChange?.(this.getValue());
    });

    labelEl.appendChild(input);

    const textSpan = document.createElement('span');
    textSpan.textContent = this._label;
    labelEl.appendChild(textSpan);

    container.appendChild(labelEl);
    return container;
  }

  async renderView() {
    const span = document.createElement('span');
    span.textContent = this.getValue() ? this._label : '(disabled)';
    return span;
  }

  getValue() {
    return this._value
  }
}
