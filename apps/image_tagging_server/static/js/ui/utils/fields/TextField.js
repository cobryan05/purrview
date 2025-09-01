import { Field } from './Field.js';

export class TextField extends Field {
  renderEdit() {
    const input = document.createElement('input');
    input.type = 'text';
    input.value = this._value || '';
    input.placeholder = this._placeholder;
    input.addEventListener('input', () => {
      this._value = input.value;
      this._onChange?.(this._value);
    });
    return input;
  }

  renderView() {
    const span = document.createElement('span');
    span.textContent = this._value || '(empty)';
    return span;
  }

  getValue() {
    return {
      text: this._value
    };
  }
}