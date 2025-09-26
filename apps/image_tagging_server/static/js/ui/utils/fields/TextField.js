import { Field } from './Field.js';

export class TextField extends Field {
  // store type and send ..rest to field in constructor
  constructor({ type = 'text', min = null, max = null, step = null, ...rest }) {
    super(rest);
    this._type = type;
    this._min = min;
    this._max = max;
    this._step = step;
  }

  async renderEdit() {
    const input = document.createElement('input');
    input.type = this._type;

    if (this._type == 'number') {
      if (this._min !== null) {
        input.min = this._min;
      }
      if (this._max !== null) {
        input.max = this._max;
      }
      if (this._step !== null) {
        input.step = this._step;
      } else {
        input.step = '1';
      }
    }

    if (typeof this._value === 'string') {
      input.value = this._value;
    } else if (typeof this._value === 'object') {
      input.value = JSON.stringify(this._value);
    } else if (typeof this._value === 'number') {
      input.value = parseFloat(this._value);
    } else {
      input.value = '';
    }

    input.placeholder = this._placeholder;
    input.addEventListener('input', () => {
      this._value = input.value;
      this._onChange?.(this._value);
    });
    return input;
  }

  async renderView() {
    const span = document.createElement('span');
    span.textContent = this._value || '(empty)';
    return span;
  }

  getValue() {
    return this._value;
  }
}
