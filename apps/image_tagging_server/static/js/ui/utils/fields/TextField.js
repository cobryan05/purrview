import { Field } from './Field.js';

export class TextField extends Field {
  // store type and send ..rest to field in constructor
  constructor({ type = 'text', min = null, max = null, step = null, ...rest }) {
    super(rest);
    this._type = type;
    this._min = min;
    this._max = max;
    this._step = step;
    this._el = null;
  }

  setValue(value) {
    this._value = value;
    if (this._el) this._el.value = value;
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
    this._el = input;
    return input;
  }

  async renderView() {
    const span = document.createElement('span');
    span.textContent = this._value != null && this._value !== '' ? this._value : '(empty)';
    return span;
  }

  getValue() {
    // Numeric inputs store the raw string typed by the user (or a number set
    // programmatically via setValue/the constructor) — parse here so callers
    // consistently get a number rather than sometimes a string, sometimes a
    // number depending on whether the user touched the field.
    if (this._type === 'number') {
      if (this._value === '' || this._value === null || this._value === undefined) return null;
      const parsed = parseFloat(this._value);
      return Number.isNaN(parsed) ? null : parsed;
    }
    return this._value;
  }
}
