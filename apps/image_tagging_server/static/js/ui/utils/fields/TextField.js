import { Field } from './Field.js';

export class TextField extends Field {
  renderEdit() {
    const input = document.createElement('input');
    input.type = 'text';
    input.value = this.value || '';
    input.placeholder = this.placeholder;
    input.addEventListener('input', () => {
      this.value = input.value;
      this.onChange?.(this.value);
    });
    return input;
  }

  renderView() {
    const span = document.createElement('span');
    span.textContent = this.value || '(empty)';
    return span;
  }

  getValue() {
    return {
      text: this.value
    };
  }
}