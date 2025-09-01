import { Field } from './Field.js';

export class RadioField extends Field {
  renderEdit() {
    const container = document.createElement('div');
    const groupName = `radio-${Math.random().toString(36).slice(2)}`;

    this._options.forEach(opt => {
      const label = document.createElement('label');
      const input = document.createElement('input');
      input.type = 'radio';
      input.name = groupName;
      input.value = opt;
      input.checked = this._value === opt;
      input.addEventListener('change', () => {
        this._value = opt;
        this._onChange?.(this._value);
      });
      label.appendChild(input);
      label.append(opt);
      container.appendChild(label);
      container.appendChild(document.createElement('br'));
    });

    return container;
  }

  renderView() {
    const span = document.createElement('span');
    span.textContent = this._value || '(none)';
    return span;
  }

  getValue() {
    return { option: this._value || null };
  }
}
