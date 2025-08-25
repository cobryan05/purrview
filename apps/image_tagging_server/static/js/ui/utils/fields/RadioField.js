import { Field } from './Field.js';

export class RadioField extends Field {
  renderEdit() {
    const container = document.createElement('div');
    const groupName = `radio-${Math.random().toString(36).slice(2)}`;

    this.options.forEach(opt => {
      const label = document.createElement('label');
      const input = document.createElement('input');
      input.type = 'radio';
      input.name = groupName;
      input.value = opt;
      input.checked = this.value === opt;
      input.addEventListener('change', () => {
        this.value = opt;
        this.onChange?.(this.value);
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
    span.textContent = this.value || '(none)';
    return span;
  }
}
