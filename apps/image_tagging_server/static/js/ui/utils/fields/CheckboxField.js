import { Field } from './Field.js';

export class CheckboxField extends Field {
  renderEdit() {
    const container = document.createElement('div');
    this.value = Array.isArray(this.value) ? this.value : [];

    this.options.forEach(opt => {
      const label = document.createElement('label');
      const input = document.createElement('input');
      input.type = 'checkbox';
      input.value = opt;
      input.checked = this.value.includes(opt);
      input.addEventListener('change', () => {
        if (input.checked) this.value.push(opt);
        else this.value = this.value.filter(v => v !== opt);
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
    span.textContent = this.value?.length ? this.value.join(', ') : '(none)';
    return span;
  }
}
