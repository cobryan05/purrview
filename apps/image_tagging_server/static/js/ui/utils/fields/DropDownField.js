import { Field } from './Field.js';

export class DropDownField extends Field {
  renderEdit() {
    const select = document.createElement('select');

    // Add placeholder if no value is set
    if (!this.value) {
      const placeholderOption = document.createElement('option');
      placeholderOption.value = '';
      placeholderOption.textContent = this.placeholder || 'Select...';
      placeholderOption.disabled = true;
      placeholderOption.selected = true;
      select.appendChild(placeholderOption);
    }

    this.options.forEach(opt => {
      const optionEl = document.createElement('option');
      optionEl.value = opt;
      optionEl.textContent = opt;
      if (this.value && this.value === opt) {
        optionEl.selected = true;
      }
      select.appendChild(optionEl);
    });

    select.addEventListener('change', () => {
      this.value = select.value;
      this.onChange?.(this.value);
    });

    return select;
  }

  renderView() {
    const span = document.createElement('span');
    span.textContent = this.value || '(none)';
    return span;
  }
}
