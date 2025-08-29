import { Field } from './Field.js';

export class DropDownField extends Field {
  constructor(params = {}) {
    super(params);
  }

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

      if (typeof opt === 'string') {
        optionEl.value = opt;
        optionEl.textContent = opt;
      } else if (typeof opt === 'object') {
        optionEl.value = opt.key ?? opt.text;
        optionEl.textContent = opt.text;
        if (opt.color) optionEl.style.color = opt.color;
      }

      if (this.isMatchingOption(opt, this.value)) {
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
    const selectedOption = this.options.find(opt => this.isMatchingOption(opt, this.value));

    if (selectedOption) {
      if (typeof selectedOption === 'string') {
        span.textContent = selectedOption;
      } else {
        span.textContent = selectedOption.text;
        if (selectedOption.color) span.style.color = selectedOption.color;
      }
    } else {
      span.textContent = '(none)';
    }

    return span;
  }

  getValue() {
    const selectedOption = this.options.find(opt => this.isMatchingOption(opt, this.value));
    return { option: selectedOption ?? null };
  }

  isMatchingOption(option, value) {
    if (typeof option === 'string') {
      return option === value;
    } else if (typeof option === 'object') {
      return option.key === (typeof value === 'object' ? value?.key : value);
    }
    return false;
  }
}
