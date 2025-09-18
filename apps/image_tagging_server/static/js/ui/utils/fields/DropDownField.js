import { Field } from './Field.js';

export class DropDownField extends Field {
  constructor(params = {}) {
    super(params);
  }

  async renderEdit() {
    const select = document.createElement('select');

    // Add placeholder if no value is set
    if (!this._value) {
      const placeholderOption = document.createElement('option');
      placeholderOption.value = '';
      placeholderOption.textContent = this._placeholder || 'Select...';
      placeholderOption.disabled = true;
      placeholderOption.selected = true;
      select.appendChild(placeholderOption);
    }

    this._options.forEach(opt => {
      const optionEl = document.createElement('option');

      if (typeof opt === 'string') {
        optionEl.value = opt;
        optionEl.textContent = opt;
      } else if (typeof opt === 'object') {
        optionEl.value = opt.key ?? opt.text;
        optionEl.textContent = opt.text;
        if (opt.color) optionEl.style.color = opt.color;
      }

      if (this._isMatchingOption(opt, this._value)) {
        optionEl.selected = true;
      }

      select.appendChild(optionEl);
    });

    select.addEventListener('change', () => {
      this._value = select.value;
      this._onChange?.(this._value);
    });

    return select;
  }

  async renderView() {
    const span = document.createElement('span');
    const selectedOption = this._options.find(opt => this._isMatchingOption(opt, this._value));

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
    const selectedOption = this._options.find(opt => this._isMatchingOption(opt, this._value));
    return { option: selectedOption ?? null };
  }

  _isMatchingOption(option, value) {
    if (typeof option === 'string') {
      return option === value;
    } else if (typeof option === 'object') {
      return option.key === (typeof value === 'object' ? value?.key : value);
    }
    return false;
  }
}
