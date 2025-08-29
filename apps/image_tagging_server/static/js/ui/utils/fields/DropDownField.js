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

      // Handle flat strings or objects
      if (typeof opt === 'string') {
        optionEl.value = opt;
        optionEl.textContent = opt;
      } else if (typeof opt === 'object') {
        const key = opt.key || opt.text; // Use key if provided, otherwise fallback to text
        optionEl.value = key;
        optionEl.textContent = opt.text;
        if (opt.color) {
          optionEl.style.color = opt.color; // Apply color to the dropdown option
        }
      }

      // Mark the selected option
      if (this.value && this.value === optionEl.value) {
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

    // Find the selected option
    const selectedOption = this.options.find(opt => {
      if (typeof opt === 'string') {
        return opt === this.value;
      } else if (typeof opt === 'object') {
        return (opt.key || opt.text) === this.value;
      }
      return false;
    });

    if (selectedOption) {
      if (typeof selectedOption === 'string') {
        span.textContent = selectedOption;
      } else if (typeof selectedOption === 'object') {
        span.textContent = selectedOption.text;
        if (selectedOption.color) {
          span.style.color = selectedOption.color; // Apply color to the label in view mode
        }
      }
    } else {
      span.textContent = '(none)';
    }

    return span;
  }

  getValue() {
    return { option: this.value || null }; // Return the current selection or null if no value is selected
  }
}