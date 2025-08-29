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
      optionEl.dataset.value = opt
      if (typeof opt === 'string') {
        optionEl.value = opt
        optionEl.textContent = opt;
      } else if (typeof opt === 'object') {
        const key = opt.key || opt.text; // Use key if provided, otherwise fallback to text
        optionEl.value = opt.key
        optionEl.textContent = opt.text;
        if (opt.color) {
          optionEl.style.color = opt.color; // Apply color to the dropdown option
        }
      }

      // Mark the selected option
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

    // Find the selected option
    const selectedOption = this.options.find(opt => this.isMatchingOption(opt, this.value));

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
    const selectedOption = this.options.find(opt => this.isMatchingOption(opt, this.value));
    return { option: selectedOption || null }; // Return the current selection or null if no value is selected
  }

  /**
 * Checks if an option matches a given value.
 * @param {string|object} option - The option to check (string or object with a `key` property).
 * @param {string|object} value - The value to match (string or object with a `key` property).
 * @returns {boolean} - True if the option matches the value, false otherwise.
 */
  isMatchingOption(option, value) {
    if (typeof option === 'string') {
      return option === value;
    } else if (typeof option === 'object') {
      return option.key === (typeof value === 'object' ? value?.key : value);
    }
    return false;
  }
}
