import { Field } from './Field.js';
import { LabelDropDownField } from './LabelDropDownField.js';
import { fetchClassByUuid, fetchClasses } from '/app-static/js/ui/utils/index.js';

// Takes options as an array of strings or objects
// Each object can have 'key', 'text', and 'color' properties

export class BboxInfoField extends Field {
  constructor(params = {}) {
    super(params);
    this._metadata = params.bbox.metadata;
    this._classField = new LabelDropDownField({ label: 'Class', value: this._metadata.class });
  }

  async renderEdit() {
    const span = document.createElement('span');
    // add title text element Bounding Box to the span
    const title = document.createElement('span');
    title.textContent = 'Bounding Box Info';
    span.appendChild(title);

    const select = document.createElement('select');
    fetchClassByUuid(this._metadata.classUuid).then(cls => {
      if (!cls) return;

      let optionEl = document.createElement('option');
      optionEl.textContent = cls.metadata.name;
      select.appendChild(optionEl);
    });

    // // Add placeholder if no value is set
    // if (!this._value) {
    //   const placeholderOption = document.createElement('option');
    //   placeholderOption.value = '';
    //   placeholderOption.textContent = this._placeholder || 'Select...';
    //   placeholderOption.disabled = true;
    //   placeholderOption.selected = true;
    //   select.appendChild(placeholderOption);
    // }

    // this._options.forEach(opt => {
    //   const optionEl = document.createElement('option');

    //   if (typeof opt === 'string') {
    //     optionEl.value = opt;
    //     optionEl.textContent = opt;
    //   } else if (typeof opt === 'object') {
    //     optionEl.value = opt.key ?? opt.text;
    //     optionEl.textContent = opt.text;
    //     if (opt.color) optionEl.style.color = opt.color;
    //   }

    //   if (this._isMatchingOption(opt, this._value)) {
    //     optionEl.selected = true;
    //   }

    //   select.appendChild(optionEl);
    // });

    // select.addEventListener('change', () => {
    //   this._value = select.value;
    //   this._onChange?.(this._value);
    // });
    span.appendChild(select);
    return span;
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
    if (typeof selectedOption === 'string') {
      return selectedOption;
    }
    if (selectedOption && typeof selectedOption === 'object') {
      return selectedOption.key ?? selectedOption.text ?? null;
    }
    return null;
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
