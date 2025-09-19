import { DropDownField } from './DropDownField.js';
import { Field } from './Field.js';


export class LabelDropDownField extends Field {
  constructor({ labelText, ...rest }) {
    super(rest);
    this._labelText = labelText;        // model class name
    this._dropdown = new DropDownField({
      value: this._value,
      options: this._options,
      placeholder: this._placeholder,
      onChange: this._onChange
    })
  }

  async renderEdit() {
    const container = document.createElement('div');
    container.style.display = 'flex';
    container.style.alignItems = 'center';
    container.style.gap = '0.5em';

    const classSpan = document.createElement('span');
    classSpan.textContent = this._labelText;
    classSpan.style.fontWeight = 'bold';
    container.appendChild(classSpan);

    container.appendChild(await this._dropdown.renderEdit());
    return container;
  }

  async renderView() {
    const container = document.createElement('div');
    container.style.display = 'flex';
    container.style.alignItems = 'center';
    container.style.gap = '0.5em';

    // Model class label
    const classSpan = document.createElement('span');
    classSpan.textContent = this._labelText;
    classSpan.style.fontWeight = 'bold';
    container.appendChild(classSpan);

    // Assigned label
    const assignedSpan = document.createElement('span');
    assignedSpan.textContent = this._value?.text || this._value || '(unassigned)';

    if (this._value) {
      let matchingOption = null;

      if (typeof this._value === 'object' && this._value.key) {
        // If value is an object, search for a matching 'key' in options
        matchingOption = this._options.find(option => option.key === this._value.key);
      } else if (typeof this._value === 'string') {
        // If value is a string, search for a matching string in options
        matchingOption = this._options.find(option => option === this._value);
      }

      if (matchingOption) {
        if (matchingOption.color) assignedSpan.style.color = matchingOption.color;
        assignedSpan.style.fontWeight = 'bold';
      }
    }

    container.appendChild(assignedSpan);

    return container;
  }

  getValue() {
    return this._dropdown.getValue();
  }
}
