import { createButton, DropDownField, Field } from '/app-static/js/ui/utils/index.js';

export class LabelDropDownField extends Field {
  constructor({ labelText, value, options, placeholder, onChange, onEdit }) {
    super();
    this.labelText = labelText;        // model class name
    this.value = value;                // currently assigned label
    this.options = options;            // all available labels
    this.placeholder = placeholder;
    this.dropdown = new DropDownField({
      value: this.value,
      options: this.options,
      placeholder: this.placeholder,
      onChange: this.onChange
    })

    this.onChange = onChange;
    this.onEdit = onEdit;              // callback when "Edit" button clicked
  }

  renderEdit() {
    const container = document.createElement('div');
    container.style.display = 'flex';
    container.style.alignItems = 'center';
    container.style.gap = '0.5em';

    const classSpan = document.createElement('span');
    classSpan.textContent = this.labelText;
    classSpan.style.fontWeight = 'bold';
    container.appendChild(classSpan);

    container.appendChild(this.dropdown.renderEdit());
    return container;
  }

  renderView() {
    const container = document.createElement('div');
    container.style.display = 'flex';
    container.style.alignItems = 'center';
    container.style.gap = '0.5em';

    // Model class label
    const classSpan = document.createElement('span');
    classSpan.textContent = this.labelText;
    classSpan.style.fontWeight = 'bold';
    container.appendChild(classSpan);

    // Assigned label
    const assignedSpan = document.createElement('span');
    assignedSpan.textContent = this.value?.text || this.value || '(unassigned)';

    if (this.value) {
      let matchingOption = null;

      if (typeof this.value === 'object' && this.value.key) {
        // If value is an object, search for a matching 'key' in options
        matchingOption = this.options.find(option => option.key === this.value.key);
      } else if (typeof this.value === 'string') {
        // If value is a string, search for a matching string in options
        matchingOption = this.options.find(option => option === this.value);
      }

      if (matchingOption) {
        if (matchingOption.color) assignedSpan.style.color = matchingOption.color;
        assignedSpan.style.fontWeight = 'bold';
      }
    }

    container.appendChild(assignedSpan);

    // Edit button
    if (this.onEdit) {
      const editBtn = createButton('emoji-button', 'Edit', '✏️');
      editBtn.onclick = this.onEdit;
      container.appendChild(editBtn);
    }

    return container;
  }

  getValue() {
    return this.dropdown.getValue();
  }
}
