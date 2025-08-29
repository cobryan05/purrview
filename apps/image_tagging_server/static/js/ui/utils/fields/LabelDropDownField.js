import { Field } from './Field.js';
import { DropDownField } from './DropDownField.js';
import { createButton } from '../createButton.js';

export class LabelDropDownField extends Field {
  constructor({ labelText, value, options, placeholder, onChange, onEdit }) {
    super();
    this.labelText = labelText;        // model class name
    this.value = value;                // currently assigned label
    this.options = options;            // all available labels
    this.placeholder = placeholder;
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

    const dropdown = new DropDownField({
      value: this.value,
      options: this.options,
      placeholder: this.placeholder,
      onChange: this.onChange
    }).renderEdit();

    container.appendChild(dropdown);
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
    assignedSpan.textContent = this.value || '(unassigned)';
    if (this.value && this.options.includes(this.value)) {
      const labelMeta = _cachedLabels && Array.from(_cachedLabels.values()).find(l => l.name === this.value);
      if (labelMeta?.color) assignedSpan.style.color = labelMeta.color;
      assignedSpan.style.fontWeight = 'bold';
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
    return { option: this.value || null };
  }
}
