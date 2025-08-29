import { createEmojiButton } from '/app-static/js/ui/utils/index.js';
import { Field } from './Field.js';

export class EditableField extends Field {
  constructor({ field, onSave = null, onCancel = null, editMode = false, buttonsLast = false } = {}) {
    super({ editMode });
    this.wrappedField = field;
    this.onSave = onSave;
    this.onCancel = onCancel;
    this.buttonsLast = buttonsLast;
  }

  renderView() {
    const container = document.createElement('div');
    container.style.display = 'flex';
    container.style.alignItems = 'center';
    container.style.gap = '0.5em';

    const editButton = createEmojiButton({
      text: 'Edit', emoji: '✏️', onClick: () => {
        this.editMode = true;
        this._rerender(container);
      }
    });
    const fieldView = this.wrappedField.renderView();

    const elements = [];
    if (this.onSave && this.onCancel) {
      elements.push(editButton);
    }
    elements.push(fieldView);
    if (this.buttonsLast) {
      elements.reverse();
    }

    elements.forEach(el => container.appendChild(el));
    return container;
  }

  renderEdit() {
    const container = document.createElement('div');
    container.style.display = 'grid';
    container.style.gridTemplateColumns = 'auto 1fr auto'; // left, middle, right
    container.style.gap = '0.5em';
    container.style.alignItems = 'start'; // top-align buttons

    const fieldEdit = this.wrappedField.renderEdit();

    const saveButton = createEmojiButton({
      text: 'Save', emoji: '✅', onClick: () => {
        const newValue = this.wrappedField.getValue();
        this.onSave?.(newValue);
        this.editMode = false;
        this._rerender(container);
      }
    });

    const cancelButton = createEmojiButton({
      text: 'Cancel', emoji: '❌', onClick: () => {
        this.onCancel?.();
        this.editMode = false;
        this._rerender(container);
      }
    });

    // Always create placeholders so the grid stays 3 columns
    const left = document.createElement('div');
    const middle = document.createElement('div');
    const right = document.createElement('div');

    // Put the field in the middle column
    middle.appendChild(fieldEdit);

    // Decide where to put buttons
    if (this.onSave && this.onCancel) {
      if (this.buttonsLast) {
        // Put both buttons to the right of the field
        right.appendChild(saveButton);
        right.appendChild(cancelButton);
      } else {
        // Put buttons on either side of the field
        left.appendChild(saveButton);
        left.appendChild(cancelButton);
      }
    }

    container.appendChild(left);
    container.appendChild(middle);
    container.appendChild(right);

    return container;
  }

  _rerender(container) {
    container.innerHTML = '';
    const newContent = this.editMode ? this.renderEdit() : this.renderView();
    container.replaceWith(newContent);
  }

  getValue() {
    return this.wrappedField.getValue();
  }
}
