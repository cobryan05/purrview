import { Field } from './Field.js';

import { createEmojiButton } from '/app-static/js/ui/utils/index.js';

export class EditableField extends Field {
  constructor({ field, onSave = null, onCancel = null, editMode = false, buttonsLast = false } = {}) {
    super({ editMode });
    this._wrappedField = field;
    this._onSave = onSave;
    this._onCancel = onCancel;
    this._buttonsLast = buttonsLast;
  }

  renderView() {
    const container = document.createElement('div');
    container.style.display = 'flex';
    container.style.alignItems = 'center';
    container.style.gap = '0.5em';

    const editButton = createEmojiButton({
      text: 'Edit', emoji: '✏️', onClick: () => {
        this._editMode = true;
        this._rerender(container);
      }
    });
    const fieldView = this._wrappedField.renderView();

    const elements = [];
    if (this._onSave && this._onCancel) {
      elements.push(editButton);
    }
    elements.push(fieldView);
    if (this._buttonsLast) {
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

    const fieldEdit = this._wrappedField.renderEdit();

    const saveButton = createEmojiButton({
      text: 'Save', emoji: '✅', onClick: () => {
        const newValue = this._wrappedField.getValue();
        this._onSave?.(newValue);
        this._editMode = false;
        this._rerender(container);
      }
    });

    const cancelButton = createEmojiButton({
      text: 'Cancel', emoji: '❌', onClick: () => {
        this._onCancel?.();
        this._editMode = false;
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
    if (this._onSave && this._onCancel) {
      if (this._buttonsLast) {
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
    const newContent = this._editMode ? this.renderEdit() : this.renderView();
    container.replaceWith(newContent);
  }

  getValue() {
    return this._wrappedField.getValue();
  }
}
