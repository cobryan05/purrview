import { Field } from './Field.js';

import { createEmojiButton } from '/app-static/js/ui/utils/index.js';

export class ArrayField extends Field {
  constructor({ fieldFactory, value = [], ...rest } = {}) {
    super({ value, ...rest });
    this._fieldFactory = fieldFactory; // function that returns a new Field instance
    this._value = Array.isArray(value) ? value : [];
    this._fields = []; // store wrapped subfields
  }

  async renderEdit() {
    const container = document.createElement('div');
    container.classList.add('array-field');

    // Ensure fields array matches value array
    this._fields = this._fields || [];

    const renderItems = () => {
      container.innerHTML = '';

      this._value.forEach(async (val, idx) => {
        const row = document.createElement('div');
        row.style.display = 'flex';
        row.style.alignItems = 'center';
        row.style.gap = '0.5em';

        // Reuse existing subfield if present, else create new
        let subfield = this._fields[idx];
        if (!subfield) {
          subfield = await this._fieldFactory(val);
          this._fields[idx] = subfield;
        }

        const fieldEdit = await subfield.renderEdit();

        const deleteBtn = createEmojiButton({
          text: 'Delete',
          emoji: '🗑️',
          onClick: () => {
            this._value.splice(idx, 1);
            this._fields.splice(idx, 1); // also remove corresponding subfield
            renderItems();
            this._onChange?.(this._value);
          }
        });

        row.appendChild(deleteBtn);
        row.appendChild(fieldEdit);
        container.appendChild(row);
      });

      // Add "Add new item" button
      const addBtn = createEmojiButton({
        text: 'Add',
        emoji: '➕',
        onClick: async () => {
          // push null to value, create a corresponding subfield
          this._value.push(null);
          this._fields.push(await this._fieldFactory(null));
          renderItems();
          this._onChange?.(this._value);
        }
      });

      container.appendChild(addBtn);
    };

    renderItems();
    return container;
  }


  async renderView() {
    const container = document.createElement('div');
    if (this._value.length === 0) {
      container.textContent = '(none)';
      return container;
    }

    const list = document.createElement('ul');
    this._value.forEach(async val => {
      const li = document.createElement('li');
      const subfield = await this._fieldFactory(val);
      li.appendChild(await subfield.renderView());
      list.appendChild(li);
    });
    container.appendChild(list);
    return container;
  }

  getValue() {
    // Pull values from subfields
    const values = this._fields.map(f => f.getValue());
    return { items: values };
  }
}
