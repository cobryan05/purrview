import { Field } from './Field.js';
import { fieldFactories } from './Factories.js';

import { createEmojiButton } from '/app-static/js/ui/utils/index.js';

export class ArrayField extends Field {
  constructor({ ...rest }) {
    super(rest);
  }

  static async create({ fieldFactory, value = [], ...rest } = {}) {
    const instance = new ArrayField({ value, ...rest });
    instance._fieldFactory = fieldFactory; // function that returns a new Field instance
    instance._value = Array.isArray(value) ? value : [];
    instance._fields = []; // store wrapped subfields

    for (let idx = 0; idx < instance._value.length; idx++) {
      instance._fields[idx] = await instance._makeSubfield(instance._value[idx]);
    }
    return instance;
  }

  /**
   * Build a subfield whose change handler resolves its own current index at call
   * time (via indexOf) instead of capturing a fixed index. This keeps `_value` in
   * sync even after rows are deleted or reordered.
   */
  async _makeSubfield(val) {
    let subfield;
    subfield = await this._fieldFactory(val, (newValue) => {
      const i = this._fields.indexOf(subfield);
      if (i >= 0) this._value[i] = newValue;
      this._onChange?.(this._value);
    });
    return subfield;
  }

  async renderEdit() {
    const container = document.createElement('div');
    container.classList.add('array-field');

    // Ensure fields array matches value array
    this._fields = this._fields || [];

    // Swap items at `from`/`to` in both the value and field arrays so order is
    // preserved on getValue() and re-render.
    const move = (from, to) => {
      if (to < 0 || to >= this._value.length) return;
      [this._value[from], this._value[to]] = [this._value[to], this._value[from]];
      [this._fields[from], this._fields[to]] = [this._fields[to], this._fields[from]];
      renderItems();
      this._onChange?.(this._value);
    };

    const renderItems = async () => {
      container.innerHTML = '';

      for (let idx = 0; idx < this._value.length; idx++) {
        const row = document.createElement('div');
        row.style.display = 'flex';
        row.style.alignItems = 'center';
        row.style.gap = '0.5em';

        const fieldEdit = await this._fields[idx].renderEdit();
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

        // Reorder controls (right side). Hidden when there's nothing to reorder.
        if (this._value.length > 1) {
          const upBtn = createEmojiButton({
            text: 'Move up', emoji: '🔼',
            onClick: () => move(idx, idx - 1),
          });
          upBtn.disabled = idx === 0;
          const downBtn = createEmojiButton({
            text: 'Move down', emoji: '🔽',
            onClick: () => move(idx, idx + 1),
          });
          downBtn.disabled = idx === this._value.length - 1;
          upBtn.style.marginLeft = 'auto'; // push the arrow pair to the right edge
          row.appendChild(upBtn);
          row.appendChild(downBtn);
        }

        container.appendChild(row);
      }

      const addBtn = createEmojiButton({
        text: 'Add',
        emoji: '➕',
        onClick: async () => {
          this._value.push(null);
          this._fields.push(await this._makeSubfield(null));
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
    return this._fields.map(f => f.getValue());
  }
}
