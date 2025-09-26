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
      const val = instance._value[idx];
      const subfield = await instance._fieldFactory(val, (newValue) => {
        instance._value[idx] = newValue;
        instance._onChange?.(instance._value);
      });
      instance._fields[idx] = subfield;
    }
    return instance;
  }

  async renderEdit() {
    const container = document.createElement('div');
    container.classList.add('array-field');

    // Ensure fields array matches value array
    this._fields = this._fields || [];

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
        container.appendChild(row);
      }

      const addBtn = createEmojiButton({
        text: 'Add',
        emoji: '➕',
        onClick: async () => {
          this._value.push(null);
          const idx = this._value.length - 1;
          this._fields.push(await this._fieldFactory(null, (newValue) => {
            this._value[idx] = newValue;
            this._onChange?.(this._value);
          }));
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
