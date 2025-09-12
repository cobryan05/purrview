import { Field } from './Field.js';
import { TextField } from './TextField.js';

export class ActiveTaskField extends Field {
  constructor({ ...rest }) {
    super(rest);
  }

  static async create({ name = '', typename = '', schema = null, ...rest } = {}) {
    const instance = new ActiveTaskField({ ...rest });
    instance._textField = new TextField({ value: name, placeholder: 'Enter new task name' });

    return instance;
  }

  renderEdit() {
    const container = document.createElement('div');
    container.style.display = 'flex';
    container.style.flexDirection = 'column';
    container.style.width = '100%';

    // Row 1: dropdown + text
    const row1 = document.createElement('div');
    row1.style.display = 'flex';
    row1.style.gap = '0.5em';
    row1.appendChild(this._textField.renderEdit());

    container.appendChild(row1);
    return container;
  }

  renderView() {
    const container = document.createElement('div');
    container.style.display = 'flex';
    container.style.alignItems = 'center';
    container.style.gap = '0.5em';

    container.appendChild(this._textField.renderView());

    return container;
  }

  getValue() {
    return {
      ...this._textField.getValue(),
    };
  }
}
