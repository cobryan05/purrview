import { Field } from './Field.js';
import { TextField } from './TextField.js';

export class ActiveTaskField extends Field {
  constructor({ ...rest }) {
    super(rest);
  }

  static async create({ ...rest } = {}) {
    const instance = new ActiveTaskField({ ...rest });
    const { name = '', id, status, message } = instance._value || {};
    instance._name = name || instance._value;
    instance._id = id;
    instance._status = status;
    instance._message = message;

    return instance;
  }

  async renderEdit() {
    const container = document.createElement('div');
    container.innerHTML = "Not Implemented";
    return container;
  }

  async renderView() {
    const container = document.createElement('div');
    container.style.display = 'flex';
    container.style.alignItems = 'center';
    container.style.gap = '0.5em';

    const label = document.createElement('span');
    label.textContent = `${this._id}: ${this._value.config_metadata?.name} [${this._value.config_metadata?.typename}] - ${this._status} ("${this._message}")`;
    container.appendChild(label);

    return container;
  }

  getValue() {
    return {};
  }
}
