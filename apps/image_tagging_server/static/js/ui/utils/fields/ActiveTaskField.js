import { Field } from './Field.js';
import { TextField } from './TextField.js';

export class ActiveTaskField extends Field {
  constructor({ ...rest }) {
    super(rest);
  }

  static async create({ ...rest } = {}) {
    const instance = new ActiveTaskField({ ...rest });
    const { uuid, status, error_message: message } = instance._value.instance || {};
    const { name, typename } = instance._value.config || {};
    instance._name = name || typename || instance._value;
    instance._uuid = uuid;
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
    label.textContent = `${this._id}: ${this._value.config?.name} [${this._value.config?.typename}] - ${this._status} ("${this._message}")`;
    container.appendChild(label);

    return container;
  }

  getValue() {
    return {};
  }
}
