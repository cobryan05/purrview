import { DropDownField } from './DropDownField.js';
import { Field } from './Field.js';
import { fetchSourceList } from '/app-static/js/ui/utils/index.js'

export class SourceSelectField extends Field {
  constructor({ ...rest }) {
    super(rest);
  }

  static async create({ ...rest } = {}) {
    const instance = new SourceSelectField({ ...rest });
    const sourceMap = await fetchSourceList();
    const options = [...sourceMap.values()].map(input => ({
      text: input.name,
      key: input.uuid,
    }));
    instance._dropDownField = new DropDownField({
      options,
      value: instance._value,
      onChange: (val) => {
        instance._value = val;
        instance._onChange?.(val);
      }
    });
    return instance;
  }

  async renderEdit() {
    const container = document.createElement('div');
    container.style.display = 'flex';
    container.style.flexDirection = 'column';
    container.style.width = '100%';

    container.appendChild(await this._dropDownField.renderEdit());
    return container;
  }

  async renderView() {
    const container = document.createElement('div');
    container.style.display = 'flex';
    container.style.alignItems = 'center';
    container.style.gap = '0.5em';

    container.appendChild(await this._dropDownField.renderView());

    return container;
  }

  getValue() {
    return this._dropDownField.getValue();
  }
}
