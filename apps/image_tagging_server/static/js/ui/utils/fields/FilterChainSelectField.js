import { DropDownField } from './DropDownField.js';
import { Field } from './Field.js';
import { fetchFilterChains } from '/app-static/js/shared/api/filterChains.js';


export class FilterChainSelectField extends Field {
  constructor({ ...rest }) {
    super(rest);
  }

  static async create({ ...rest } = {}) {
    const instance = new FilterChainSelectField({ ...rest });
    const chainMap = await fetchFilterChains();
    const options = [...chainMap.values()].map(chain => ({
      text: chain.name,
      key: chain.uuid,
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
