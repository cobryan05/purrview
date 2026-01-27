import { DropDownField } from './DropDownField.js';
import { Field } from './Field.js';
import { fetchImageProviderList } from '/app-static/js/shared/api/sources.js';

export class ImageProviderSelectField extends Field {
  constructor({ ...rest }) {
    super(rest);
  }

  static async create({ onChange, ...rest } = {}) {
    const instance = new ImageProviderSelectField({ ...rest, onChange });

    // providerList is just an array of names
    const providerList = await fetchImageProviderList();

    instance._dropDownField = new DropDownField({
      value: instance._value,
      options: providerList,
      onChange: (val) => {
        instance._value = val;
        onChange?.(val);
      },
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
    return this._dropDownField.getValue() ?? null;
  }
}
