import { DropDownField } from './DropDownField.js';
import { Field } from './Field.js';
import { fetchFilterChains } from '/app-static/js/shared/api/filterChains.js';


/** Key of the explicit "no chain" option. Empty string so the task params carry
 *  `filter_chain_uuid: ''`, which every consumer already reads as "unset"
 *  (`params.get("filter_chain_uuid") or None` on the task side). */
const NO_CHAIN_KEY = '';

export class FilterChainSelectField extends Field {
  constructor({ ...rest }) {
    super(rest);
  }

  /**
   * `allowNone` adds an explicit "No filter chain" option, and is the only way to
   * clear a chain once one is picked: DropDownField's placeholder is `disabled`
   * and is only rendered while the value is empty, so a field whose options are
   * just the existing chains is a one-way door — selecting a chain permanently
   * removes the ability to select nothing. Gated on the schema's `required` flag
   * rather than always on, because ApplyFilterChainTask has nothing to do without
   * a chain and must not offer the option.
   */
  static async create({ allowNone = false, ...rest } = {}) {
    const instance = new FilterChainSelectField({ ...rest });
    const chainMap = await fetchFilterChains();
    const options = [...chainMap.values()].map(chain => ({
      text: chain.name,
      key: chain.uuid,
    }));
    if (allowNone) {
      options.unshift({ text: 'No filter chain', key: NO_CHAIN_KEY });
    }
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
