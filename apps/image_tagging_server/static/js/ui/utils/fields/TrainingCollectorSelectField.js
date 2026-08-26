import { fetchTrainingCollectors } from '/app-static/js/shared/api/trainingCollectors.js';
import { DropDownField } from './DropDownField.js';
import { Field } from './Field.js';

/** Picks the collector a job runs. Unlike FilterChainSelectField there is no
 *  "none" option: a collect job with no collector has nothing to collect with,
 *  and the task refuses to start rather than falling back to defaults. */
export class TrainingCollectorSelectField extends Field {
  static async create({ ...rest } = {}) {
    const instance = new TrainingCollectorSelectField({ ...rest });
    const collectorsMap = await fetchTrainingCollectors();
    // Disabled collectors are hidden from new picks, but a task already pointing at one
    // keeps showing it (labeled) rather than silently losing its selection.
    const options = [...collectorsMap.values()]
      .filter(c => c.enabled !== false || c.uuid === instance._value)
      .map(c => ({ text: c.enabled === false ? `${c.name} (disabled)` : c.name, key: c.uuid }));
    instance._dropDownField = new DropDownField({
      options,
      value: instance._value,
      onChange: (val) => {
        instance._value = val;
        instance._onChange?.(val);
      },
    });
    return instance;
  }

  async renderEdit() {
    return this._dropDownField.renderEdit();
  }

  async renderView() {
    return this._dropDownField.renderView();
  }

  getValue() {
    return this._dropDownField.getValue();
  }
}
