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
    instance._dropDownField = new DropDownField({
      options: [...collectorsMap.values()].map(c => ({ text: c.name, key: c.uuid })),
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
