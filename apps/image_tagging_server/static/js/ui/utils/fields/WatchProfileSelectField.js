import { DropDownField } from './DropDownField.js';
import { Field } from './Field.js';
import { fetchWatchProfiles } from '/app-static/js/shared/api/watchProfiles.js';

/**
 * Picks the Watch Profile a StreamWatcherTask executes.
 *
 * Mirrors FilterChainSelectField: the task's params carry only the profile uuid,
 * because the curated definition is its own entity and the task is merely a run of it.
 */
export class WatchProfileSelectField extends Field {
  constructor({ ...rest }) {
    super(rest);
  }

  static async create({ ...rest } = {}) {
    const instance = new WatchProfileSelectField({ ...rest });
    // fetchWatchProfiles() returns a Map, not an array.
    const profileMap = await fetchWatchProfiles();
    const options = [...profileMap.values()].map(profile => ({
      text: profile.name,
      key: profile.uuid,
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
