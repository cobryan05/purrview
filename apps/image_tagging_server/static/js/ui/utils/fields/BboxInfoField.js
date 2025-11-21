import { Logger } from '../logging.js';
import { Field } from './Field.js';
import { LabelDropDownField } from './LabelDropDownField.js';
import { ArrayField } from './ArrayField.js';
import { TextField } from './TextField.js';
import { fetchClassByUuid, fetchClasses, fetchTags } from '/app-static/js/ui/utils/index.js';
import { fieldFactories } from './Factories.js';

// Takes options as an array of strings or objects
// Each object can have 'key', 'text', and 'color' properties

export class BboxInfoField extends Field {
  constructor(params = {}) {
    super(params);
  }

  static async create({ bbox, classes, value = [], ...rest } = {}) {
    const instance = new BboxInfoField({ value, ...rest });
    instance._metadata = bbox.metadata;

    classes = classes ?? await fetchClasses();
    const classItems = Array.from(classes.values()).map(cls => ({
      key: cls.metadata.uuid,
      text: cls.metadata.name,
      color: cls.metadata.color || '#cccccc'
    }));
    const tags = await fetchTags();
    const tagItems = Array.from(tags.values()).map(tag => ({
      key: tag.uuid,
      text: tag.name,
      color: tag.color || '#cccccc'
    }));
    instance._classField = new LabelDropDownField({
      label: 'Class', value: instance._metadata.classUuid, options: classItems,
      onChange: async (newProvider) => { Logger.warn(newProvider); }
    });
    const tagFactory = (val, onChange) => {
      return new LabelDropDownField({
        value: val || '',
        options: tagItems,
        placeholder: val,
        onChange: onChange
      });
    };
    instance._tagArrayField = await ArrayField.create({
      label: 'Tags', value: instance._metadata.tags || [],
      fieldFactory: tagFactory
    });
    instance._value = Array.isArray(value) ? value : [];
    return instance;
  }

  async renderEdit() {
    const span = document.createElement('span');
    span.appendChild(document.createElement('br'));
    // add title text element Bounding Box to the span
    const title = document.createElement('span');
    title.textContent = 'Bounding Box Info';
    span.appendChild(title);

    const classSelect = await this._classField.renderEdit();
    span.appendChild(classSelect);

    const tags = await this._tagArrayField.renderEdit();
    span.appendChild(tags);
    return span;
  }

  async renderView() {
    const span = document.createElement('span');
    const selectedOption = this._options.find(opt => this._isMatchingOption(opt, this._value));

    if (selectedOption) {
      if (typeof selectedOption === 'string') {
        span.textContent = selectedOption;
      } else {
        span.textContent = selectedOption.text;
        if (selectedOption.color) span.style.color = selectedOption.color;
      }
    } else {
      span.textContent = '(none)';
    }

    return span;
  }

  getValue() {
    const selectedOption = this._options.find(opt => this._isMatchingOption(opt, this._value));
    if (typeof selectedOption === 'string') {
      return selectedOption;
    }
    if (selectedOption && typeof selectedOption === 'object') {
      return selectedOption.key ?? selectedOption.text ?? null;
    }
    return null;
  }


  _isMatchingOption(option, value) {
    if (typeof option === 'string') {
      return option === value;
    } else if (typeof option === 'object') {
      return option.key === (typeof value === 'object' ? value?.key : value);
    }
    return false;
  }
}
