// fieldFactories.js
import { ArrayField } from './ArrayField.js'
import { CheckboxField } from './CheckboxField.js';
import { TextField } from './TextField.js'

// Factory map
export const fieldFactories = {
  string(fieldName, fieldMeta, values, onChange) {
    const { type, required, description, items, options, ...rest } = fieldMeta;

    return new TextField({
      ...rest,
      placeholder: description || '',
      value: values[fieldName] || '',
      onChange: (val) => {
        values[fieldName] = val;
        onChange?.(values);
      }
    });
  },

  int(fieldName, fieldMeta, values, onChange) {
    const { type, required, description, items, options, ...rest } = fieldMeta;

    return new TextField({
      ...rest,
      placeholder: description || '',
      type: 'number',
      value: values[fieldName] || '',
      onChange: (val) => {
        values[fieldName] = parseInt(val, 10);
        onChange?.(values);
      }
    });
  },

  boolean(fieldName, fieldMeta, values, onChange) {
    const { type, required, description, items, options, ...rest } = fieldMeta;

    return new CheckboxField({
      ...rest,
      value: values[fieldName] ? [fieldName] : [],
      options: [{ key: fieldName, text: description || fieldName }],
      onChange: (val) => {
        values[fieldName] = !!val.length;
        onChange?.(values);
      }
    });
  },

  select(fieldName, fieldMeta, values, onChange) {
    const { type, required, description, items, options = [], ...rest } = fieldMeta;

    const select = document.createElement('select');
    Object.assign(select, { name: fieldName, ...rest });

    options.forEach(option => {
      const opt = document.createElement('option');
      opt.value = option.value || option;
      opt.textContent = option.label || option;
      select.appendChild(opt);
    });

    select.value = values[fieldName] || '';
    select.addEventListener('change', () => {
      values[fieldName] = select.value;
      onChange?.(values);
    });

    return select;
  },

  array(fieldName, fieldMeta, values, onChange) {
    const { type, required, description, items, options, ...rest } = fieldMeta;

    return new ArrayField({
      ...rest,
      description,
      value: values[fieldName] || [],
      onChange: (arr) => {
        values[fieldName] = arr;
        onChange?.(values);
      },
      fieldFactory: (val) => {
        const itemType = items?.type || 'string';
        const factory = fieldFactories[itemType];

        if (!factory) {
          return new TextField({
            ...items,
            value: val || '',
            placeholder: `Unhandled array item type: ${itemType}`,
            onChange: () => {}
          });
        }

        return factory(
          fieldName,
          { ...items },
          { [fieldName]: val },
          () => {}
        );
      }
    });
  },
};
