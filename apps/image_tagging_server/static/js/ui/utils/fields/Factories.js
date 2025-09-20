// fieldFactories.js
import { ArrayField } from './ArrayField.js'
import { SourceSelectField } from './SourceSelectField.js';
import { ImageProviderSelectField } from './ImageProviderSelectField.js';
import { BooleanCheckboxField } from './BooleanField.js';
import { TextField } from './TextField.js'
import { ModelLabelSelectField } from './ModelLabelSelectField.js';

// Factory map
export const fieldFactories = {
  async string(fieldName, fieldMeta, values, onChange) {
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

  async int(fieldName, fieldMeta, values, onChange) {
    const { type, required, description, items, options, ...rest } = fieldMeta;

    return new TextField({
      ...rest,
      placeholder: description || '',
      type: 'number',
      value: values[fieldName] || '',
      onChange: (val) => {
        values[fieldName] = parseFloat(val);
        onChange?.(values);
      }
    });
  },

  async boolean(fieldName, fieldMeta, values, onChange) {
    const { type, required, description, ...rest } = fieldMeta;

    return new BooleanCheckboxField({
      label: description || fieldName,
      value: !!values[fieldName],
      ...rest,
      onChange: (val) => {
        values[fieldName] = !!val;
        onChange?.(values);
      }
    });
  },


  async source_uuid(fieldName, fieldMeta, values, onChange) {
    const { ...rest } = fieldMeta;

    return await SourceSelectField.create({
      value: values[fieldName],
      onChange: (val) => {
        values[fieldName] = val;
        onChange?.(values);
      },
      ...rest
    });
  },

  async image_provider_name(fieldName, fieldMeta, values, onChange) {
    const { ...rest } = fieldMeta;

    return await ImageProviderSelectField.create({
      value: values[fieldName],
      onChange: (val) => {
        values[fieldName] = val;
        onChange?.(values);
      },
      ...rest
    });
  },

  async model_label(fieldName, fieldMeta, values, onChange) {
    const { ...rest } = fieldMeta;

    return await ModelLabelSelectField.create({
      onChange: (val) => {
        values[fieldName] = !!val;
        onChange?.(values);
      },
      ...rest
    });
  },

  async array(fieldName, fieldMeta, values, onChange) {
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
            onChange: () => { }
          });
        }

        return factory(
          fieldName,
          { ...items },
          { [fieldName]: val },
          () => { }
        );
      }
    });
  },
};
