// fieldFactories.js
import { ArrayField } from './ArrayField.js'
import { BboxInfoField } from './BboxInfoField.js';
import { SourceConfigField } from './SourceConfigField.js';
import { SourceSelectField } from './SourceSelectField.js';
import { ImageProviderSelectField } from './ImageProviderSelectField.js';
import { BooleanCheckboxField } from './BooleanField.js';
import { TextField } from './TextField.js'
import { ModelLabelSelectField } from './ModelLabelSelectField.js';
import { CheckboxField } from './CheckboxField.js';
import { DropDownField } from './DropDownField.js';
import { fetchTags } from '/app-static/js/shared/api/tags.js';
import { fetchLabels } from '/app-static/js/shared/api/labels.js';
// Factory map
export const fieldFactories = {
  async array(fieldName, fieldMeta, values, onChange) {
    const { type, required, description, items, options, ...rest } = fieldMeta;

    return ArrayField.create({
      ...rest,
      description,
      value: values[fieldName] || [],
      onChange: (arr) => {
        onChange?.(arr);
      },
      fieldFactory: (val, onChange) => {
        const itemType = items?.type || 'string';
        const factory = fieldFactories[itemType];

        if (!factory) {
          return new TextField({
            ...items,
            value: val || '',
            placeholder: `Unhandled array item type: ${itemType}`,
            onChange: onChange
          });
        }

        return factory(
          fieldName,
          { ...items },
          { [fieldName]: val },
          onChange
        );
      }
    });
  },

  async bbox_info(fieldName, fieldMeta, values, onChange) {
    const { ...rest } = fieldMeta;

    return new BboxInfoField({
      label: fieldName,
      value: !!values[fieldName],
      ...rest,
      onChange: (val) => {
        onChange?.(!!val);
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
        onChange?.(!!val);
      }
    });
  },

  async label(fieldName, fieldMeta, values, onChange) {
    const { ...rest } = fieldMeta;

    const labelList = await fetchLabels();
    const labelOptions = Array.from(labelList.values()).map(label => ({
      key: label.metadata.uuid,
      text: label.metadata.name,
      color: label.metadata.color || '#cccccc'
    }))

    return await new DropDownField({
      value: values[fieldName],
      onChange: (val) => {
        onChange?.(val);
      },
      options: labelOptions,
      ...rest
    });
  },

  async image_provider_name(fieldName, fieldMeta, values, onChange) {
    const { ...rest } = fieldMeta;

    return await ImageProviderSelectField.create({
      value: values[fieldName],
      onChange: (val) => {
        onChange?.(val);
      },
      ...rest
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
        onChange?.(parseFloat(val));
      }
    });
  },

  async model_label(fieldName, fieldMeta, values, onChange) {
    const { ...rest } = fieldMeta;

    return await ModelLabelSelectField.create({
      onChange: (val) => {
        onChange?.(val);
      },
      value: values[fieldName],
      ...rest
    });
  },

  async source_config(fieldName, fieldMeta, values, onChange) {
    const { ...rest } = fieldMeta;

    return await SourceConfigField.create({
      value: values[fieldName],
      onChange: (val) => {
        onChange?.(val);
      },
      ...rest
    });
  },


  async source_uuid(fieldName, fieldMeta, values, onChange) {
    const { ...rest } = fieldMeta;

    return await SourceSelectField.create({
      value: values[fieldName],
      onChange: (val) => {
        onChange?.(val);
      },
      ...rest
    });
  },

  async string(fieldName, fieldMeta, values, onChange) {
    const { type, required, description, items, options, ...rest } = fieldMeta;

    return new TextField({
      ...rest,
      placeholder: description || '',
      value: values[fieldName] || '',
      onChange: (val) => {
        onChange?.(val);
      }
    });
  },

  async tags(fieldName, fieldMeta, values, onChange) {
    const { ...rest } = fieldMeta;

    const tags = await fetchTags();

    const tagOptions =  Array.from(tags.values()).map(tag => ({
      key: tag.uuid,
      text: tag.name,
      color: tag.color || '#cccccc'
    }));

    return await new CheckboxField({
      value: values[fieldName],
      onChange: (val) => {
        onChange?.(val);
      },
      options: tagOptions,
      ...rest
    });
  },
};