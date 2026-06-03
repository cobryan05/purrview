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
import { fetchTags, createNewTag } from '/app-static/js/shared/api/tags.js';
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
      key: label.uuid,
      text: label.name,
      color: label.color || '#cccccc'
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

  async select(fieldName, fieldMeta, values, onChange) {
    const { options = [], ...rest } = fieldMeta;
    // Schema options use { value, label }; DropDownField expects { key, text }.
    const dropdownOptions = options.map(opt =>
      typeof opt === 'string' ? opt : { key: opt.value ?? opt.key, text: opt.label ?? opt.text }
    );
    return new DropDownField({
      value: values[fieldName],
      onChange: (val) => onChange?.(val),
      options: dropdownOptions,
      ...rest,
    });
  },

  async tags(fieldName, fieldMeta, values, onChange) {
    const { allow_add_tag, ...rest } = fieldMeta;

    const tags = await fetchTags();

    const tagOptions = Array.from(tags.values()).map(tag => ({
      key: tag.uuid,
      text: (tag.icon ? tag.icon + ' ' : '') + tag.name,
      color: tag.color || '#cccccc',
    }));

    const checkboxField = new CheckboxField({
      value: values[fieldName],
      onChange: (val) => {
        onChange?.(val);
      },
      options: tagOptions,
      ...rest
    });

    if (!allow_add_tag) return checkboxField;

    // Wrap with an inline "add new tag" affordance.
    return {
      getEditMode: () => true,
      getValue: () => checkboxField.getValue(),
      renderView: async () => checkboxField.renderView(),
      renderEdit: async () => {
        const wrapper = document.createElement('div');

        let checkboxEl = await checkboxField.renderEdit();
        wrapper.appendChild(checkboxEl);

        const addBtn = document.createElement('button');
        addBtn.type = 'button';
        addBtn.textContent = '+ New tag';
        addBtn.style.cssText = 'font-size:0.8em; cursor:pointer; padding:0.15em 0.5em; margin-top:0.4em; display:block;';

        const inputRow = document.createElement('div');
        inputRow.style.cssText = 'display:none; align-items:center; gap:0.4em; margin-top:0.3em;';
        const nameInput = document.createElement('input');
        nameInput.type = 'text';
        nameInput.placeholder = 'Tag name…';
        nameInput.style.cssText = 'font-size:0.85em; padding:0.2em 0.4em;';
        const saveBtn = document.createElement('button');
        saveBtn.type = 'button';
        saveBtn.textContent = 'Save';
        saveBtn.style.cssText = 'font-size:0.8em; cursor:pointer;';
        const cancelBtn = document.createElement('button');
        cancelBtn.type = 'button';
        cancelBtn.textContent = 'Cancel';
        cancelBtn.style.cssText = 'font-size:0.8em; cursor:pointer; color:#888;';
        inputRow.append(nameInput, saveBtn, cancelBtn);

        addBtn.addEventListener('click', () => {
          addBtn.style.display = 'none';
          inputRow.style.display = 'flex';
          nameInput.focus();
        });

        cancelBtn.addEventListener('click', () => {
          nameInput.value = '';
          inputRow.style.display = 'none';
          addBtn.style.display = '';
        });

        async function doSave() {
          const name = nameInput.value.trim();
          if (!name) return;
          saveBtn.disabled = true;
          try {
            // createNewTag returns the created tag and already clears the cache.
            const newTag = await createNewTag(name, '#888888');
            if (newTag) {
              const opt = { key: newTag.uuid, text: newTag.name, color: newTag.color || '#cccccc' };
              checkboxField._options.push(opt);
              checkboxField._optionsMap.set(newTag.uuid, opt);
              checkboxField._value = [...(checkboxField._value || []), opt];
              onChange?.(checkboxField.getValue());
              // Re-render the checkbox list in-place.
              const newCheckboxEl = await checkboxField.renderEdit();
              checkboxEl.replaceWith(newCheckboxEl);
              checkboxEl = newCheckboxEl;
            }
          } catch (err) {
            console.error('Failed to create tag:', err);
          } finally {
            saveBtn.disabled = false;
            nameInput.value = '';
            inputRow.style.display = 'none';
            addBtn.style.display = '';
          }
        }

        saveBtn.addEventListener('click', doSave);
        nameInput.addEventListener('keydown', (e) => {
          if (e.key === 'Enter') doSave();
          if (e.key === 'Escape') cancelBtn.click();
        });

        wrapper.appendChild(addBtn);
        wrapper.appendChild(inputRow);
        return wrapper;
      },
    };
  },
};