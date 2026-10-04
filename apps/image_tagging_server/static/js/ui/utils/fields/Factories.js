// fieldFactories.js
import { ArrayField } from './ArrayField.js'
import { BboxInfoField } from './BboxInfoField.js';
import { DirBrowserField } from './DirBrowserField.js';
import { FilterChainSelectField } from './FilterChainSelectField.js';
import { TrainingCollectorSelectField } from './TrainingCollectorSelectField.js';
import { WatchProfileSelectField } from './WatchProfileSelectField.js';
import { SourceConfigField } from './SourceConfigField.js';
import { SourceSelectField } from './SourceSelectField.js';
import { ImageProviderSelectField } from './ImageProviderSelectField.js';
import { BooleanCheckboxField } from './BooleanField.js';
import { TextField } from './TextField.js'
import { ModelLabelSelectField } from './ModelLabelSelectField.js';
import { MultiSelectField } from './MultiSelectField.js';
import { DropDownField } from './DropDownField.js';
import { fetchTags, createNewTag, refreshTagsIfUnknown } from '/app-static/js/shared/api/tags.js';
import { fetchLabels } from '/app-static/js/shared/api/labels.js';
import { filterVisibleLabels, labelPickerText, filterVisibleTags } from '/app-static/js/shared/activeModel.js';
import { GroupField } from './GroupField.js';
// Factory map
export const fieldFactories = {
  async array(fieldName, fieldMeta, values, onChange) {
    const { type, required, description, items, options, allow_group_insert, ...rest } = fieldMeta;

    return ArrayField.create({
      ...rest,
      description,
      allowGroupInsert: !!allow_group_insert,
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
    // The stored value's own label stays in the options even when dynamic labels are
    // hidden — DropDownField.getValue() returns null for a value it cannot match, and
    // this field's value is persisted straight back into the task config.
    const labelOptions = filterVisibleLabels(labelList.values(), values[fieldName])
      .map(label => ({
        key: label.uuid,
        text: labelPickerText(label),
        color: label.color || '#cccccc',
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
      // `??` (not `||`) — a falsy default of 0 must render as "0", not fall
      // through to '' and later be sent as an empty string.
      value: values[fieldName] ?? '',
      onChange: (val) => {
        onChange?.(parseFloat(val));
      }
    });
  },

  async model_label(fieldName, fieldMeta, values, onChange) {
    // The band's two bounds mean whatever the owning task says they mean, so the
    // column headers and help copy come from the schema. snake_case -> camelCase
    // is done by hand here, the same way `allow_group_insert` is above; omitted
    // keys fall back to ModelLabelSelectField's neutral Min/Max defaults.
    const { name_label, min_label, max_label, band_help, ...rest } = fieldMeta;

    return await ModelLabelSelectField.create({
      onChange: (val) => {
        onChange?.(val);
      },
      value: values[fieldName],
      nameLabel: name_label,
      minLabel: min_label,
      maxLabel: max_label,
      bandHelp: band_help,
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

  async filter_chain_uuid(fieldName, fieldMeta, values, onChange) {
    const { type, required, description, ...rest } = fieldMeta;

    return await FilterChainSelectField.create({
      value: values[fieldName],
      // An optional chain must be un-selectable again; a required one must not be.
      allowNone: !required,
      onChange: (val) => {
        onChange?.(val);
      },
      ...rest
    });
  },

  async training_collector_uuid(fieldName, fieldMeta, values, onChange) {
    const { type, required, description, ...rest } = fieldMeta;

    return await TrainingCollectorSelectField.create({
      value: values[fieldName],
      onChange: (val) => {
        onChange?.(val);
      },
      ...rest
    });
  },

  async watch_profile_uuid(fieldName, fieldMeta, values, onChange) {
    const { type, required, description, ...rest } = fieldMeta;

    return await WatchProfileSelectField.create({
      value: values[fieldName],
      onChange: (val) => {
        onChange?.(val);
      },
      ...rest
    });
  },

  async directory(fieldName, fieldMeta, values, onChange) {
    const { type, required, description, ...rest } = fieldMeta;
    return new DirBrowserField({
      ...rest,
      placeholder: description || 'Select a directory…',
      value: values[fieldName] || '',
      onChange: val => onChange?.(val),
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
    const { type, required, description, label, allow_add_tag, ...rest } = fieldMeta;

    // A task's stored tag uuids can name a tag this session has never fetched —
    // CollectTrainingImagesTask mints its own TASK_GENERATED tag server-side,
    // and renaming it mints another. Without this the option list lacks the
    // tag and MultiSelectField.renderView falls back to printing the raw uuid.
    const tags = await refreshTagsIfUnknown(
      Array.isArray(values[fieldName]) ? values[fieldName] : [],
    ).catch(() => fetchTags());

    const tagOptions = filterVisibleTags(tags.values())
      .map(tag => ({
        key: tag.uuid,
        text: (tag.icon ? tag.icon + ' ' : '') + tag.name,
        color: tag.color || '#cccccc',
      }));

    // `label` is deliberately not forwarded: SchemaField already draws the
    // field's heading above the control, and the dropdown would repeat it
    // inside the trigger button.
    const tagField = new MultiSelectField({
      value: values[fieldName],
      onChange: (val) => {
        onChange?.(val);
      },
      options: tagOptions,
      emptyText: 'No tags defined yet',
      ...rest
    });

    if (!allow_add_tag) return tagField;

    // Wrap with an inline "add new tag" affordance.
    // flattenIntoParent must be present: SchemaField calls it on every field instance.
    return {
      getEditMode: () => true,
      flattenIntoParent: () => false,
      getValue: () => tagField.getValue(),
      renderView: async () => tagField.renderView(),
      renderEdit: async () => {
        const wrapper = document.createElement('div');

        wrapper.appendChild(await tagField.renderEdit());

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
              // addOption appends to the live dropdown and checks it — no
              // re-render, so an open panel stays open and scrolled.
              tagField.addOption(
                { key: newTag.uuid, text: newTag.name, color: newTag.color || '#cccccc' },
                { select: true },
              );
              onChange?.(tagField.getValue());
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

  async group(fieldName, fieldMeta, values, onChange) {
    const { type, label = '', collapsed = false, fields = {}, ...rest } = fieldMeta;
    return GroupField.create({
      label,
      collapsed,
      fields,
      // Pass the full parent values dict so each sub-field can find its own key.
      values,
      onChange,
      ...rest,
    });
  },
};

// 'number' and 'float' are aliases for 'int' — all render a numeric text input.
// ('int' already parses with parseFloat, so fractional values round-trip fine.)
// Without the 'float' entry, min_capture_interval falls through to the
// "Unhandled type" text field in SchemaField._createInputField.
fieldFactories.number = fieldFactories.int;
fieldFactories.float = fieldFactories.int;

// Python-spelled aliases. `params_schema()` is written in Python, so schema authors
// reach for the builtin's name — StreamWatcherTask's DRY_RUN was declared `bool` and
// rendered as a *text box* reading "Unhandled type: bool", which silently coerces to a
// truthy string. `str` had the same mismatch (TrainModelTask, RunClassifierTask); its
// fallback happened to be a text input too, so it only ever looked ugly.
// Accepting both spellings is cheaper than policing the naming across every task.
fieldFactories.bool = fieldFactories.boolean;
fieldFactories.str = fieldFactories.string;