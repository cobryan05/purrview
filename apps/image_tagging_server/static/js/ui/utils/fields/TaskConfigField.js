import { DropDownField } from './DropDownField.js';
import { Field } from './Field.js';
import { SchemaField } from './SchemaField.js';
import { TextField } from './TextField.js';
import { fetchTaskConfigs, fetchTaskTypeList, fetchTaskTypeSchemas } from '/app-static/js/shared/api/tasks.js';
import { Logger } from '/app-static/js/ui/utils/index.js';
import { pickExistingModal } from '/app-static/js/ui/utils/pickExistingModal.js';

const FROM_EXISTING_SENTINEL = '__from_existing__';

// Mirrors Task.InternalKeys.PERSISTENT ("_persistent") in apps/tasks_server/tasks/Task.py.
// The backend stuffs this into a task config's `params` dict; there's no dedicated
// field on the update endpoint, so it has to travel inside `params` on edits.
const PERSISTENT_PARAM_KEY = '_persistent';

export class TaskConfigField extends Field {
  constructor({ ...rest }) {
    super(rest);
  }

  static async create({ name = '', typename = null, schema = null, uuid = null, ...rest } = {}) {
    const instance = new TaskConfigField({ ...rest });
    let isInvalid = false;

    // If no schema passed but a typename is provided, fetch the schema now
    if (!schema && typename) {
      try {
        schema = await fetchTaskTypeSchemas([typename]).then(schemas => schemas.get(typename));
      } catch (err) {
        isInvalid = true;
      }
    }

    instance._uuid = uuid;
    instance._isInvalid = isInvalid;
    instance._invalidTypename = isInvalid ? typename : null;
    instance._textField = new TextField({ value: name, placeholder: 'Enter new config name' });
    instance._schemaField = await SchemaField.create({ schema: schema ?? {}, values: instance._value || {} });
    // Default false — this must be explicit opt-in (see PERSISTENT_PARAM_KEY comment above).
    instance._persistent = (instance._value || {})[PERSISTENT_PARAM_KEY] === true;
    const taskTypes = await fetchTaskTypeList();
    const taskTypeOptions = [
      { key: FROM_EXISTING_SENTINEL, text: 'From Existing...' },
      ...(isInvalid && typename ? [{ key: typename, text: `${typename} (unknown type)` }] : []),
      ...taskTypes,
    ];

    instance._dropDownField = new DropDownField({
      options: taskTypeOptions,
      value: typename,
      onChange: async (newTypename) => {
        if (newTypename === FROM_EXISTING_SENTINEL) {
          try {
            const configs = await fetchTaskConfigs();
            const picked = await pickExistingModal({ items: configs, getLabel: c => c.name });
            if (picked) {
              await instance._prefillFrom(picked);
            } else {
              instance._dropDownField.setValue('');
            }
          } catch (err) {
            Logger.error('Failed to copy from existing task config:', err);
          }
          return;
        }
        try {
          const newSchema = await fetchTaskTypeSchemas([newTypename]).then(schemas => schemas.get(newTypename));
          instance._schemaField = await SchemaField.create({
            schema: newSchema,
            values: instance._schemaField.getValue()
          });
          await instance._rerenderSchema();
        } catch (err) {
          Logger.error('Failed to fetch schema:', err);
        }
      }
    });

    return instance;
  }

  async _prefillFrom({ typename, params, name }) {
    this._dropDownField.setValue(typename);
    this._textField.setValue(`Copy of ${name}`);
    this._persistent = (params || {})[PERSISTENT_PARAM_KEY] === true;
    if (this._persistentChk) this._persistentChk.checked = this._persistent;
    try {
      const schema = await fetchTaskTypeSchemas([typename]).then(s => s.get(typename));
      this._schemaField = await SchemaField.create({ schema, values: params });
      await this._rerenderSchema();
    } catch (err) {
      Logger.error('Failed to load schema for copied task config:', err);
    }
  }

  // Helper to re-render the schemaField in the DOM
  async _rerenderSchema() {
    if (!this._schemaContainer) return;
    this._schemaContainer.innerHTML = '';
    this._schemaContainer.appendChild(await this._schemaField.renderEdit());
  }

  async renderEdit() {
    const container = document.createElement('div');
    container.style.display = 'flex';
    container.style.flexDirection = 'column';
    container.style.width = '100%';

    if (this._isInvalid) {
      const warning = document.createElement('div');
      warning.textContent = `⚠ Unknown task type: "${this._invalidTypename}". Delete this config or reassign it to a valid type.`;
      warning.style.color = 'var(--pico-color-red-500, #c0392b)';
      warning.style.fontSize = '0.85em';
      warning.style.marginBottom = '0.25em';
      container.appendChild(warning);
    }

    // Row 1: dropdown + text
    const topRow = document.createElement('div');
    topRow.style.display = 'flex';
    topRow.style.gap = '0.5em';
    topRow.appendChild(await this._dropDownField.renderEdit());
    topRow.appendChild(await this._textField.renderEdit());

    const uuidInfo = document.createElement('div');
    uuidInfo.textContent = `UUID: ${this._uuid || '(none)'}`;
    uuidInfo.style.fontSize = '0.75em';
    uuidInfo.style.color = '#888';
    uuidInfo.style.marginLeft = '0.25em';
    uuidInfo.style.userSelect = 'text';
    uuidInfo.style.fontFamily = 'monospace';
    uuidInfo.style.overflowWrap = 'anywhere'; // wrap long UUID if needed

    const schemaRow = document.createElement('div');
    this._schemaContainer = schemaRow; // remember container for re-render
    await this._rerenderSchema();

    // Persistent (auto-restart-after-reboot) opt-in checkbox.
    const persistentRow = document.createElement('div');
    persistentRow.style.cssText = 'margin:0.5em 0; font-size:0.85em;';
    const persistentLabel = document.createElement('label');
    persistentLabel.style.cssText = 'display:flex; align-items:center; gap:0.4em; cursor:pointer;';
    const persistentChk = document.createElement('input');
    persistentChk.type = 'checkbox';
    persistentChk.checked = this._persistent === true;
    persistentChk.addEventListener('change', () => {
      this._persistent = persistentChk.checked;
    });
    persistentLabel.append(persistentChk, 'Restart automatically after a server restart');
    persistentRow.appendChild(persistentLabel);
    const persistentHelp = document.createElement('div');
    persistentHelp.textContent = 'If checked, this task resumes on its own after the container restarts (e.g. crash, redeploy). Leave unchecked for one-off or transient runs.';
    persistentHelp.style.cssText = 'color:#888; margin-left:1.5em;';
    persistentRow.appendChild(persistentHelp);
    this._persistentChk = persistentChk;

    container.appendChild(topRow);
    if (this._uuid) {
      container.appendChild(uuidInfo);
    }
    container.appendChild(schemaRow);
    container.appendChild(persistentRow);
    return container;
  }

  async renderView() {
    const container = document.createElement('div');
    container.style.display = 'flex';
    container.style.alignItems = 'center';
    container.style.gap = '0.5em';

    if (this._isInvalid) {
      const warning = document.createElement('span');
      warning.textContent = `⚠ Unknown type`;
      warning.style.color = 'var(--pico-color-red-500, #c0392b)';
      warning.style.fontSize = '0.85em';
      container.appendChild(warning);
    }

    container.appendChild(await this._dropDownField.renderView());
    container.appendChild(await this._textField.renderView());

    if (this._persistent) {
      const badge = document.createElement('span');
      badge.textContent = '🔁 persistent';
      badge.title = 'Restarts automatically after a server restart';
      badge.style.cssText = 'font-size:0.75em; background:#e0e0e0; border-radius:3px; padding:1px 5px; color:#444;';
      container.appendChild(badge);
    }

    return container;
  }

  getValue() {
    const persistent = this._persistent === true;
    return {
      name: this._textField.getValue(),
      typename: this._dropDownField.getValue(),
      // `_persistent` travels inside params too — the update-config endpoint has no
      // dedicated `persistent` field and just replaces `params` wholesale, so this is
      // the only way the flag survives an edit. `persistent` (top-level) is what the
      // create-config endpoint reads.
      params: { ...this._schemaField.getValue(), [PERSISTENT_PARAM_KEY]: persistent },
      persistent,
    };
  }
}
