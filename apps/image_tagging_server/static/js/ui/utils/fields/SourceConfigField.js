import { DropDownField } from './DropDownField.js';
import { Field } from './Field.js';
import { SchemaField } from './SchemaField.js';
import { TextField } from './TextField.js';
import { fetchImageProviderList, fetchImageProviderSchema } from '/app-static/js/shared/api/sources.js';
import { Logger } from '/app-static/js/ui/utils/index.js';
export class SourceConfigField extends Field {
  constructor({ ...rest }) {
    super(rest);
  }

  static async create({ ...rest } = {}) {
    const instance = new SourceConfigField({ ...rest });
    const { sourceName = '', provider = '', providerParams = {}, uuid = null } = instance._value || {};

    instance._uuid = uuid;
    instance._textField = new TextField({ value: sourceName, placeholder: 'Enter new source name' });

    const imageProviders = await fetchImageProviderList();
    instance._dropDownField = new DropDownField({
      options: imageProviders,
      value: provider,
      onChange: async (newProvider) => {
        try {
          const fetchedSchema = await fetchImageProviderSchema(newProvider);
          instance._schemaField = await SchemaField.create({ schema: fetchedSchema });
          instance._rerenderSchema();
        } catch (err) {
          Logger.error('Failed to fetch schema:', err);
        }
      }
    });

    let fetchedSchema = {};
    if (provider) {
      try {
        fetchedSchema = await fetchImageProviderSchema(provider);
      } catch (err) {
        Logger.error('Failed to fetch schema:', err);
      }
    }
    instance._schemaField = await SchemaField.create({ schema: fetchedSchema, values: providerParams });

    return instance;
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

    // Row 1: dropdown + text
    const row1 = document.createElement('div');
    row1.style.display = 'flex';
    row1.style.gap = '0.5em';
    row1.appendChild(await this._dropDownField.renderEdit());
    row1.appendChild(await this._textField.renderEdit());

    const uuidInfo = document.createElement('div');
    uuidInfo.textContent = `UUID: ${this._uuid || '(none)'}`;
    uuidInfo.style.fontSize = '0.75em';
    uuidInfo.style.color = '#888';
    uuidInfo.style.marginLeft = '0.25em';
    uuidInfo.style.userSelect = 'text';
    uuidInfo.style.fontFamily = 'monospace';
    uuidInfo.style.overflowWrap = 'anywhere'; // wrap long UUID if needed


    // Row 2: schema field
    const row2 = document.createElement('div');
    this._schemaContainer = row2; // remember container for re-render
    row2.appendChild(await this._schemaField.renderEdit());

    container.appendChild(row1);
    if (this._uuid) {
      container.appendChild(uuidInfo);
    }
    container.appendChild(row2);
    return container;
  }

  async renderView() {
    const container = document.createElement('div');
    container.style.display = 'flex';
    container.style.alignItems = 'center';
    container.style.gap = '0.5em';

    container.appendChild(await this._dropDownField.renderView());
    container.appendChild(await this._textField.renderView());

    return container;
  }

  getValue() {
    return {
      sourceName: this._textField.getValue(),
      provider: this._dropDownField.getValue(),
      providerParams: this._schemaField.getValue()
    };
  }
}
