import { Field, DropDownField, TextField, SchemaField, getImageProviderList, fetchImageProviderSchema } from '/app-static/js/ui/utils/index.js';

export class SourceConfigField extends Field {
  constructor({ name = '', typename = '', schema = null } = {}) {
    super();

    const imageProviders = getImageProviderList();
    this.textField = new TextField({ value: name, placeholder: 'Enter new source name' });
    this.schemaField = new SchemaField({ schema: schema ?? {} });

    // Initialize DropDownField with onChange
    this.dropDownField = new DropDownField({
      options: imageProviders,
      value: typename,
      onChange: async (newTypename) => {
        try {
          const fetchedSchema = await fetchImageProviderSchema(newTypename);
          this.schemaField = new SchemaField({ schema: fetchedSchema });
          this._rerenderSchema();
        } catch (error) {
          console.error('Failed to fetch schema:', error);
        }
      }
    });

    // If no schema passed but a typename is provided, fetch the schema now
    if (!schema && typename) {
      fetchImageProviderSchema(typename)
        .then(fetchedSchema => {
          this.schemaField = new SchemaField({ schema: fetchedSchema });
          this._rerenderSchema();
        })
        .catch(err => console.error('Failed to fetch initial schema:', err));
    }
  }

  // Helper to re-render the schemaField in the DOM
  _rerenderSchema() {
    if (!this._schemaContainer) return;
    this._schemaContainer.innerHTML = '';
    this._schemaContainer.appendChild(this.schemaField.renderEdit());
  }

  renderEdit() {
    const container = document.createElement('div');
    container.style.display = 'flex';
    container.style.flexDirection = 'column';
    container.style.width = '100%';

    // Row 1: dropdown + text
    const row1 = document.createElement('div');
    row1.style.display = 'flex';
    row1.style.gap = '0.5em';
    row1.appendChild(this.dropDownField.renderEdit());
    row1.appendChild(this.textField.renderEdit());

    // Row 2: schema field
    const row2 = document.createElement('div');
    this._schemaContainer = row2; // remember container for re-render
    row2.appendChild(this.schemaField.renderEdit());

    container.appendChild(row1);
    container.appendChild(row2);
    return container;
  }

  renderView() {
    const container = document.createElement('div');
    container.style.display = 'flex';
    container.style.alignItems = 'center';
    container.style.gap = '0.5em';

    container.appendChild(this.dropDownField.renderView());
    container.appendChild(this.textField.renderView());

    return container;
  }

  getValue() {
    return {
      ...this.textField.getValue(),
      typename: this.dropDownField.getValue().option ?? null,
      schema: this.schemaField.getValue()
    };
  }
}
