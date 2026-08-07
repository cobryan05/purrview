import { Field } from './Field.js';
import { fieldFactories } from './Factories.js';
import { TextField } from './TextField.js'

export class SchemaField extends Field {
  constructor({ ...rest }) {
    super(rest);
  }

  static async create({ schema = {}, values = {}, ...rest } = {}) {
    const instance = new SchemaField({ ...rest });
    instance._schema = schema; // The schema definition
    instance._init_values = structuredClone(values); // Clone to prevent mutating the caller's object
    instance._fields = {};

    for (const [fieldName, fieldMeta] of Object.entries(schema)) {
      if (fieldName === 'meta') continue;

      // If schema specified a default value then set it here.
      if (fieldMeta.hasOwnProperty('default') && !instance._init_values.hasOwnProperty(fieldName)) {
        instance._init_values[fieldName] = fieldMeta.default;
      }

      const inputField = await instance._createInputField(fieldName, fieldMeta);
      instance._fields[fieldName] = inputField;
    }

    return instance;
  }

  /**
   * Renders the form in edit mode based on the schema.
   * @returns {HTMLElement} The form container.
   */
  async renderEdit() {
    return this._renderFieldsWith((field) => field.renderEdit());
  }

  /**
   * Renders the form in view mode (read-only).
   * @returns {HTMLElement} The form container.
   */
  async renderView() {
    return this._renderFieldsWith((field) => field.renderView());
  }

  /**
   * Field types whose control needs the full width of the form to be usable —
   * multi-row lists, nested forms, checkbox grids, path pickers. Everything else
   * (text boxes, numbers, selects) is a single control that reads fine in a
   * ~240px grid cell, so the form flows them two-or-more across instead of
   * stacking every field down one long column.
   */
  static FULL_WIDTH_TYPES = new Set([
    'array', 'tags', 'group', 'source_config', 'bbox_info', 'directory',
    // model_label renders a grid of per-label min/max confidence bands — it is
    // the widest control in any schema and overflows a grid track visibly.
    'model_label',
  ]);

  /**
   * Types that render their own visible label as part of the control, so the
   * form must not add a second one above it. `boolean` is the case that
   * mattered: its factory passes `label: description || fieldName` into
   * BooleanCheckboxField, which draws that text beside the checkbox — the
   * schema-level label above it was a duplicate, and turned every one-line
   * toggle into a three-line block.
   */
  static SELF_LABELLING_TYPES = new Set(['boolean', 'bbox_info']);

  /**
   * Renders the form children using the callback for each item.
   *
   * Emits semantic classes and lets `.schema-form` in app.css own the layout,
   * rather than the inline flex-column this used to hard-code. That column was
   * the main reason a task or source form scrolled: every field became two full
   * rows (bold label, then a 100%-wide input) no matter how small the control.
   *
   * @returns {HTMLElement} The form container.
   */
  async _renderFieldsWith(fieldRenderCallback) {
    const container = document.createElement('div');
    container.className = 'schema-form';

    const order = this._schema.meta?.order;
    let fieldEntries;

    if (order && Array.isArray(order)) {
      const orderedFields = order
        .filter((field) => field in this._fields)
        .map((field) => [field, this._fields[field]]);

      const remainingFields = Object.entries(this._fields).filter(
        ([key]) => key !== 'meta' && !order.includes(key)
      );

      fieldEntries = [...orderedFields, ...remainingFields];
    } else {
      fieldEntries = Object.entries(this._fields).filter(([key]) => key !== 'meta');
    }

    for (const [fieldName, fieldInstance] of fieldEntries) {
      if (fieldInstance.flattenIntoParent()) {
        // The group renders its own <details><summary> header — no outer wrapper needed.
        const node = await fieldRenderCallback(fieldInstance);
        node.classList.add('schema-field--full');
        container.appendChild(node);
        continue;
      }

      const meta = this._schema[fieldName] ?? {};
      const type = meta.type;
      const isFullWidth = SchemaField.FULL_WIDTH_TYPES.has(type);
      const selfLabels = SchemaField.SELF_LABELLING_TYPES.has(type);

      const fieldContainer = document.createElement('div');
      fieldContainer.className = 'schema-field';
      if (isFullWidth || selfLabels) fieldContainer.classList.add('schema-field--full');
      if (selfLabels) fieldContainer.classList.add('schema-field--self-labelled');

      if (!selfLabels) {
        const label = document.createElement('label');
        label.className = 'schema-field__label';
        label.textContent = meta.label || fieldName;
        // `description` is already consumed as the placeholder for text/number
        // fields, so surface it here as a tooltip rather than another line.
        if (meta.description) label.title = meta.description;
        fieldContainer.appendChild(label);
      }

      const node = await fieldRenderCallback(fieldInstance);
      fieldContainer.appendChild(node);

      // `help` is the visible counterpart to `description`'s tooltip: prose that
      // has to be readable without hovering (e.g. "how do I choose this value?").
      // Blank lines in the source string become paragraph breaks.
      if (meta.help) {
        const help = document.createElement('div');
        help.className = 'schema-field__help';
        for (const para of String(meta.help).split(/\n\s*\n/)) {
          const p = document.createElement('p');
          p.textContent = para.trim();
          help.appendChild(p);
        }
        fieldContainer.appendChild(help);
      }

      container.appendChild(fieldContainer);
    }

    return container;
  }

  /**
   * Creates an input field based on the schema metadata.
   * @param {string} fieldName - The name of the field.
   * @param {object} fieldMeta - Metadata for the field (e.g., type, required).
   * @returns {Field} The input field
   */
  async _createInputField(fieldName, fieldMeta) {
    const factory = fieldFactories[fieldMeta.type];
    let field;

    if (factory) {
      field = await factory(fieldName, fieldMeta, this._init_values, (value) => {
        if (field.flattenIntoParent()) {
          // Groups return a flat dict — merge it into the parent values rather
          // than nesting it under the group's own key.
          Object.assign(this._init_values, value);
        } else {
          this._init_values[fieldName] = value;
        }
        this._onChange?.(this._init_values);
      });
    } else {
      // Fallback generic input
      field = new TextField({
        name: fieldName,
        placeholder: `Unhandled type: ${fieldMeta.type}`,
        value: this._init_values[fieldName],
        onChange: (value) => {
          this._init_values[fieldName] = value;
          this._onChange?.(this._init_values);
        },
      });
    }
    field.name = fieldName;
    return field
  }


  /**
   * Returns the current values of the form.
   * Group fields are spread into the flat result rather than nested under their key.
   * @returns {object} The current values.
   */
  getValue() {
    const values = {};
    for (const [fieldName, fieldInstance] of Object.entries(this._fields)) {
      if (fieldInstance.flattenIntoParent()) {
        Object.assign(values, fieldInstance.getValue());
      } else {
        values[fieldName] = fieldInstance.getValue();
      }
    }
    return values;
  }
}
