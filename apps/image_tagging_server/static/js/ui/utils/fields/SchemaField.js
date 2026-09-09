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
    // 'tags' is deliberately absent: it renders as a single compact dropdown
    // trigger now (MultiSelectField), not the old full-width checkbox column.
    'array', 'group', 'source_config', 'bbox_info', 'directory',
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
   *
   * 'bool' is the Python-spelled alias of 'boolean' (see the alias block at the
   * foot of Factories.js) and must appear here too — routing it to the checkbox
   * factory while omitting it from this set is exactly how it would come back as
   * the three-line block described above.
   */
  static SELF_LABELLING_TYPES = new Set(['boolean', 'bool', 'bbox_info']);

  /**
   * The hidden prose block for a field's `description`. Blank lines become
   * paragraphs; single newlines are kept (the longer descriptions are written as
   * bullet runs), which is what `white-space: pre-line` on the block is for.
   */
  static _buildDescBody(text) {
    const body = document.createElement('div');
    body.className = 'schema-field__desc';
    body.dataset.testid = 'schema-field-desc';
    body.hidden = true;
    for (const para of String(text).split(/\n\s*\n/)) {
      const p = document.createElement('p');
      p.textContent = para.trim();
      body.appendChild(p);
    }
    return body;
  }

  /** The `ⓘ` button that shows/hides one description block. */
  static _buildInfoToggle(descBody, fieldLabel) {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'schema-field__info';
    btn.dataset.testid = 'schema-field-info';
    btn.textContent = 'i';
    btn.title = 'What does this do?';
    btn.setAttribute('aria-label', `What does “${fieldLabel}” do?`);
    btn.setAttribute('aria-expanded', 'false');
    btn.addEventListener('click', (ev) => {
      ev.preventDefault();
      ev.stopPropagation();
      const open = descBody.hidden;
      descBody.hidden = !open;
      btn.setAttribute('aria-expanded', String(open));
      btn.classList.toggle('is-open', open);
    });
    return btn;
  }

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

      // `description` is the schema's own explanation of the field. It used to be
      // *only* a `title` on the label: invisible until hovered, unreachable on a
      // touch screen, truncated with the label's ellipsis, and — for a boolean,
      // which draws its own label and so gets no label element here — dropped on
      // the floor entirely. The collector form is a dozen such fields, every one
      // of them documented in the schema and none of that visible on screen.
      // So it becomes a toggle: an `ⓘ` beside the label that opens the prose
      // under the control. Closed by default, so no form gets taller until the
      // reader asks a question; a click, not a hover, so it works on a phone.
      // A boolean with no `label` of its own draws its *description* beside the
      // checkbox (see the boolean factory), so offering the same text again under
      // an `ⓘ` would just repeat the line the reader is already looking at.
      const descIsTheControlLabel = selfLabels && !meta.label;
      const descText = meta.description && !descIsTheControlLabel ? String(meta.description).trim() : '';
      const descBody = descText ? SchemaField._buildDescBody(descText) : null;

      if (!selfLabels) {
        const labelRow = document.createElement('div');
        labelRow.className = 'schema-field__label-row';

        const label = document.createElement('label');
        label.className = 'schema-field__label';
        label.textContent = meta.label || fieldName;
        if (descText) label.title = descText;
        labelRow.appendChild(label);

        if (descBody) labelRow.appendChild(SchemaField._buildInfoToggle(descBody, meta.label || fieldName));
        fieldContainer.appendChild(labelRow);

        // The desc block must sit right under the toggle that opens it, not
        // after the field's own content — a full-width field like model_label
        // can render a whole screen (or several) of configured rows, and a
        // description appended after that content would open far below the
        // button that revealed it.
        if (descBody) fieldContainer.appendChild(descBody);
      }

      const node = await fieldRenderCallback(fieldInstance);

      if (selfLabels && descBody) {
        // A self-labelling control owns the line it is on, so the toggle sits
        // beside the control itself rather than above it. Wrapping keeps the
        // checkbox and its `ⓘ` on one row without the control knowing.
        const selfRow = document.createElement('div');
        selfRow.className = 'schema-field__self-row';
        selfRow.appendChild(node);
        selfRow.appendChild(SchemaField._buildInfoToggle(descBody, meta.label || fieldName));
        fieldContainer.appendChild(selfRow);
        fieldContainer.appendChild(descBody);
      } else {
        fieldContainer.appendChild(node);
      }

      // `help` is the discoverable counterpart to `description`'s tooltip: prose
      // that has to be readable without hovering (e.g. "how do I choose this
      // value?"). It is a *disclosure*, closed by default — the calibration help
      // on the dedup fields runs to two paragraphs, and rendering it inline meant
      // scrolling past the same wall of text on every visit to the form. A schema
      // may set `help_open: true` (camelCase accepted too) for the rare case where
      // the prose has to be read before the field can be answered at all.
      // Blank lines in the source string become paragraph breaks.
      if (meta.help) {
        const help = document.createElement('details');
        help.className = 'schema-field__help';
        if (meta.help_open === true || meta.helpOpen === true) help.open = true;

        const summary = document.createElement('summary');
        summary.className = 'schema-field__help-toggle';
        summary.textContent = 'Help';
        help.appendChild(summary);

        const helpBody = document.createElement('div');
        helpBody.className = 'schema-field__help-body';
        for (const para of String(meta.help).split(/\n\s*\n/)) {
          const p = document.createElement('p');
          p.textContent = para.trim();
          helpBody.appendChild(p);
        }
        help.appendChild(helpBody);
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
