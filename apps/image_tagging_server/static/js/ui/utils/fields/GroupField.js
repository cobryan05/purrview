import { Field } from './Field.js';
// SchemaField is imported here — the SchemaField→Factories→GroupField→SchemaField cycle is
// safe because GroupField only uses SchemaField inside an async method body, never at
// module-evaluation time.  ES6 live-bindings ensure SchemaField is fully defined by
// the time create() is first called.
import { SchemaField } from './SchemaField.js';

/**
 * A collapsible group of schema fields rendered with a native <details> element.
 *
 * The group has no key of its own in the output — its sub-field values are merged
 * directly into the parent SchemaField's flat value dict.
 *
 * Schema example (Python side):
 *   "sidecar_sync": {
 *     "type": "group",
 *     "label": "Sidecar Sync",
 *     "collapsed": False,          // optional; default False (starts open)
 *     "fields": {
 *       "sync_jsons": { "type": "boolean", ... },
 *       "sync_mode":  { "type": "select",  ... },
 *     },
 *   }
 */
export class GroupField extends Field {
  constructor({ collapsed = false, ...rest } = {}) {
    super(rest);
    this._collapsed = collapsed;
  }

  /** @override — GroupField values are spread into the parent dict, not nested. */
  flattenIntoParent() { return true; }

  /**
   * @param {string}   label     - Header text shown in the <summary>.
   * @param {boolean}  collapsed - Whether the group starts closed.
   * @param {object}   fields    - Nested schema dict (same format as a full params schema).
   * @param {object}   values    - Full parent values dict; sub-fields pull their own keys.
   * @param {Function} onChange  - Called with the group's flat value dict on any change.
   */
  static async create({
    label = '', description = '', collapsed = false, fields = {}, values = {}, onChange = null, ...rest
  } = {}) {
    const instance = new GroupField({ collapsed, ...rest });
    instance._label = label;
    instance._description = description;
    instance._schemaField = await SchemaField.create({
      schema: fields,
      values,
      onChange,
    });
    return instance;
  }

  /** Returns a flat dict of this group's sub-field values. */
  getValue() {
    return this._schemaField.getValue();
  }

  async renderEdit() {
    return this._renderWith((sf) => sf.renderEdit());
  }

  async renderView() {
    return this._renderWith((sf) => sf.renderView());
  }

  async _renderWith(callback) {
    const details = document.createElement('details');
    if (!this._collapsed) details.open = true;
    details.style.cssText =
      'border:1px solid var(--pico-muted-border-color,#ccc);' +
      'border-radius:4px;padding:0.25em 0.75em 0.75em;';

    const summary = document.createElement('summary');
    summary.textContent = this._label;
    summary.style.cssText = 'cursor:pointer;font-weight:bold;padding:0.3em 0;user-select:none;';
    details.appendChild(summary);

    // Shown unconditionally rather than behind the per-field `ⓘ`: a group's
    // description answers "should I even open this?", which is useless if you
    // have to open it first. It costs one muted line, and only while the group
    // is expanded.
    if (this._description) {
      const desc = document.createElement('p');
      desc.className = 'group-field__desc';
      desc.textContent = this._description;
      details.appendChild(desc);
    }

    const content = await callback(this._schemaField);
    content.style.marginTop = '0.5em';
    details.appendChild(content);
    return details;
  }
}
