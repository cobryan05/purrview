/**
 * Bubbling event a field fires when it changes its value without any native
 * `input`/`change` event happening — button-driven edits like deleting,
 * reordering or appending array rows.
 *
 * Form-level dirty tracking (`watchForEdits` in ui/tasks.js and ui/sources.js)
 * is event delegation on the form root, so an edit that fires no event is
 * invisible to it and the Save button stays disabled.
 */
export const FIELD_EDIT_EVENT = 'field-edit';

/** Announce a non-native edit to whatever is watching the enclosing form. */
export function notifyFieldEdit(el) {
  el?.dispatchEvent(new CustomEvent(FIELD_EDIT_EVENT, { bubbles: true }));
}

export class Field {
  constructor({ value = null, options = [], placeholder = '', onChange = null, editMode = false } = {}) {
    this._value = value;
    this._options = options;
    this._placeholder = placeholder;
    this._onChange = onChange;
    this._editMode = editMode;
  }

  async renderEdit() { throw new Error('renderEdit not implemented'); }
  async renderView() { throw new Error('renderView not implemented'); }
  getValue() { return this._value; }
  getEditMode() { return this._editMode; }
  /**
   * When true, SchemaField spreads this field's getValue() dict into the parent
   * values dict instead of nesting it under the field's own key.
   * Override in subclasses that represent logical groupings (e.g. GroupField).
   */
  flattenIntoParent() { return false; }
}
