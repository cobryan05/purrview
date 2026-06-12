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
