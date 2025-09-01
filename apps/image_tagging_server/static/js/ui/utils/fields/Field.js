export class Field {
  constructor({ value = null, options = [], placeholder = '', onChange = null, editMode = false } = {}) {
    this._value = value;
    this._options = options;
    this._placeholder = placeholder;
    this._onChange = onChange;
    this._editMode = editMode;
  }

  renderEdit() { throw new Error('renderEdit not implemented'); }
  renderView() { throw new Error('renderView not implemented'); }
  getValue() { return this._value; }
  getEditMode() { return this._editMode; }
}
