export class Field {
  constructor({ value = null, options = [], placeholder = '', onChange = null, editMode = false } = {}) {
    this.value = value;
    this.options = options;
    this.placeholder = placeholder;
    this.onChange = onChange;
    this.editMode = editMode;
  }

  renderEdit() { throw new Error('renderEdit not implemented'); }
  renderView() { throw new Error('renderView not implemented'); }
  getValue() { return this.value; }
  getEditMode() { return this.editMode; }
}
