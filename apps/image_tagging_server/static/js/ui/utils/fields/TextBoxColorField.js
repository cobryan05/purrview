import { Field } from './Field.js';
import { TextField } from './TextField.js';
import { ColorSwatchField } from './ColorSwatchField.js';

export class TextBoxColorField extends Field {
  constructor({ text = '', color = '#cccccc', uuid = null, textOnLeft = false, onTextChange, onColorChange, ...rest } = {}) {
    super(rest);
    this._textField = new TextField({ value: text, placeholder: this._placeholder, onChange: onTextChange });
    this._colorSwatchField = new ColorSwatchField({ color, onChange: onColorChange });
    this._textOnLeft = textOnLeft
    this._uuid = uuid;
  }

  async renderEdit() {
    const container = document.createElement('div');
    container.style.display = 'flex';
    container.style.flexDirection = 'column'; // ⬅️ stack children vertically
    container.style.alignItems = 'flex-start';
    container.style.gap = '0.2em';

    const row1 = document.createElement('div');
    row1.style.display = 'flex';
    row1.style.gap = '0.2em';
    row1.appendChild(await this._colorSwatchField.renderEdit());
    row1.appendChild(await this._textField.renderEdit());

    const uuidInfo = document.createElement('div');
    uuidInfo.textContent = `UUID: ${this._uuid || '(none)'}`;
    uuidInfo.style.fontSize = '0.75em';
    uuidInfo.style.color = '#888';
    uuidInfo.style.marginLeft = '0.25em';
    uuidInfo.style.userSelect = 'text';
    uuidInfo.style.fontFamily = 'monospace';
    uuidInfo.style.overflowWrap = 'anywhere'; // wrap long UUID if needed

    container.appendChild(row1);
    if (this._uuid) {
      container.appendChild(uuidInfo);
    }
    return container;
  }

  async renderView() {
    const container = document.createElement('div');
    container.style.display = 'flex';
    container.style.alignItems = 'flex-start'; // align all children to top
    container.style.gap = '0.5em';

    const textView = await this._textField.renderView();
    const swatchView = await this._colorSwatchField.renderView();

    if (this._textOnLeft) {
      container.appendChild(textView);
      container.appendChild(swatchView);
    } else {
      container.appendChild(swatchView);
      container.appendChild(textView);
    }

    return container;
  }

  getValue() {
    return {
      text: this._textField.getValue(),
      color: this._colorSwatchField.getValue(),
    };
  }
}