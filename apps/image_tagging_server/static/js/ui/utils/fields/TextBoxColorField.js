import { Field } from './Field.js';
import { TextField } from './TextField.js';
import { ColorSwatchField } from './ColorSwatchField.js';

export class TextBoxColorField extends Field {
  constructor({ text = '', color = '#cccccc', textOnLeft = false, onTextChange, onColorChange, ...rest } = {}) {
    super(rest);
    this._textField = new TextField({ value: text, placeholder: this._placeholder, onChange: onTextChange });
    this._colorSwatchField = new ColorSwatchField({ color, onChange: onColorChange });
    this._textOnLeft = textOnLeft
  }

  async renderEdit() {
    const container = document.createElement('div');
    container.style.display = 'flex';
    container.style.alignItems = 'flex-start'; // align all children to top
    container.style.gap = '0.5em';

    container.appendChild(await this._colorSwatchField.renderEdit());
    container.appendChild(await this._textField.renderEdit());

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
      ...this._textField.getValue(),
      ...this._colorSwatchField.getValue(),
    };
  }
}