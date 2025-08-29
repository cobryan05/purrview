import { Field } from './Field.js';
import { TextField } from './TextField.js';
import { ColorSwatchField } from './ColorSwatchField.js';

export class TextBoxColorField extends Field {
  constructor({ text = '', color = '#cccccc', placeholder = '', textOnLeft = false, onTextChange, onColorChange } = {}) {
    super();
    this.textField = new TextField({ value: text, placeholder, onChange: onTextChange });
    this.colorSwatchField = new ColorSwatchField({ color, onChange: onColorChange });
    this.textOnLeft = textOnLeft
  }

  renderEdit() {
    const container = document.createElement('div');
    container.style.display = 'flex';
    container.style.alignItems = 'flex-start'; // align all children to top
    container.style.gap = '0.5em';

    container.appendChild(this.colorSwatchField.renderEdit());
    container.appendChild(this.textField.renderEdit());

    return container;
  }

  renderView() {
    const container = document.createElement('div');
    container.style.display = 'flex';
    container.style.alignItems = 'flex-start'; // align all children to top
    container.style.gap = '0.5em';

    if (this.textOnLeft) {
      container.appendChild(this.textField.renderView());
      container.appendChild(this.colorSwatchField.renderView());
    } else {
      container.appendChild(this.colorSwatchField.renderView());
      container.appendChild(this.textField.renderView());
    }

    return container;
  }

  getValue() {
    return {
      ...this.textField.getValue(),
      ...this.colorSwatchField.getValue(),
    };
  }
}