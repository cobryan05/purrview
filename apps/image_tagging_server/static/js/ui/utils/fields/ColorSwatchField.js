import { Field } from './Field.js';

export class ColorSwatchField extends Field {
  constructor({ color = '#cccccc', ...rest } = {}) {
    super(rest);
    this._color = color;
  }

  async renderEdit() {
    const container = document.createElement('div');
    container.style.position = 'relative';
    container.style.width = '1em';
    container.style.height = '1em';
    container.style.borderRadius = '3px';
    container.style.backgroundColor = this._color;
    container.style.cursor = 'pointer';

    const colorInput = document.createElement('input');
    colorInput.type = 'color';
    colorInput.value = this._color;
    colorInput.style.cssText = `
      opacity: 0;
      pointer-events: 'auto';
      position: absolute;
      left: 0;
      top: 0;
      width: 100%;
      height: 100%;
      cursor: 'pointer';
    `;
    colorInput.disabled = false;

    colorInput.addEventListener('input', () => {
      this._color = colorInput.value;
      container.style.backgroundColor = this._color;
      this._onChange?.(this._color);
    });

    container.appendChild(colorInput);
    return container;
  }

  async renderView() {
    const container = document.createElement('div');
    container.style.width = '1em';
    container.style.height = '1em';
    container.style.borderRadius = '3px';
    container.style.backgroundColor = this._color;
    return container;
  }

  getValue() {
    return this._color;
  }
}