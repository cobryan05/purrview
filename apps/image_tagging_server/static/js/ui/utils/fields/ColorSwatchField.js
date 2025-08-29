import { Field } from './Field.js';

export class ColorSwatchField extends Field {
  constructor({ color = '#cccccc', onChange } = {}) {
    super();
    this.color = color;
    this.onChange = onChange;
  }

  renderEdit() {
    const container = document.createElement('div');
    container.style.position = 'relative';
    container.style.width = '1em';
    container.style.height = '1em';
    container.style.borderRadius = '3px';
    container.style.backgroundColor = this.color;
    container.style.cursor = 'pointer';

    const colorInput = document.createElement('input');
    colorInput.type = 'color';
    colorInput.value = this.color;
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
      this.color = colorInput.value;
      container.style.backgroundColor = this.color;
      this.onChange?.(this.color);
    });

    container.appendChild(colorInput);
    return container;
  }

  renderView() {
    const container = document.createElement('div');
    container.style.width = '1em';
    container.style.height = '1em';
    container.style.borderRadius = '3px';
    container.style.backgroundColor = this.color;
    return container;
  }

  getValue() {
    return {
      color: this.color
    };
  }
}