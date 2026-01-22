import { Field } from './Field.js';

export class CheckboxField extends Field {
  constructor(params = {}) {
    super(params);
    // Generate a key: value map from options
    this._optionsMap = new Map();
    this._options.forEach(opt => {
      if (typeof opt === 'string') {
        this._optionsMap.set(opt, opt);
      } else if (typeof opt === 'object') {
        this._optionsMap.set(opt.key, opt);
      }
    });
  }


  async renderEdit() {
    const container = document.createElement('div');
    this._value = Array.isArray(this._value) ? this._value : [];

    const renderedKeys = new Set();

    this._options.forEach(opt => {
      const label = document.createElement('label');
      const input = document.createElement('input');
      input.type = 'checkbox';

      // normalize option
      let optionKey, optionText, optionColor;
      if (typeof opt === 'string') {
        optionKey = opt;
        optionText = opt;
      } else if (typeof opt === 'object') {
        optionKey = opt.key ?? opt.text;
        optionText = opt.text;
        optionColor = opt.color;
      }

      renderedKeys.add(optionKey);

      input.value = optionKey;
      input.checked = this._value.some(v =>
        typeof v === 'object' ? v.key === optionKey : v === optionKey
      );

      input.addEventListener('change', () => {
        if (input.checked) {
          this._value.push(
            typeof opt === 'string'
              ? optionKey
              : { key: optionKey, text: optionText, color: optionColor }
          );
        } else {
          this._value = this._value.filter(v =>
            typeof v === 'object' ? v.key !== optionKey : v !== optionKey
          );
        }
        this._onChange?.(this._value);
      });

      label.appendChild(input);

      const textSpan = document.createElement('span');
      textSpan.textContent = optionText;
      if (optionColor) textSpan.style.color = optionColor;
      label.appendChild(textSpan);

      container.appendChild(label);
      container.appendChild(document.createElement('br'));
    });

    // ---- render values not present in options ----
    this._value.forEach(v => {
      const key = typeof v === 'object' ? v.key : v;
      if (renderedKeys.has(key)) return;

      const label = document.createElement('label');
      const input = document.createElement('input');
      input.type = 'checkbox';
      input.value = key;
      input.checked = true;

      input.addEventListener('change', () => {
        if (!input.checked) {
          this._value = this._value.filter(val =>
            typeof val === 'object' ? val.key !== key : val !== key
          );
          this._onChange?.(this._value);
        }
      });

      label.appendChild(input);

      const textSpan = document.createElement('span');
      textSpan.textContent =
        typeof v === 'object' ? v.text ?? key : key;
      if (typeof v === 'object' && v.color) {
        textSpan.style.color = v.color;
      }

      label.appendChild(textSpan);
      container.appendChild(label);
      container.appendChild(document.createElement('br'));
    });

    return container;
  }


  async renderView() {
    const span = document.createElement('span');

    if (this._value?.length) {
      span.innerHTML = this._value
        .map(v => { return this._getOptionText(v); }).join(', ');
    } else {
      span.textContent = '(none)';
    }

    return span;
  }

  _getOptionText(optionKey) {
    let retText = null;
    if (optionKey && typeof optionKey === 'object') {
      retText = optionKey.text;
    }
    if (retText == null && typeof optionKey === 'string') {
      retText = this._optionsMap.get(optionKey)?.text;
    }
    return retText ?? optionKey
  }

  getValue() {
    return (this._value || []).map(v => typeof v === 'string' ? v : v.key);
  }
}
