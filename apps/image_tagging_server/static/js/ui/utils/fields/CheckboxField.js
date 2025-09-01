import { Field } from './Field.js';

export class CheckboxField extends Field {
  renderEdit() {
    const container = document.createElement('div');
    this._value = Array.isArray(this._value) ? this._value : [];

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

      input.value = optionKey;
      input.checked = this._value.some(v =>
        typeof v === 'object' ? v.key === optionKey : v === optionKey
      );

      input.addEventListener('change', () => {
        if (input.checked) {
          // ensure consistent stored value (object if option was object)
          this._value.push(typeof opt === 'string' ? optionKey : { key: optionKey, text: optionText, color: optionColor });
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

    return container;
  }

  renderView() {
    const span = document.createElement('span');

    if (this._value?.length) {
      span.innerHTML = this._value
        .map(v => {
          if (typeof v === 'string') {
            return v;
          } else {
            const colored = v.color ? `<span style="color:${v.color}">${v.text}</span>` : v.text;
            return colored;
          }
        })
        .join(', ');
    } else {
      span.textContent = '(none)';
    }

    return span;
  }

  getValue() {
    return { options: this._value || [] };
  }
}
