import { TextField } from './TextField.js';
import { Field } from './Field.js';

/**
 * Normalizes an incoming value into a `{min, max}` confidence band.
 *
 * Accepts either a legacy bare scalar confidence floor (e.g. `0.25`) or an
 * already-banded `{min, max}` object. Mirrors the backend's
 * `_parse_confidence_band` convention (CollectTrainingImagesTask.py): a bare
 * scalar loads as `{min: scalar, max: 1.0}`. Handles null/undefined/garbage
 * without throwing.
 */
function normalizeBand(value) {
  if (value != null && typeof value === 'object') {
    const min = Number(value.min);
    const max = Number(value.max);
    return {
      min: Number.isFinite(min) ? min : 0,
      max: Number.isFinite(max) ? max : 1,
    };
  }

  const scalar = Number(value);
  return {
    min: Number.isFinite(scalar) ? scalar : 0,
    max: 1,
  };
}

export class LabelNumberField extends Field {
  constructor({ label, min = null, max = null, step = null, color = "#cccc", ...rest }) {
    super(rest);
    this._labelText = label;
    this._labelColor = color;

    const band = normalizeBand(this._value);

    this._minField = new TextField({
      type: 'number',
      value: band.min,
      placeholder: this._placeholder,
      onChange: this._onChange,
      min: min,
      max: max,
      step: step,
    });
    this._maxField = new TextField({
      type: 'number',
      value: band.max,
      placeholder: this._placeholder,
      onChange: this._onChange,
      min: min,
      max: max,
      step: step,
    });
  }

  async _renderFields() {
    const container = document.createElement('div');
    container.style.display = 'flex';
    container.style.alignItems = 'center';
    container.style.gap = '0.35em';

    const minWrap = document.createElement('label');
    minWrap.style.display = 'flex';
    minWrap.style.alignItems = 'center';
    minWrap.style.gap = '0.25em';
    const minTag = document.createElement('span');
    minTag.textContent = 'min';
    minTag.style.fontSize = '0.8em';
    minTag.style.color = '#aaa';
    minWrap.appendChild(minTag);
    const minInput = await this._minField.renderEdit();
    minInput.style.width = '4.5em';
    minWrap.appendChild(minInput);

    const maxWrap = document.createElement('label');
    maxWrap.style.display = 'flex';
    maxWrap.style.alignItems = 'center';
    maxWrap.style.gap = '0.25em';
    const maxTag = document.createElement('span');
    maxTag.textContent = 'max';
    maxTag.style.fontSize = '0.8em';
    maxTag.style.color = '#aaa';
    maxWrap.appendChild(maxTag);
    const maxInput = await this._maxField.renderEdit();
    maxInput.style.width = '4.5em';
    maxWrap.appendChild(maxInput);

    container.appendChild(minWrap);
    container.appendChild(maxWrap);
    return container;
  }

  async renderEdit() {
    const container = document.createElement('div');
    container.style.display = 'flex';
    container.style.alignItems = 'center';
    container.style.gap = '0.5em';

    const classSpan = document.createElement('span');
    classSpan.textContent = this._labelText;
    classSpan.style.fontWeight = 'bold';
    container.appendChild(classSpan);

    container.appendChild(await this._renderFields());
    return container;
  }

  async renderView() {
    const container = document.createElement('div');
    container.style.display = 'flex';
    container.style.alignItems = 'center';
    container.style.gap = '0.5em';

    const labelSpan = document.createElement('span');
    labelSpan.textContent = this._labelText;
    labelSpan.style.fontWeight = 'bold';
    labelSpan.style.color = this._labelColor;
    container.appendChild(labelSpan);
    container.appendChild(await this._renderFields());

    return container;
  }

  getValue() {
    const min = Number(this._minField.getValue());
    const max = Number(this._maxField.getValue());
    return {
      min: Number.isFinite(min) ? min : 0,
      max: Number.isFinite(max) ? max : 1,
    };
  }
}
