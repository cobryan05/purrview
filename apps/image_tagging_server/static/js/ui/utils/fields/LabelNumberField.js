import { TextField } from './TextField.js';
import { Field } from './Field.js';

/**
 * Coerces one bound to a number, falling back when it is absent or unparseable.
 *
 * `Number()` alone is not enough: `Number(null)`, `Number('')` and
 * `Number(false)` are all `0` — finite, so an `isFinite` guard never fires and
 * the fallback is silently skipped. Python's `float()` raises on all three, so
 * the backend takes its own fallback. Blanking the max input has to mean
 * "unbounded", not "max = 0".
 */
function coerceBound(value, fallback) {
  if (value === null || value === undefined || value === '') return fallback;
  const n = Number(value);
  return Number.isFinite(n) ? n : fallback;
}

/**
 * Normalizes an incoming value into a `{min, max}` confidence band.
 *
 * Accepts either a legacy bare scalar confidence floor (e.g. `0.25`) or an
 * already-banded `{min, max}` object. Mirrors the backend's
 * `_parse_confidence_band` convention (CollectTrainingImagesTask.py) exactly —
 * a bare scalar loads as `{min: scalar, max: 1.0}`, bounds are clamped into
 * [0, 1], and an inverted band collapses to `max = min` rather than being
 * swapped. Any divergence here means the band the user sees is not the band the
 * collect task enforces. Handles null/undefined/garbage without throwing.
 */
function normalizeBand(value) {
  let min;
  let max;
  if (value != null && typeof value === 'object') {
    min = coerceBound(value.min, 0);
    max = coerceBound(value.max, 1);
  } else {
    min = coerceBound(value, 0);
    max = 1;
  }

  // Same defensive clamps as `_parse_confidence_band`, in the same order.
  min = Math.min(1, Math.max(0, min));
  max = Math.min(1, Math.max(0, max));
  if (max < min) max = min;
  return { min, max };
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
    // Routed through normalizeBand so what is saved is what the backend will
    // read back — a blanked max is unbounded, not 0, and an inverted or
    // out-of-range band is corrected here rather than only inside the task.
    return normalizeBand({
      min: this._minField.getValue(),
      max: this._maxField.getValue(),
    });
  }
}
