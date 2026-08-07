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

/**
 * One "label — min — max" line of the model's confidence-band table.
 *
 * Renders as a `display: contents` wrapper rather than a self-contained box, so
 * its three cells become cells of the *parent* grid in ModelLabelSelectField.
 * That is what keeps every model's min/max inputs on the same two vertical
 * lines regardless of how long the label names are; a wrapper box would only
 * align each row against itself. The trade-off is that these rows are only
 * meaningful inside `.mlsf-bands` — they are not a general-purpose field.
 */
export class LabelNumberField extends Field {
  constructor({ label, min = null, max = null, step = null, color = '#cccc', ...rest }) {
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

  async _renderRow() {
    const row = document.createElement('div');
    row.className = 'mlsf-band-row';

    const name = document.createElement('span');
    name.className = 'mlsf-band-row__name';
    name.textContent = this._labelText;
    name.title = this._labelText;
    // The label's own colour is the only thing tying a row back to the label
    // list, so it stays on both the view and edit renderings.
    name.style.color = this._labelColor;
    row.appendChild(name);

    // Min before max, in DOM order: the column headers, the saved band and the
    // Playwright coverage all index these two inputs positionally.
    for (const field of [this._minField, this._maxField]) {
      const input = await field.renderEdit();
      input.classList.add('mlsf-band-input');
      row.appendChild(input);
    }

    return row;
  }

  async renderEdit() {
    return this._renderRow();
  }

  async renderView() {
    return this._renderRow();
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
