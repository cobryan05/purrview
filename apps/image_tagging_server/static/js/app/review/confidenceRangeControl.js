/**
 * Shared model-confidence range control for the Review filters.
 *
 * Renders "Confidence: [min %] – [max %] [∅]", where ∅ opts boxes that carry no
 * recorded confidence at all into the match. Maps onto the server-side
 * `bboxConfidence` filter term (apps/helpers/db/types.py) and its client-side
 * mirror in filterPredicate.js.
 *
 * ⚠️ The control talks **percent** (what the card badge and canvas label show);
 * the state it emits is **0–1**, the unit the metadata is stored in. Everything
 * outside this module — filter state, the wire term, the predicates — is 0–1.
 *
 * Built in the same deliberately dependency-light style as
 * bboxVerificationControl.js (no filter-panel coupling, only shared class names
 * that already exist in app.css) so it can be dropped into the main filter panel
 * and the similarity/dissimilarity modals' curated surface alike.
 *
 * Usage:
 *   const ctl = createConfidenceRangeControl({
 *     onChange: ({ min, max, includeMissing }) => { ... },
 *   });
 *   container.appendChild(ctl.element);
 *   ctl.setValue({ min: 0.5, max: null, includeMissing: false });
 */

/**
 * Percent string from an input → a 0–1 number, or null when blank/unparseable.
 * Out-of-range entries clamp rather than reject: a 0–1 filter that silently
 * matched nothing because someone typed 150 would be worse than one that reads
 * it as 100%.
 */
function _pctToUnit(raw) {
  const s = String(raw ?? '').trim();
  if (!s) return null;
  const n = parseFloat(s);
  if (!Number.isFinite(n)) return null;
  return Math.min(1, Math.max(0, n / 100));
}

/** 0–1 number → percent string for an input, or '' for null/undefined. */
function _unitToPct(value) {
  return Number.isFinite(value) ? String(Math.round(value * 100)) : '';
}

/**
 * @param {{
 *   min?: ?number, max?: ?number, includeMissing?: boolean,
 *   onChange: (value: {min: ?number, max: ?number, includeMissing: boolean}) => void,
 *   label?: string,
 * }} opts
 */
export function createConfidenceRangeControl({
  min = null,
  max = null,
  includeMissing = false,
  onChange,
  label = 'Confidence',
} = {}) {
  const wrap = document.createElement('div');
  wrap.className = 'filter-section confidence-range-control';

  const labelEl = document.createElement('div');
  labelEl.className = 'filter-section-label';
  labelEl.textContent = label;
  wrap.appendChild(labelEl);

  const row = document.createElement('div');
  row.className = 'confidence-range-row';

  function _makeInput(placeholder, title) {
    const input = document.createElement('input');
    input.type = 'number';
    input.min = '0';
    input.max = '100';
    input.step = '1';
    input.placeholder = placeholder;
    input.title = title;
    input.className = 'confidence-range-input';
    return input;
  }

  const minInput = _makeInput('0', 'Minimum confidence, in percent (blank = no lower bound)');
  const maxInput = _makeInput('100', 'Maximum confidence, in percent (blank = no upper bound)');
  minInput.value = _unitToPct(min);
  maxInput.value = _unitToPct(max);

  const missingLabel = document.createElement('label');
  missingLabel.className = 'itc-negate-label confidence-range-missing';
  missingLabel.title = 'Also match boxes with no recorded confidence (hand-drawn boxes, older imports)';
  const missingCb = document.createElement('input');
  missingCb.type = 'checkbox';
  missingCb.checked = !!includeMissing;
  missingLabel.append(missingCb, document.createTextNode('∅'));

  row.append(minInput, document.createTextNode('–'), maxInput, missingLabel);
  wrap.appendChild(row);

  function getValue() {
    return {
      min: _pctToUnit(minInput.value),
      max: _pctToUnit(maxInput.value),
      includeMissing: missingCb.checked,
    };
  }

  // `input` rather than `change` would fire on every keystroke of a two-digit
  // number, i.e. a full re-query for the intermediate value — the filename
  // filter debounces for the same reason. A number input's `change` fires on
  // blur and on the spinner, which is the right granularity here.
  const emit = () => onChange?.(getValue());
  for (const el of [minInput, maxInput, missingCb]) el.addEventListener('change', emit);

  function setValue({ min: nextMin, max: nextMax, includeMissing: nextMissing } = {}, { emit: doEmit = false } = {}) {
    if (nextMin !== undefined) minInput.value = _unitToPct(nextMin);
    if (nextMax !== undefined) maxInput.value = _unitToPct(nextMax);
    if (nextMissing !== undefined) missingCb.checked = !!nextMissing;
    if (doEmit) emit();
  }

  function destroy() {
    // Plain DOM control with no subscriptions; kept for interface symmetry with
    // the other filter controls (see createBboxVerificationControl.destroy).
  }

  return { element: wrap, getValue, setValue, destroy };
}
