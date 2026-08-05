/**
 * Shared bbox-verification control — V1 of the "unify Review filters" plan.
 *
 * Renders a labeled `<select>` with options Any / Verified / Unverified,
 * mapped to the same `bboxVerification` state vocabulary consumed by
 * `filterPredicate.js` (`'any' | 'verified' | 'unverified'`).
 *
 * This is distinct from the whole-image "Config Ready" label-completeness
 * check (a separate control) — this one filters bboxes by their verification
 * TAG (`VERIFIED_TAG_UUID` / `UNVERIFIED_TAG_UUID`).
 *
 * Deliberately dependency-light (no filter-panel coupling, no CSS import
 * beyond a couple of shared class names already defined in app.css) so it can
 * be dropped into similarity/dissimilarity modals (unit M1) as well as the
 * main-page filter panel.
 *
 * Usage:
 *   const ctl = createBboxVerificationControl({
 *     onChange: (value) => { ... },
 *   });
 *   container.appendChild(ctl.element);
 *   ctl.getValue();       // 'any' | 'verified' | 'unverified'
 *   ctl.setValue('verified');
 */

const OPTIONS = [
  ['any', 'Any'],
  ['verified', 'Verified'],
  ['unverified', 'Unverified'],
];

/**
 * @param {{
 *   value?: 'any'|'verified'|'unverified',
 *   onChange: (value: 'any'|'verified'|'unverified') => void,
 *   label?: string,
 * }} opts
 * @returns {{
 *   element: HTMLElement,
 *   getValue: () => string,
 *   setValue: (value: string, { emit?: boolean }) => void,
 *   destroy: () => void,
 * }}
 */
export function createBboxVerificationControl({ value = 'any', onChange, label = 'Box status' } = {}) {
  const wrap = document.createElement('div');
  wrap.className = 'filter-section bbox-verification-control';

  const labelEl = document.createElement('div');
  labelEl.className = 'filter-section-label';
  labelEl.textContent = label;
  wrap.appendChild(labelEl);

  const select = document.createElement('select');
  select.className = 'filter-verification-select bbox-verification-select';
  for (const [val, text] of OPTIONS) {
    const opt = document.createElement('option');
    opt.value = val;
    opt.textContent = text;
    select.appendChild(opt);
  }
  select.value = value;
  wrap.appendChild(select);

  select.addEventListener('change', () => {
    onChange?.(select.value);
  });

  function getValue() {
    return select.value;
  }

  function setValue(v, { emit = false } = {}) {
    select.value = v;
    if (emit) onChange?.(select.value);
  }

  function destroy() {
    // Plain DOM control, no subscriptions to tear down; kept for interface
    // symmetry with other filter controls (createMultiSelectDropdown, etc.).
  }

  return { element: wrap, getValue, setValue, destroy };
}
