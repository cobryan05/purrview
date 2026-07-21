/**
 * Shared pagination utilities used by similarSearch.js and dissimilarSearch.js.
 */

/**
 * Return the start index for a keyset-paginated slice of an ordered array.
 * Each element must have an `imageUuid` property.
 *
 * @param {Array<{imageUuid: string}>} order
 * @param {string|null} afterUuid
 * @returns {number}
 */
export function pageStartIdx(order, afterUuid) {
  return afterUuid !== null ? order.findIndex(e => e.imageUuid === afterUuid) + 1 : 0;
}

/**
 * Build a "Top N" number input control.
 * Returns { element, getValue }.
 * `onChange` is called when the value changes (on blur / Enter).
 */
export function buildCountControl(initial, { min = 10, max = 500, onChange }) {
  const el = document.createElement('div');
  el.className = 'modal-toolbar-control';

  const label = document.createElement('span');
  label.className = 'modal-toolbar-control__label';
  label.textContent = 'Result limit:';

  const input = document.createElement('input');
  input.type = 'number';
  input.min = String(min);
  input.max = String(max);
  input.step = '10';
  input.value = String(initial);
  input.className = 'modal-count-input';
  input.title = `Maximum number of most-distinct results to return (${min}–${max})`;

  function _emit() {
    const v = Math.max(min, Math.min(max, parseInt(input.value, 10) || initial));
    input.value = String(v);
    onChange(v);
  }
  input.addEventListener('change', _emit);
  input.addEventListener('keydown', e => { if (e.key === 'Enter') { e.preventDefault(); _emit(); } });

  el.append(label, input);
  return { element: el, getValue: () => parseInt(input.value, 10) || initial };
}

/**
 * Build a small button for switching between similar and dissimilar modals.
 * `onClick` may be async; errors are silently swallowed (not user-actionable).
 */
export function buildSwitchModeButton(label, onClick) {
  const btn = document.createElement('button');
  btn.type = 'button';
  btn.textContent = label;
  btn.className = 'modal-switch-btn';
  btn.addEventListener('click', () => Promise.resolve(onClick()).catch(() => {}));
  return btn;
}
