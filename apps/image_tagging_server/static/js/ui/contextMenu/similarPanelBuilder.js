/**
 * Shared helper for building the threshold-picker submenu panel used by
 * bbox, image, and canvas context-menu providers.
 */

/**
 * Build a threshold-picker panel for the similar-search submenu.
 * Calls `onSubmit(threshold)` when the user confirms via button click or Enter.
 *
 * @param {(threshold: number) => void} onSubmit
 * @param {number} [defaultThreshold=5]
 * @returns {HTMLElement}
 */
export function buildSimilarPanel(onSubmit, defaultThreshold = 5) {
  const wrap = document.createElement('div');
  wrap.style.cssText = 'min-width:180px;padding:8px 10px;display:flex;flex-direction:column;gap:6px;';

  const label = document.createElement('label');
  label.style.cssText = 'font-size:0.85em;';
  label.textContent = 'Distance threshold:';

  const input = document.createElement('input');
  input.type = 'number';
  input.min = '1';
  input.max = '64';
  input.value = String(defaultThreshold);
  input.style.cssText = 'width:100%;box-sizing:border-box;font-size:0.85em;';

  const btn = document.createElement('button');
  btn.textContent = 'Search';
  btn.style.cssText = 'font-size:0.85em;padding:0.25em 0.6em;';

  function _submit() {
    const t = Math.max(1, Math.min(64, parseInt(input.value, 10) || defaultThreshold));
    onSubmit(t);
  }

  btn.addEventListener('mousedown', e => { e.preventDefault(); _submit(); });
  input.addEventListener('keydown', e => { if (e.key === 'Enter') { e.preventDefault(); _submit(); } });

  // Auto-focus input after mount
  setTimeout(() => input.focus(), 30);

  wrap.append(label, input, btn);
  return wrap;
}
