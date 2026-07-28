/**
 * Builds a colored `.label-chip` for a candidate row's current/proposed label.
 * Shared by runModelOnResults.js's relabel confirmation grid and
 * runClassifierOnResults.js's classifier-match confirmation grid.
 *
 * @param {?string} name
 * @param {?string} uuid
 * @param {Map<string, object>} labels
 * @returns {HTMLSpanElement}
 */
export function labelChip(name, uuid, labels) {
  const chip = document.createElement('span');
  chip.className = 'label-chip relabel-modal-chip';
  chip.textContent = name ?? '(none)';
  const color = (uuid && labels.get(uuid)?.color) || '#888';
  chip.style.cssText = `background:${color}33;border:1px solid ${color};color:${color};`;
  return chip;
}
