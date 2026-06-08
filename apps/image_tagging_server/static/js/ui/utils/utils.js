// Utility functions

/**
 * Creates a labelled field row for panel forms.
 * @param {string} labelText
 * @param {HTMLElement} input
 * @returns {HTMLElement} div.panel-field-row
 */
export function makeFieldRow(labelText, input) {
  const row = document.createElement('div');
  row.className = 'panel-field-row';
  const lbl = document.createElement('label');
  lbl.textContent = labelText;
  row.appendChild(lbl);
  row.appendChild(input);
  return row;
}

/**
 * Wraps a given DOM node in a styled box with a border and padding.
 * @param {HTMLElement} node - The DOM node to be wrapped.
 * @param {Object} [opts] - Optional style overrides.
 * @param {string} [opts.border] - CSS border (default: "1px solid #cccccc").
 * @param {string} [opts.borderRadius] - CSS border-radius (default: "4px").
 * @param {string} [opts.padding] - CSS padding (default: "0.5em").
 * @param {string} [opts.background] - CSS background (default: "transparent").
 * @param {boolean} [opts.fullWidth] - Whether the box should expand horizontally. **Note: currently does not work.**
 * @param {boolean} [opts.fullHeight] - Whether the box should expand vertically. Defaults to false.
 * @returns {HTMLElement} The wrapped node inside a styled container.
 */
export function renderBoxed(node, opts = {}) {
  const {
    border = '1px solid #cccccc',
    borderRadius = '4px',
    padding = '0.5em',
    background = 'transparent',
    fullWidth = false,
    fullHeight = false,
  } = opts;

  const span = document.createElement('span');
  span.appendChild(document.createElement('br'));

  const box = document.createElement('div');
  box.style.border = border;
  box.style.borderRadius = borderRadius;
  box.style.padding = padding;
  box.style.background = background;

  if (fullWidth) {
    box.style.width = '100%';
  }
  if (fullHeight) {
    box.style.height = '100%';
  }

  box.appendChild(node);
  span.appendChild(box);

  return span;
}

/**
 * Returns the text for a label-filter notice, or '' if hiddenCount is 0.
 * @param {number} hiddenCount
 * @returns {string}
 */
export function labelFilterNoticeText(hiddenCount) {
  if (hiddenCount <= 0) return '';
  return `⚠ ${hiddenCount} label${hiddenCount !== 1 ? 's' : ''} hidden by model filter`;
}

/**
 * Creates a <span class="label-filter-notice"> element for use when the active
 * model filter hides some labels. Returns null when hiddenCount is 0.
 * @param {number} hiddenCount
 * @returns {HTMLElement|null}
 */
export function makeLabelFilterNotice(hiddenCount) {
  const text = labelFilterNoticeText(hiddenCount);
  if (!text) return null;
  const el = document.createElement('span');
  el.className = 'label-filter-notice';
  el.textContent = text;
  return el;
}

/**
 * Generates a UUID (Universally Unique Identifier).
 * Uses the browser's crypto API if available, otherwise falls back to a random implementation.
 * @returns {string} A UUID string.
 */
export function generateUUID() {
  if (window.crypto && typeof window.crypto.randomUUID === 'function') {
    return window.crypto.randomUUID();
  }
  // Fallback: RFC4122 version 4 compliant UUID
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function (c) {
    const r = (Math.random() * 16) | 0;
    const v = c === 'x' ? r : (r & 0x3) | 0x8;
    return v.toString(16);
  });
}
