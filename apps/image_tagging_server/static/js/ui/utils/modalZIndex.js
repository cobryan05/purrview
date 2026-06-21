/**
 * Shared modal z-index manager and ESC-key close stack.
 *
 * nextModalZIndex()        — returns a z-index strictly above all currently open modals.
 * registerEscHandler(fn)  — push a close function onto the ESC stack.
 * unregisterEscHandler(fn) — remove it (call when the modal closes).
 *
 * ESC always closes only the topmost (last-registered) modal.
 */

let _zIndex = 1000;
const _escStack = [];

export function nextModalZIndex() {
  _zIndex += 10;
  return _zIndex;
}

const _CASCADE_STEP   = 20;
const _CASCADE_MARGIN = 8;

/**
 * Applies a cascade (staircase) offset to a dialog that is already in the DOM,
 * bouncing back toward centre when the dialog would otherwise leave the viewport.
 *
 * Call AFTER appending the overlay to document.body and BEFORE calling
 * registerEscHandler for the new modal (depth is read from _escStack.length).
 */
export function applyModalCascade(dialog) {
  const depth = _escStack.length;
  if (depth === 0) return;
  const rect = dialog.getBoundingClientRect();
  const x = _bounce(depth, _CASCADE_STEP, rect.left  - _CASCADE_MARGIN);
  const y = _bounce(depth, _CASCADE_STEP, rect.top   - _CASCADE_MARGIN);
  if (x || y) dialog.style.transform = `translate(${x}px, ${y}px)`;
}

/**
 * Triangle-wave bounce: 0 → +halfRange → 0 → -halfRange → 0 → …
 * Returns an integer pixel offset for the given depth.
 */
function _bounce(depth, step, halfRange) {
  if (halfRange < step) return 0;
  const hs = Math.floor(halfRange / step); // half-steps to edge
  const t  = depth % (4 * hs);
  if (t <= hs)      return t * step;            // going positive
  if (t <= 3 * hs)  return (2 * hs - t) * step; // going negative
  return (t - 4 * hs) * step;                   // returning to 0
}

export function registerEscHandler(fn) {
  _escStack.push(fn);
}

export function unregisterEscHandler(fn) {
  const idx = _escStack.lastIndexOf(fn);
  if (idx !== -1) _escStack.splice(idx, 1);
}

// One global listener in capture phase so it fires before any element's own keydown handlers.
window.addEventListener('keydown', (e) => {
  if (e.key !== 'Escape') return;
  const top = _escStack[_escStack.length - 1];
  if (top) {
    e.preventDefault();
    e.stopPropagation();
    top();
  }
}, true);
