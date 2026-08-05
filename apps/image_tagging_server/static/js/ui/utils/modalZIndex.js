/**
 * Shared overlay z-index manager and ESC-key close stack.
 *
 * nextModalZIndex()        — returns a z-index strictly above all currently open overlays.
 * registerEscHandler(fn)  — push a close function onto the ESC stack.
 * unregisterEscHandler(fn) — remove it (call when the overlay closes).
 *
 * ESC always closes only the topmost (last-registered) overlay — modals,
 * anchored popups and context menus all share this one stack, so their
 * relative order is defined rather than incidental.
 *
 * ⚠️ The counter is **bounded and self-resetting**. It used to increment
 * forever, which made `#bottom-bars`' documented "above all modals (max
 * ~9999)" arithmetic rather than an invariant: ~900 opens in one session
 * inverted it. Allocation is now clamped to [base, max] from the --layer-*
 * tokens, and rewinds to base whenever the last overlay closes.
 */

/** Layer scale, read from tokens.css so the order lives in exactly one place. */
function _token(name, fallback) {
  const raw = getComputedStyle(document.documentElement).getPropertyValue(name);
  const n = parseInt(raw, 10);
  return Number.isFinite(n) ? n : fallback;
}

/** Only the layers JS actually allocates within or asserts against live here —
 *  --layer-dropdown and --layer-hover-popover are consumed from CSS alone, so
 *  duplicating their values in JS would be two more constants to keep in sync
 *  for no consumer. */
export const LAYERS = {
  modalBase: _token('--layer-modal-base', 1000),
  modalMax: _token('--layer-modal-max', 1890),
  bottomBars: _token('--layer-bottom-bars', 2000),
};

const _STEP = 10;

let _zIndex = LAYERS.modalBase;
const _escStack = [];

export function nextModalZIndex() {
  // Clamp rather than wrap: at the top of the band successive overlays share
  // the ceiling, so DOM order decides. That is a cosmetic tie at absurd
  // nesting depth; escaping the band would put overlays over the bottom bars.
  _zIndex = Math.min(_zIndex + _STEP, LAYERS.modalMax);
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
  // Nothing is open, so the band can start over — this is what keeps the
  // allocation bounded across a long session rather than only within one
  // nesting chain.
  if (_escStack.length === 0) _zIndex = LAYERS.modalBase;
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
