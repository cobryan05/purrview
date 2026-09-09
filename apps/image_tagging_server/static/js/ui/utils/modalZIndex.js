/**
 * Shared overlay z-index manager and ESC-key close stack.
 *
 * acquireModalLayer()      — reserve a stacking slot; returns { zIndex, depth, release }.
 * registerEscHandler(fn)   — push a close function onto the ESC stack.
 * unregisterEscHandler(fn) — remove it (call when the overlay closes).
 *
 * ESC always closes only the topmost (last-registered) overlay — modals,
 * anchored popups and context menus all share this one stack, so their
 * relative order is defined rather than incidental.
 *
 * ⚠️ The counter is **bounded and self-resetting**, clamped to [base, max] from
 * the --layer-* tokens and rewound to base once every acquired layer releases.
 * Unbounded incrementing used to invert `#bottom-bars`' "above all modals (max
 * ~9999)" arithmetic after ~900 opens in one session.
 *
 * ⚠️ Layer lifetime is tracked by the acquire/release pair, **not** the ESC
 * stack — overlays often allocate their z-index and append before registering
 * an ESC handler (see the review canvas modal), so rewinding on an empty ESC
 * stack could hand out a z-index already in use on screen.
 *
 * ⚠️ Backdrop dimming is owned here too, for the same reason: dialogs used to
 * each append their own full-viewport dimmed overlay, so nested dialogs
 * compounded into near-black. Pass the overlay to acquireModalLayer to join
 * the shared backdrop stack — only the topmost dialog dims
 * (`data-backdrop="dim"`), others go `"clear"` and re-dim as the stack unwinds,
 * driven off acquire/release rather than the ESC stack.
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

/** Layers acquired and not yet released, all kinds. Drives the band rewind. */
let _openLayers = 0;
/** Of those, the ones that are dialogs. Drives the cascade offset. */
let _openDialogs = 0;

// ── Background scroll lock ──────────────────────────────────────────────
// trapModalFocus keeps focus inside the dialog, but arrow keys/Space/PageUp-Down
// still fall through to the browser's default viewport-scroll action when nothing
// focused consumes them (a dialog full of buttons never does) — so without this
// lock, the page behind an open dialog scrolled on every arrow-key press. Locked
// on both html and body since which one is document.scrollingElement varies by
// browser/quirks-mode.
let _scrollLockCount = 0;
let _prevHtmlOverflow = null;
let _prevBodyOverflow = null;

function _lockBackgroundScroll() {
  if (_scrollLockCount === 0) {
    _prevHtmlOverflow = document.documentElement.style.overflow;
    _prevBodyOverflow = document.body.style.overflow;
    document.documentElement.style.overflow = 'hidden';
    document.body.style.overflow = 'hidden';
  }
  _scrollLockCount++;
}

function _unlockBackgroundScroll() {
  _scrollLockCount = Math.max(0, _scrollLockCount - 1);
  if (_scrollLockCount === 0) {
    document.documentElement.style.overflow = _prevHtmlOverflow ?? '';
    document.body.style.overflow = _prevBodyOverflow ?? '';
  }
}

// ── Backdrop stack ──────────────────────────────────────────────────────
// Dialog overlays that opted in by passing themselves to acquireModalLayer,
// in open order. Only the last one dims; see the module docstring.
const _backdropStack = [];

function _syncBackdrops() {
  for (let i = 0; i < _backdropStack.length; i++) {
    _backdropStack[i].dataset.backdrop = i === _backdropStack.length - 1 ? 'dim' : 'clear';
  }
}

/**
 * Reserve a stacking slot strictly above every layer currently held.
 *
 * @param {Object}  [opts]
 * @param {boolean} [opts.dialog] - true for centred dialogs, which cascade off
 *   one another and lock background scroll. Popups, menus and flyouts are
 *   anchored to something and must not count toward that depth.
 * @param {HTMLElement} [opts.overlay] - this dialog's full-viewport overlay. Pass
 *   it to join the shared backdrop stack, so only the topmost dialog dims the
 *   page. Optional: an overlay that doesn't pass one keeps whatever dimming its
 *   own CSS gives it, exactly as before.
 * @returns {{ zIndex: number, depth: number, release: () => void }}
 *   `depth` is how many dialogs were already open; pass it to applyModalCascade.
 *   `release` is idempotent — call it from the overlay's close path.
 */
export function acquireModalLayer({ dialog = false, overlay = null } = {}) {
  // Clamp rather than wrap: at the top of the band successive overlays share
  // the ceiling, so DOM order decides. That is a cosmetic tie at absurd
  // nesting depth; escaping the band would put overlays over the bottom bars.
  _zIndex = Math.min(_zIndex + _STEP, LAYERS.modalMax);
  _openLayers++;
  const depth = dialog ? _openDialogs++ : 0;
  if (dialog) _lockBackgroundScroll();
  if (dialog && overlay) {
    _backdropStack.push(overlay);
    _syncBackdrops();
  }

  let _released = false;
  return {
    zIndex: _zIndex,
    depth,
    release() {
      if (_released) return;
      _released = true;
      _openLayers--;
      if (dialog) {
        _openDialogs--;
        _unlockBackgroundScroll();
      }
      if (dialog && overlay) {
        // Splice by identity rather than pop: modals do not always close in
        // open order (a background one can be torn down programmatically).
        const idx = _backdropStack.lastIndexOf(overlay);
        if (idx !== -1) _backdropStack.splice(idx, 1);
        _syncBackdrops();
      }
      // Nothing is open, so the band can start over — this is what keeps the
      // allocation bounded across a long session rather than only within one
      // nesting chain.
      if (_openLayers === 0) _zIndex = LAYERS.modalBase;
    },
  };
}

const _CASCADE_STEP   = 20;
const _CASCADE_MARGIN = 8;

/**
 * Applies a cascade (staircase) offset to a dialog that is already in the DOM,
 * bouncing back toward centre when the dialog would otherwise leave the viewport.
 *
 * Call AFTER appending the overlay to document.body, passing the `depth` from
 * this dialog's own acquireModalLayer({ dialog: true }) handle. Depth must count
 * only dialogs: menus and anchored popups share the ESC stack but are anchored
 * to a trigger, so counting them staircased first-level dialogs off-centre
 * whenever a context-menu item opened one (the item's onSelect is awaited before
 * the menu closes, so the menu is still open at that point).
 */
export function applyModalCascade(dialog, depth) {
  if (!depth) return;
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

/**
 * How many centred dialogs (createModalShell / hand-rolled modals sharing its
 * acquire/release pair) are currently open. Anchored popups and context menus
 * don't count — they're transient and dismiss on outside click, not something
 * a background hotkey should be blocked by.
 *
 * hotkeys.js records this count at attach time and compares against it on
 * every keydown, rather than gating on `> 0` — a dialog like the canvas editor
 * modal must keep its own hotkeys working while only it is open, and stop only
 * once something opens on top of it (e.g. an "unsaved changes?" confirm).
 * Focus can't do this job: a dialog full of buttons never becomes "text entry"
 * focus, so without this, hotkeys like ']' (next image) reached the canvas
 * underneath an open confirm dialog.
 */
export function openDialogCount() {
  return _openDialogs;
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
