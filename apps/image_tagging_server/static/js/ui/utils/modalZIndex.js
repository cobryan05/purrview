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
 * ⚠️ The counter is **bounded and self-resetting**. It used to increment
 * forever, which made `#bottom-bars`' documented "above all modals (max
 * ~9999)" arithmetic rather than an invariant: ~900 opens in one session
 * inverted it. Allocation is now clamped to [base, max] from the --layer-*
 * tokens, and rewinds to base once every acquired layer has been released.
 *
 * ⚠️ Layer lifetime is tracked by the acquire/release pair, **not** by the ESC
 * stack. The two are not the same interval: overlays routinely allocate their
 * z-index, append themselves, and only register an ESC handler after some
 * awaits (see the review canvas modal). Rewinding the band when the ESC stack
 * happened to empty therefore handed the same z-index to an overlay that had to
 * stack above one already on screen.
 *
 * ⚠️ Backdrop dimming is owned here too, for the same reason: every dialog used
 * to append its **own** full-viewport dimmed overlay, so two nested dialogs
 * compounded two translucent black layers into near-black and three were
 * effectively opaque. Pass the overlay element to acquireModalLayer and only the
 * topmost dialog overlay dims (`data-backdrop="dim"`); the ones below it go
 * transparent (`data-backdrop="clear"`) and are re-dimmed as the stack unwinds.
 * It is driven off acquire/release — the authoritative lifetime — not the ESC
 * stack, so closing the top modal restores the one underneath it exactly.
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
// Trapping focus inside a dialog (trapModalFocus) stops the browser from
// scrolling some OTHER focused/scrollable element, but it does not stop
// arrow keys / Space / Page Up-Down from falling through to their *default*
// action of scrolling the viewport itself — that fallback fires whenever the
// key isn't otherwise handled, independent of what has focus, as long as the
// focused element doesn't consume it. A dialog full of buttons (or nothing
// focusable at all) never consumes those keys, so without this the page
// behind an open dialog kept scrolling on every arrow-key press. Locked on
// html AND body — whichever one is document.scrollingElement varies by
// browser/quirks-mode, and locking both is cheap and safe either way.
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
 * How many centred dialogs (createModalShell / the hand-rolled modals that
 * share its acquire/release pair) are currently open. Anchored popups and
 * context menus don't count (see acquireModalLayer's `dialog` param) — those
 * are transient and dismiss on outside click, not the kind of thing a
 * background hotkey should be blocked by.
 *
 * Consumed by hotkeys.js as a per-context baseline, not a plain "is anything
 * open" flag: the canvas editor modal is itself a dialog, so its own hotkeys
 * must keep working while only it is open — they should stop only once
 * something opens *on top* of it (e.g. the "unsaved changes?" confirm shown
 * mid-navigation). hotkeys.js records this count at attach time and compares
 * against it on every keydown, rather than gating on `> 0`.
 *
 * Focus alone can't do this job: a dialog full of buttons never becomes
 * "text entry" focus, so without this a hotkey like ']' (next image) kept
 * reaching the canvas underneath an open confirm dialog — each press re-ran
 * the same guarded navigation and popped another stacked copy of it.
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
