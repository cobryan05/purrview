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
