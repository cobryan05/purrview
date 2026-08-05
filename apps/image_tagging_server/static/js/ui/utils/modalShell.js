import { nextModalZIndex, applyModalCascade, registerEscHandler, unregisterEscHandler } from './modalZIndex.js';

// Elements that can hold focus. Ordered as the DOM presents them, which is the
// order Tab already follows, so the trap doesn't reorder anything.
const FOCUSABLE = [
  'a[href]', 'button:not([disabled])', 'input:not([disabled])',
  'select:not([disabled])', 'textarea:not([disabled])',
  '[tabindex]:not([tabindex="-1"])',
].join(',');

/**
 * Build the shared overlay + dialog scaffold used by app modals. Backdrop click
 * and a programmatic close() both tear the modal down and run onClose exactly once.
 * Styling lives in app.css (.app-modal-overlay / .app-modal-dialog); tweak one-off
 * sizing via the returned `dialog.style`.
 *
 * @param {Object} [opts]
 * @param {() => void} [opts.onClose] - Run when the modal is dismissed (backdrop click or close()).
 * @returns {{ overlay: HTMLDivElement, dialog: HTMLDivElement, open: () => void, close: () => void }}
 */
export function createModalShell({ onClose } = {}) {
  const overlay = document.createElement('div');
  overlay.className = 'app-modal-overlay';
  overlay.dataset.testid = 'modal-shell-overlay';

  const dialog = document.createElement('div');
  dialog.className = 'app-modal-dialog';
  dialog.dataset.testid = 'modal-shell-dialog';
  overlay.appendChild(dialog);

  function _focusable() {
    return [...dialog.querySelectorAll(FOCUSABLE)]
      .filter(el => el.offsetParent !== null || el === document.activeElement);
  }

  /** Keep Tab inside the dialog — without this, tabbing walks the page behind it. */
  function _trapTab(e) {
    if (e.key !== 'Tab') return;
    const items = _focusable();
    if (items.length === 0) return;
    const first = items[0];
    const last = items[items.length - 1];
    if (e.shiftKey && document.activeElement === first) {
      e.preventDefault();
      last.focus();
    } else if (!e.shiftKey && document.activeElement === last) {
      e.preventDefault();
      first.focus();
    }
  }

  let _returnFocusTo = null;

  let _closed = false;
  function close() {
    if (_closed) return;
    _closed = true;
    unregisterEscHandler(close);
    dialog.removeEventListener('keydown', _trapTab);
    if (overlay.parentNode) document.body.removeChild(overlay);
    // Restore focus to whatever opened the modal, so keyboard users aren't
    // dumped back at the top of the document.
    if (_returnFocusTo?.isConnected) _returnFocusTo.focus();
    onClose?.();
  }

  function open() {
    _returnFocusTo = document.activeElement instanceof HTMLElement ? document.activeElement : null;
    overlay.style.zIndex = nextModalZIndex();
    document.body.appendChild(overlay);
    applyModalCascade(dialog);
    registerEscHandler(close);
    dialog.addEventListener('keydown', _trapTab);
    overlay.addEventListener('click', e => { if (e.target === overlay) close(); });
  }

  return { overlay, dialog, open, close };
}
