import { acquireModalLayer, applyModalCascade, registerEscHandler, unregisterEscHandler } from './modalZIndex.js';

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
  // Focusable so open() can move focus inside even when the dialog holds no
  // focusable content of its own; -1 keeps it out of the Tab order itself.
  dialog.tabIndex = -1;
  overlay.appendChild(dialog);

  let _releaseTrap = null;

  let _layer = null;

  let _closed = false;
  function close() {
    if (_closed) return;
    _closed = true;
    unregisterEscHandler(close);
    _layer?.release();
    _releaseTrap?.();
    if (overlay.parentNode) document.body.removeChild(overlay);
    onClose?.();
  }

  function open() {
    _layer = acquireModalLayer({ dialog: true, overlay });
    overlay.style.zIndex = _layer.zIndex;
    document.body.appendChild(overlay);
    applyModalCascade(dialog, _layer.depth);
    registerEscHandler(close);
    overlay.addEventListener('click', e => { if (e.target === overlay) close(); });
    _releaseTrap = trapModalFocus(dialog);
  }

  return { overlay, dialog, open, close };
}

/**
 * Move focus inside `dialog`, trap Tab within it, and restore focus to whatever
 * was focused before once released. Shared by createModalShell and the handful
 * of modals still built by hand (canvas editor, bulk-action modals, review-grid
 * browse modal) so every modal gets the same keyboard behavior from one place —
 * without this, a modal can open while focus stays on a background element
 * (e.g. a search input), so arrow keys/Tab silently act on the page behind it
 * instead of the modal that's visually on top.
 *
 * Call once `dialog` is attached to the document and open for input. Call the
 * returned release() from the modal's own close/teardown path.
 *
 * @param {HTMLElement} dialog
 * @returns {() => void} release
 */
export function trapModalFocus(dialog) {
  const returnFocusTo = document.activeElement instanceof HTMLElement ? document.activeElement : null;
  // Focusable even with no focusable content of its own (see FOCUSABLE below);
  // -1 keeps it out of the page's own Tab order.
  if (dialog.tabIndex < 0) dialog.tabIndex = -1;

  function focusable() {
    return [...dialog.querySelectorAll(FOCUSABLE)]
      .filter(el => el.offsetParent !== null || el === document.activeElement);
  }

  function onKeydown(e) {
    if (e.key !== 'Tab') return;
    const items = focusable();
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

  dialog.addEventListener('keydown', onKeydown);
  (focusable()[0] ?? dialog).focus();

  let _released = false;
  return function release() {
    if (_released) return;
    _released = true;
    dialog.removeEventListener('keydown', onKeydown);
    // Restore focus to whatever opened the modal, so keyboard users aren't
    // dumped back at the top of the document.
    if (returnFocusTo?.isConnected) returnFocusTo.focus();
  };
}
