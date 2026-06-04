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

  const dialog = document.createElement('div');
  dialog.className = 'app-modal-dialog';
  overlay.appendChild(dialog);

  let _closed = false;
  function close() {
    if (_closed) return;
    _closed = true;
    if (overlay.parentNode) document.body.removeChild(overlay);
    onClose?.();
  }

  function open() {
    document.body.appendChild(overlay);
    overlay.addEventListener('click', e => { if (e.target === overlay) close(); });
  }

  return { overlay, dialog, open, close };
}
