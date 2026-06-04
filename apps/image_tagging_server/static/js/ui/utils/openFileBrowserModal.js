import { FileBrowser } from '/app-static/js/ui/fileBrowser.js';
import { createModalShell } from '/app-static/js/ui/utils/modalShell.js';

/**
 * Open a FileBrowser inside a modal dialog and resolve with the selected directory path.
 *
 * @param {Object} [opts]
 * @param {string}  [opts.initialPath='/']      - Directory the browser opens at.
 * @param {boolean} [opts.allowCreateDir=false]  - Show the "New Folder" affordance.
 * @param {boolean} [opts.showDbStatus=false]    - Show DB-presence badges on entries.
 * @param {boolean} [opts.showThumbnails=false]  - Enable grid/thumbnail view toggle.
 * @returns {Promise<string|null>} Resolves to the selected directory path, or null if cancelled.
 */
export function openFileBrowserModal({
  initialPath = '/',
  allowCreateDir = false,
  showDbStatus = false,
  showThumbnails = false,
} = {}) {
  return new Promise(resolve => {
    let _settled = false;
    const finish = val => {
      if (_settled) return;
      _settled = true;
      resolve(val);
    };

    const { dialog, open, close } = createModalShell({ onClose: () => finish(null) });

    // Size the dialog for comfortable browsing
    Object.assign(dialog.style, {
      width: '600px',
      maxWidth: '92vw',
      maxHeight: '80vh',
      padding: '0',
      overflow: 'hidden',
      display: 'flex',
      flexDirection: 'column',
    });

    // Header bar
    const header = document.createElement('div');
    header.style.cssText =
      'display:flex; align-items:center; justify-content:space-between;' +
      ' padding:0.75em 1em; border-bottom:1px solid var(--pico-muted-border-color,#ccc); flex-shrink:0;';

    const title = document.createElement('strong');
    title.textContent = 'Select Directory';

    const closeBtn = document.createElement('button');
    closeBtn.type = 'button';
    closeBtn.textContent = '✕';
    closeBtn.style.cssText = 'background:none; border:none; cursor:pointer; font-size:1.1em; line-height:1; padding:0;';
    closeBtn.addEventListener('click', () => close());

    header.append(title, closeBtn);

    // Scrollable browser area
    const body = document.createElement('div');
    body.style.cssText = 'flex:1; overflow:auto; padding:0.75em 1em;';

    const browser = new FileBrowser({
      container: body,
      initialPath,
      mode: 'dir',
      allowCreateDir,
      showDbStatus,
      showThumbnails,
      onDirSelect: path => {
        finish(path);
        close();
      },
    });

    dialog.append(header, body);
    open();
    browser.init();
  });
}
