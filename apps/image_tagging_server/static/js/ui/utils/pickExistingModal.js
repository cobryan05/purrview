/**
 * Shows a modal that lets the user pick one item from a Map of existing items.
 *
 * @param {Object} options
 * @param {Map<string, any>} options.items - Map of key → item
 * @param {function(any): string} options.getLabel - Returns display label for an item
 * @returns {Promise<any|null>} Resolves to the selected item, or null if cancelled
 */
export function pickExistingModal({ items, getLabel }) {
  return new Promise((resolve) => {
    const overlay = document.createElement('div');
    overlay.style.cssText = `
      position: fixed; inset: 0;
      background: rgba(0,0,0,0.5);
      display: flex; align-items: center; justify-content: center;
      z-index: 9999;
    `;

    const dialog = document.createElement('div');
    dialog.style.cssText = `
      background: var(--pico-background-color, #fff);
      border-radius: 8px;
      padding: 1.5em;
      min-width: 280px;
      max-width: 480px;
      display: flex;
      flex-direction: column;
      gap: 1em;
    `;

    const title = document.createElement('strong');
    title.textContent = 'Copy From Existing';

    const select = document.createElement('select');
    select.style.width = '100%';

    for (const [key, item] of items) {
      const opt = document.createElement('option');
      opt.value = key;
      opt.textContent = getLabel(item);
      select.appendChild(opt);
    }

    const buttons = document.createElement('div');
    buttons.style.cssText = 'display: flex; gap: 0.5em; justify-content: flex-end;';

    const cancelBtn = document.createElement('button');
    cancelBtn.textContent = 'Cancel';
    cancelBtn.onclick = () => {
      document.body.removeChild(overlay);
      resolve(null);
    };

    const okBtn = document.createElement('button');
    okBtn.textContent = 'Copy';
    okBtn.onclick = () => {
      const selectedKey = select.value;
      document.body.removeChild(overlay);
      resolve(items.get(selectedKey) ?? null);
    };

    buttons.appendChild(cancelBtn);
    buttons.appendChild(okBtn);
    dialog.appendChild(title);
    dialog.appendChild(select);
    dialog.appendChild(buttons);
    overlay.appendChild(dialog);
    document.body.appendChild(overlay);

    // Close on backdrop click
    overlay.addEventListener('click', (e) => {
      if (e.target === overlay) {
        document.body.removeChild(overlay);
        resolve(null);
      }
    });
  });
}
