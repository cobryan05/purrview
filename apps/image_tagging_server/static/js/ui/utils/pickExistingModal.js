import { createModalShell } from './modalShell.js';

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
    let result = null;
    const { dialog, open, close } = createModalShell({ onClose: () => resolve(result) });
    dialog.style.minWidth = '280px';

    const finish = (value) => { result = value; close(); };

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
    cancelBtn.onclick = () => finish(null);

    const okBtn = document.createElement('button');
    okBtn.textContent = 'Copy';
    okBtn.onclick = () => finish(items.get(select.value) ?? null);

    buttons.append(cancelBtn, okBtn);
    dialog.append(title, select, buttons);
    open();
  });
}
