import { createModalShell } from './modalShell.js';

/**
 * Modal asking the user what to do about unsaved annotation changes.
 *
 * @returns {Promise<'save'|'discard'|'cancel'>}
 */
export function showUnsavedChangesModal() {
  return new Promise((resolve) => {
    let result = 'cancel';
    const { dialog, open, close } = createModalShell({ onClose: () => resolve(result) });

    const title = document.createElement('strong');
    title.textContent = 'Unsaved changes';

    const message = document.createElement('div');
    message.textContent = 'You have unsaved annotation changes. What would you like to do?';

    const buttons = document.createElement('div');
    buttons.style.cssText = 'display: flex; gap: 0.5em; justify-content: flex-end; flex-wrap: wrap;';

    /** @param {'save'|'discard'|'cancel'} value */
    const finish = (value) => { result = value; close(); };

    const cancelBtn = document.createElement('button');
    cancelBtn.textContent = 'Cancel';
    cancelBtn.className = 'secondary';
    cancelBtn.onclick = () => finish('cancel');

    const discardBtn = document.createElement('button');
    discardBtn.textContent = 'Discard & continue';
    discardBtn.className = 'contrast';
    discardBtn.onclick = () => finish('discard');

    const saveBtn = document.createElement('button');
    saveBtn.textContent = 'Save & continue';
    saveBtn.onclick = () => finish('save');

    buttons.append(cancelBtn, discardBtn, saveBtn);
    dialog.append(title, message, buttons);
    open();
    saveBtn.focus();
  });
}
