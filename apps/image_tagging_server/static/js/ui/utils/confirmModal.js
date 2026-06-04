/**
 * Modal asking the user what to do about unsaved annotation changes.
 *
 * @returns {Promise<'save'|'discard'|'cancel'>}
 */
export function showUnsavedChangesModal() {
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
      min-width: 320px;
      max-width: 480px;
      display: flex;
      flex-direction: column;
      gap: 1em;
    `;

    const title = document.createElement('strong');
    title.textContent = 'Unsaved changes';

    const message = document.createElement('div');
    message.textContent = 'You have unsaved annotation changes. What would you like to do?';

    const buttons = document.createElement('div');
    buttons.style.cssText = 'display: flex; gap: 0.5em; justify-content: flex-end; flex-wrap: wrap;';

    /** @param {'save'|'discard'|'cancel'} value */
    const close = (value) => {
      if (overlay.parentNode) document.body.removeChild(overlay);
      resolve(value);
    };

    const cancelBtn = document.createElement('button');
    cancelBtn.textContent = 'Cancel';
    cancelBtn.className = 'secondary';
    cancelBtn.onclick = () => close('cancel');

    const discardBtn = document.createElement('button');
    discardBtn.textContent = 'Discard & continue';
    discardBtn.className = 'contrast';
    discardBtn.onclick = () => close('discard');

    const saveBtn = document.createElement('button');
    saveBtn.textContent = 'Save & continue';
    saveBtn.onclick = () => close('save');

    buttons.append(cancelBtn, discardBtn, saveBtn);
    dialog.append(title, message, buttons);
    overlay.appendChild(dialog);
    document.body.appendChild(overlay);

    overlay.addEventListener('click', (e) => {
      if (e.target === overlay) close('cancel');
    });
    saveBtn.focus();
  });
}
