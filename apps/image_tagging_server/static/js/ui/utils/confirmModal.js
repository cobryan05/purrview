import { createModalShell } from './modalShell.js';

/**
 * Internal builder shared by all exported modal functions.
 *
 * @param {Object} opts
 * @param {string} opts.title
 * @param {string} opts.message
 * @param {Array<{text: string, value: any, className?: string}>} opts.buttons
 * @param {number} [opts.defaultFocusIndex=0]  — which button receives focus after open
 * @param {any}    [opts.defaultCloseValue]     — resolved value when modal is dismissed without a button
 * @returns {Promise<any>}
 */
function _buildModal({ title, message, buttons, defaultFocusIndex = 0, defaultCloseValue = null }) {
  return new Promise((resolve) => {
    let resolved = false;
    const finish = (value) => { resolved = true; close(); resolve(value); };
    const { dialog, open, close } = createModalShell({
      onClose: () => { if (!resolved) resolve(defaultCloseValue); },
    });

    const titleEl = document.createElement('strong');
    titleEl.textContent = title;

    const msgEl = document.createElement('div');
    msgEl.style.cssText = 'font-size: 0.9em;';
    msgEl.textContent = message;

    const btnRow = document.createElement('div');
    btnRow.style.cssText = 'display: flex; gap: 0.5em; justify-content: flex-end; flex-wrap: wrap;';

    const btnEls = buttons.map(({ text, value, className }) => {
      const btn = document.createElement('button');
      btn.textContent = text;
      if (className) btn.className = className;
      btn.onclick = () => finish(value);
      btnRow.appendChild(btn);
      return btn;
    });

    dialog.append(titleEl, msgEl, btnRow);
    open();
    btnEls[defaultFocusIndex]?.focus();
  });
}

/**
 * Generic single-confirm modal.
 *
 * @param {Object} opts
 * @param {string} opts.title
 * @param {string} opts.message
 * @param {string} [opts.confirmText='Confirm']
 * @param {string} [opts.cancelText='Cancel']
 * @param {boolean} [opts.danger=false] — styles confirm button as destructive (contrast)
 * @returns {Promise<boolean>} true = confirmed, false = cancelled/dismissed
 */
export function showConfirmModal({
  title,
  message,
  confirmText = 'Confirm',
  cancelText = 'Cancel',
  danger = false,
} = {}) {
  return _buildModal({
    title,
    message,
    buttons: [
      { text: cancelText, value: false, className: 'secondary' },
      { text: confirmText, value: true, className: danger ? 'contrast' : undefined },
    ],
    defaultFocusIndex: 0,   // default to safe 'Cancel' action
    defaultCloseValue: false,
  });
}

/**
 * Modal asking the user what to do about unsaved changes.
 *
 * @param {Object} [opts]
 * @param {string} [opts.message]      — body text (default: annotation-specific wording)
 * @param {string} [opts.discardText]  — label for the discard button
 * @returns {Promise<'save'|'discard'|'cancel'>}
 */
export function showUnsavedChangesModal({
  message = 'You have unsaved annotation changes. What would you like to do?',
  discardText = 'Discard & continue',
} = {}) {
  return _buildModal({
    title: 'Unsaved changes',
    message,
    buttons: [
      { text: 'Cancel', value: 'cancel', className: 'secondary' },
      { text: discardText, value: 'discard', className: 'contrast' },
      { text: 'Save & continue', value: 'save' },
    ],
    defaultFocusIndex: 2,
    defaultCloseValue: 'cancel',
  });
}

/**
 * Show a conflict dialog when external bbox changes overlap with unsaved canvas edits.
 * Returns 'reload' (discard canvas changes, fetch fresh) or 'keep' (ignore external changes).
 * @param {{ conflictCount?: number }} [opts]
 * @returns {Promise<'reload'|'keep'>}
 */
export function showExternalChangesModal({ conflictCount = 0 } = {}) {
  const noun = conflictCount === 1 ? 'bbox' : 'bboxes';
  const verb = conflictCount === 1 ? 'was' : 'were';
  return _buildModal({
    title: 'External changes detected',
    message: `${conflictCount} ${noun} you are editing ${verb} modified by another panel. What would you like to do?`,
    buttons: [
      { text: 'Keep my changes', value: 'keep', className: 'secondary' },
      { text: 'Reload image', value: 'reload', className: 'contrast' },
    ],
    defaultFocusIndex: 0,
    defaultCloseValue: 'keep',
  });
}

/**
 * Variant of showUnsavedChangesModal for config panels (datasets, model configs, etc.).
 * @returns {Promise<'save'|'discard'|'cancel'>}
 */
export function showUnsavedConfigChangesModal() {
  return showUnsavedChangesModal({
    message: 'You have unsaved changes. What would you like to do?',
    discardText: "Don't save & continue",
  });
}
