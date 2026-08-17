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
 * @param {?Node}  [opts.bodyNode]              — extra content between the message and the buttons
 * @param {?{index: number, ms: number}} [opts.arm] — hold one button disabled for `ms`,
 *   counting down in its label. For confirmations that must not be dismissed by
 *   reflex; see showEmphasisConfirmModal.
 * @returns {Promise<any>}
 */
function _buildModal({
  title, message, buttons, defaultFocusIndex = 0, defaultCloseValue = null,
  bodyNode = null, arm = null,
}) {
  return new Promise((resolve) => {
    let resolved = false;
    let armTimer = null;
    const finish = (value) => {
      resolved = true;
      if (armTimer !== null) clearInterval(armTimer);
      close();
      resolve(value);
    };
    const { dialog, open, close } = createModalShell({
      onClose: () => {
        // Dismissal (Esc / backdrop) skips finish(), so the countdown has to be
        // stopped here too or it ticks against a detached button forever.
        if (armTimer !== null) clearInterval(armTimer);
        if (!resolved) resolve(defaultCloseValue);
      },
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

    dialog.append(titleEl, msgEl, ...(bodyNode ? [bodyNode] : []), btnRow);

    if (arm) {
      const armed = btnEls[arm.index];
      const armedText = buttons[arm.index]?.text ?? 'Confirm';
      let remaining = Math.ceil(arm.ms / 1000);
      armed.disabled = true;
      armed.dataset.testid = 'confirm-armed';
      const tick = () => {
        // Written as "Delete (3)" rather than a bare number so the button still
        // says what it does while it is counting.
        armed.textContent = remaining > 0 ? `${armedText} (${remaining})` : armedText;
        if (remaining > 0) return;
        clearInterval(armTimer);
        armTimer = null;
        armed.disabled = false;
      };
      tick();
      armTimer = setInterval(() => { remaining -= 1; tick(); }, 1000);
    }

    open();
    btnEls[defaultFocusIndex]?.focus();
  });
}

/**
 * A confirmation whose *number* is the argument.
 *
 * Deleting 40 images and deleting 40,000 are the same click through the same
 * dialog, and the count is the only thing that distinguishes them — so here it
 * is the loudest thing on screen, grouped with thousands separators (nobody
 * reads 41273 correctly at a glance) and stated in one unmistakable line.
 *
 * Two dials, because "large" and "costly" are not the same claim:
 *
 *  - `tone` — 'danger' when the action destroys something that cannot be
 *    remade, 'caution' when it is merely big. Colouring a bulk removal of
 *    machine-generated images the same red as one that throws away a week of
 *    human labelling teaches people to ignore the red.
 *  - `cooldownMs` — 0 by default. A cooldown only earns its annoyance when
 *    there is work to lose: this dialog appears *after* an ordinary confirm the
 *    user has already dismissed, so without one the second dialog goes the same
 *    way as the first by pure reflex. It is short on purpose — it interrupts a
 *    reflex, it does not punish someone who did mean it.
 *
 * @param {Object} opts
 * @param {string} opts.title
 * @param {number} opts.count       — the number the user needs to actually read
 * @param {string} opts.countNoun   — what is being counted, e.g. 'images'
 * @param {string} opts.message     — the consequence, in the user's terms
 * @param {string} [opts.note]      — the line under the count
 * @param {'danger'|'caution'} [opts.tone='danger']
 * @param {string} [opts.confirmText='Confirm']
 * @param {string} [opts.cancelText='Cancel']
 * @param {number} [opts.cooldownMs=0]
 * @returns {Promise<boolean>}
 */
export function showEmphasisConfirmModal({
  title,
  count,
  countNoun = 'items',
  message,
  note = '',
  tone = 'danger',
  confirmText = 'Confirm',
  cancelText = 'Cancel',
  cooldownMs = 0,
} = {}) {
  const callout = document.createElement('div');
  callout.className = `high-risk-callout is-${tone}`;
  callout.dataset.testid = 'high-risk-callout';
  callout.dataset.tone = tone;

  const countEl = document.createElement('strong');
  countEl.className = 'high-risk-callout__count';
  countEl.dataset.testid = 'high-risk-count';
  countEl.textContent = count.toLocaleString();

  const nounEl = document.createElement('span');
  nounEl.className = 'high-risk-callout__noun';
  nounEl.textContent = countNoun;

  const line = document.createElement('div');
  line.className = 'high-risk-callout__headline';
  line.append(countEl, nounEl);

  callout.appendChild(line);

  if (note) {
    const noteEl = document.createElement('div');
    noteEl.className = 'high-risk-callout__note';
    noteEl.dataset.testid = 'high-risk-note';
    noteEl.textContent = note;
    callout.appendChild(noteEl);
  }

  return _buildModal({
    title,
    message,
    bodyNode: callout,
    buttons: [
      { text: cancelText, value: false, className: 'secondary' },
      { text: confirmText, value: true, className: 'contrast' },
    ],
    defaultFocusIndex: 0,   // still the safe action
    defaultCloseValue: false,
    arm: cooldownMs > 0 ? { index: 1, ms: cooldownMs } : null,
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
 * Show a modal to choose how to delete an image.
 * @param {string} filename  - Display name of the image being deleted.
 * @returns {Promise<'disk'|'db'|null>}  'disk' = delete from disk+DB, 'db' = DB only, null = cancelled.
 */
export function showDeleteImageModal(filename) {
  return _buildModal({
    title: 'Delete image',
    message: `"${filename}" — choose how to remove this image:`,
    buttons: [
      { text: 'Cancel',          value: null,   className: 'secondary' },
      { text: 'Remove from DB',  value: 'db',   className: undefined },
      { text: 'Delete from disk', value: 'disk', className: 'contrast' },
    ],
    defaultFocusIndex: 0,
    defaultCloseValue: null,
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

/**
 * Variant for the "you pressed Start while the form is dirty" case.
 *
 * Distinct from showUnsavedConfigChangesModal because the stakes are different:
 * navigating away only loses edits, whereas starting launches server-side work
 * against the *last saved* config, so "discard" silently runs something other
 * than what is on screen. The wording says so, and 'discard' is spelled as
 * "Start with saved settings" rather than a generic continue.
 *
 * @param {{ actionText?: string }} [opts] - e.g. 'Start Training'; defaults to 'Start'.
 * @returns {Promise<'save'|'discard'|'cancel'>}
 */
export function showUnsavedBeforeStartModal({ actionText = 'Start' } = {}) {
  return _buildModal({
    title: 'Unsaved changes',
    message: 'This config has unsaved changes. Starting now would run the last '
      + 'saved settings, not the ones on screen.',
    buttons: [
      { text: 'Cancel', value: 'cancel', className: 'secondary' },
      { text: `${actionText} with saved settings`, value: 'discard', className: 'contrast' },
      { text: `Save & ${actionText.toLowerCase()}`, value: 'save' },
    ],
    defaultFocusIndex: 2,
    defaultCloseValue: 'cancel',
  });
}
