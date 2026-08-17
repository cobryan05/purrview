/**
 * Creates a button DOM element with specified class, title, and text content.
 * @param {string} className - The CSS class to apply.
 * @param {string} title - Tooltip/title text.
 * @param {string} text - Visible button label.
 * @returns {HTMLButtonElement} The created button element.
 */
export function createButton(className, title, text) {
  const btn = document.createElement('button');
  btn.className = className;
  btn.title = title;
  btn.textContent = text;
  return btn;
}

/**
 * Creates a styled button with an optional emoji and text. The text is used as the tooltip and label if no emoji is provided.
 * @param {string} text - The visible button label and tooltip text.
 * @param {string} [emoji] - Optional emoji to display as the button's label.
 * @param {Function} [onClick] - Optional callback function to execute when the button is clicked.
 * @returns {HTMLButtonElement} The created button element.
 */
export function createEmojiButton({text, emoji = null, onClick = null}) {
  const btn = document.createElement('button');
  btn.className = 'emoji-button';
  btn.style.cssText = `
    display: inline-flex;
    align-items: center;
    justify-content: center;
    padding: 0 0.3em;
    font-size: 0.9em;
    height: 28px;
    line-height: 1;
    border: 1px solid #ccc;
    border-radius: 3px;
    cursor: pointer;
    flex-shrink: 0;
    box-sizing: border-box;
  `;
  btn.title = text; // Tooltip is always the text
  btn.textContent = emoji || text; // Use emoji if provided, otherwise fallback to text
  // The button is passed to the handler so an async action can put it into a busy
  // state (see withButtonBusy). Without it a handler has no way to acknowledge the
  // click, and a slow action looks like a button that did nothing.
  btn.onclick = () => onClick?.(btn);
  return btn;
}

/** Run an async action with the button visibly busy, and never leave it stuck.
 *
 * Disables the button and swaps its face for an hourglass for the duration, so the
 * click is acknowledged the instant it happens rather than only once the server
 * answers. Restores the original state in a `finally`, guarded on the button still
 * being in the document — a background refresh may have rebuilt the row underneath a
 * long-running action, and writing to a detached node is harmless but pointless.
 *
 * Re-entrancy is handled by the disable itself: a second click on a busy button is
 * swallowed by the browser, so double-stops can't be issued.
 *
 * @param {HTMLButtonElement} btn - The button to mark busy (may be undefined).
 * @param {Function} run - The async action to run. Its result is returned.
 */
export async function withButtonBusy(btn, run) {
  if (!btn) return run();
  const previousText = btn.textContent;
  const previousTitle = btn.title;
  btn.disabled = true;
  btn.setAttribute('aria-busy', 'true');
  btn.dataset.busy = 'true';
  btn.textContent = '⏳';
  btn.title = `${previousTitle}…`;
  try {
    return await run();
  } finally {
    delete btn.dataset.busy;
    btn.removeAttribute('aria-busy');
    btn.disabled = false;
    btn.textContent = previousText;
    btn.title = previousTitle;
  }
}
