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
  btn.onclick = () => onClick?.();
  return btn;
}
