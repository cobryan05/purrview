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
