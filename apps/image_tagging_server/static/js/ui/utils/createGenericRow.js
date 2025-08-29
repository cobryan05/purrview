import { createEmojiButton } from '/app-static/js/ui/utils/index.js'

/**
 * Creates a generic row with optional color swatch, label/input field, and action buttons.
 *
 * @param {Object} config
 * @param {import('./fields/Field.js').Field} config.field - A Field instance (TextField, RadioField, CheckboxField, etc.)
 * @param {number} [config.indentLevel=0] - Indentation level.
 * @param {Array} [config.leftButtons=[]] - Left-side button configs.
 * @param {Array} [config.rightButtons=[]] - Right-side button configs.
 * @returns {HTMLDivElement} The created row element.
 */
export function createGenericRow({
  field,
  indentLevel = 0,
  leftButtons = [],
  rightButtons = []
} = {}) {
  const row = document.createElement('div');
  row.className = 'label-row';
  row.style.display = 'flex';
  row.style.flexDirection = 'column';
  row.style.gap = '0.15em';
  row.style.alignItems = 'flex-start';

  // Indentation container
  const indentContainer = document.createElement('div');
  indentContainer.style.display = 'flex';
  indentContainer.style.flexDirection = 'column';
  indentContainer.style.paddingLeft = `${indentLevel * 1.5}em`;

  // Main row
  const mainRow = document.createElement('div');
  mainRow.style.display = 'flex';
  mainRow.style.alignItems = 'center';
  mainRow.style.gap = '0.25em';
  mainRow.style.width = '100%';

  // --- Helper to create button sets ---
  const createButtonsInContainer = btns => {
    const container = document.createElement('div');
    container.style.display = 'flex';
    container.style.alignItems = 'center';
    container.style.gap = '0.15em';
    const inputHeight = 28;
    btns.forEach(({ text, emoji, onClick }) => {
      const btn = createEmojiButton({ text, emoji, onClick: () => onClick?.({ field }) });
      container.appendChild(btn);
    });
    return container;
  };

  // --- Left buttons ---
  const leftContainer = document.createElement('div');
  leftContainer.style.display = 'flex';
  leftContainer.style.gap = '0.15em';

  // --- Right buttons ---
  const rightContainer = document.createElement('div');
  rightContainer.style.display = 'flex';
  rightContainer.style.gap = '0.15em';

  leftContainer.appendChild(createButtonsInContainer(leftButtons));
  rightContainer.appendChild(createButtonsInContainer(rightButtons));

  mainRow.appendChild(leftContainer);
  mainRow.appendChild( field.getEditMode() ? field.renderEdit() : field.renderView());
  mainRow.appendChild(rightContainer);

  indentContainer.appendChild(mainRow);
  row.appendChild(indentContainer);

  return row;
}
