import { createEmojiButton } from './createButton.js'

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
  mainRow.style.alignItems = 'flex-start'; // top align
  mainRow.style.gap = '0.25em';
  mainRow.style.width = '100%';

  // --- Helper to create button sets ---
  const createButtonsRow = (btns) => {
    const container = document.createElement('div');
    container.style.display = 'flex';
    container.style.flexDirection = 'row';  // horizontal
    container.style.alignItems = 'flex-start';
    container.style.gap = '0.25em';

    btns.forEach(({ text, emoji, onClick }) => {
      const btn = createEmojiButton({ text, emoji, onClick: () => onClick?.({ field }) });
      container.appendChild(btn);
    });

    return container;
  };

  const leftContainer = createButtonsRow(leftButtons);
  const rightContainer = createButtonsRow(rightButtons);

  mainRow.appendChild(leftContainer);
  mainRow.appendChild(field.getEditMode() ? field.renderEdit() : field.renderView());
  mainRow.appendChild(rightContainer);

  indentContainer.appendChild(mainRow);
  row.appendChild(indentContainer);

  return row;
}