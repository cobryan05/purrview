/**
 * Creates a generic row with optional color swatch, label/input field, and action buttons.
 *
 * @param {Object} config
 * @param {import('./fields/Field.js').Field} config.field - A Field instance (TextField, RadioField, CheckboxField, etc.)
 * @param {boolean} [config.editable=false] - Whether the row is in edit mode.
 * @param {string|null} [config.colorSwatchColor=null] - Color swatch hex.
 * @param {number} [config.indentLevel=0] - Indentation level.
 * @param {Array} [config.leftButtons=[]] - Left-side button configs.
 * @param {Array} [config.rightButtons=[]] - Right-side button configs.
 * @returns {HTMLDivElement} The created row element.
 */
export function createGenericRow({
  field,
  editable = false,
  indentLevel = 0,
  leftButtons = [],
  rightButtons = [],
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
  const createButtons = btns => {
    const container = document.createElement('div');
    container.style.display = 'flex';
    container.style.alignItems = 'center';
    container.style.gap = '0.15em';
    const inputHeight = 28;
    btns.forEach(({ text, emoji, onClick }) => {
      const btn = document.createElement('button');
      btn.className = 'emoji-button';
      btn.style.cssText = `
        display: inline-flex;
        align-items: center;
        justify-content: center;
        padding: 0 0.3em;
        font-size: 0.9em;
        height: ${inputHeight}px;
        line-height: 1;
        border: 1px solid #ccc;
        border-radius: 3px;
        cursor: pointer;
        flex-shrink: 0;
        box-sizing: border-box;
      `;
      btn.textContent = emoji || text;
      btn.title = text;
      btn.onclick = () => onClick?.({ row, field });
      container.appendChild(btn);
    });
    return container;
  };

  // --- Left buttons ---
  const leftContainer = document.createElement('div');
  leftContainer.style.display = 'flex';
  leftContainer.style.gap = '0.15em';
  leftButtons.forEach(btnRow => {
    const leftRow = Array.isArray(btnRow) ? btnRow : [btnRow];
    leftContainer.appendChild(createButtons(leftRow));
  });

  // --- Right buttons ---
  const rightContainer = document.createElement('div');
  rightContainer.style.display = 'flex';
  rightContainer.style.gap = '0.15em';
  rightButtons.forEach(btnRow => {
    const rightRow = Array.isArray(btnRow) ? btnRow : [btnRow];
    rightContainer.appendChild(createButtons(rightRow));
  });

  // --- Field ---
  const fieldElement = editable ? field.renderEdit() : field.renderView();

  const fieldWrapper = document.createElement('div');
  fieldWrapper.style.flexGrow = '1';
  fieldWrapper.appendChild(fieldElement);

  // Assemble main row: left buttons, field, right buttons
  if (leftButtons.length) mainRow.appendChild(leftContainer);
  mainRow.appendChild(fieldWrapper);
  if (rightButtons.length) mainRow.appendChild(rightContainer);

  indentContainer.appendChild(mainRow);
  row.appendChild(indentContainer);

  return row;
}
