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
  rightButtons = [],
  alignLeftButtons = false,
  alignRightButtons = false
} = {}) {
  const row = document.createElement('div');
  row.className = 'label-row';
  row.style.display = 'flex';
  row.style.flexDirection = 'row';
  row.style.alignItems = 'flex-start';
  row.style.gap = '0.25em';
  row.style.width = '100%';

  const createButtonsRow = (btns) => {
    const container = document.createElement('div');
    container.style.display = 'flex';
    container.style.flexDirection = 'row';
    container.style.alignItems = 'flex-start';
    container.style.gap = '0.25em';
    // Buttons keep their size; the field column is what gives way (see below).
    container.style.flexShrink = '0';

    btns.forEach(({ text, emoji, onClick }) => {
      const btn = createEmojiButton({ text, emoji });
      btn.onclick = async () => {
        if (btn.disabled) return;
        btn.disabled = true;
        try {
          await onClick?.({ field, row });
        } finally {
          btn.disabled = false;
        }
      };
      container.appendChild(btn);
    });

    return container;
  };

  const leftContainer = createButtonsRow(leftButtons);
  const rightContainer = createButtonsRow(rightButtons);

  // Indented container holds field + (optionally) left/right buttons
  const indentContainer = document.createElement('div');
  indentContainer.style.display = 'flex';
  indentContainer.style.flexDirection = 'row';
  indentContainer.style.alignItems = 'flex-start';
  indentContainer.style.gap = '0.25em';
  indentContainer.style.flex = '1';
  indentContainer.style.minWidth = '0';
  indentContainer.style.paddingLeft = `${indentLevel * 1.5}em`;

  // Add left buttons: either flush left or indented with field
  if (alignLeftButtons) {
    row.appendChild(leftContainer); // stays left-aligned, outside indent
  } else if (leftButtons.length > 0) {
    indentContainer.appendChild(leftContainer); // indented with field
  }

  // Add field
  // A flex item defaults to `min-width: auto`, i.e. "never shrink below my own
  // content". With buttons either side taking their share first, a wide field —
  // an expanded task card's histogram, sweep table and stat grid — had nowhere
  // to give and pushed the row out past the card's border instead. `0` lets the
  // column shrink to the space actually left over; nothing changes when the
  // content already fits, so the other callers (labels, models, inspector) lay
  // out exactly as before.
  const fieldDiv = document.createElement("div");
  fieldDiv.style.minWidth = '0';
  const renderPromise = field.getEditMode() ? field.renderEdit() : field.renderView();
  renderPromise.then(renderedField => {
    fieldDiv.appendChild(renderedField);
  });
  indentContainer.appendChild(fieldDiv);

  // Add right buttons
  if (rightButtons.length > 0) {
    if (alignRightButtons) {
      rightContainer.style.marginLeft = 'auto'; // push to far right
    }
    indentContainer.appendChild(rightContainer);
  }

  row.appendChild(indentContainer);
  return row;
}
