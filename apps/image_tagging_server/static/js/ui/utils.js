// ================= Utility Functions =================

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

export function createGenericRow({
  labelText = '',
  colorSwatchColor,
  indentLevel = 0,
  leftButtons = [],
  rightButtons = [],
  editable = false,
} = {}) {
  const row = document.createElement('div');
  row.className = 'label-row';
  row.style.display = 'flex';
  row.style.flexDirection = 'column';
  row.style.gap = '0.15em';
  row.style.alignItems = 'flex-start';

  // Indentation
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

  // Color swatch
  let colorWrapper, colorInput;
  if (colorSwatchColor !== undefined) {
    colorWrapper = document.createElement('span');
    colorWrapper.className = 'label-color';
    colorWrapper.style.cssText = `
      position: relative;
      display: inline-block;
      width: 1em;
      height: 1em;
      border-radius: 3px;
      cursor: ${editable ? 'pointer' : 'default'};
      background-color: ${colorSwatchColor};
      flex-shrink: 0;
    `;

    colorInput = document.createElement('input');
    colorInput.type = 'color';
    colorInput.value = colorSwatchColor;
    Object.assign(colorInput.style, {
      opacity: editable ? '0' : '0',
      pointerEvents: editable ? 'auto' : 'none',
      position: 'absolute',
      left: 0,
      top: 0,
      width: '100%',
      height: '100%',
      cursor: editable ? 'pointer' : 'default',
    });
    colorInput.disabled = !editable;

    if (editable) {
      colorInput.addEventListener('input', () => {
        colorWrapper.style.backgroundColor = colorInput.value;
      });
    }

    colorWrapper.appendChild(colorInput);
  }

  // --- Label / Input ---
  let nameElement;
  if (editable) {
    const nameInput = document.createElement('input');
    nameInput.type = 'text';
    nameInput.value = labelText;
    nameInput.placeholder = 'Label name';
    Object.assign(nameInput.style, {
      minWidth: '5em',
      height: '1.8em',
      padding: '0 0.25em',
      flexGrow: 1,
      marginLeft: colorSwatchColor !== undefined ? '0.25em' : '0',
      boxSizing: 'border-box',
      lineHeight: '1.2em',
    });
    nameElement = nameInput;
  } else {
    const nameLabel = document.createElement('span');
    nameLabel.textContent = labelText === undefined || labelText === null ? '(unnamed)' : labelText;
    Object.assign(nameLabel.style, {
      flexGrow: 1,
      marginLeft: colorSwatchColor !== undefined ? '0.25em' : '0',
      whiteSpace: 'nowrap',
      overflow: 'hidden',
      textOverflow: 'ellipsis',
    });
    nameElement = nameLabel;
  }

  // Helper to create buttons
  const createButtons = btns => {
    const container = document.createElement('div');
    container.style.display = 'flex';
    container.style.alignItems = 'center';
    container.style.gap = '0.15em';
    const inputHeight = editable ? (nameElement.clientHeight || 28) : 28;
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
      btn.onclick = () => onClick?.({ row, nameInput: editable ? nameElement : null, colorInput });
      container.appendChild(btn);
    });
    return container;
  };

  // Assemble row
  leftButtons.forEach(btnRow => {
    const leftRow = Array.isArray(btnRow) ? btnRow : [btnRow];
    mainRow.appendChild(createButtons(leftRow));
  });

  if (colorSwatchColor !== undefined) mainRow.appendChild(colorWrapper);
  mainRow.appendChild(nameElement);

  rightButtons.forEach(btnRow => {
    const rightRow = Array.isArray(btnRow) ? btnRow : [btnRow];
    mainRow.appendChild(createButtons(rightRow));
  });

  indentContainer.appendChild(mainRow);
  row.appendChild(indentContainer);
  return row;
}
