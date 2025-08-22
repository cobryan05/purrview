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
  labelPlaceholder = null,
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
    nameInput.placeholder = labelPlaceholder || 'Label name';
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

/**
 * Extracts form parameters from a given container element.
 *
 * This function iterates over all input, select, and textarea elements
 * within the specified container and collects their values into an object.
 * Checkbox values are stored as booleans, number inputs are parsed as floats,
 * and other input types are stored as strings.
 *
 * @param {HTMLElement} container - The container element containing form inputs.
 * @returns {Object} An object where keys are input names and values are their corresponding values.
 */
export function getParamsFromForm(container) {
  const params = {};
  container.querySelectorAll('input, select, textarea').forEach(el => {
    if (!el.name) return;
    if (el.type === 'checkbox') params[el.name] = el.checked;
    else if (el.type === 'number') params[el.name] = parseFloat(el.value);
    else params[el.name] = el.value;
  });
  return params;
}


/**
 * Utility: Fetch list of image providers
 */
export async function fetchImageProviderList() {
  const res = await fetch('/api/sources/image-providers/list');
  const data = await res.json();
  return data.providers;
}

/**
 * Renders form inputs from JSON schema
 */
export function renderSchemaForm(schema, container) {
  container.innerHTML = '';
  Object.entries(schema).forEach(([name, field]) => {
    const wrapper = document.createElement('div');
    wrapper.style.display = 'flex';
    wrapper.style.flexDirection = 'column';
    wrapper.style.marginBottom = '1em';

    const label = document.createElement('label');
    label.textContent = field.title || name;
    label.htmlFor = name;
    if (field.required) label.innerHTML += ' <span style="color: red">*</span>';
    wrapper.appendChild(label);

    let input;
    if (field.type === 'boolean') {
      const checkboxWrapper = document.createElement('label');
      checkboxWrapper.style.display = 'flex';
      checkboxWrapper.style.alignItems = 'center';
      checkboxWrapper.style.gap = '0.5em';

      input = document.createElement('input');
      input.type = 'checkbox';
      input.name = name;
      input.checked = !!field.default;
      checkboxWrapper.appendChild(input);
      checkboxWrapper.appendChild(document.createTextNode(field.description || field.title || name));
      wrapper.appendChild(checkboxWrapper);
    } else if (field.type === 'array' && field.items?.type === 'string') {
      input = document.createElement('textarea');
      input.name = name;
      input.placeholder = (field.items.description || field.description || '') + ' (one per line)';
      input.rows = 3;
      wrapper.appendChild(input);
    } else {
      input = document.createElement('input');
      input.type = 'text';
      input.name = name;
      input.placeholder = field.description || '';
      if (field.default !== undefined) input.value = field.default;
      wrapper.appendChild(input);
    }

    if (field.required) input.required = true;
    container.appendChild(wrapper);
  });
}

export function pollTaskStatus(taskUuid, resultBox, onComplete, timeoutMs = 30000) {
  const interval = setInterval(async () => {
    try {
      const res = await fetch(`/api/tasks/status`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ task_ids: [taskUuid] }),
      });
      const result = await res.json();

      if (result.status === "success") {
        const taskStatus = result.tasks[taskUuid];
        resultBox.textContent = taskStatus.message;

        if (taskStatus.status === 'completed' || taskStatus.status === 'error') {
          clearInterval(interval);
          clearTimeout(timeout);
          const taskRes = await fetch(`/api/tasks/result`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ task_id: taskUuid }),
          });
          const taskResult = await taskRes.json();
          onComplete(taskResult.result);
        }
      }
    } catch (err) {
      console.error("Polling error:", err);
    }
  }, 1000);

  const timeout = setTimeout(() => {
    clearInterval(interval);
    resultBox.textContent = "Task timed out.";
    onComplete({ status: 'timeout', message: 'The task did not complete in time.' });
  }, timeoutMs);
}
