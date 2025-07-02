// canvas_ui_main.js

import { addRecognizedBoxes, loadImageAndMetadata } from '/app-static/js/canvas/io.js';
import { getCurrentLabelUuid, getLayer, setLabelList } from '/app-static/js/canvas/state.js';
import { getTool, setTool } from '/app-static/js/canvas/tools.js';

// --- Utility functions ---

/**
 * Creates a button element with given class, title, and text.
 */
function createButton(className, title, text) {
  const btn = document.createElement('button');
  btn.className = className;
  btn.title = title;
  btn.textContent = text;
  return btn;
}

/**
 * Creates a row with inputs for label name and color, plus Save/Cancel buttons.
 * Calls onSave(newName, newColor) on save, onCancel() on cancel.
 */
function createInputRow({ defaultName = '', defaultColor = '#cccccc', onSave, onCancel, indentLevel = 0 }) {
  const row = document.createElement('div');
  row.className = 'label-row';
  if (indentLevel > 0) row.style.paddingLeft = `${indentLevel * 1.5}em`;

  const saveBtn = createButton('emoji-button', 'Save', '✅');
  const cancelBtn = createButton('emoji-button', 'Cancel', '❌');

  // Color swatch wrapper with hidden color input
  const colorWrapper = document.createElement('span');
  colorWrapper.className = 'label-color';
  colorWrapper.style = `position: relative; display: inline-block; cursor: pointer; background-color: ${defaultColor}`;

  const colorInput = document.createElement('input');
  colorInput.type = 'color';
  colorInput.value = defaultColor;
  Object.assign(colorInput.style, {
    opacity: '0',
    position: 'absolute',
    left: '0',
    top: '0',
    width: '100%',
    height: '100%',
    cursor: 'pointer',
  });
  colorInput.addEventListener('input', () => {
    colorWrapper.style.backgroundColor = colorInput.value;
  });
  colorWrapper.appendChild(colorInput);

  // Name input
  const nameInput = document.createElement('input');
  nameInput.type = 'text';
  nameInput.value = defaultName;
  nameInput.placeholder = 'New label name';
  Object.assign(nameInput.style, {
    minWidth: '5em',
    height: '2em',
    marginRight: '0.5em',
  });

  row.append(saveBtn, cancelBtn, colorWrapper, nameInput);

  saveBtn.onclick = () => {
    const newName = nameInput.value.trim();
    if (!newName) {
      alert('Name required');
      return;
    }
    onSave?.(newName, colorInput.value);
  };

  cancelBtn.onclick = () => onCancel?.();

  return row;
}

// --- Model list UI ---

/**
 * Fetches the list of available models from the backend API,
 * populates the UI with radio buttons for model selection,
 * and selects the first model by default.
 */
export async function refreshModelList() {
  try {
    const res = await fetch('/api/models/list');
    if (!res.ok) throw new Error(`Failed to fetch models: ${res.status}`);
    const data = await res.json();

    const modelListContainer = document.getElementById('model-list');
    if (!modelListContainer) {
      console.error("modelListContainer is null");
      return;
    }

    modelListContainer.innerHTML = '';

    data.models.forEach(model => {
      const label = document.createElement('label');
      label.style.display = 'block';

      const radio = document.createElement('input');
      radio.type = 'radio';
      radio.name = 'model';
      radio.value = model;

      label.appendChild(radio);
      label.appendChild(document.createTextNode(` ${model}`));
      modelListContainer.appendChild(label);
    });

    const firstRadio = modelListContainer.querySelector('input[type="radio"]');
    if (firstRadio) firstRadio.checked = true;

  } catch (err) {
    console.error('Failed to fetch models:', err);
  }
}

// --- Label list UI ---

/**
 * Fetches labels list from the API and renders it in the specified container.
 * Supports editable mode with buttons for editing, deleting, adding sublabels.
 * @param {Object} options
 * @param {string} options.target - The id of the container element to render labels into.
 * @param {boolean} options.editable - Whether label editing controls are shown.
 */
export async function refreshLabelList({ target = "labels-list", editable = true }) {
  try {
    const res = await fetch('/api/labels/list');
    if (!res.ok) throw new Error(`Failed to fetch labels: ${res.status}`);
    const data = await res.json();
    const labelListContainer = document.getElementById(target);
    if (!labelListContainer) {
      console.error("labelListContainer is null");
      return;
    }

    setLabelList(data.labels);

    labelListContainer.innerHTML = '';

    // Recursively render labels and children with indentation
    function renderLabel(label, indentLevel = 0) {
      const { name, color, uuid } = label.metadata;

      const labelRow = document.createElement('div');
      labelRow.className = 'label-row';
      labelRow.dataset.uuid = uuid;
      labelRow.style.paddingLeft = `${indentLevel * 1.5}em`;

      if (!editable) {
        labelRow.style.cursor = 'pointer';
        labelRow.onclick = () => {
          setTool(`bbox:${uuid}`);
          highlightSelectedLabel(uuid);
        };
      }

      if (editable) {
        // Edit button
        const editBtn = createButton('label-edit-btn emoji-button', 'Edit label', '✏️');
        editBtn.onclick = () => {
          const inputRow = createInputRow({
            defaultName: name,
            defaultColor: color,
            indentLevel,
            onSave: async (newName, newColor) => {
              const updatedLabel = { label_uuid: uuid, name: newName, color: newColor };
              const response = await fetch('/api/labels/update', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(updatedLabel),
              });
              if (response.ok) refreshLabelList({ target, editable });
              else alert('Failed to update label');
            },
            onCancel: () => refreshLabelList({ target, editable }),
          });
          labelRow.replaceWith(inputRow);
        };

        // Delete button
        const deleteBtn = createButton('label-remove-btn emoji-button', 'Delete label', '🗑️');
        deleteBtn.onclick = async () => {
          if (!window.confirm(`Delete label "${name}"?`)) return;
          const response = await fetch('/api/labels/delete', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ label_uuid: uuid }),
          });
          if (response.ok) refreshLabelList({ target, editable });
          else alert('Failed to delete label');
        };

        // Add sublabel button
        const addChildBtn = createButton('label-add-child emoji-button', 'Add sublabel', '➕');
        addChildBtn.onclick = () => {
          const childRow = createInputRow({
            indentLevel: indentLevel + 1,
            onSave: async (newName, newColor) => {
              const newLabel = { name: newName, color: newColor, parent_uuid: uuid };
              const response = await fetch('/api/labels/add', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(newLabel),
              });
              if (response.ok) refreshLabelList({ target, editable });
              else alert('Failed to create label');
            },
            onCancel: () => childRow.remove(),
          });
          labelRow.after(childRow);
        };

        labelRow.append(editBtn, deleteBtn, addChildBtn);
      }

      // Color swatch and label name
      const colorSwatch = document.createElement('span');
      colorSwatch.className = 'label-color';
      colorSwatch.style.background = color;

      const nameSpan = document.createElement('span');
      nameSpan.className = 'label-name';
      nameSpan.textContent = name;

      labelRow.append(colorSwatch, nameSpan);
      labelListContainer.appendChild(labelRow);

      label.children.forEach(child => renderLabel(child, indentLevel + 1));

      return labelRow;
    }

    /**
     * Highlights the selected label in the labels-tool-list panel
     * @param {string|null} selectedUuid Label ID to highlight
     */
    function highlightSelectedLabel(selectedUuid = null) {
      const currentTool = getTool();
      const selectedLabelUuid = selectedUuid ?? (currentTool.startsWith('bbox:') ? currentTool.split(':')[1] : null);

      document.querySelectorAll('#labels-tool-list .label-row').forEach(row => {
        row.style.outline = row.dataset.uuid == selectedLabelUuid ? '2px solid #ff0033' : '';
      });
    }

    data.labels.forEach(label => renderLabel(label, 0));

    // Add new label row at bottom
    const newLabelRow = document.createElement('div');
    newLabelRow.className = 'label-row';
    if (editable) {
      const newLabelBtn = createButton('label-add-child emoji-button', 'Add label', '➕');
      newLabelBtn.onclick = () => {
        const childRow = createInputRow({
          onSave: async (newName, newColor) => {
            const newLabel = { name: newName, color: newColor, parent_id: null };
            const response = await fetch('/api/labels/add', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify(newLabel),
            });
            if (response.ok) refreshLabelList({ target, editable });
            else alert('Failed to create label');
          },
          onCancel: () => refreshLabelList({ target, editable }),
        });
        newLabelRow.before(childRow);
      };
      newLabelRow.appendChild(newLabelBtn);
    }

    const refreshBtn = createButton("emoji-button", "Refresh", "🔄");
    refreshBtn.onclick = () => refreshLabelList({ target, editable });
    newLabelRow.appendChild(refreshBtn);

    labelListContainer.appendChild(newLabelRow);

    const selectedLabelUuid = getCurrentLabelUuid();
    if (selectedLabelUuid) {
      highlightSelectedLabel(selectedLabelUuid);
    }

  } catch (err) {
    console.error('Failed to fetch labels:', err);
  }
}

// --- Image recognition ---

/**
 * Sends the current canvas image to the recognition API using the selected model.
 * Adds recognized bounding boxes to the canvas on success.
 */
export async function recognizeImage() {
  try {
    const selected = document.querySelector('input[name="model"]:checked');
    if (!selected) {
      alert('Please select a model');
      return;
    }

    const modelName = selected.value;
    const layer = getLayer();
    const background = layer.findOne('.background');
    if (!background || !background.image()) {
      throw new Error('No background image found on canvas');
    }

    // Draw background image to a temporary canvas
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = background.image().width;
    tempCanvas.height = background.image().height;
    const ctx = tempCanvas.getContext('2d');
    ctx.drawImage(background.image(), 0, 0);

    // Convert to Blob
    const blob = await new Promise(resolve => tempCanvas.toBlob(resolve, 'image/png'));
    if (!blob) {
      throw new Error('Failed to convert canvas image to Blob');
    }

    // Prepare form data for POST
    const formData = new FormData();
    formData.append('model_name', modelName);
    formData.append('conf_thresh', '0.25');
    formData.append('return_annotated', 'false');
    formData.append('image', blob, 'canvas_image.png');

    // Send recognition request
    const response = await fetch('/api/recognize', {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const text = await response.text();
      throw new Error(`Recognition API error: ${response.status} ${text}`);
    }

    const result = await response.json();
    console.log('Recognize response:', result);

    if (result.status === 'success') {
      addRecognizedBoxes(result.results);
    }
  } catch (err) {
    console.error('Failed to recognize image:', err);
  }
}

// --- File browser setup ---

let fileBrowserInitialized = false;
let currentFileBrowserPath = '/';

/**
 * Sets up the files tab UI with a container and an Open All button.
 */
function setupFilesTab() {
  const filesTab = document.getElementById('tab-files');
  if (!filesTab) return;

  const browser = document.createElement('div');
  browser.id = 'file-browser';
  filesTab.innerHTML = '';
  filesTab.appendChild(browser);

  const openAllBtn = document.createElement('button');
  openAllBtn.textContent = 'Open All';
  openAllBtn.style.marginBottom = '0.5em';
  openAllBtn.onclick = () => {
    alert('TODO: Open all files in this directory as a queue');
  };
  filesTab.appendChild(openAllBtn);
}

/**
 * Loads and displays the file browser for the given path.
 * Supports directory navigation and file selection.
 */
async function loadFileBrowser(path) {
  currentFileBrowserPath = path;
  const browser = document.getElementById('file-browser');
  if (!browser) return;
  browser.innerHTML = 'Loading...';

  try {
    const res = await fetch(`/api/images/list?path=${encodeURIComponent(path)}`);
    if (!res.ok) throw new Error('Failed to fetch file list');
    const data = await res.json();

    const currentPath = path || '/';
    const breadcrumb = document.createElement('div');
    breadcrumb.style.marginBottom = '0.5em';

    let parts = currentPath.split('/').filter(Boolean);
    let accum = '';
    breadcrumb.appendChild(makeBreadcrumbLink('/', '/'));
    parts.forEach((part) => {
      accum += '/' + part;
      breadcrumb.appendChild(document.createTextNode(' / '));
      breadcrumb.appendChild(makeBreadcrumbLink(part, accum));
    });

    browser.innerHTML = '';
    browser.appendChild(breadcrumb);

    const list = document.createElement('ul');
    list.style.listStyle = 'none';
    list.style.padding = '0';

    data.files.forEach(entry => {
      const li = document.createElement('li');
      li.style.margin = '0.2em 0';

      if (entry.type === 'dir') {
        li.innerHTML = `📁 <a href="#">${entry.name}</a>`;
        li.querySelector('a').onclick = (e) => {
          e.preventDefault();
          loadFileBrowser(entry.path);
        };
      } else {
        li.innerHTML = `🖼️ <a href="#">${entry.name}</a>`;
        li.querySelector('a').onclick = (e) => {
          e.preventDefault();
          loadImageByName(entry.name, currentPath);
        };
      }
      list.appendChild(li);
    });

    browser.appendChild(list);
  } catch (err) {
    browser.innerHTML = 'Failed to load files.';
    console.error(err);
  }
}

/**
 * Creates a breadcrumb link element for navigation.
 */
function makeBreadcrumbLink(label, path) {
  const a = document.createElement('a');
  a.href = '#';
  a.textContent = label;
  a.onclick = (e) => {
    e.preventDefault();
    loadFileBrowser(path);
  };
  return a;
}

/**
 * Loads an image by filename and directory path,
 * switches to the canvas tab, and triggers image loading.
 */
async function loadImageByName(filename, dirPath) {
  const imageName = (dirPath === '/' ? '' : dirPath + '/') + filename;

  const canvasTabBtn = document.querySelector('.tab-button[data-tab="tab-canvas"]');
  const canvasPane = document.getElementById('tab-canvas');
  if (canvasTabBtn && canvasPane) {
    document.querySelectorAll('.tab-button').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));

    canvasTabBtn.classList.add('active');
    canvasPane.classList.add('active');
  }

  // Wait for next animation frame so canvas layout updates
  await new Promise(requestAnimationFrame);

  loadImageAndMetadata(imageName);
}

// --- Inspector tab control ---

/**
 * Opens the inspector tab in the sidebar and highlights it briefly.
 */
export function openInspectorTab() {
  const sidebar = document.getElementById('sidebar');
  if (!sidebar) return;

  const tabs = sidebar.querySelectorAll('.tab-button');
  const panes = sidebar.querySelectorAll('.tab-pane');

  tabs.forEach(tab => {
    const tabName = tab.getAttribute('data-tab');
    tab.classList.toggle('active', tabName === 'tab-inspector');
  });

  panes.forEach(pane => {
    pane.classList.toggle('active', pane.id === 'tab-inspector');
  });

  document.getElementById('labelInput')?.focus();

  const inspector = document.getElementById('tab-inspector');
  if (inspector) {
    inspector.style.outline = '2px solid #ff0033';
    setTimeout(() => inspector.style.outline = '', 1000);
  }
}

// --- Initial setup on DOM ready ---

document.addEventListener('DOMContentLoaded', () => {
  document.querySelectorAll('.tab-header, .primary-tab-header').forEach(tabHeader => {
    const tabs = tabHeader.querySelectorAll('.tab-button');
    const tabContainer = tabHeader.parentElement;
    const tabPanes = tabContainer.querySelectorAll('.tab-pane');

    tabs.forEach(tab => {
      tab.addEventListener('click', () => {
        tabs.forEach(t => t.classList.remove('active'));
        tab.classList.add('active');

        const tabName = tab.getAttribute('data-tab');
        tabPanes.forEach(tc => tc.classList.toggle('active', tc.id === tabName));

        if (tabName === 'tab-labels') {
          refreshLabelList({ target: "labels-list" });
        } else if (tabName === 'tab-labels-tool') {
          refreshLabelList({ target: "labels-tool-list", editable: false });
        } else if (tabName === 'tab-files' && !fileBrowserInitialized) {
          loadFileBrowser(currentFileBrowserPath);
          fileBrowserInitialized = true;
        }
      });
    });
  });

  refreshModelList();

  document.getElementById('recognize-button')?.addEventListener('click', recognizeImage);

  setupFilesTab();
});
