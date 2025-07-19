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
export async function renderLabelList({ target = "labels-list", editable = true }) {
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
              if (response.ok) renderLabelList({ target, editable });
              else alert('Failed to update label');
            },
            onCancel: () => renderLabelList({ target, editable }),
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
          if (response.ok) renderLabelList({ target, editable });
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
              if (response.ok) renderLabelList({ target, editable });
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

    data.labels
      .filter(label => !label.metadata.parent_uuid)
      .forEach(label => renderLabel(label, 0));

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
            if (response.ok) renderLabelList({ target, editable });
            else alert('Failed to create label');
          },
          onCancel: () => renderLabelList({ target, editable }),
        });
        newLabelRow.before(childRow);
      };
      newLabelRow.appendChild(newLabelBtn);
    }

    const refreshBtn = createButton("emoji-button", "Refresh", "🔄");
    refreshBtn.onclick = () => renderLabelList({ target, editable });
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

  // First activate the "Image Tagging" outer tab
  const imageTaggingOuterTabBtn = document.querySelector('.outer-tab-button[data-tab="outer-tab-image-tagging"]');
  const imageTaggingOuterPane = document.getElementById('outer-tab-image-tagging');
  if (imageTaggingOuterTabBtn && imageTaggingOuterPane) {
    document.querySelectorAll('.outer-tab-button').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.outer-tab-pane').forEach(p => p.classList.remove('active'));
    imageTaggingOuterTabBtn.classList.add('active');
    imageTaggingOuterPane.classList.add('active');
  }

  // Then activate the "Canvas" inner tab within "Image Tagging"
  // Get the primary-tab-header that contains the canvas and files tabs
  const primaryTabHeader = document.querySelector('#main-stage .primary-tab-header');
  const canvasTabBtn = primaryTabHeader.querySelector('.tab-button[data-tab="tab-canvas"]');
  const canvasPane = document.getElementById('tab-canvas');

  if (canvasTabBtn && canvasPane && primaryTabHeader) {
    // Deactivate all sibling tabs in the same header (Canvas, Files)
    primaryTabHeader.querySelectorAll('.tab-button').forEach(t => t.classList.remove('active'));
    canvasTabBtn.classList.add('active');

    // Deactivate all sibling panes (tab-canvas, tab-files, tab-label-management)
    // These are direct children of #main-stage
    document.querySelectorAll('#main-stage > .tab-pane').forEach(p => p.classList.remove('active'));
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

// Global cache for labels metadata
let _cachedLabels = null;

async function renderModelLabelAssignments({ target = "model-labels-box", editable = true, preselectedModel = null } = {}) {
  const container = document.getElementById(target);
  container.innerHTML = '';

  // Load and cache labels metadata only once
  if (!_cachedLabels) {
    const labelRes = await fetch('/api/labels/list');
    const { labels } = await labelRes.json();
    // Create a map from UUID to label metadata for quick lookup
    _cachedLabels = new Map(labels.map(label => [label.metadata.uuid, label.metadata]));
  }

  const headerRow = document.createElement('div');
  headerRow.style.display = 'flex';
  headerRow.style.alignItems = 'center';
  headerRow.style.gap = '0.5em';
  headerRow.style.marginBottom = '1em';

  const modelSelect = document.createElement('select');

  if (!preselectedModel) {
    const placeholderOption = document.createElement('option');
    placeholderOption.value = '';
    placeholderOption.textContent = 'Select a model';
    placeholderOption.disabled = true;
    placeholderOption.selected = true;
    modelSelect.appendChild(placeholderOption);
  }

  const refreshBtn = createButton('emoji-button', 'Refresh', '🔄');
  refreshBtn.onclick = () => {
    _cachedLabels = null; // Clear cache on refresh so new labels are fetched
    renderModelLabelAssignments({ target, editable });
  };

  headerRow.appendChild(modelSelect);
  headerRow.appendChild(refreshBtn);
  container.appendChild(headerRow);

  // Load models
  const modelRes = await fetch('/api/models/list');
  const { models } = await modelRes.json();
  models.forEach(model => {
    const opt = document.createElement('option');
    opt.value = model;
    opt.textContent = model;
    modelSelect.appendChild(opt);
  });

  if (preselectedModel) {
    modelSelect.value = preselectedModel;
  }

  modelSelect.onchange = () => {
    const selectedModel = modelSelect.value;
    if (!selectedModel) return;

    setTimeout(() => {
      renderModelLabelAssignments({ target, editable, preselectedModel: selectedModel });
    }, 0);
  };

  // Exit early if no model is selected
  if (!preselectedModel) return;

  const res = await fetch('/api/models/labels/get', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model_name: preselectedModel }),
  });

  const { labels: classMap } = await res.json();

  Object.entries(classMap).forEach(([cls, uuid]) => {
    const row = document.createElement('div');
    row.style.marginBottom = '0.5em';

    const labelSpan = document.createElement('span');
    labelSpan.style.marginRight = '0.5em';

    if (uuid && _cachedLabels.has(uuid)) {
      const labelMeta = _cachedLabels.get(uuid);
      labelSpan.textContent = cls + ' ';
      // Create span for assigned label name with color
      const assignedLabelSpan = document.createElement('span');
      assignedLabelSpan.textContent = labelMeta.name;
      assignedLabelSpan.style.color = labelMeta.color || 'inherit';
      assignedLabelSpan.style.fontWeight = 'bold';

      labelSpan.appendChild(assignedLabelSpan);
    } else {
      labelSpan.textContent = cls + ' (unassigned)';
    }

    const labelRow = document.createElement('div');
    labelRow.style.display = 'flex';
    labelRow.style.alignItems = 'center';
    labelRow.style.gap = '0.5em';

    labelRow.appendChild(labelSpan);

    if (editable) {
      const editBtn = createButton('emoji-button', 'Edit', '✏️');
      labelRow.appendChild(editBtn);

      const labelListContainer = document.createElement('div');
      labelListContainer.style.marginTop = '0.5em';

      editBtn.onclick = async () => {
        labelListContainer.innerHTML = '';

        const labels = Array.from(_cachedLabels.values());

        const labelList = document.createElement('div');
        labelList.style.marginTop = '0.5em';

        const helpContainer = document.createElement('div');
        helpContainer.style.display = 'flex';
        helpContainer.style.alignItems = 'center';
        helpContainer.style.marginBottom = '0.5em';

        const closeBtn = document.createElement('button');
        closeBtn.textContent = 'X';
        closeBtn.style.cursor = 'pointer';
        closeBtn.style.background = 'transparent';
        closeBtn.style.border = 'none';
        closeBtn.style.color = 'red';
        closeBtn.style.fontWeight = 'bold';
        closeBtn.style.fontSize = '1em';
        closeBtn.style.padding = '0 0.3em';
        closeBtn.style.lineHeight = '1';
        closeBtn.style.marginRight = '0.3em';

        closeBtn.onclick = () => {
          labelListContainer.innerHTML = '';
        };

        const helpText = document.createElement('div');
        helpText.textContent = 'Click a label to assign it to ' + cls;
        helpText.style.fontStyle = 'italic';
        helpText.style.fontSize = '0.9em';

        helpContainer.appendChild(closeBtn);
        helpContainer.appendChild(helpText);
        labelList.appendChild(helpContainer);

        const labelTextContainer = document.createElement('span');
        labelTextContainer.style.display = 'flex';
        labelTextContainer.style.flexWrap = 'wrap';
        labelTextContainer.style.gap = '0.5em';
        labelTextContainer.style.alignItems = 'center';
        labelTextContainer.style.marginTop = '0.3em';

        labels.forEach((label, index) => {
          const labelSpan = document.createElement('span');
          labelSpan.style.cursor = 'pointer';
          labelSpan.style.display = 'inline-block';

          if (label.color) {
            labelSpan.style.color = label.color;
          }

          labelSpan.textContent = label.name;

          labelSpan.onclick = async () => {
            await fetch('/api/models/labels/associate', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({
                model_name: preselectedModel,
                model_class: cls,
                label_uuid: label.uuid,
              }),
            });
            renderModelLabelAssignments({ target, editable, preselectedModel });
          };

          labelTextContainer.appendChild(labelSpan);

          if (index < labels.length - 1) {
            const comma = document.createElement('span');
            comma.textContent = ',';
            labelTextContainer.appendChild(comma);
          }
        });

        labelList.appendChild(labelTextContainer);
        labelListContainer.appendChild(labelList);
      };

      row.appendChild(labelListContainer);
    }

    row.appendChild(labelRow);
    container.appendChild(row);
  });
}

// Helper: flatten label tree into flat list with indent info
function flattenLabels(labels, depth = 0) {
  let flat = [];
  for (const label of labels) {
    flat.push({ ...label, indent: depth });
    if (label.children?.length) {
      flat = flat.concat(flattenLabels(label.children, depth + 1));
    }
  }
  return flat;
}

// --- Initial setup on DOM ready ---

document.addEventListener('DOMContentLoaded', () => {
  // New: Outer tab handling
  document.querySelectorAll('.outer-tab-header').forEach(tabHeader => {
    const tabs = tabHeader.querySelectorAll('.outer-tab-button');
    const tabContainer = document.querySelector('main.container'); // The main container holds outer panes
    const tabPanes = tabContainer.querySelectorAll('.outer-tab-pane');

    tabs.forEach(tab => {
      tab.addEventListener('click', () => {
        tabs.forEach(t => t.classList.remove('active'));
        tab.classList.add('active');

        const tabName = tab.getAttribute('data-tab');
        tabPanes.forEach(tc => tc.classList.toggle('active', tc.id === tabName));

        // If "Settings" tab is opened, ensure "Label Management" is active within it
        if (tabName === 'outer-tab-settings') {
          const labelManagementTabBtn = document.querySelector('#outer-tab-settings .tab-button[data-tab="tab-label-management"]');
          const labelManagementPane = document.getElementById('tab-label-management');
          if (labelManagementTabBtn && labelManagementPane) {
            // Ensure only tabs within this specific tab header are activated
            labelManagementTabBtn.closest('.primary-tab-header').querySelectorAll('.tab-button').forEach(t => t.classList.remove('active'));
            labelManagementTabBtn.classList.add('active');
            labelManagementPane.classList.add('active');
          }
          renderLabelList({ target: "labels-list" });
          renderModelLabelAssignments({ target: "model-labels-box" });
        }
      });
    });
  });


  // Primary tab handling (now within outer panes)
  document.querySelectorAll('.primary-tab-header').forEach(tabHeader => {
    const tabs = tabHeader.querySelectorAll('.tab-button');
    const tabContainer = tabHeader.parentElement; // The parent of primary-tab-header is now the main-stage or outer-tab-settings
    const tabPanes = tabContainer.querySelectorAll('.tab-pane'); // Get panes specific to this tab container

    tabs.forEach(tab => {
      tab.addEventListener('click', () => {
        // Deactivate all sibling tabs in the same header
        tabs.forEach(t => t.classList.remove('active'));
        tab.classList.add('active');

        const tabName = tab.getAttribute('data-tab');
        // Deactivate all sibling panes and activate the relevant one
        tabPanes.forEach(tc => tc.classList.toggle('active', tc.id === tabName));

        // Conditional rendering/loading based on active tab
        if (tabName === 'tab-label-management') {
          renderLabelList({ target: "labels-list" });
          renderModelLabelAssignments({ target: "model-labels-box" });
        } else if (tabName === 'tab-source-management') {
          renderSourceManagement({ target: "source-management-box" });
        } else if (tabName === 'tab-files' && !fileBrowserInitialized) {
          loadFileBrowser(currentFileBrowserPath);
          fileBrowserInitialized = true;
        }
      });
    });
  });

  // Sidebar tab handling
  document.querySelectorAll('#sidebar .tab-header').forEach(tabHeader => {
    const tabs = tabHeader.querySelectorAll('.tab-button');
    const tabContainer = tabHeader.parentElement; // sidebar
    const tabPanes = tabContainer.querySelectorAll('.tab-pane');

    tabs.forEach(tab => {
      tab.addEventListener('click', () => {
        tabs.forEach(t => t.classList.remove('active'));
        tab.classList.add('active');

        const tabName = tab.getAttribute('data-tab');
        tabPanes.forEach(tc => tc.classList.toggle('active', tc.id === tabName));

        if (tabName === 'tab-labels-tool') {
          renderLabelList({ target: "labels-tool-list", editable: false });
        }
      });
    });
  });

  // Initial load for default active tabs
  refreshModelList();
  document.getElementById('recognize-button')?.addEventListener('click', recognizeImage);
  document.getElementById('refresh-models')?.addEventListener('click', refreshModelList);
  setupFilesTab();

  // Manually trigger initial rendering for the default active tabs
  // This will activate 'Image Tagging' and then 'Canvas'
  document.querySelector('.outer-tab-button[data-tab="outer-tab-image-tagging"]').click();
  document.querySelector('#outer-tab-image-tagging .tab-button[data-tab="tab-canvas"]').click();
});