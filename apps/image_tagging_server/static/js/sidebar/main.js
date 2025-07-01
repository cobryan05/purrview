import { getLayer } from '/app-static/js/canvas/state.js';
import { addRecognizedBoxes, loadImageAndMetadata } from '/app-static/js/canvas/io.js';

async function refreshModelList() {
  try {
    const res = await fetch('/api/models/list');
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
    if (firstRadio) {
      firstRadio.checked = true;
    }
  } catch (err) {
    console.error('Failed to fetch models:', err);
  }
}

async function refreshLabelList({ target = "labels-list", editable = true }) {
  try {
    const res = await fetch('/api/labels/list');
    const data = await res.json();

    const labelListContainer = document.getElementById(target);
    if (!labelListContainer) {
      console.error("labelListContainer is null");
      return;
    }
    labelListContainer.innerHTML = '';

    // --- Helper functions ---
    const createButton = (className, title, text) => {
      const btn = document.createElement('button');
      btn.className = className;
      btn.title = title;
      btn.textContent = text;
      return btn;
    };

    function createInputRow({ defaultName = '', defaultColor = '#cccccc', onSave, onCancel, indentLevel = 0 }) {
      const row = document.createElement('div');
      row.className = 'label-row';
      if (indentLevel > 0) row.style.paddingLeft = `${indentLevel * 1.5}em`;

      const saveBtn = createButton('emoji-button', onSave ? 'Save' : 'Create', '✅');
      const cancelBtn = createButton('emoji-button', 'Cancel', '❌');

      // Custom-styled color swatch + hidden color input
      const colorWrapper = document.createElement('span');
      colorWrapper.className = 'label-color';
      colorWrapper.style.position = 'relative';
      colorWrapper.style.display = 'inline-block';
      colorWrapper.style.cursor = 'pointer';
      colorWrapper.style.backgroundColor = defaultColor;

      const colorInput = document.createElement('input');
      colorInput.type = 'color';
      colorInput.value = defaultColor;
      colorInput.style.opacity = '0';
      colorInput.style.position = 'absolute';
      colorInput.style.left = '0';
      colorInput.style.top = '0';
      colorInput.style.width = '100%';
      colorInput.style.height = '100%';
      colorInput.style.cursor = 'pointer';

      colorInput.addEventListener('input', () => {
        colorWrapper.style.backgroundColor = colorInput.value;
      });

      colorWrapper.appendChild(colorInput);

      // Label name input
      const nameInput = document.createElement('input');
      nameInput.type = 'text';
      nameInput.value = defaultName;
      nameInput.placeholder = 'New label name';
      nameInput.className = 'label-name';
      nameInput.style.minWidth = '5em';
      nameInput.style.height = '2em';
      nameInput.style.marginRight = '0.5em';

      // Append all elements
      row.appendChild(saveBtn);
      row.appendChild(cancelBtn);
      row.appendChild(colorWrapper);  // styled color input
      row.appendChild(nameInput);

      saveBtn.onclick = () => {
        const newName = nameInput.value.trim();
        if (!newName) {
          alert('Name required');
          return;
        }
        onSave?.(newName, colorInput.value);
      };

      cancelBtn.onclick = () => {
        onCancel?.();
      };

      return row;
    }

    function renderLabel(label, indentLevel = 0) {
      const { id, name, color } = label.metadata;

      const labelRow = document.createElement('div');
      labelRow.className = 'label-row';
      labelRow.dataset.labelId = id;
      labelRow.style.paddingLeft = `${indentLevel * 1.5}em`;

      if (editable) {
        const editBtn = createButton('label-edit-btn emoji-button', 'Edit label', '✏️');
        const deleteBtn = createButton('label-remove-btn emoji-button', 'Delete label', '🗑️');
        const addChildBtn = createButton('label-add-child emoji-button', 'Add sublabel', '➕');

        editBtn.onclick = () => {
          const inputRow = createInputRow({
            defaultName: name,
            defaultColor: color,
            indentLevel,
            onSave: async (newName, newColor) => {
              const updatedLabel = { label_id: id, name: newName, color: newColor };
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

        deleteBtn.onclick = async () => {
          if (!window.confirm(`Delete label "${name}"?`)) return;
          const response = await fetch('/api/labels/delete', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ label_id: id }),
          });
          if (response.ok) refreshLabelList({ target, editable });
          else alert('Failed to delete label');
        };

        addChildBtn.onclick = () => {
          const childRow = createInputRow({
            indentLevel: indentLevel + 1,
            onSave: async (newName, newColor) => {
              const newLabel = { name: newName, color: newColor, parent_id: id };
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

        labelRow.appendChild(editBtn);
        labelRow.appendChild(deleteBtn);
        labelRow.appendChild(addChildBtn);
      }

      const colorSwatch = document.createElement('span');
      colorSwatch.className = 'label-color';
      colorSwatch.style.background = color;

      const nameSpan = document.createElement('span');
      nameSpan.className = 'label-name';
      nameSpan.textContent = name;

      labelRow.appendChild(colorSwatch);
      labelRow.appendChild(nameSpan);

      labelListContainer.appendChild(labelRow);

      label.children.forEach(child => renderLabel(child, indentLevel + 1));
    }

    data.labels.forEach(label => renderLabel(label, 0));

    const newLabelRow = document.createElement('div');
    newLabelRow.className = 'label-row';
    if (editable) {
      const newLabelBtn = createButton('label-add-child emoji-button', 'Add label', '➕');
      newLabelRow.appendChild(newLabelBtn);
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
    }

    const refreshBtn = createButton("emoji-button", "Refresh", "🔄");
    refreshBtn.onclick = () => refreshLabelList({ target, editable });
    newLabelRow.appendChild(refreshBtn);
    labelListContainer.appendChild(newLabelRow);

  } catch (err) {
    console.error('Failed to fetch labels:', err);
  }
}


async function recognizeImage() {
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

    // Draw background image to a canvas element
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = background.image().width;
    tempCanvas.height = background.image().height;
    const ctx = tempCanvas.getContext('2d');
    ctx.drawImage(background.image(), 0, 0);

    // Convert to Blob
    const blob = await new Promise(resolve => {
      tempCanvas.toBlob(resolve, 'image/png');
    });

    if (!blob) {
      throw new Error('Failed to convert canvas image to Blob');
    }

    // Create FormData
    const formData = new FormData();
    formData.append('model_name', modelName);
    formData.append('conf_thresh', '0.25');
    formData.append('return_annotated', 'false');
    formData.append('image', blob, 'canvas_image.png');

    // Send to recognition API
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

export function openInspectorTab() {
  // Scope tab switching to the sidebar tab group only
  const sidebar = document.getElementById('sidebar');
  if (!sidebar) return;

  const tabs = sidebar.querySelectorAll('.tab-button');
  const panes = sidebar.querySelectorAll('.tab-pane');

  tabs.forEach(tab => {
    const tabName = tab.getAttribute('data-tab');
    const isInspector = tabName === 'tab-inspector';
    tab.classList.toggle('active', isInspector);
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


let fileBrowserInitialized = false;
let currentFileBrowserPath = '/';

function setupFilesTab() {
  const filesTab = document.getElementById('tab-files');
  if (!filesTab) return;

  // Create container for file browser
  let browser = document.createElement('div');
  browser.id = 'file-browser';
  filesTab.innerHTML = ''; // Clear placeholder
  filesTab.appendChild(browser);

  // Add "Open All" button
  const openAllBtn = document.createElement('button');
  openAllBtn.textContent = 'Open All';
  openAllBtn.style.marginBottom = '0.5em';
  openAllBtn.onclick = () => {
    // TODO: Implement queue loading of all files in current directory
    alert('TODO: Open all files in this directory as a queue');
  };
  filesTab.appendChild(openAllBtn);
}

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

    // Breadcrumb navigation
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

    // File/folder list
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

async function loadImageByName(filename, dirPath) {
  const imageName = (dirPath === '/' ? '' : dirPath + '/') + filename;

  // Switch to the Canvas tab
  const canvasTabBtn = document.querySelector('.tab-button[data-tab="tab-canvas"]');
  const canvasPane = document.getElementById('tab-canvas');
  if (canvasTabBtn && canvasPane) {
    // Deactivate all tabs
    document.querySelectorAll('.tab-button').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));

    // Activate Canvas tab
    canvasTabBtn.classList.add('active');
    canvasPane.classList.add('active');
  }

  // Wait for next animation frame so canvas layout updates
  await new Promise(requestAnimationFrame);

  // Now load the image
  loadImageAndMetadata(imageName);
}



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
        tabPanes.forEach(tc => {
          tc.classList.toggle('active', tc.id === tabName);
        });
        // Call refreshLabelList when Labels tab is activated
        if (tabName === 'tab-labels') {
          refreshLabelList({ target: "labels-list" })
        } else if (tabName == "tab-labels-tool") {
          refreshLabelList({ target: "labels-tool-list", editable: false });
        } else if (tabName === 'tab-files' && !fileBrowserInitialized) {
          loadFileBrowser(currentFileBrowserPath);
          fileBrowserInitialized = true;
        }
      });
    });
  });

  refreshModelList();
  document.getElementById('recognize-button').addEventListener('click', recognizeImage);

  setupFilesTab()
});