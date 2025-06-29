import { getLayer } from '/app-static/js/canvas/state.js';
import { addRecognizedBoxes, loadImageAndMetadata } from '/app-static/js/canvas/io.js';

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
          refreshLabelList();
        } else if (tabName === 'tab-files' && !fileBrowserInitialized) {
          loadFileBrowser(currentFileBrowserPath);
          fileBrowserInitialized = true;
        }
      });
    });
  });

  refreshModelList();
  document.getElementById('recognize-button').addEventListener('click', recognizeImage);

  const addLabelForm = document.getElementById('add-label-form');
  if (addLabelForm) {
    addLabelForm.addEventListener('submit', async (e) => {
      e.preventDefault();
      const name = document.getElementById('new-label-name').value.trim();
      const color = document.getElementById('new-label-color').value;
      if (!name) return;
      const res = await fetch('/api/labels/add', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name, color })
      });
      if (res.ok) {
        document.getElementById('new-label-name').value = '';
        refreshLabelList();
      } else {
        error('Failed to add label');
      }
    });
  }

  setupFilesTab()
});

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

async function refreshLabelList() {
  try {
    const res = await fetch('/api/labels/list');
    const data = await res.json();

    const labelListContainer = document.getElementById('labels-list');
    if (!labelListContainer) {
      console.error("labelListContainer is null");
      return;
    }

    labelListContainer.innerHTML = '';

    const renderLabel = (label, indentLevel = 0) => {
      const { id, name, color } = label.metadata;

      const row = document.createElement('div');
      row.className = 'label-row';
      row.dataset.labelId = id;
      row.style.paddingLeft = `${indentLevel * 1.5}em`;

      const colorSwatch = document.createElement('span');
      colorSwatch.className = 'label-color';
      colorSwatch.style.background = color;

      const nameSpan = document.createElement('span');
      nameSpan.className = 'label-name';
      nameSpan.textContent = name;

      const editBtn = document.createElement('button');
      editBtn.className = 'label-edit-btn emoji-button';
      editBtn.title = 'Edit label';
      editBtn.textContent = '✏️';

      const deleteBtn = document.createElement('button');
      deleteBtn.className = 'label-remove-btn emoji-button';
      deleteBtn.title = 'Delete label';
      deleteBtn.textContent = '🗑️';

      const addChildBtn = document.createElement('button');
      addChildBtn.className = 'label-add-child emoji-button';
      addChildBtn.title = 'Add sublabel';
      addChildBtn.textContent = '➕';

      row.appendChild(editBtn);
      row.appendChild(deleteBtn);
      row.appendChild(addChildBtn);
      row.appendChild(colorSwatch);
      row.appendChild(nameSpan);
      labelListContainer.appendChild(row);

      // DELETE handler
      deleteBtn.onclick = async () => {
        const confirmed = window.confirm(`Delete label "${name}"?`);
        if (!confirmed) return;
        const response = await fetch('/api/labels/delete', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ label_id: id })
        });
        if (response.ok) refreshLabelList();
        else alert('Failed to delete label');
      };

      // EDIT handler
      editBtn.onclick = () => {
        row.innerHTML = '';

        const colorInput = document.createElement('input');
        colorInput.type = 'color';
        colorInput.value = color;
        colorInput.className = 'label-color';
        colorInput.style.width = '3em';
        colorInput.style.height = '3em';

        const nameInput = document.createElement('input');
        nameInput.type = 'text';
        nameInput.value = name;
        nameInput.className = 'label-name';
        nameInput.style.flex = '1';
        nameInput.style.minWidth = '5em';
        nameInput.style.height = '2em';
        nameInput.style.marginRight = '0.5em';

        const saveBtn = document.createElement('button');
        saveBtn.title = 'Save';
        saveBtn.textContent = '✅';
        saveBtn.className = 'emoji-button';

        const discardBtn = document.createElement('button');
        discardBtn.title = 'Discard';
        discardBtn.textContent = '❌';
        discardBtn.className = 'emoji-button';

        row.appendChild(saveBtn);
        row.appendChild(discardBtn);
        row.appendChild(colorInput);
        row.appendChild(nameInput);

        saveBtn.onclick = async () => {
          const updatedLabel = {
            label_id: id,
            name: nameInput.value.trim(),
            color: colorInput.value
          };
          const response = await fetch('/api/labels/update', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(updatedLabel)
          });

          if (response.ok) refreshLabelList();
          else alert('Failed to update label');
        };

        discardBtn.onclick = () => refreshLabelList();
      };

      // SUBLABEL handler
      addChildBtn.onclick = () => {
        const childRow = document.createElement('div');
        childRow.className = 'label-row';
        childRow.style.paddingLeft = `${(indentLevel + 1) * 1.5}em`;

        const nameInput = document.createElement('input');
        nameInput.type = 'text';
        nameInput.placeholder = 'New label name';
        nameInput.className = 'label-name';
        nameInput.style.minWidth = '5em';
        nameInput.style.height = '2em';
        nameInput.style.marginRight = '0.5em';

        const colorInput = document.createElement('input');
        colorInput.type = 'color';
        colorInput.value = '#cccccc';
        colorInput.className = 'label-color';
        colorInput.style.width = '3em';
        colorInput.style.height = '3em';

        const saveBtn = document.createElement('button');
        saveBtn.title = 'Create';
        saveBtn.textContent = '✅';
        saveBtn.className = 'emoji-button';

        const cancelBtn = document.createElement('button');
        cancelBtn.title = 'Cancel';
        cancelBtn.textContent = '❌';
        cancelBtn.className = 'emoji-button';

        childRow.appendChild(saveBtn);
        childRow.appendChild(cancelBtn);
        childRow.appendChild(colorInput);
        childRow.appendChild(nameInput);
        row.after(childRow);

        saveBtn.onclick = async () => {
          const newName = nameInput.value.trim();
          if (!newName) {
            alert("Name required");
            return;
          }

          const newLabel = {
            name: newName,
            color: colorInput.value,
            parent_id: id
          };

          const response = await fetch('/api/labels/add', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(newLabel)
          });

          if (response.ok) {
            refreshLabelList();
          } else {
            alert('Failed to create label');
          }
        };

        cancelBtn.onclick = () => {
          childRow.remove();
        };
      };

      // Recurse into children
      label.children.forEach(child => renderLabel(child, indentLevel + 1));
    };

    data.labels.forEach(label => renderLabel(label, 0));
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
