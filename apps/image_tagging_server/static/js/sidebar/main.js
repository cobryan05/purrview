// canvas_ui_main.js

import { addRecognizedBoxes, loadImageAndMetadata } from '/app-static/js/canvas/io.js';
import { getLayer, } from '/app-static/js/canvas/state.js';
import { renderLabelList, renderModelLabelAssignments, renderSourceManager } from '/app-static/js/ui/renderers.js';
import { setTool } from '/app-static/js/canvas/tools.js';

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
          renderSourceManager({ target: "source-management-box" });
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
          renderLabelList({ target: "labels-tool-list", editable: false, clickable: true, onSelectCallback: (uuid) => { setTool(`bbox:${uuid}`) } });
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