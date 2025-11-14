// canvas_ui_main.js

import { addRecognizedBoxes, loadImageAndMetadata } from '/app-static/js/canvas/io.js';
import { getLayer, } from '/app-static/js/canvas/state.js';
import { setTool } from '/app-static/js/canvas/tools.js';
import { renderClassList } from '/app-static/js/ui/classes.js';
import { renderModelClassAssignments } from '/app-static/js/ui/models.js';
import { renderSourceManager } from '/app-static/js/ui/sources.js';
import { renderActiveTasks, renderTaskConfigs } from '/app-static/js/ui/tasks.js';
import { Logger, clearModelsCache, fetchImageList, fetchModelsList, toast } from '/app-static/js/ui/utils/index.js';
// --- Model list UI ---

/**
 * Fetches the list of available models from the backend API,
 * populates the UI with radio buttons for model selection,
 * and selects the first model by default.
 */
export async function refreshModelList() {
  try {
    await clearModelsCache();
    const models = await fetchModelsList();

    const modelListContainer = document.getElementById('model-list');
    if (!modelListContainer) {
      console.error("modelListContainer is null");
      return;
    }

    modelListContainer.innerHTML = '';

    models.forEach(model => {
      const cls = document.createElement('class');
      cls.style.display = 'block';

      const radio = document.createElement('input');
      radio.type = 'radio';
      radio.name = 'model';
      radio.value = model;

      cls.appendChild(radio);
      cls.appendChild(document.createTextNode(` ${model}`));
      modelListContainer.appendChild(cls);
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
    const img = background.image();
    if (!img.src.startsWith("data:")) {
      throw new Error("Image is not a base64 data URI");
    }
    const base64String = img.src.split(',')[1]; // strip "data:image/png;base64,"

    // Prepare JSON payload
    const payload = {
      model_name: modelName,
      conf_thresh: 0.25,
      return_annotated: false,
      pin_id: null, // optional
      image_base64: base64String,
    };

    toast("Sending recognition request...");
    // Send recognition request
    const response = await fetch('/api/recognize', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      throw new Error(`Recognition request failed: ${response.statusText}`);
    }

    const result = await response.json();
    Logger.debug('Recognize response:', result);

    if (result.status != 'success') {
      throw new Error(`Recognition failed: ${result.message}`);
    }
    toast(`${result.detections.length} objects detected`);
    addRecognizedBoxes(result.detections);
  } catch (err) {
    Logger.error('Failed to recognize image:', err);
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
    const fileList = await fetchImageList(path);
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

    fileList.forEach(entry => {
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

        // If "Settings" tab is opened, trigger the Label Management tab click
        if (tabName === 'outer-tab-settings') {
          const defaultTabBtn = document.querySelector('#outer-tab-settings .tab-button[data-tab="tab-label-management"]');
          if (defaultTabBtn) {
            // Only trigger if no tab is currently active inside this header
            const primaryHeader = defaultTabBtn.closest('.primary-tab-header');
            const activeInner = primaryHeader.querySelector('.tab-button.active');
            if (!activeInner) {
              defaultTabBtn.click(); // trigger the click handler
            }
          }
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
          renderClassList({ target: "classes-list" });
          renderModelClassAssignments({ target: "model-classes-box" });
        } else if (tabName === 'tab-source-management') {
          renderSourceManager({ target: "source-management-box" });
        } else if (tabName === 'tab-files' && !fileBrowserInitialized) {
          loadFileBrowser(currentFileBrowserPath);
          fileBrowserInitialized = true;
        } else if (tabName === 'tab-task-management') {
          const refreshTaskTab = () => {
            renderActiveTasks({ target: 'active-tasks-list', refresh: refreshTaskTab });
            renderTaskConfigs({
              target: 'task-config-list',
              refresh: refreshTaskTab
            });
          };
          refreshTaskTab();
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

        if (tabName === 'tab-classes-tool') {
          renderClassList({ target: "classes-tool-list", editable: false, clickable: true, onSelectCallback: (uuid) => { setTool(`bbox:${uuid}`) } });
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