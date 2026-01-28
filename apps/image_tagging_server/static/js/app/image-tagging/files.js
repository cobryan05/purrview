// files.js
import { loadImageAndMetadata } from './canvas/image.js';
import { fetchImageList } from '/app-static/js/shared/api/images.js';
import { Logger } from '/app-static/js/ui/utils/index.js';

export async function init() {
  await initFileBrowser();
}

let currentFileBrowserPath = '/';
let _initialized = false;

export async function initFileBrowser() {
  if (_initialized) return;
  _initialized = true;

  const browserContainer = document.getElementById('tab-files');
  if (!browserContainer) return;

  // Optional: load root folder on init
  await loadFileBrowser('/');
}

export async function loadFileBrowser(path) {
  currentFileBrowserPath = path;
  const browser = document.getElementById('tab-files');
  if (!browser) return;
  browser.innerHTML = 'Loading...';

  try {
    const files = await fetchImageList(path);
    browser.innerHTML = '';

    // Breadcrumb
    const breadcrumb = document.createElement('div');
    breadcrumb.style.marginBottom = '0.5em';
    let accum = '';
    breadcrumb.appendChild(makeBreadcrumbLink('/', '/'));
    path.split('/').filter(Boolean).forEach(part => {
      accum += '/' + part;
      breadcrumb.appendChild(document.createTextNode(' / '));
      breadcrumb.appendChild(makeBreadcrumbLink(part, accum));
    });
    browser.appendChild(breadcrumb);

    // File list
    const list = document.createElement('ul');
    list.style.listStyle = 'none';
    list.style.padding = 0;

    files.forEach(entry => {
      const li = document.createElement('li');
      li.style.margin = '0.2em 0';

      const a = document.createElement('a');
      a.href = '#';
      a.textContent = entry.name;
      a.onclick = e => {
        e.preventDefault();
        if (entry.type === 'dir') loadFileBrowser(entry.path);
        else loadImageByName(entry.name, path);
      };

      li.innerHTML = entry.type === 'dir' ? '📁 ' : '🖼️ ';
      li.appendChild(a);
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
  a.onclick = e => {
    e.preventDefault();
    loadFileBrowser(path);
  };
  return a;
}

async function loadImageByName(filename, dirPath) {
  const imageName = (dirPath === '/' ? '' : dirPath + '/') + filename;

  // Activate Canvas tab in workspace
  document.querySelectorAll('.main-tab-button').forEach(t => t.classList.remove('active'));
  const mainTab = document.querySelector('.main-tab-button[data-tab="tab-image-tagging"]');
  mainTab?.classList.add('active');

  document.querySelectorAll('.main-tab-pane').forEach(p => p.classList.remove('active'));
  document.getElementById('tab-image-tagging')?.classList.add('active');

  const workspaceHeader = document.querySelector('#workspace .workspace-tab-header');
  workspaceHeader.querySelectorAll('.workspace-tab-button').forEach(t => t.classList.remove('active'));
  const canvasBtn = workspaceHeader.querySelector('.workspace-tab-button[data-tab="tab-canvas"]');
  canvasBtn?.classList.add('active');

  document.querySelectorAll('#workspace .workspace-tab-pane').forEach(p => p.classList.remove('active'));
  document.getElementById('tab-canvas')?.classList.add('active');

  await new Promise(requestAnimationFrame);
  loadImageAndMetadata(imageName);
}

export { _initialized as isInitialized };
