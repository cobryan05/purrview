import { fetchImageList, createDirectory } from '/app-static/js/shared/api/images.js';
import { toast } from '/app-static/js/ui/utils/logging.js';

/**
 * @typedef {import('@ui_types').FileNavigation} FileNavigation
 * @typedef {{ name: string, path: string, isDir: boolean, uuid?: string|null }} FileEntry
 */

/**
 * FileBrowser renders a navigable directory listing.
 *
 * Modes
 *   'file'  (default) — clicking a file fires onFileClick; directories navigate in-place.
 *   'dir'             — shows a "Select This Directory" button; onDirSelect fires when clicked.
 *
 * Optional features (all off by default)
 *   showThumbnails  — enables a grid/list view toggle; image files render as thumbnails.
 *   showDbStatus    — shows a "DB" badge on entries whose uuid is populated from the server.
 *   allowCreateDir  — shows an inline "New Folder" affordance in the toolbar.
 */
export class FileBrowser {
  /**
   * @param {{
   *   container: HTMLElement,
   *   initialPath?: string,
   *   filter?: (entry: FileEntry) => boolean,
   *   fileNavigation?: FileNavigation,
   *   onFileClick?: (entry: FileEntry, currentPath: string) => void,
   *   onDirClick?: (path: string) => void,
   *   onDirSelect?: (path: string) => void,
   *   mode?: 'file' | 'dir',
   *   showThumbnails?: boolean,
   *   showDbStatus?: boolean,
   *   allowCreateDir?: boolean,
   * }} options
   */
  constructor({
    container,
    initialPath = '/',
    filter,
    fileNavigation,
    onFileClick,
    onDirClick,
    onDirSelect,
    mode = 'file',
    showThumbnails = false,
    showDbStatus = false,
    allowCreateDir = false,
  }) {
    this.container = container;
    this.currentPath = initialPath;
    this.filter = filter ?? (() => true);
    this.fileNavigation = fileNavigation;
    this.onFileClick = onFileClick;
    this.onDirClick = onDirClick;
    this.onDirSelect = onDirSelect;
    this.mode = mode;
    this.showThumbnails = showThumbnails;
    this.showDbStatus = showDbStatus;
    this.allowCreateDir = allowCreateDir;
    this._viewMode = 'list'; // 'list' | 'grid'
    this._entries = [];
  }

  /** Initialize the browser by loading the initial path. */
  async init() {
    await this.load(this.currentPath);
  }

  /**
   * Load a directory and re-render.
   * @param {string} path
   */
  async load(path) {
    this.currentPath = path;
    this.container.innerHTML = '<div class="fb-loading">Loading…</div>';
    try {
      const entries = await fetchImageList(path, '*', false, this.showDbStatus);
      this._entries = entries.filter(this.filter);
      if (this.fileNavigation) {
        this.fileNavigation.setDirectory(path, this._entries.map(e => e.name));
      }
      this.rerender();
    } catch (err) {
      this.container.innerHTML = '<div class="fb-error">Failed to load directory.</div>';
      console.error(err);
    }
  }

  /** Re-render current entries (e.g. after external navigation state changes). */
  rerender() {
    this.container.innerHTML = '';
    const wrapper = document.createElement('div');
    wrapper.className = 'fb-container';

    this._renderToolbar(wrapper);
    this._renderBreadcrumb(wrapper);

    if (this._viewMode === 'grid' && this.showThumbnails) {
      this._renderGrid(wrapper, this._entries);
    } else {
      this._renderList(wrapper, this._entries);
    }

    this.container.appendChild(wrapper);
  }

  /** Returns the currently loaded entries. */
  getEntries() {
    return this._entries;
  }

  // ─── private helpers ────────────────────────────────────────────────────────

  _renderToolbar(parent) {
    const toolbar = document.createElement('div');
    toolbar.className = 'fb-toolbar';

    // Dir-select mode: prominent "Select This Directory" button
    if (this.mode === 'dir' && this.onDirSelect) {
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'fb-select-btn';
      btn.textContent = 'Select This Directory';
      btn.addEventListener('click', () => this.onDirSelect(this.currentPath));
      toolbar.appendChild(btn);
    }

    // New Folder affordance
    if (this.allowCreateDir) {
      const newBtn = document.createElement('button');
      newBtn.type = 'button';
      newBtn.className = 'fb-new-folder-btn';
      newBtn.textContent = '+ New Folder';

      const inputRow = document.createElement('div');
      inputRow.className = 'fb-new-folder-input';
      inputRow.style.display = 'none';

      const nameInput = document.createElement('input');
      nameInput.type = 'text';
      nameInput.placeholder = 'Folder name…';

      const saveBtn = document.createElement('button');
      saveBtn.type = 'button';
      saveBtn.textContent = 'Create';

      const cancelBtn = document.createElement('button');
      cancelBtn.type = 'button';
      cancelBtn.textContent = 'Cancel';
      cancelBtn.style.color = 'var(--pico-muted-color, #888)';

      inputRow.append(nameInput, saveBtn, cancelBtn);

      newBtn.addEventListener('click', () => {
        newBtn.style.display = 'none';
        inputRow.style.display = 'flex';
        nameInput.focus();
      });

      const resetForm = () => {
        nameInput.value = '';
        nameInput.style.outline = '';
        inputRow.style.display = 'none';
        newBtn.style.display = '';
      };

      cancelBtn.addEventListener('click', resetForm);

      const doCreate = async () => {
        const name = nameInput.value.trim();
        if (!name) return;
        if (name.includes('/') || name.includes('\\')) {
          nameInput.style.outline = '2px solid red';
          return;
        }
        saveBtn.disabled = true;
        try {
          const newPath = this.currentPath === '/' ? `/${name}` : `${this.currentPath}/${name}`;
          await createDirectory(newPath);
          resetForm();
          await this.load(this.currentPath);
        } catch (err) {
          nameInput.style.outline = '2px solid red';
          toast(err.message || 'Failed to create directory', 5000, 'error');
        } finally {
          saveBtn.disabled = false;
        }
      };

      saveBtn.addEventListener('click', doCreate);
      nameInput.addEventListener('keydown', e => {
        if (e.key === 'Enter') doCreate();
        if (e.key === 'Escape') cancelBtn.click();
        nameInput.style.outline = '';
      });

      toolbar.appendChild(newBtn);
      toolbar.appendChild(inputRow);
    }

    // View toggle (only when thumbnails feature is enabled)
    if (this.showThumbnails) {
      const toggle = document.createElement('div');
      toggle.className = 'fb-view-toggle';

      const listBtn = document.createElement('button');
      listBtn.type = 'button';
      listBtn.textContent = '☰';
      listBtn.title = 'List view';
      if (this._viewMode === 'list') listBtn.classList.add('active');
      listBtn.addEventListener('click', () => { this._viewMode = 'list'; this.rerender(); });

      const gridBtn = document.createElement('button');
      gridBtn.type = 'button';
      gridBtn.textContent = '⊞';
      gridBtn.title = 'Grid view';
      if (this._viewMode === 'grid') gridBtn.classList.add('active');
      gridBtn.addEventListener('click', () => { this._viewMode = 'grid'; this.rerender(); });

      toggle.append(listBtn, gridBtn);
      toolbar.appendChild(toggle);
    }

    if (toolbar.childElementCount > 0) {
      parent.appendChild(toolbar);
    }
  }

  _renderBreadcrumb(parent) {
    const bc = document.createElement('div');
    bc.className = 'fb-breadcrumb';
    bc.appendChild(this._makeCrumbLink('/', '/'));

    let accum = '';
    this.currentPath.split('/').filter(Boolean).forEach(part => {
      accum += '/' + part;
      bc.appendChild(document.createTextNode(' / '));
      bc.appendChild(this._makeCrumbLink(part, accum));
    });

    parent.appendChild(bc);
  }

  _makeCrumbLink(label, path) {
    const a = document.createElement('a');
    a.href = '#';
    a.textContent = label;
    a.addEventListener('click', e => { e.preventDefault(); this.load(path); });
    return a;
  }

  _renderList(parent, entries) {
    const ul = document.createElement('ul');
    ul.className = 'fb-list';

    entries.forEach(entry => {
      const li = document.createElement('li');
      li.className = 'fb-entry';
      if (this.fileNavigation?.current() === entry.name) li.classList.add('fb-entry--active');

      const icon = document.createElement('span');
      icon.className = 'fb-entry__icon';
      icon.textContent = entry.isDir ? '📁' : '🖼️';

      // Name + badge sit together in a flex wrapper so the badge follows the text
      const nameRow = document.createElement('span');
      nameRow.className = 'fb-entry__name-row';

      const name = document.createElement('span');
      name.className = 'fb-entry__name';
      name.textContent = entry.name;
      nameRow.appendChild(name);

      if (this.showDbStatus && entry.uuid) {
        const badge = document.createElement('span');
        badge.className = 'fb-entry__db-badge';
        const count = entry.bboxCount ?? entry.bbox_count;
        badge.textContent = count > 0 ? `${count} box${count === 1 ? '' : 'es'}` : 'DB';
        badge.title = count > 0
          ? `In database — ${count} bounding box${count === 1 ? '' : 'es'}`
          : 'Registered in database (no boxes)';
        nameRow.appendChild(badge);
      }

      li.append(icon, nameRow);

      li.addEventListener('click', () => this._handleEntryClick(entry));
      ul.appendChild(li);
    });

    parent.appendChild(ul);
  }

  _renderGrid(parent, entries) {
    const grid = document.createElement('div');
    grid.className = 'fb-grid';

    entries.forEach(entry => {
      const cell = document.createElement('div');
      cell.className = 'fb-grid-cell';
      if (this.fileNavigation?.current() === entry.name) cell.classList.add('fb-grid-cell--active');

      if (entry.isDir) {
        const icon = document.createElement('div');
        icon.className = 'fb-grid-cell__icon';
        icon.textContent = '📁';
        cell.appendChild(icon);
      } else {
        const img = document.createElement('img');
        img.className = 'fb-grid-cell__thumb';
        img.loading = 'lazy';
        img.src = `/api/images/${encodeURIComponent(entry.path)}/thumbnail?width=80`;
        img.alt = entry.name;
        cell.appendChild(img);
      }

      const label = document.createElement('div');
      label.className = 'fb-grid-cell__name';
      label.textContent = entry.name;
      label.title = entry.name;
      cell.appendChild(label);

      if (this.showDbStatus && entry.uuid) {
        const badge = document.createElement('div');
        badge.className = 'fb-grid-cell__db-badge';
        const count = entry.bboxCount ?? entry.bbox_count;
        badge.textContent = count > 0 ? `${count}` : '✓';
        badge.title = count > 0
          ? `In database — ${count} bounding box${count === 1 ? '' : 'es'}`
          : 'Registered in database (no boxes)';
        cell.appendChild(badge);
      }

      cell.addEventListener('click', () => this._handleEntryClick(entry));
      grid.appendChild(cell);
    });

    parent.appendChild(grid);
  }

  _handleEntryClick(entry) {
    if (entry.isDir) {
      this.onDirClick?.(entry.path);
      this.load(entry.path);
    } else {
      this.fileNavigation?.select(entry.name);
      this.onFileClick?.(entry, this.currentPath);
    }
  }
}
