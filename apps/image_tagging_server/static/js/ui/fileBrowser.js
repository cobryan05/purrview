import { fetchImageList, createDirectory } from '/app-static/js/shared/api/images.js';
import { toast } from '/app-static/js/ui/utils/logging.js';
import { attachFileEntryHoverPreview } from '/app-static/js/ui/thumbnailHoverPreview.js';
import { createEntryView } from '/app-static/js/ui/entryView.js';

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
    this._entries = [];
    this._offset = 0;
    this._hasMore = false;

    this._view = createEntryView({
      storageKey: 'fb-view',
      defaultView: 'list',
      showToggle: this.showThumbnails,
      listClass: 'fb-list',
      gridClass: 'fb-grid',
      renderRow: (entry) => this._makeListEntry(entry),
      renderCell: (entry) => this._makeGridCell(entry),
      renderLeading: (view) => this._makeNavLeading(view),
    });
  }

  static get PAGE_SIZE() { return 200; }

  /** Initialize the browser by loading the initial path. */
  async init() {
    await this.load(this.currentPath);
  }

  /**
   * Load a directory from the beginning and re-render.
   * @param {string} path
   */
  async load(path) {
    this.currentPath = path;
    this._entries = [];
    this._offset = 0;
    this._hasMore = false;
    this.fileNavigation?.setDirectory(path, []);
    this.container.innerHTML = '<div class="fb-loading">Loading…</div>';
    await this._loadPage();
  }

  /**
   * Load the next page of entries if more are available.
   * @returns {Promise<boolean>} True if a page was fetched, false if already at end.
   */
  async loadNextPage() {
    if (!this._hasMore) return false;
    await this._loadPage();
    return true;
  }

  /**
   * Fetch the next page of entries and update the view.
   * First page: full rerender (toolbar, breadcrumb, entries, load-more button).
   * Subsequent pages: append entries without rebuilding the toolbar or breadcrumb,
   * and surgically update the load-more button instead of replacing it.
   * @private
   */
  async _loadPage() {
    const isAppend = this._offset > 0;
    try {
      const { files, hasMore } = await fetchImageList(
        this.currentPath, '*', false, this.showDbStatus,
        { offset: this._offset, limit: FileBrowser.PAGE_SIZE },
      );
      const filtered = files.filter(this.filter);
      this._entries.push(...filtered);
      this._hasMore = hasMore;
      this._offset += files.length;
      this.fileNavigation?.appendFiles(filtered.map(e => e.name));
      if (isAppend) {
        // Append-only update: avoid rebuilding toolbar/breadcrumb on every page load.
        this._view.appendEntries(filtered);
        const footer = this.container.querySelector('.fb-load-more');
        if (hasMore) {
          // Re-enable the existing button; _renderLoadMore already wired the click handler.
          const btn = footer?.querySelector('.fb-load-more-btn');
          if (btn) { btn.disabled = false; btn.textContent = 'Load more…'; }
        } else {
          footer?.remove();
        }
      } else {
        this.rerender();
      }
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
    wrapper.appendChild(this._view.element);
    this._renderBreadcrumb(wrapper);
    this._view.setEntries(this._entries);
    wrapper.appendChild(this._view.layout);

    if (this._hasMore) {
      this._renderLoadMore(wrapper);
    }

    this.container.appendChild(wrapper);
  }

  /** Returns the currently loaded entries. */
  getEntries() {
    return this._entries;
  }

  /**
   * Scroll the active entry into view and briefly flash it.
   * No-op if there is no active entry in the current render.
   */
  scrollToActive() {
    this._view.scrollToActive('.fb-entry--active, .fb-grid-cell--active');
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

    if (toolbar.childElementCount > 0) {
      parent.appendChild(toolbar);
    }
  }

  /** @returns {string|null} Parent directory path, or null when already at root. */
  _parentPath() {
    if (this.currentPath === '/') return null;
    const parts = this.currentPath.split('/').filter(Boolean);
    parts.pop();
    return parts.length === 0 ? '/' : '/' + parts.join('/');
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

  /** @private Append `.` and `..` virtual nav entries using the supplied factory. */
  _appendNavItems(container, makeItem) {
    if (this.mode === 'dir' && this.onDirSelect) {
      container.appendChild(makeItem('.', () => this.onDirSelect(this.currentPath)));
    }
    const parentPath = this._parentPath();
    if (parentPath !== null) {
      container.appendChild(makeItem('..', () => this.load(parentPath)));
    }
  }

  /** @private Build the list-view row for an entry (passed to `entryView` as `renderRow`). */
  _makeListEntry(entry) {
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
    attachFileEntryHoverPreview(name, entry);
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
    return li;
  }

  /** @private Build the grid-view cell for an entry (passed to `entryView` as `renderCell`). */
  _makeGridCell(entry) {
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
      img.src = `/api/images/${encodeURIComponent(entry.path)}/thumbnail?width=120`;
      img.alt = entry.name;
      cell.appendChild(img);
    }

    const label = document.createElement('div');
    label.className = 'fb-grid-cell__name';
    label.textContent = entry.name;
    label.title = entry.name;
    cell.appendChild(label);

    // Hovering anywhere over the cell (including the thumbnail) shows a larger
    // annotated preview. Attaching to the cell rather than just the label means
    // the hover fires when the cursor is over the image area, not only the text.
    attachFileEntryHoverPreview(cell, entry);

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
    return cell;
  }

  /**
   * @private Build the `.`/`..` virtual nav items for the current view mode,
   * wrapped in a fragment so `entryView` can prepend them to the layout.
   * @param {'list'|'grid'} view
   * @returns {DocumentFragment|null}
   */
  _makeNavLeading(view) {
    const fragment = document.createDocumentFragment();
    const makeItem = view === 'grid'
      ? (label, cb) => this._makeNavCell(label, cb)
      : (label, cb) => this._makeNavEntry(label, cb);
    this._appendNavItems(fragment, makeItem);
    return fragment.childElementCount > 0 ? fragment : null;
  }

  /** @private Build a navigation row/cell for `.` or `..` entries. */
  _makeNavItem(outerTag, prefix, innerTag, label, onClick) {
    const outer = document.createElement(outerTag);
    outer.className = `${prefix} ${prefix}--nav`;
    const icon = document.createElement(innerTag);
    icon.className = `${prefix}__icon`;
    icon.textContent = '📁';
    const name = document.createElement(innerTag);
    name.className = `${prefix}__name`;
    name.textContent = label;
    outer.append(icon, name);
    outer.addEventListener('click', onClick);
    return outer;
  }

  /** Build a list-view navigation row (`.` or `..`). */
  _makeNavEntry(label, onClick) { return this._makeNavItem('li', 'fb-entry', 'span', label, onClick); }

  /** Build a grid-view navigation cell (`.` or `..`). */
  _makeNavCell(label, onClick)  { return this._makeNavItem('div', 'fb-grid-cell', 'div', label, onClick); }

  _renderLoadMore(parent) {
    const footer = document.createElement('div');
    footer.className = 'fb-load-more';
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'fb-load-more-btn';
    btn.textContent = `Load more…`;
    btn.addEventListener('click', async () => {
      btn.disabled = true;
      btn.textContent = 'Loading…';
      await this._loadPage();
    });
    footer.appendChild(btn);
    parent.appendChild(footer);
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
