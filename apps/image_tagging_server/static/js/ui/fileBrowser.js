import { fetchImageList } from '/app-static/js/shared/api/images.js';

/**
 * @typedef {import('@ui_types').FileNavigation} FileNavigation
 */

/**
 * FileBrowser renders a directory listing and handles user interaction.
 * It can optionally update a FileNavigation instance when files or directories are clicked.
 */
export class FileBrowser {
  /**
   * @param {{
   *   container: HTMLElement,
   *   initialPath?: string,
   *   filter?: (entry: {name: string, type: string, path: string}) => boolean,
   *   fileNavigation?: { setDirectory: (dir: string, files: string[]) => void, select: (file: string) => void },
   *   onFileClick?: (entry: any, currentPath: string) => void,
   *   onDirClick?: (path: string) => void,
   * }} options
   */
  constructor({ container, initialPath = '/', filter, fileNavigation, onFileClick, onDirClick }) {
    this.container = container;
    this.currentPath = initialPath;
    this.filter = filter ?? (() => true);
    this.fileNavigation = fileNavigation; // injected navigation state
    this.onFileClick = onFileClick;
    this.onDirClick = onDirClick;
    this._entries = [];
  }

  /** Initialize the browser by loading the initial path */
  async init() {
    await this.load(this.currentPath);
  }

  /**
   * Load a directory and render its contents
   * @param {string} path
   */
  async load(path) {
    this.currentPath = path;
    this.container.innerHTML = 'Loading...';
    try {
      const entries = await fetchImageList(path);
      this._entries = entries.filter(this.filter);
      this.container.innerHTML = '';

      // Update navigation if injected
      if (this.fileNavigation) {
        this.fileNavigation.setDirectory(path, this._entries.map(e => e.name));
      }

      this.renderBreadcrumb(path);
      this.renderList(this._entries);
    } catch (err) {
      this.container.innerHTML = 'Failed to load files.';
      console.error(err);
    }
  }

  /** Returns the currently loaded entries */
  getEntries() {
    return this._entries;
  }

  /**
   * Render the list of entries in the container
   * @param {Array<{name:string,type:string,path:string}>} entries
   */
  renderList(entries) {
    const list = document.createElement('ul');
    list.style.listStyle = 'none';
    list.style.padding = 0;

    entries.forEach(entry => {
      const li = document.createElement('li');
      li.style.margin = '0.2em 0';

      const link = document.createElement('a');
      link.href = '#';
      link.textContent = entry.name;
      link.onclick = e => {
        e.preventDefault();

        if (entry.type === 'dir') {
          this.onDirClick?.(entry.path);
          this.load(entry.path);
        } else {
          this.fileNavigation?.select(entry.name);
          this.onFileClick?.(entry, this.currentPath);
        }
      };

      li.textContent = entry.type === 'dir' ? '📁 ' : '🖼️ ';
      li.appendChild(link);
      list.appendChild(li);
    });

    this.container.appendChild(list);
  }

  /**
   * Utility for breadcrumb navigation
   * @param {string} label
   * @param {string} path
   */
  makeBreadcrumbLink(label, path) {
    const a = document.createElement('a');
    a.href = '#';
    a.textContent = label;
    a.onclick = e => {
      e.preventDefault();
      this.load(path);
    };
    return a;
  }

  /**
   * Render a breadcrumb for the current path
   * @param {string} path
   */
  renderBreadcrumb(path) {
    const breadcrumb = document.createElement('div');
    breadcrumb.style.marginBottom = '0.5em';

    let accum = '';
    breadcrumb.appendChild(this.makeBreadcrumbLink('/', '/'));

    path.split('/').filter(Boolean).forEach(part => {
      accum += '/' + part;
      breadcrumb.appendChild(document.createTextNode(' / '));
      breadcrumb.appendChild(this.makeBreadcrumbLink(part, accum));
    });

    this.container.appendChild(breadcrumb);
  }
}
