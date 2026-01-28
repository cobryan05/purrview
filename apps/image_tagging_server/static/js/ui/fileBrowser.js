// ui/fileBrowser.js
import { fetchImageList } from '/app-static/js/shared/api/images.js';

export class FileBrowser {
  /**
   * @param {{
   *   container: HTMLElement,
   *   initialPath?: string,
   *   onFileClick?: (entry, currentPath) => void,
   *   onDirClick?: (path) => void,
   *   filter?: (entry) => boolean,
   * }} options
   */
  constructor(options) {
    this.container = options.container;
    this.currentPath = options.initialPath ?? '/';
    this.onFileClick = options.onFileClick;
    this.onDirClick = options.onDirClick;
    this.filter = options.filter ?? (() => true);
  }

  async init() {
    await this.load(this.currentPath);
  }

  async load(path) {
    this.currentPath = path;
    this.container.innerHTML = 'Loading...';

    try {
      const entries = await fetchImageList(path);
      this.container.innerHTML = '';

      this.renderBreadcrumb(path);
      this.renderList(entries.filter(this.filter));
    } catch (err) {
      this.container.innerHTML = 'Failed to load files.';
      console.error(err);
    }
  }

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
          this.onFileClick?.(entry, this.currentPath);
        }
      };

      li.textContent = entry.type === 'dir' ? '📁 ' : '🖼️ ';
      li.appendChild(link);
      list.appendChild(li);
    });

    this.container.appendChild(list);
  }

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
}
