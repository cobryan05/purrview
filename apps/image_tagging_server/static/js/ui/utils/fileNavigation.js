/**
 * Class representing file navigation state.
 * Can be used anywhere you need to keep track of the current file in a directory,
 * independent of UI components like a FileBrowser.
 */
export class FileNavigation {
  /** @type {string} */
  _dir = '/';

  /** @type {string[]} */
  _files = [];

  /** @type {number} */
  _index = -1;

  /**
   * Create a FileNavigation instance
   * @param {string[]} [files=[]] - Initial list of file names
   * @param {string} [dir='/'] - Initial directory
   */
  constructor(files = [], dir = '/') {
    this.setDirectory(dir, files);
  }

  /**
   * Set a new directory and file list
   * @param {string} dir - Directory path
   * @param {string[]} files - Array of file names
   */
  setDirectory(dir, files) {
    this._dir = dir;
    this._files = files.filter(f => f.match(/\.(jpg|jpeg|png)$/i));
    this._index = this._files.length ? 0 : -1;
  }

  /**
   * Select a specific file by name
   * @param {string} file - Name of the file to select
   */
  select(file) {
    const idx = this._files.indexOf(file);
    if (idx >= 0) this._index = idx;
  }

  /**
   * Move to the next file in the list
   * @returns {string|null} The selected file name, or null if none
   */
  next() {
    if (!this._files.length) return null;
    this._index = Math.min(this._index + 1, this._files.length - 1);
    return this.current();
  }

  /**
   * Move to the previous file in the list
   * @returns {string|null} The selected file name, or null if none
   */
  prev() {
    if (!this._files.length) return null;
    this._index = Math.max(this._index - 1, 0);
    return this.current();
  }

  /**
   * Get the currently selected file name
   * @returns {string|null} The current file, or null if none
   */
  current() {
    if (this._index < 0 || !this._files.length) return null;
    return this._files[this._index];
  }

  /**
   * Get the full path to the currently selected file
   * @returns {string|null} Full path including directory, or null if none
   */
  currentPath() {
    const file = this.current();
    if (!file) return null;
    return this._dir === '/' ? file : `${this._dir}/${file}`;
  }

  /**
   * Get a copy of the current file list
   * @returns {string[]}
   */
  getFiles() {
    return [...this._files];
  }

  /**
   * Get the current directory path
   * @returns {string}
   */
  getDirectory() {
    return this._dir;
  }
}
