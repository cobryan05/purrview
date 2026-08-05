import { prefs } from '/app-static/js/shared/prefs.js';

/**
 * Attaches a drag-resize handler to handleEl that controls the width of sidebarEl.
 * The sidebar is assumed to be on the RIGHT side of the layout (dragging left = wider).
 * Persists the chosen width in localStorage when a storageKey is supplied.
 *
 * @param {HTMLElement} handleEl   Thin drag-handle element between the two panes
 * @param {HTMLElement} sidebarEl  The element whose width is controlled
 * @param {{
 *   storageKey?: string|null,
 *   defaultWidth?: number,
 *   min?: number,
 *   max?: number,
 * }} opts
 * @returns {{ destroy: () => void }}
 */
export function initSplitResize(handleEl, sidebarEl, {
  storageKey = null,
  defaultWidth = 240,
  min = 120,
  max = 600,
} = {}) {
  // Restore persisted width, falling back to defaultWidth.
  const stored = storageKey ? prefs.readRaw(storageKey) : null;
  const initWidth = stored ? parseInt(stored, 10) : defaultWidth;
  sidebarEl.style.width = `${Math.min(max, Math.max(min, initWidth))}px`;

  let dragging = false;
  let startX = 0;
  let startWidth = 0;

  function onMouseDown(e) {
    dragging = true;
    startX = e.clientX;
    startWidth = sidebarEl.getBoundingClientRect().width;
    handleEl.classList.add('split-resize-handle--active');
    document.body.style.cursor = 'col-resize';
    document.body.style.userSelect = 'none';
    e.preventDefault();
  }

  function onMouseMove(e) {
    if (!dragging) return;
    // Sidebar is on the right: dragging leftward (smaller clientX) increases width.
    const delta = startX - e.clientX;
    const newWidth = Math.min(max, Math.max(min, startWidth + delta));
    sidebarEl.style.width = `${newWidth}px`;
  }

  function onMouseUp() {
    if (!dragging) return;
    dragging = false;
    handleEl.classList.remove('split-resize-handle--active');
    document.body.style.cursor = '';
    document.body.style.userSelect = '';
    if (storageKey) {
      prefs.writeRaw(storageKey, Math.round(parseFloat(sidebarEl.style.width)));
    }
  }

  handleEl.addEventListener('mousedown', onMouseDown);
  document.addEventListener('mousemove', onMouseMove);
  document.addEventListener('mouseup', onMouseUp);

  return {
    destroy() {
      handleEl.removeEventListener('mousedown', onMouseDown);
      document.removeEventListener('mousemove', onMouseMove);
      document.removeEventListener('mouseup', onMouseUp);
    },
  };
}
