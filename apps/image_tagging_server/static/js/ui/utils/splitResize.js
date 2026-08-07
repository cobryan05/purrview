import { prefs } from '/app-static/js/shared/prefs.js';

/**
 * Attaches a drag-resize handler to handleEl that controls the width of sidebarEl.
 * Persists the chosen width in localStorage when a storageKey is supplied.
 *
 * The resized pane sits on the RIGHT by default (dragging left = wider), which is
 * the Image Tagging and Review sidebars. Pass `side: 'left'` for a pane on the
 * left of its handle — the list panel of a list-editor screen — where dragging
 * right is what widens it.
 *
 * `sidebarEl` must not have a fixed `flex-basis`, or the inline width this sets
 * is ignored: give it `flex: 0 0 auto` and let width govern.
 *
 * @param {HTMLElement} handleEl   Thin drag-handle element between the two panes
 * @param {HTMLElement} sidebarEl  The element whose width is controlled
 * @param {{
 *   storageKey?: string|null,
 *   defaultWidth?: number,
 *   min?: number,
 *   max?: number,
 *   side?: 'left' | 'right',
 *   signal?: AbortSignal|null,
 * }} opts
 * @returns {{ destroy: () => void }}
 */
export function initSplitResize(handleEl, sidebarEl, {
  storageKey = null,
  defaultWidth = 240,
  min = 120,
  max = 600,
  side = 'right',
  // Two of the three listeners live on `document`, so a screen that re-renders
  // (every tab switch) stacks a fresh set on top of the old ones unless the
  // caller can tear them down. Callers that already own an AbortController for
  // re-render teardown pass its signal here.
  signal = null,
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
    // Right-hand pane: dragging leftward (smaller clientX) increases width.
    // Left-hand pane: the reverse.
    const delta = side === 'left' ? e.clientX - startX : startX - e.clientX;
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

  const listenerOpts = signal ? { signal } : undefined;
  handleEl.addEventListener('mousedown', onMouseDown, listenerOpts);
  document.addEventListener('mousemove', onMouseMove, listenerOpts);
  document.addEventListener('mouseup', onMouseUp, listenerOpts);

  return {
    destroy() {
      handleEl.removeEventListener('mousedown', onMouseDown);
      document.removeEventListener('mousemove', onMouseMove);
      document.removeEventListener('mouseup', onMouseUp);
    },
  };
}
