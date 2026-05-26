import { reloadImage, clearAnnotations, deleteSelected, saveAnnotations } from './canvas/image.js';
import { events, EventTypes } from '/app-static/js/shared/events/index.js';
import { setTool } from './canvas/tools.js';

const ALL_BUTTONS = ['reload', 'select', 'bbox', 'clear', 'delete', 'save'];

/**
 * Initialises a canvas toolbar into an explicit container element.
 *
 * @param {import('@image_tagging_types').ImageTaggingRuntime} runtime
 * @param {HTMLElement} container  - Element to render buttons into.
 * @param {{ buttons?: string[] }} [options]
 *   buttons: subset of ['reload','select','bbox','clear','delete','save'].
 *            Defaults to all buttons.
 * @returns {{ destroy: () => void }}
 */
export function init(runtime, container, options = {}) {
  const enabledButtons = options.buttons ?? ALL_BUTTONS;

  if (!container) {
    console.error('Toolbar container not found!');
    return { destroy: () => {} };
  }

  container.innerHTML = '';
  Object.assign(container.style, {
    display: 'flex',
    gap: '0.5em',
    padding: '5px',
    width: '100%',
    boxSizing: 'border-box',
  });

  const buttonDefs = [
    { key: 'reload', label: 'Reload',  onClick: () => reloadImage(runtime) },
    { key: 'select', label: 'Select',  onClick: () => setTool(runtime, 'select'), dataTool: 'select' },
    { key: 'bbox',   label: 'BBox',    onClick: () => setTool(runtime, 'bbox'),   dataTool: 'bbox' },
    { key: 'clear',  label: 'Clear',   onClick: () => clearAnnotations(runtime) },
    { key: 'delete', label: 'Delete',  onClick: () => deleteSelected(runtime) },
    { key: 'save',   label: 'Save',    onClick: () => saveAnnotations(runtime) },
  ];

  buttonDefs
    .filter(btn => enabledButtons.includes(btn.key))
    .forEach(btn => {
      const b = document.createElement('button');
      b.textContent = btn.label;
      b.style.flex = '1';
      b.addEventListener('click', btn.onClick);
      if (btn.dataTool) b.dataset.tool = btn.dataTool;
      container.appendChild(b);
    });

  const unsub = events.subscribe(EventTypes.CANVAS_TOOL_CHANGED, ({ tool }) => {
    container.querySelectorAll('button[data-tool]').forEach(btn => {
      const toolBase = tool.split(':')[0]; // handles 'bbox:label-uuid'
      btn.classList.toggle('selected', btn.dataset.tool === toolBase);
    });
  });

  return { destroy: unsub };
}
