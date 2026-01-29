import { reloadImage, clearAnnotations, deleteSelected, saveAnnotations } from './canvas/image.js';
import { events, EventTypes } from '/app-static/js/shared/events/index.js';
import { setTool } from './canvas/tools.js';
/**
 * @param {import('@image_tagging_types').ImageTaggingState} state
 */
export function init(state) {
  const container = document.getElementById('tab-canvas');
  const nav = document.createElement('nav');
  nav.id = 'toolbar';
  nav.style.display = 'flex';
  nav.style.gap = '0.5em';
  nav.style.marginBottom = '0.5em';
  nav.style.width = '100%';

  const buttons = [
    { label: 'Reload Image', onClick: () => reloadImage(state) },
    { label: 'Select', onClick: () => setTool(state, 'select'), dataTool: 'select' },
    { label: 'BBox', onClick: () => setTool(state, `bbox:`), dataTool: 'bbox' },
    { label: 'Clear Annotations', onClick: () => clearAnnotations(state) },
    { label: 'Delete Selected', onClick: () => deleteSelected(state) },
    { label: 'Save', onClick: () => saveAnnotations(state) }
  ];

  buttons.forEach(btn => {
    const b = document.createElement('button');
    b.textContent = btn.label;
    b.addEventListener('click', btn.onClick);
    if (btn.dataTool) {
      b.dataset.tool = btn.dataTool;
    }
    nav.appendChild(b);
  });

  container.prepend(nav);

  events.subscribe(EventTypes.CANVAS_TOOL_CHANGED, ({ tool }) => {
    document.querySelectorAll('#toolbar button[data-tool]').forEach(btn => {
      const isActive = btn.dataset.tool === tool || (tool.split(':')[0] === 'bbox' && btn.dataset.tool === 'bbox');
      btn.classList.toggle('selected', isActive);
    });
  });

}
