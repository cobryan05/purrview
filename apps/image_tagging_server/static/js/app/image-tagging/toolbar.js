import { reloadImage, clearAnnotations, deleteSelected, saveAnnotations } from './canvas/image.js';
import { events, EventTypes } from '/app-static/js/shared/events/index.js';
import { setTool } from './canvas/tools.js';
/**
 * @param {import('@image_tagging_types').ImageTaggingRuntime} runtime
 */
export function init(runtime) {
  const pane = runtime.imageTaggingPane;
  const nav = pane?.querySelector('.canvas-toolbar');

  if (!nav) {
    console.error("Toolbar element not found!");
    return;
  }

  nav.innerHTML = '';
  Object.assign(nav.style, {
    display: 'flex',
    gap: '0.5em',
    padding: '5px',
    width: '100%',
    boxSizing: 'border-box'
  });

  const buttons = [
    { label: 'Reload', onClick: () => reloadImage(runtime) },
    { label: 'Select', onClick: () => setTool(runtime, 'select'), dataTool: 'select' },
    { label: 'BBox', onClick: () => setTool(runtime, 'bbox'), dataTool: 'bbox' },
    { label: 'Clear', onClick: () => clearAnnotations(runtime) },
    { label: 'Delete', onClick: () => deleteSelected(runtime) },
    { label: 'Save', onClick: () => saveAnnotations(runtime) }
  ];

  buttons.forEach(btn => {
    const b = document.createElement('button');
    b.textContent = btn.label;
    b.style.flex = "1";
    b.addEventListener('click', btn.onClick);
    if (btn.dataTool) b.dataset.tool = btn.dataTool;
    nav.appendChild(b);
  });

  events.subscribe(EventTypes.CANVAS_TOOL_CHANGED, ({ tool }) => {
    nav.querySelectorAll('button[data-tool]').forEach(btn => {
      const toolBase = tool.split(':')[0]; // handles 'bbox:label-uuid'
      const isActive = btn.dataset.tool === toolBase;
      btn.classList.toggle('selected', isActive);
    });
  });
}
