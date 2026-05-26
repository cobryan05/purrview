import { setTool } from './tools.js';
import { fetchLabels } from '/app-static/js/shared/api/labels.js';
import { events, EventTypes } from '/app-static/js/shared/events/index.js';

/**
 * Renders a compact, clickable label list into `container`.
 *
 * Clicking a label row switches the canvas to bbox-draw mode for that label
 * (`setTool(runtime, 'bbox:<uuid>')`).  The active label row is highlighted
 * whenever the tool changes.
 *
 * @param {import('@image_tagging_types').ImageTaggingRuntime} runtime
 * @param {HTMLElement} container
 * @returns {Promise<{ destroy: () => void }>}
 */
export async function initLabelPanel(runtime, container) {
  container.innerHTML = '';
  Object.assign(container.style, {
    overflowY: 'auto',
    padding: '4px',
  });

  const header = document.createElement('div');
  header.textContent = 'Labels';
  Object.assign(header.style, {
    fontWeight: 'bold',
    fontSize: '0.8em',
    textTransform: 'uppercase',
    letterSpacing: '0.05em',
    padding: '4px 6px 6px',
    color: '#aaa',
  });
  container.appendChild(header);

  const labels = await fetchLabels();

  // Build hierarchy: root labels first, children indented below
  /** @type {Map<string|null, Array>} */
  const byParent = new Map();
  for (const label of labels.values()) {
    const key = label.parent_uuid ?? null;
    if (!byParent.has(key)) byParent.set(key, []);
    byParent.get(key).push(label);
  }

  function renderRows(parentUuid, indent) {
    const children = byParent.get(parentUuid) ?? [];
    children.sort((a, b) => a.name.localeCompare(b.name));
    for (const label of children) {
      const row = document.createElement('div');
      row.className = 'label-panel-row';
      row.dataset.uuid = label.uuid;
      Object.assign(row.style, {
        display: 'flex',
        alignItems: 'center',
        gap: '0.4em',
        padding: '4px 6px',
        paddingLeft: `${6 + indent * 14}px`,
        cursor: 'pointer',
        borderRadius: '3px',
        userSelect: 'none',
      });

      const swatch = document.createElement('span');
      Object.assign(swatch.style, {
        width: '10px',
        height: '10px',
        borderRadius: '50%',
        flexShrink: '0',
        background: label.color ?? '#888',
        display: 'inline-block',
      });

      const name = document.createElement('span');
      name.textContent = label.name;
      name.style.fontSize = '0.88em';

      row.appendChild(swatch);
      row.appendChild(name);

      row.addEventListener('click', () => {
        setTool(runtime, `bbox:${label.uuid}`);
      });

      container.appendChild(row);
      renderRows(label.uuid, indent + 1);
    }
  }

  renderRows(null, 0);

  // Highlight the active label whenever the tool changes
  function syncHighlight(activeLabelUuid) {
    container.querySelectorAll('.label-panel-row').forEach(row => {
      const isActive = row.dataset.uuid === activeLabelUuid;
      row.style.outline = isActive ? '2px solid #ff0033' : '';
      row.style.background = isActive ? '#ff003322' : '';
    });
  }

  const unsub = events.subscribe(EventTypes.CANVAS_TOOL_CHANGED, ({ tool }) => {
    const activeLabelUuid = tool.startsWith('bbox:') ? tool.slice(5) : null;
    syncHighlight(activeLabelUuid);
  });

  // Reflect tool that's already active when the panel mounts
  const currentTool = runtime.tool ?? '';
  if (currentTool.startsWith('bbox:')) {
    syncHighlight(currentTool.slice(5));
  }

  return { destroy: unsub };
}
