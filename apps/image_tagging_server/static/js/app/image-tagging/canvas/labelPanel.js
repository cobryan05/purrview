import { setTool } from './tools.js';
import { fetchLabels } from '/app-static/js/shared/api/labels.js';
import { fetchImageLabels, setImageLabel } from '/app-static/js/shared/api/imageLabels.js';
import { events, EventTypes } from '/app-static/js/shared/events/index.js';

const STATUS_CFG = {
  present:   { text: '✓', color: '#4c4', locked: true  },
  absent:    { text: '✗', color: '#e55', locked: false },
  uncertain: { text: '?', color: '#888', locked: false },
};

const STATUS_CYCLE = { uncertain: 'absent', absent: 'uncertain' };

/**
 * Renders a compact, clickable label list into `container`.
 *
 * Clicking a label row switches the canvas to bbox-draw mode for that label.
 * A tri-state button on each row shows/toggles the label's presence status
 * for the currently loaded image (PRESENT=locked, ABSENT↔UNCERTAIN toggleable).
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

  let currentImagePath = null;
  let _loadSeq = 0;
  /** @type {Map<string, 'present'|'absent'>} */
  let labelStates = new Map();

  /** @type {Map<string, (status: string) => void>} */
  const rowUpdaters = new Map();

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

      // Status button (right side)
      const btn = document.createElement('button');
      Object.assign(btn.style, {
        marginLeft: 'auto',
        flexShrink: '0',
        width: '22px',
        height: '20px',
        fontSize: '0.8em',
        border: '1px solid #444',
        borderRadius: '3px',
        padding: '0',
        lineHeight: '1',
        background: 'transparent',
      });

      function applyStatus(status) {
        const cfg = STATUS_CFG[status] ?? STATUS_CFG.uncertain;
        btn.textContent = cfg.text;
        btn.style.color = cfg.color;
        btn.style.cursor = cfg.locked ? 'default' : 'pointer';
        btn.style.opacity = cfg.locked ? '0.7' : '1';
      }

      applyStatus(labelStates.get(label.uuid) ?? 'uncertain');
      rowUpdaters.set(label.uuid, applyStatus);

      btn.addEventListener('click', async (e) => {
        e.stopPropagation(); // don't trigger row click (tool switch)
        if (!currentImagePath) return;

        const current = labelStates.get(label.uuid) ?? 'uncertain';
        const cfg = STATUS_CFG[current] ?? STATUS_CFG.uncertain;
        if (cfg.locked) return;

        const next = STATUS_CYCLE[current] ?? 'uncertain';
        try {
          await setImageLabel(currentImagePath, label.uuid, next);
          if (next === 'uncertain') {
            labelStates.delete(label.uuid);
          } else {
            labelStates.set(label.uuid, next);
          }
          applyStatus(next);
        } catch (err) {
          console.error('Failed to set image label:', err);
        }
      });

      row.appendChild(swatch);
      row.appendChild(name);
      row.appendChild(btn);

      row.addEventListener('click', () => {
        setTool(runtime, `bbox:${label.uuid}`);
      });

      container.appendChild(row);
      renderRows(label.uuid, indent + 1);
    }
  }

  renderRows(null, 0);

  function refreshButtons() {
    for (const [uuid, update] of rowUpdaters) {
      update(labelStates.get(uuid) ?? 'uncertain');
    }
  }

  // Sync status buttons whenever a new image is loaded
  const unsubImage = events.subscribeToCanvas(
    runtime.id,
    EventTypes.LOAD_IMAGE_ONTO_CANVAS,
    async ({ path }) => {
      currentImagePath = path;
      const seq = ++_loadSeq;
      let states;
      try {
        states = await fetchImageLabels(path);
      } catch (err) {
        console.error('Failed to fetch image labels:', err);
        states = new Map();
      }
      if (seq !== _loadSeq) return; // a newer image load superseded this one
      labelStates = states;
      refreshButtons();
    }
  );

  // Highlight the active label whenever the tool changes
  function syncHighlight(activeLabelUuid) {
    container.querySelectorAll('.label-panel-row').forEach(row => {
      const isActive = row.dataset.uuid === activeLabelUuid;
      row.style.outline = isActive ? '2px solid #ff0033' : '';
      row.style.background = isActive ? '#ff003322' : '';
    });
  }

  const unsubTool = events.subscribeToCanvas(runtime.id, EventTypes.CANVAS_TOOL_CHANGED, ({ tool }) => {
    const activeLabelUuid = tool.startsWith('bbox:') ? tool.slice(5) : null;
    syncHighlight(activeLabelUuid);
  });

  // Reflect tool that's already active when the panel mounts
  const currentTool = runtime.tool ?? '';
  if (currentTool.startsWith('bbox:')) {
    syncHighlight(currentTool.slice(5));
  }

  return {
    destroy: () => {
      unsubImage();
      unsubTool();
    },
  };
}
