/**
 * Unified label list component.
 *
 *   mode='edit'   — editable CRUD list (config page)
 *   mode='select' — compact click-to-apply list with per-image presence status buttons (canvas panels)
 *
 * Usage:
 *   const handle = await createLabelList(containerEl, { mode: 'select', runtime });
 *   // later:
 *   handle.destroy();          // unsubscribe events, clean up
 *   await handle.refresh();    // re-fetch labels and re-render
 */

import { fetchLabels, clearLabelsCache, createNewLabel, deleteLabel, updateLabel } from '/app-static/js/shared/api/labels.js';
import { fetchImageLabels, setImageLabel } from '/app-static/js/shared/api/imageLabels.js';
import { VERIFIED_TAG_UUID } from '/app-static/js/shared/api/review.js';
import { events, EventTypes } from '/app-static/js/shared/events/index.js';
import { setTool } from '../app/image-tagging/canvas/tools.js';
import { EditableField, TextBoxColorField } from '/app-static/js/ui/utils/fields/index.js';
import { createGenericRow, toast } from '/app-static/js/ui/utils/index.js';

const DEFAULT_COLOR = '#cccccc';

const STATUS_CFG = Object.freeze({
  present:   { text: '✓', color: '#4c4', locked: true  },
  absent:    { text: '✗', color: '#e55', locked: false },
  uncertain: { text: '?', color: '#888', locked: false },
});
const STATUS_CYCLE = Object.freeze({ uncertain: 'absent', absent: 'uncertain' });

// ─── Shared hierarchy builder ────────────────────────────────────────────────

/**
 * Returns a Map<string|null, label[]> where null → root labels.
 * Each group is sorted by name. Filtered labels are excluded along with their subtrees.
 *
 * @param {Map<string, object>} labelsMap
 * @param {((label: object) => boolean)|null} filter
 * @returns {Map<string|null, object[]>}
 */
function buildHierarchy(labelsMap, filter) {
  /** @type {Map<string|null, object[]>} */
  const byParent = new Map();
  for (const label of labelsMap.values()) {
    if (filter && !filter(label)) continue;
    const key = label.parent_uuid ?? null;
    if (!byParent.has(key)) byParent.set(key, []);
    byParent.get(key).push(label);
  }
  for (const children of byParent.values()) {
    children.sort((a, b) => a.name.localeCompare(b.name));
  }
  return byParent;
}

// ─── Edit mode ───────────────────────────────────────────────────────────────

function renderEditRows(byParent, container, parentUuid, indentLevel, onRerender) {
  const children = byParent.get(parentUuid ?? null) ?? [];
  for (const label of children) {
    const { name, color, uuid } = label;

    const addChildButton = {
      text: 'Add Child',
      emoji: '➕',
      onClick: ({ row }) => {
        const childRow = createGenericRow({
          field: new EditableField({
            buttonsLast: true,
            field: new TextBoxColorField({ placeholder: 'New Label Name', color: DEFAULT_COLOR }),
            editMode: true,
            onSave: async ({ text: newText, color: newColor }) => {
              try {
                await createNewLabel(newText, newColor, uuid);
                onRerender();
              } catch (err) {
                toast(err.message, 5000, 'error');
              }
            },
            onCancel: () => onRerender(),
          }),
          indentLevel: indentLevel + 1,
        });
        row.after(childRow);
      },
    };

    const deleteButton = {
      text: 'Delete',
      emoji: '🗑️',
      onClick: async () => {
        if (!window.confirm(`Delete label "${name}"?`)) return;
        try {
          await deleteLabel(uuid);
          onRerender();
        } catch (err) {
          toast(err.message, 5000, 'error');
        }
      },
    };

    const labelRow = createGenericRow({
      field: new EditableField({
        buttonsLast: true,
        field: new TextBoxColorField({ text: name, uuid, placeholder: 'New Label Name', color }),
        onSave: async ({ text: newText, color: newColor }) => {
          try {
            await updateLabel(uuid, newText, newColor);
            onRerender();
          } catch (err) {
            toast(err.message, 5000, 'error');
          }
        },
        onCancel: () => onRerender(),
      }),
      indentLevel,
      alignLeftButtons: true,
      leftButtons: [addChildButton, deleteButton],
      rightButtons: [],
    });

    labelRow.dataset.uuid = uuid;
    container.appendChild(labelRow);

    renderEditRows(byParent, container, uuid, indentLevel + 1, onRerender);
  }
}

async function mountEditMode(container, { filter, onRerender }) {
  const inner = document.createElement('div');
  Object.assign(inner.style, {
    display: 'inline-block',
    verticalAlign: 'top',
    width: 'max-content',
    minWidth: '0',
  });
  container.appendChild(inner);

  const labelsMap = await fetchLabels();
  const byParent = buildHierarchy(labelsMap, filter);

  renderEditRows(byParent, inner, null, 0, onRerender);

  const divider = document.createElement('hr');
  divider.style.margin = '10px 0';
  inner.appendChild(divider);

  const newRootRow = createGenericRow({
    field: new EditableField({
      buttonsLast: true,
      field: new TextBoxColorField({ text: '', placeholder: 'New Root Label Name', color: DEFAULT_COLOR }),
      editMode: true,
    }),
    indentLevel: 0,
    leftButtons: [
      {
        text: 'Add label',
        emoji: '➕',
        onClick: async ({ field }) => {
          const val = field.getValue();
          const name = val.text?.trim();
          const color = val.color || DEFAULT_COLOR;
          if (!name) { toast('Error: Name required', 5000, 'error'); return; }
          try {
            await createNewLabel(name, color);
            onRerender();
          } catch (err) {
            toast(err.message, 5000, 'error');
          }
        },
      },
    ],
    rightButtons: [
      {
        text: 'Refresh',
        emoji: '🔄',
        onClick: async () => { clearLabelsCache(); onRerender(); },
      },
    ],
  });
  inner.appendChild(newRootRow);

  return { destroy() {} };
}

// ─── Select mode ─────────────────────────────────────────────────────────────

/**
 * Returns the set of label UUIDs that are PRESENT based on current canvas bbox state.
 * A label is present if at least one verified bbox in the image carries that label.
 *
 * @param {Map<string, object>|undefined} bboxes
 * @returns {Set<string>}
 */
function computeDerivedPresent(bboxes) {
  const present = new Set();
  for (const bbox of (bboxes ?? new Map()).values()) {
    if (bbox.label?.uuid && (bbox.tags ?? []).some(t => t.uuid === VERIFIED_TAG_UUID)) {
      present.add(bbox.label.uuid);
    }
  }
  return present;
}

async function mountSelectMode(container, { runtime, filter }) {
  Object.assign(container.style, { overflowY: 'auto', padding: '4px' });

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

  const labelsMap = await fetchLabels();
  const byParent = buildHierarchy(labelsMap, filter);

  let currentImagePath = null;
  let _loadSeq = 0;
  let _lastSelectedUuid = null;
  /** @type {Map<string, 'absent'>} — explicit absence marks from the server */
  let serverStates = new Map();
  let derivedPresent = computeDerivedPresent(runtime?.state?.image?.bboxes);

  /** @type {Map<string, (status: string) => void>} */
  const rowUpdaters = new Map();

  function effectiveStatus(uuid) {
    if (derivedPresent.has(uuid)) return 'present';
    return serverStates.get(uuid) ?? 'uncertain';
  }

  function renderSelectRows(parentUuid, indent) {
    const children = byParent.get(parentUuid ?? null) ?? [];
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

      const nameEl = document.createElement('span');
      nameEl.textContent = label.name;
      nameEl.style.fontSize = '0.88em';

      const btn = document.createElement('button');
      Object.assign(btn.style, {
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

      applyStatus(effectiveStatus(label.uuid));
      rowUpdaters.set(label.uuid, applyStatus);

      btn.addEventListener('click', async (e) => {
        e.stopPropagation();
        if (!currentImagePath) return;
        const current = effectiveStatus(label.uuid);
        if (STATUS_CFG[current]?.locked) return;
        const next = STATUS_CYCLE[current] ?? 'uncertain';
        try {
          await setImageLabel(currentImagePath, label.uuid, next);
          if (next === 'uncertain') {
            serverStates.delete(label.uuid);
          } else {
            serverStates.set(label.uuid, /** @type {'absent'} */ (next));
          }
          applyStatus(effectiveStatus(label.uuid));
        } catch (err) {
          console.error('Failed to set image label:', err);
        }
      });

      row.appendChild(btn);
      row.appendChild(swatch);
      row.appendChild(nameEl);

      row.addEventListener('click', () => {
        _lastSelectedUuid = label.uuid;
        setTool(runtime, `bbox:${label.uuid}`);
      });

      container.appendChild(row);
      renderSelectRows(label.uuid, indent + 1);
    }
  }

  renderSelectRows(null, 0);

  function refreshStatusButtons() {
    for (const [uuid, update] of rowUpdaters) {
      update(effectiveStatus(uuid));
    }
  }

  function syncHighlight(activeLabelUuid) {
    container.querySelectorAll('.label-panel-row').forEach(row => {
      const isActive = row.dataset.uuid === activeLabelUuid;
      row.style.outline = isActive ? '2px solid #ff0033' : '';
      row.style.background = isActive ? '#ff003322' : '';
    });
  }

  // Fetch server label states when a new image is loaded
  const unsubImage = events.subscribeToCanvas(
    runtime.id,
    EventTypes.IMAGE_LOADED_ONTO_CANVAS,
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
      if (seq !== _loadSeq) return; // superseded by a newer load
      // Only retain explicit ABSENT marks; PRESENT is derived client-side from bboxes
      serverStates = new Map([...states].filter(([, v]) => v === 'absent'));
      derivedPresent = computeDerivedPresent(runtime?.state?.image?.bboxes);
      refreshStatusButtons();
    }
  );

  // Recompute derived presence whenever the bbox collection changes
  const unsubBboxes = events.subscribeToCanvas(
    runtime.id,
    EventTypes.CANVAS_BBOXES_CHANGED,
    () => {
      derivedPresent = computeDerivedPresent(runtime?.state?.image?.bboxes);
      refreshStatusButtons();
    }
  );

  // Highlight the active label row whenever the tool changes; auto-select on bare 'bbox'
  const unsubTool = events.subscribeToCanvas(
    runtime.id,
    EventTypes.CANVAS_TOOL_CHANGED,
    ({ tool }) => {
      if (tool === 'bbox') {
        // Restore the last used label, or default to the first row
        if (_lastSelectedUuid) {
          setTool(runtime, `bbox:${_lastSelectedUuid}`);
        } else {
          const firstRow = container.querySelector('.label-panel-row');
          if (firstRow) firstRow.click();
        }
        return;
      }
      if (tool.startsWith('bbox:')) {
        _lastSelectedUuid = tool.slice(5);
      }
      syncHighlight(tool.startsWith('bbox:') ? tool.slice(5) : null);
    }
  );

  // Reflect tool already active when the panel mounts
  const currentTool = runtime.tool ?? '';
  if (currentTool.startsWith('bbox:')) {
    _lastSelectedUuid = currentTool.slice(5);
    syncHighlight(_lastSelectedUuid);
  } else if (currentTool === 'bbox') {
    // Panel mounted while bbox is active but no label chosen — pick first
    const firstRow = container.querySelector('.label-panel-row');
    if (firstRow) firstRow.click();
  }

  // If an image was already loaded before this component was created (lazy init),
  // we missed its IMAGE_LOADED_ONTO_CANVAS event — fetch its states now so ✗ icons appear.
  const alreadyLoadedPath = runtime?.state?.image?.image_path;
  if (alreadyLoadedPath) {
    currentImagePath = alreadyLoadedPath;
    const seq = ++_loadSeq;
    try {
      const states = await fetchImageLabels(alreadyLoadedPath);
      if (seq === _loadSeq) {
        serverStates = new Map([...states].filter(([, v]) => v === 'absent'));
        derivedPresent = computeDerivedPresent(runtime?.state?.image?.bboxes);
        refreshStatusButtons();
      }
    } catch (err) {
      console.error('Failed to fetch initial image labels:', err);
    }
  }

  return {
    destroy() {
      unsubImage();
      unsubBboxes();
      unsubTool();
    },
  };
}

// ─── Public API ──────────────────────────────────────────────────────────────

/**
 * Create a label list component in the given container.
 *
 * @param {HTMLElement} container
 * @param {{
 *   mode?: 'edit'|'select',
 *   runtime?: object,
 *   filter?: ((label: object) => boolean)|null,
 * }} options
 * @returns {Promise<{ destroy(): void, refresh(): Promise<void> }>}
 */
export async function createLabelList(container, options = {}) {
  const { mode = 'select', runtime = null, filter = null } = options;
  let _destroyFn = () => {};

  async function mount() {
    _destroyFn();
    _destroyFn = () => {};
    container.innerHTML = '';
    const handle = mode === 'edit'
      ? await mountEditMode(container, { filter, onRerender: mount })
      : await mountSelectMode(container, { runtime, filter });
    _destroyFn = handle?.destroy ?? (() => {});
  }

  await mount();

  return {
    destroy() { _destroyFn(); },
    refresh() { return mount(); },
  };
}
