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
import { UNVERIFIED_TAG_UUID } from '/app-static/js/shared/api/review.js';
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
export function buildHierarchy(labelsMap, filter) {
  /** @type {Map<string|null, object[]>} */
  const byParent = new Map();
  for (const label of labelsMap.values()) {
    if (filter && !filter(label)) continue;
    const key = label.parentUuid ?? null;
    if (!byParent.has(key)) byParent.set(key, []);
    byParent.get(key).push(label);
  }
  for (const children of byParent.values()) {
    children.sort((a, b) => a.name.localeCompare(b.name));
  }
  return byParent;
}

/**
 * Walk the hierarchy tree depth-first and return a flat array, each item
 * augmented with a `depth` field (0 = root).  Useful for rendering tree-shaped
 * data in a flat list (e.g. multi-select dropdowns) while preserving order and
 * parent-child relationships visually via indentation.
 *
 * @param {Map<string, object>} labelsMap
 * @param {((label: object) => boolean)|null} [filter]
 * @returns {Array<object & { depth: number }>}
 */
export function flattenHierarchy(labelsMap, filter = null) {
  const byParent = buildHierarchy(labelsMap, filter);
  const result = [];
  function visit(parentUuid, depth) {
    for (const label of (byParent.get(parentUuid ?? null) ?? [])) {
      result.push({ ...label, depth });
      visit(label.uuid, depth + 1);
    }
  }
  visit(null, 0);
  return result;
}

/**
 * Expand a list of label UUIDs to include all their descendants (recursively).
 * Uses the labelsMap to build a parent→children relationship.
 *
 * @param {string[]} labelUuids
 * @param {Map<string, object>} labelsMap
 * @returns {Set<string>}
 */
export function expandWithDescendants(labelUuids, labelsMap) {
  if (!labelUuids?.length || !labelsMap?.size) return new Set(labelUuids ?? []);
  // Reuse buildHierarchy's parent→children map rather than re-deriving it.
  const byParent = buildHierarchy(labelsMap);
  const result = new Set(labelUuids);
  const queue = [...labelUuids];
  while (queue.length > 0) {
    const uuid = queue.shift();
    for (const child of (byParent.get(uuid) ?? [])) {
      if (!result.has(child.uuid)) {
        result.add(child.uuid);
        queue.push(child.uuid);
      }
    }
  }
  return result;
}

// ─── Shared row builder ──────────────────────────────────────────────────────

/**
 * Builds a base label row: a flex div with an indent, a color swatch, and a name span.
 * Callers may prepend leading controls (row.prepend) and append trailing controls (row.appendChild).
 *
 * @param {object} label
 * @param {number} indent
 * @param {{ padding?: string }} [opts]
 * @returns {{ row: HTMLDivElement, swatch: HTMLSpanElement, nameEl: HTMLSpanElement }}
 */
export function buildLabelRow(label, indent, { padding = '4px 6px' } = {}) {
  const row = document.createElement('div');
  row.dataset.uuid = label.uuid;
  Object.assign(row.style, {
    display: 'flex',
    alignItems: 'center',
    gap: '0.4em',
    padding,
    paddingLeft: `${6 + indent * 14}px`,
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

  row.appendChild(swatch);
  row.appendChild(nameEl);

  return { row, swatch, nameEl };
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
 * A label is present if at least one reviewed (non-unverified) bbox in the image carries that label.
 *
 * @param {Map<string, object>|undefined} bboxes
 * @returns {Set<string>}
 */
function computeDerivedPresent(bboxes) {
  const present = new Set();
  for (const bbox of (bboxes ?? new Map()).values()) {
    if (bbox.label?.uuid && !(bbox.tags ?? []).some(t => t.uuid === UNVERIFIED_TAG_UUID)) {
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

  const bulkRow = document.createElement('div');
  Object.assign(bulkRow.style, { display: 'flex', gap: '4px', padding: '0 6px 4px' });
  const markAbsentBtn = document.createElement('button');
  markAbsentBtn.textContent = 'Mark all absent';
  Object.assign(markAbsentBtn.style, { fontSize: '0.75em', padding: '2px 6px' });
  const resetAllBtn = document.createElement('button');
  resetAllBtn.textContent = 'Reset all';
  Object.assign(resetAllBtn.style, { fontSize: '0.75em', padding: '2px 6px' });
  bulkRow.appendChild(markAbsentBtn);
  bulkRow.appendChild(resetAllBtn);
  container.appendChild(bulkRow);

  const labelsMap = await fetchLabels();
  const byParent = buildHierarchy(labelsMap, filter);

  let currentImagePath = null;
  let _loadSeq = 0;
  let _lastSelectedUuid = null;
  /** @type {Map<string, 'absent'>} — explicit absence marks from the server */
  let serverStates = new Map();
  /**
   * Pending label-presence changes not yet flushed to the server.
   * Key = label UUID, value = status string or null (null means reset to 'uncertain').
   * Populated on user interaction; flushed when canvas annotations are saved.
   */
  let pendingFlush = new Map();
  let derivedPresent = computeDerivedPresent(runtime?.state?.image?.bboxes);

  /** @type {Map<string, (status: string) => void>} */
  const rowUpdaters = new Map();

  function effectiveStatus(uuid) {
    if (derivedPresent.has(uuid)) return 'present';
    return serverStates.get(uuid) ?? 'uncertain';
  }

  /** Update both local maps together; the single write site prevents drift between them. */
  function setLocalPresence(uuid, status) {
    if (status === 'uncertain') {
      serverStates.delete(uuid);
    } else {
      serverStates.set(uuid, /** @type {'absent'} */ (status));
    }
    pendingFlush.set(uuid, status);
  }

  function renderSelectRows(parentUuid, indent) {
    const children = byParent.get(parentUuid ?? null) ?? [];
    for (const label of children) {
      const { row } = buildLabelRow(label, indent);
      row.className = 'label-panel-row';
      row.style.cursor = 'pointer';

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

      btn.addEventListener('click', (e) => {
        e.stopPropagation();
        if (!currentImagePath) return;
        const current = effectiveStatus(label.uuid);
        if (STATUS_CFG[current]?.locked) return;
        const next = STATUS_CYCLE[current] ?? 'uncertain';
        // Update local state immediately for UI feedback; defer API call to save.
        setLocalPresence(label.uuid, next);
        applyStatus(effectiveStatus(label.uuid));
        runtime?.markDirty?.();
      });

      row.prepend(btn);

      row.addEventListener('click', () => {
        _lastSelectedUuid = label.uuid;
        setTool(runtime, `bbox:${label.uuid}`);
      });

      container.appendChild(row);
      renderSelectRows(label.uuid, indent + 1);
    }
  }

  function bulkOp(uuids, status) {
    if (!currentImagePath) { toast('No image loaded'); return; }
    uuids.forEach(uuid => {
      setLocalPresence(uuid, status);
      rowUpdaters.get(uuid)?.(effectiveStatus(uuid));
    });
    if (uuids.length) runtime?.markDirty?.();
  }

  markAbsentBtn.addEventListener('click', () => bulkOp(
    [...rowUpdaters.keys()].filter(uuid => effectiveStatus(uuid) !== 'present'),
    'absent',
  ));

  resetAllBtn.addEventListener('click', () => bulkOp(
    [...serverStates.keys()],
    'uncertain',
  ));

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

  // Fetch server label states when a new image is loaded; discard any pending unsaved changes.
  const unsubImage = events.subscribeToCanvas(
    runtime.id,
    EventTypes.IMAGE_LOADED_ONTO_CANVAS,
    async ({ path }) => {
      // Discard any pending presence changes for the previous image.
      pendingFlush.clear();
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

  // Flush pending presence changes to the server when the canvas is saved.
  const unsubSaved = events.subscribeToCanvas(
    runtime.id,
    EventTypes.CANVAS_ANNOTATIONS_SAVED,
    async () => {
      if (!currentImagePath || !pendingFlush.size) return;
      // Capture path and entries atomically; clear only after we know what succeeded.
      const pathAtSave = currentImagePath;
      const entries = [...pendingFlush];
      pendingFlush.clear();
      const results = await Promise.allSettled(
        entries.map(([uuid, status]) => setImageLabel(pathAtSave, uuid, status))
      );
      results.forEach(r => {
        if (r.status === 'rejected') console.error('Failed to persist image label:', r.reason);
      });
      const failed = entries.filter((_, i) => results[i].status === 'rejected');
      if (failed.length) {
        if (currentImagePath === pathAtSave) {
          // Re-queue only entries that haven't been superseded by a newer user
          // interaction while the save was in flight.
          failed.forEach(([uuid, status]) => {
            if (!pendingFlush.has(uuid)) pendingFlush.set(uuid, status);
          });
        }
        // Always show the error — data was lost whether or not we can retry.
        toast(`${failed.length} label state(s) failed to save`, 5000, 'error');
      }
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
      unsubSaved();
      unsubTool();
    },
  };
}

// ─── Model-config mode ───────────────────────────────────────────────────────

/**
 * Renders a fully-expanded, always-visible label tree for the model-config form.
 *
 * Each row: [checkbox] [color swatch] [label name] [☐ sub model] ← parent-only badge
 *
 * @param {HTMLElement} container
 * @param {{
 *   labelsMap?: Map<string,object>,   pre-fetched labels (skips internal fetch)
 *   initialSelected?: Set<string>,    UUIDs whose checkboxes start checked
 *   initialSubmodels?: Map<string,string>, UUID → typename for enabled submodels
 *   typenames?: string[],
 *   onSubmodelsChanged?: (submodels: Map<string,string>) => void,
 * }} opts
 */
async function mountModelConfigMode(container, {
  labelsMap: preloaded = null,
  initialSelected = new Set(),
  initialSubmodels = new Map(),
  typenames = [],
  onSubmodelsChanged = () => {},
} = {}) {
  const labelsMap = preloaded ?? await fetchLabels();
  const byParent = buildHierarchy(labelsMap, null);

  const childCountMap = new Map();
  for (const label of labelsMap.values()) {
    if (label.parentUuid) childCountMap.set(label.parentUuid, (childCountMap.get(label.parentUuid) ?? 0) + 1);
  }
  const selected = new Set(initialSelected);
  const submodels = new Map(initialSubmodels);

  function appendRows(parentUuid, indent) {
    for (const label of (byParent.get(parentUuid ?? null) ?? [])) {
      const isParent = childCountMap.has(label.uuid);
      const showSubmodel = isParent;

      const { row, nameEl } = buildLabelRow(label, indent, { padding: '3px 6px' });

      const cb = document.createElement('input');
      cb.type = 'checkbox';
      cb.checked = selected.has(label.uuid);
      cb.style.cssText = 'margin: 0; flex-shrink: 0;';

      nameEl.style.cursor = 'pointer';
      nameEl.style.whiteSpace = 'nowrap';
      nameEl.addEventListener('click', () => {
        cb.checked = !cb.checked;
        cb.dispatchEvent(new Event('change'));
      });

      row.prepend(cb);

      // "sub model" badge — only for labels with 2+ children
      let subToggle = null;
      if (showSubmodel) {
        const badge = document.createElement('label');
        badge.style.cssText = [
          'display: inline-flex', 'align-items: center', 'gap: 3px',
          'font-size: 0.72em', 'cursor: pointer', 'margin: 0',
          'padding: 1px 5px', 'border-radius: 3px',
          'border: 1px solid var(--pico-border-color)',
          'color: var(--pico-muted-color)',
          'white-space: nowrap', 'flex-shrink: 0',
        ].join('; ');

        subToggle = document.createElement('input');
        subToggle.type = 'checkbox';
        subToggle.checked = submodels.has(label.uuid);
        subToggle.style.cssText = 'margin: 0;';

        badge.appendChild(subToggle);
        badge.appendChild(document.createTextNode('sub model'));
        row.appendChild(badge);
      }

      // ---- Checkbox event -------------------------------------------------
      cb.addEventListener('change', () => {
        if (cb.checked) {
          selected.add(label.uuid);
        } else {
          // Unchecking removes any active submodel for this label too.
          if (submodels.has(label.uuid)) {
            submodels.delete(label.uuid);
            if (subToggle) subToggle.checked = false;
            onSubmodelsChanged(new Map(submodels));
          }
          selected.delete(label.uuid);
        }
      });

      // ---- Sub-model toggle event -----------------------------------------
      if (subToggle) {
        subToggle.addEventListener('change', () => {
          if (subToggle.checked) {
            // Enabling submodel auto-selects the label.
            cb.checked = true;
            selected.add(label.uuid);
            submodels.set(label.uuid, typenames[0] ?? '');
          } else {
            submodels.delete(label.uuid);
          }
          onSubmodelsChanged(new Map(submodels));
        });
      }

      container.appendChild(row);

      if (isParent) appendRows(label.uuid, indent + 1);
    }
  }

  appendRows(null, 0);

  return {
    getSelected: () => new Set(selected),
    getSubmodels: () => new Map(submodels),
    destroy() {},
  };
}

// ─── Public API ──────────────────────────────────────────────────────────────

/**
 * Create a label list component in the given container.
 *
 * @param {HTMLElement} container
 * @param {{
 *   mode?: 'edit'|'select'|'model-config',
 *   runtime?: object,
 *   filter?: ((label: object) => boolean)|null,
 *   labelsMap?: Map<string,object>,        pre-fetched labels (model-config mode)
 *   initialSelected?: Set<string>,         model-config mode
 *   initialSubmodels?: Map<string,string>, model-config mode
 *   typenames?: string[],                  model-config mode
 *   onSubmodelsChanged?: function,         model-config mode
 * }} options
 * @returns {Promise<{
 *   destroy(): void,
 *   refresh(): Promise<void>,
 *   getSelected(): Set<string>,   model-config mode only
 *   getSubmodels(): Map<string,string>, model-config mode only
 * }>}
 */
export async function createLabelList(container, options = {}) {
  const {
    mode = 'select',
    runtime = null,
    filter = null,
    labelsMap = null,
    initialSelected = new Set(),
    initialSubmodels = new Map(),
    typenames = [],
    onSubmodelsChanged = () => {},
  } = options;

  let _destroyFn = () => {};
  let _handle = null;

  async function mount() {
    _destroyFn();
    _destroyFn = () => {};
    container.innerHTML = '';

    let handle;
    if (mode === 'edit') {
      handle = await mountEditMode(container, { filter, onRerender: mount });
    } else if (mode === 'model-config') {
      handle = await mountModelConfigMode(container, {
        labelsMap, initialSelected, initialSubmodels, typenames, onSubmodelsChanged,
      });
    } else {
      handle = await mountSelectMode(container, { runtime, filter });
    }

    _handle = handle;
    _destroyFn = handle?.destroy ?? (() => {});
  }

  await mount();

  return {
    destroy() { _destroyFn(); },
    refresh() { return mount(); },
    getSelected() { return _handle?.getSelected?.() ?? new Set(); },
    getSubmodels() { return _handle?.getSubmodels?.() ?? new Map(); },
  };
}
