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
import { activeModel } from '/app-static/js/shared/activeModel.js';
import { activeTrainingConfig } from '/app-static/js/shared/activeTrainingConfig.js';
import {
  fetchTrainingConfigImagesByLabelStatus,
  fetchTrainingConfigNeedsReview,
} from '/app-static/js/shared/api/trainingConfigs.js';
import { fetchImageLabels, setImageLabel } from '/app-static/js/shared/api/imageLabels.js';
import { VERIFIED_TAG_UUID } from '/app-static/js/shared/api/review.js';
import { events, EventTypes } from '/app-static/js/shared/events/index.js';
import { setTool } from '../app/image-tagging/canvas/tools.js';
import { EditableField, TextBoxColorField } from '/app-static/js/ui/utils/fields/index.js';
import { createGenericRow, toast, makeLabelFilterNotice } from '/app-static/js/ui/utils/index.js';

const DEFAULT_COLOR = '#cccccc';

const STATUS_CFG = Object.freeze({
  present:   { text: '✓', color: '#4c4' },
  absent:    { text: '✗', color: '#e55' },
  uncertain: { text: '?', color: '#888' },
});
// Three-way cycle: uncertain → present → absent → uncertain
// PRESENT derived from verified bboxes is locked regardless of this cycle.
const STATUS_CYCLE = Object.freeze({ uncertain: 'present', present: 'absent', absent: 'uncertain' });

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
export function buildLabelRow(label, indent, { padding = '3px 5px' } = {}) {
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
  nameEl.style.fontSize = '0.82em';

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

  const allLabels = await fetchLabels();
  const hideDynamic = (label) => label.labelType !== 'dynamic';
  const composedFilter = filter ? (label) => hideDynamic(label) && filter(label) : hideDynamic;
  const labelsMap = new Map([...allLabels].filter(([, l]) => composedFilter(l)));

  function addSectionHeader(text) {
    const hdr = document.createElement('div');
    hdr.textContent = text;
    Object.assign(hdr.style, {
      fontSize: '0.72em',
      fontWeight: 'bold',
      color: '#777',
      textTransform: 'uppercase',
      letterSpacing: '0.06em',
      padding: '4px 6px 2px',
    });
    inner.appendChild(hdr);
  }

  function addCreateRow(scopeValue) {
    const placeholder = scopeValue === 'object' ? 'New Object Label' : 'New Image Label';
    const row = createGenericRow({
      field: new EditableField({
        buttonsLast: true,
        field: new TextBoxColorField({ text: '', placeholder, color: DEFAULT_COLOR }),
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
              await createNewLabel(name, color, null, 'standard', scopeValue);
              onRerender();
            } catch (err) {
              toast(err.message, 5000, 'error');
            }
          },
        },
      ],
      rightButtons: [],
    });
    inner.appendChild(row);
  }

  // Object Labels section
  addSectionHeader('Object Labels');
  const byParentObj = buildHierarchy(labelsMap, (l) => l.scope === 'object');
  renderEditRows(byParentObj, inner, null, 0, onRerender);
  addCreateRow('object');

  // Spacer between sections
  const spacer = document.createElement('div');
  spacer.style.height = '16px';
  inner.appendChild(spacer);

  // Image Labels section
  addSectionHeader('Image Labels');
  const byParentImg = buildHierarchy(labelsMap, (l) => l.scope === 'image');
  renderEditRows(byParentImg, inner, null, 0, onRerender);
  addCreateRow('image');

  // Refresh button
  const refreshWrap = document.createElement('div');
  refreshWrap.style.cssText = 'padding:6px 2px 2px;';
  const refreshBtn = document.createElement('button');
  refreshBtn.textContent = '🔄 Refresh';
  refreshBtn.style.cssText = 'font-size:0.78em;padding:2px 8px;background:#222;color:#777;border:1px solid #444;border-radius:3px;cursor:pointer;';
  refreshBtn.onclick = () => { clearLabelsCache(); onRerender(); };
  refreshWrap.appendChild(refreshBtn);
  inner.appendChild(refreshWrap);

  return { destroy() {} };
}

// ─── Select mode ─────────────────────────────────────────────────────────────

/**
 * Returns the set of label UUIDs that are PRESENT based on current canvas bbox state.
 * A label is present if at least one VERIFIED bbox in the image carries that label.
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

  const rawLabelsMap = await fetchLabels();
  // TC filter takes precedence over model-config filter when a TC is active.
  const filteredMap = activeTrainingConfig.getActive()
    ? activeTrainingConfig.applyLabelFilter(rawLabelsMap)
    : activeModel.applyLabelFilter(rawLabelsMap);
  // Split by (scope, labelType): object-scope object labels / image-scope labels / dynamic labels.
  // Dynamic labels are always shown (no visibility toggle gate) so the user can mark present/absent.
  const regularMap = new Map([...filteredMap].filter(([, l]) => l.labelType !== 'dynamic' && l.scope !== 'image'));
  const imageClassMap = new Map([...filteredMap].filter(([, l]) => l.labelType !== 'dynamic' && l.scope === 'image'));
  const dynamicMap = new Map([...filteredMap].filter(([, l]) => l.labelType === 'dynamic'));
  const filterNotice = makeLabelFilterNotice(rawLabelsMap.size - regularMap.size - imageClassMap.size - dynamicMap.size);
  if (filterNotice) container.appendChild(filterNotice);
  const byParent = buildHierarchy(regularMap, filter);
  const byParentImageClass = buildHierarchy(imageClassMap, filter);
  const byParentDynamic = buildHierarchy(dynamicMap, filter);

  // Cache a stats map keyed by labelUuid for O(1) lookup in row rendering
  function _buildStatsMap() {
    const cached = activeTrainingConfig.getCachedStats();
    if (!cached?.labelStats) return new Map();
    return new Map(cached.labelStats.map(s => [s.labelUuid, s]));
  }
  let _statsMap = _buildStatsMap();

  let currentImagePath = null;
  let _loadSeq = 0;
  let _lastSelectedUuid = null;
  /** @type {Map<string, 'present'|'absent'>} — explicit presence/absence marks from the server */
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
      serverStates.set(uuid, /** @type {'present'|'absent'} */ (status));
    }
    pendingFlush.set(uuid, status);
  }

  function renderSelectRows(parentUuid, indent, bp = byParent) {
    const children = bp.get(parentUuid ?? null) ?? [];
    for (const label of children) {
      const { row, nameEl } = buildLabelRow(label, indent);
      row.className = 'label-panel-row';
      row.style.cursor = 'pointer';

      // Mini coverage badges from active TC stats — [✓ verified] [⚠ unverified]
      const stat = _statsMap.get(label.uuid);
      if (stat && activeTrainingConfig.getActive()) {
        const tc = activeTrainingConfig.getActive();
        const isImage = stat.labelScope === 'image';

        // Rollup child UUIDs (for calls that need to include them)
        const rollupChildUuids = stat.hasRollup
          ? [..._statsMap.values()]
              .filter(s => s.rollUpParentUuid === label.uuid)
              .map(s => s.labelUuid)
          : [];

        // Verified count + optional rollup suffix
        const ownVerified = isImage ? (stat.presentImageCount ?? 0) : (stat.verifiedBboxCount ?? 0);
        let rollupVerified = 0;
        if (stat.hasRollup) {
          for (const cs of _statsMap.values()) {
            if (cs.rollUpParentUuid === label.uuid) {
              rollupVerified += isImage ? (cs.presentImageCount ?? 0) : (cs.verifiedBboxCount ?? 0);
            }
          }
        }
        const unverifiedCount = isImage
          ? (stat.unverifiedImageCount ?? 0)
          : (stat.imagesMissingVerificationCount ?? 0);

        // Badge factory — clickable when count > 0
        function _mkBadge(text, color, title, onClickFn) {
          const b = document.createElement('span');
          b.className = 'tc-stat-badge';
          b.textContent = text;
          b.style.color = color;
          b.title = title;
          if (onClickFn) {
            b.style.cursor = 'pointer';
            b.style.textDecoration = 'underline dotted';
            b.style.textUnderlineOffset = '2px';
            b.addEventListener('click', async (e) => { e.stopPropagation(); await onClickFn(); });
          }
          return b;
        }

        async function _browseVerified() {
          const { showReviewModal } = await import('/app-static/js/app/review/main.js');
          if (isImage) {
            await showReviewModal({
              title: `${label.name} — verified present (${tc.name})`,
              initialViewFilter: 'images',
              fetchImages: async ({ afterUuid = null, afterSortValue = null, limit }) =>
                fetchTrainingConfigImagesByLabelStatus(tc.uuid, {
                  labelUuid: label.uuid,
                  rollupLabelUuids: rollupChildUuids.length ? rollupChildUuids : undefined,
                  status: 'present', afterUuid, afterSortValue, limit,
                }),
            });
          } else {
            const effectiveUuids = new Set([label.uuid, ...rollupChildUuids]);
            await showReviewModal({
              title: `${label.name} — verified boxes (${tc.name})`,
              initialViewFilter: 'bboxes',
              bboxFilterFn: (b) => effectiveUuids.has(b.labelUuid) && (b.tags || []).some(t => t.uuid === VERIFIED_TAG_UUID),
              fetchImages: async ({ afterUuid = null, afterSortValue = null, limit }) =>
                fetchTrainingConfigImagesByLabelStatus(tc.uuid, {
                  labelUuid: label.uuid,
                  rollupLabelUuids: rollupChildUuids.length ? rollupChildUuids : undefined,
                  status: 'has_verified_bbox', afterUuid, afterSortValue, limit,
                }),
            });
          }
        }

        async function _browseUnverified() {
          const { showReviewModal } = await import('/app-static/js/app/review/main.js');
          if (isImage) {
            await showReviewModal({
              title: `${label.name} — unreviewed (${tc.name})`,
              initialViewFilter: 'images',
              fetchImages: async ({ afterUuid = null, afterSortValue = null, limit }) =>
                fetchTrainingConfigImagesByLabelStatus(tc.uuid, {
                  labelUuid: label.uuid,
                  rollupLabelUuids: rollupChildUuids.length ? rollupChildUuids : undefined,
                  status: 'uncertain', afterUuid, afterSortValue, limit,
                }),
            });
          } else {
            await showReviewModal({
              title: `${label.name} — needs review (${tc.name})`,
              initialViewFilter: 'bboxes',
              fetchImages: async ({ afterUuid = null, afterSortValue = null, limit }) =>
                fetchTrainingConfigNeedsReview({
                  uuid: tc.uuid,
                  labelUuids: [label.uuid, ...rollupChildUuids],
                  afterUuid, afterSortValue, limit,
                }),
            });
          }
        }

        // Build badge wrapper (pushed right via margin-left:auto)
        const badgeWrap = document.createElement('span');
        badgeWrap.style.cssText = 'margin-left:auto;display:flex;gap:3px;flex-shrink:0;align-items:center;';

        // Verified badge (always shown)
        let verifiedText = `✓ ${ownVerified.toLocaleString()}`;
        if (stat.hasRollup && rollupVerified > 0) verifiedText += ` +${rollupVerified}↑`;
        const verifiedTitle = stat.hasRollup && rollupVerified > 0
          ? `${ownVerified} ${isImage ? 'verified positives' : 'verified boxes'} (own) + ${rollupVerified} from rollup children — click to browse`
          : `${ownVerified} ${isImage ? 'verified positives' : 'verified boxes'}${ownVerified > 0 ? ' — click to browse' : ''}`;
        badgeWrap.appendChild(_mkBadge(
          verifiedText,
          ownVerified > 0 || rollupVerified > 0 ? '#4c4' : '#555',
          verifiedTitle,
          ownVerified > 0 || rollupVerified > 0 ? _browseVerified : null,
        ));

        // Unverified badge (only shown if > 0)
        if (unverifiedCount > 0) {
          badgeWrap.appendChild(_mkBadge(
            `⚠ ${unverifiedCount.toLocaleString()}`,
            '#f90',
            `${unverifiedCount} ${isImage ? 'unreviewed images' : 'images with unverified boxes'} — click to browse`,
            _browseUnverified,
          ));
        }

        nameEl.after(badgeWrap);
      }

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
        // Lock the button only when PRESENT is derived from verified bboxes, not when manually set.
        const locked = derivedPresent.has(label.uuid);
        btn.textContent = cfg.text;
        btn.style.color = cfg.color;
        btn.style.cursor = locked ? 'default' : 'pointer';
        btn.style.opacity = locked ? '0.7' : '1';
        btn.title = locked
          ? 'Present (locked — derived from verified bboxes)'
          : status === 'present'   ? 'Present — positive training example (click to cycle)'
          : status === 'absent'    ? 'Confirmed not present — negative training example (click to cycle)'
          : 'Unreviewed (click to cycle)';
      }

      applyStatus(effectiveStatus(label.uuid));
      rowUpdaters.set(label.uuid, applyStatus);

      btn.addEventListener('click', (e) => {
        e.stopPropagation();
        if (!currentImagePath) return;
        if (derivedPresent.has(label.uuid)) return; // locked when derived from verified bboxes
        const current = effectiveStatus(label.uuid);
        const next = STATUS_CYCLE[current] ?? 'uncertain';
        // Update local state immediately for UI feedback; defer API call to save.
        setLocalPresence(label.uuid, next);
        applyStatus(effectiveStatus(label.uuid));
        runtime?.markDirty?.();
      });

      row.prepend(btn);

      row.addEventListener('click', () => {
        // Image-scope labels have no bboxes — don't arm the bbox draw tool.
        if (label.scope === 'image') return;
        _lastSelectedUuid = label.uuid;
        setTool(runtime, `bbox:${label.uuid}`);
      });

      container.appendChild(row);
      renderSelectRows(label.uuid, indent + 1, bp);
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

  // UUIDs belonging to the Image Classification section — populated after that section renders.
  let imageClassUuids = new Set();

  markAbsentBtn.addEventListener('click', () => bulkOp(
    [...rowUpdaters.keys()].filter(uuid => !imageClassUuids.has(uuid) && effectiveStatus(uuid) !== 'present'),
    'absent',
  ));

  resetAllBtn.addEventListener('click', () => bulkOp(
    [...serverStates.keys()].filter(uuid => !imageClassUuids.has(uuid)),
    'uncertain',
  ));

  renderSelectRows(null, 0);

  if (imageClassMap.size > 0) {
    const hr = document.createElement('hr');
    Object.assign(hr.style, { margin: '6px 4px', borderColor: '#444' });
    container.appendChild(hr);

    const imgHeader = document.createElement('div');
    imgHeader.textContent = 'Image Classification';
    Object.assign(imgHeader.style, {
      fontWeight: 'bold',
      fontSize: '0.8em',
      textTransform: 'uppercase',
      letterSpacing: '0.05em',
      padding: '2px 6px 6px',
      color: '#88aaff',
    });
    container.appendChild(imgHeader);

    const imgBulkRow = document.createElement('div');
    Object.assign(imgBulkRow.style, { display: 'flex', gap: '4px', padding: '0 6px 4px' });
    const imgMarkAbsentBtn = document.createElement('button');
    imgMarkAbsentBtn.textContent = 'Mark all absent';
    Object.assign(imgMarkAbsentBtn.style, { fontSize: '0.75em', padding: '2px 6px' });
    const imgResetAllBtn = document.createElement('button');
    imgResetAllBtn.textContent = 'Reset all';
    Object.assign(imgResetAllBtn.style, { fontSize: '0.75em', padding: '2px 6px' });
    imgBulkRow.appendChild(imgMarkAbsentBtn);
    imgBulkRow.appendChild(imgResetAllBtn);
    container.appendChild(imgBulkRow);

    // Capture keys before rendering so we can identify which UUIDs belong to this section.
    const keysBefore = new Set(rowUpdaters.keys());
    renderSelectRows(null, 0, byParentImageClass);
    imageClassUuids = new Set([...rowUpdaters.keys()].filter(k => !keysBefore.has(k)));

    imgMarkAbsentBtn.addEventListener('click', () => bulkOp(
      [...imageClassUuids].filter(uuid => effectiveStatus(uuid) !== 'present'),
      'absent',
    ));
    imgResetAllBtn.addEventListener('click', () => bulkOp(
      [...imageClassUuids].filter(uuid => serverStates.has(uuid)),
      'uncertain',
    ));
  }

  if (dynamicMap.size > 0) {
    const hr = document.createElement('hr');
    Object.assign(hr.style, { margin: '6px 4px', borderColor: '#444' });
    container.appendChild(hr);

    const dynHeader = document.createElement('div');
    dynHeader.textContent = 'Dynamic Labels';
    Object.assign(dynHeader.style, {
      fontWeight: 'bold',
      fontSize: '0.8em',
      textTransform: 'uppercase',
      letterSpacing: '0.05em',
      padding: '2px 6px 6px',
      color: '#888',
      fontStyle: 'italic',
    });
    container.appendChild(dynHeader);

    renderSelectRows(null, 0, byParentDynamic);
  }

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
      // Retain explicit PRESENT and ABSENT marks from the server.
      // Derived PRESENT (from verified bboxes) is computed separately via computeDerivedPresent.
      serverStates = new Map([...states].filter(([, v]) => v !== 'uncertain'));
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
        serverStates = new Map([...states].filter(([, v]) => v !== 'uncertain'));
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
 *   defaultSpec?: {modelType:string,architecture:string}|null,
 *   onSubmodelsChanged?: (submodels: Map<string,string>) => void,
 * }} opts
 */
async function mountModelConfigMode(container, {
  labelsMap: preloaded = null,
  initialSelected = new Set(),
  initialSubmodels = new Map(),
  initialRollUpUnchecked = new Set(),
  defaultSpec = null,
  onSubmodelsChanged = () => {},
  onRollUpUncheckedChanged = () => {},
  scopeFilter = null,  // 'object' | 'image' | null (null = show all)
} = {}) {
  const rawMap = preloaded ?? await fetchLabels();

  // Filter to the relevant scope, then remap orphaned parents to root so every
  // filtered label is reachable in the hierarchy tree.
  // Labels without an explicit scope (e.g. legacy data) default to 'object'.
  // Always exclude dynamic labels from the main tree — they appear in their own section below.
  let displayMap = new Map([...rawMap].filter(([, l]) => l.labelType !== 'dynamic'));
  if (scopeFilter) {
    const filteredEntries = [...displayMap].filter(([, l]) => (l.scope ?? 'object') === scopeFilter);
    const filteredUuids = new Set(filteredEntries.map(([uuid]) => uuid));
    displayMap = new Map(filteredEntries.map(([uuid, l]) => [
      uuid,
      (!l.parentUuid || filteredUuids.has(l.parentUuid)) ? l : { ...l, parentUuid: null },
    ]));
  }

  // Dynamic labels are always shown in a separate flat section below the main tree.
  // They're object-level bbox annotations so they're not relevant for image classifiers.
  const dynamicMap = scopeFilter !== 'image'
    ? new Map([...rawMap].filter(([, l]) => l.labelType === 'dynamic'))
    : new Map();

  const byParent = buildHierarchy(displayMap, null);

  const childCountMap = new Map();
  for (const label of displayMap.values()) {
    if (label.parentUuid) childCountMap.set(label.parentUuid, (childCountMap.get(label.parentUuid) ?? 0) + 1);
  }
  const selected = new Set(initialSelected);
  const submodels = new Map(initialSubmodels);
  const rollUpUnchecked = new Set(initialRollUpUnchecked);

  // Image classifiers operate on whole images — sub-model overrides don't apply.
  const allowSubmodels = scopeFilter !== 'image';

  function appendRows(parentUuid, indent) {
    for (const label of (byParent.get(parentUuid ?? null) ?? [])) {
      const isParent = childCountMap.has(label.uuid);
      const showSubmodel = isParent && allowSubmodels;

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

      // "roll-up" badge — only for labels with children
      let rollUpToggle = null;
      if (showSubmodel) {
        const rollUpBadge = document.createElement('label');
        rollUpBadge.style.cssText = [
          'display: inline-flex', 'align-items: center', 'gap: 3px',
          'font-size: 0.72em', 'cursor: pointer', 'margin: 0',
          'padding: 1px 5px', 'border-radius: 3px',
          'border: 1px solid var(--pico-border-color)',
          'color: var(--pico-muted-color)',
          'white-space: nowrap', 'flex-shrink: 0',
        ].join('; ');
        rollUpBadge.title = 'During training, attribute images from unchecked child labels to this label';

        rollUpToggle = document.createElement('input');
        rollUpToggle.type = 'checkbox';
        rollUpToggle.checked = rollUpUnchecked.has(label.uuid);
        rollUpToggle.style.cssText = 'margin: 0;';

        rollUpBadge.appendChild(rollUpToggle);
        rollUpBadge.appendChild(document.createTextNode('roll-up'));
        row.appendChild(rollUpBadge);
      }

      // "sub model" badge — only for labels with children
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
        badge.title = 'Use a separate sub-model for this label and its children';

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
          // Unchecking removes any active submodel and roll-up for this label.
          if (submodels.has(label.uuid)) {
            submodels.delete(label.uuid);
            if (subToggle) subToggle.checked = false;
            onSubmodelsChanged(new Map(submodels));
          }
          if (rollUpUnchecked.has(label.uuid)) {
            rollUpUnchecked.delete(label.uuid);
            if (rollUpToggle) rollUpToggle.checked = false;
            onRollUpUncheckedChanged(new Set(rollUpUnchecked));
          }
          selected.delete(label.uuid);
        }
      });

      // ---- Roll-up toggle event -------------------------------------------
      if (rollUpToggle) {
        rollUpToggle.addEventListener('change', () => {
          if (rollUpToggle.checked) {
            // Enabling roll-up auto-selects the label.
            cb.checked = true;
            selected.add(label.uuid);
            rollUpUnchecked.add(label.uuid);
          } else {
            rollUpUnchecked.delete(label.uuid);
          }
          onRollUpUncheckedChanged(new Set(rollUpUnchecked));
        });
      }

      // ---- Sub-model toggle event -----------------------------------------
      if (subToggle) {
        subToggle.addEventListener('change', () => {
          if (subToggle.checked) {
            // Enabling submodel auto-selects the label and seeds the default spec.
            cb.checked = true;
            selected.add(label.uuid);
            submodels.set(label.uuid, defaultSpec ?? { modelType: 'crop_classifier', architecture: 'efficientnet_b0' });
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

  // ── Dynamic Labels section ────────────────────────────────────────────────
  // Rendered as a flat list with checkboxes only — no sub-model or roll-up
  // toggles since dynamic labels don't have a structured parent/child hierarchy.
  if (dynamicMap.size > 0) {
    const hr = document.createElement('hr');
    hr.style.cssText = 'margin: 0.5em 0; opacity: 0.3;';
    container.appendChild(hr);

    const dynHdr = document.createElement('p');
    dynHdr.className = 'panel-section-label';
    dynHdr.style.cssText = 'margin: 0.3em 0 0.2em; font-style: italic; color: var(--pico-muted-color);';
    dynHdr.textContent = 'Dynamic Labels';
    container.appendChild(dynHdr);

    for (const label of dynamicMap.values()) {
      const { row, nameEl } = buildLabelRow(label, 0, { padding: '3px 6px' });

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

      cb.addEventListener('change', () => {
        if (cb.checked) {
          selected.add(label.uuid);
        } else {
          selected.delete(label.uuid);
        }
      });

      container.appendChild(row);
    }
  }

  return {
    getSelected: () => new Set(selected),
    getSubmodels: () => new Map(submodels),
    getRollUpUnchecked: () => new Set(rollUpUnchecked),
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
 *   defaultSpec?: {modelType:string,architecture:string}|null,                  model-config mode
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
    initialRollUpUnchecked = new Set(),
    defaultSpec = null,
    onSubmodelsChanged = () => {},
    onRollUpUncheckedChanged = () => {},
    scopeFilter = null,
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
        labelsMap, initialSelected, initialSubmodels, initialRollUpUnchecked,
        defaultSpec, onSubmodelsChanged, onRollUpUncheckedChanged, scopeFilter,
      });
    } else {
      handle = await mountSelectMode(container, { runtime, filter });
    }

    _handle = handle;
    _destroyFn = handle?.destroy ?? (() => {});
  }

  await mount();

  // Reactive subscriptions vary by mode.
  let _unsubMode = null;
  if (mode === 'select') {
    _unsubMode = events.subscribeMany([
      [EventTypes.ACTIVE_MODEL_CHANGED,              () => mount()],
      [EventTypes.ACTIVE_TRAINING_CONFIG_CHANGED,    () => mount()],
      [EventTypes.DYNAMIC_LABELS_VISIBILITY_CHANGED, () => mount()],
      [EventTypes.LABELS_UPDATED,                    () => mount()],
    ]);
  } else if (mode === 'model-config') {
    // Re-render when labels are created, renamed, or deleted while the editor is open.
    _unsubMode = events.subscribe(EventTypes.LABELS_UPDATED, () => mount());
  }

  return {
    destroy() { _destroyFn(); _unsubMode?.(); },
    refresh() { return mount(); },
    getSelected() { return _handle?.getSelected?.() ?? new Set(); },
    getSubmodels() { return _handle?.getSubmodels?.() ?? new Map(); },
    getRollUpUnchecked() { return _handle?.getRollUpUnchecked?.() ?? new Set(); },
  };
}
