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
import { activeTrainingConfig } from '/app-static/js/shared/activeTrainingConfig.js';
import {
  fetchTrainingConfigImagesByLabelStatus,
  fetchTrainingConfigNeedsReview,
} from '/app-static/js/shared/api/trainingConfigs.js';
import { fetchImageLabels, setImageLabel } from '/app-static/js/shared/api/imageLabels.js';
import { VERIFIED_TAG_UUID, UNVERIFIED_TAG_UUID } from '/app-static/js/shared/api/review.js';
import { events, EventTypes } from '/app-static/js/shared/events/index.js';
import { setTool } from '../app/image-tagging/canvas/tools.js';
import { verifyAllBboxes } from '../app/image-tagging/canvas/image.js';
import { EditableField, TextBoxColorField } from '/app-static/js/ui/utils/fields/index.js';
import { createGenericRow, toast, makeLabelFilterNotice } from '/app-static/js/ui/utils/index.js';
import { openAnchoredPopup } from '/app-static/js/ui/utils/anchoredPopup.js';

const DEFAULT_COLOR = '#cccccc';

const STATUS_CFG = Object.freeze({
  present:   { text: '✓', color: '#4c4' },
  absent:    { text: '✗', color: '#e55' },
  uncertain: { text: '?', color: '#888' },
});
// Three-way cycle: uncertain → absent → present → uncertain
// Absent comes first because a click is more likely to be "confirmed not
// present" than "present" — a present label is usually already established
// via a drawn bbox rather than this toggle.
// PRESENT derived from verified bboxes is locked regardless of this cycle.
const STATUS_CYCLE = Object.freeze({ uncertain: 'absent', absent: 'present', present: 'uncertain' });

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

// ─── Re-parent ("Move to…") picker ──────────────────────────────────────────
// Shared between edit mode and model-config mode: the invalid-target set
// (self + descendants) is computed client-side for a snappy, testable picker,
// but the server-side cycle guard (Manager._assert_no_reparent_cycle) is still
// the authority — this is a UX nicety, not the enforcement point.

/**
 * Labels that would create a cycle if `uuid` were re-parented under them: the
 * label itself and every one of its descendants.
 *
 * @param {Map<string, object>} labelsMap
 * @param {string} uuid
 * @returns {Set<string>}
 */
function invalidMoveTargets(labelsMap, uuid) {
  const byParent = buildHierarchy(labelsMap, null);
  const invalid = new Set([uuid]);
  const queue = [uuid];
  while (queue.length > 0) {
    const cur = queue.shift();
    for (const child of (byParent.get(cur) ?? [])) {
      if (!invalid.has(child.uuid)) {
        invalid.add(child.uuid);
        queue.push(child.uuid);
      }
    }
  }
  return invalid;
}

/**
 * Build a <select> of valid re-parent targets: "(root)" plus every label in
 * `labelsMap` not in `excludeUuids`, indented to show hierarchy depth.
 *
 * @param {Map<string, object>} labelsMap
 * @param {Set<string>} excludeUuids
 * @param {string|null} currentParentUuid
 * @returns {HTMLSelectElement}
 */
function buildMoveTargetSelect(labelsMap, excludeUuids, currentParentUuid) {
  const select = document.createElement('select');
  select.className = 'label-move-select';

  const rootOpt = document.createElement('option');
  rootOpt.value = '';
  rootOpt.textContent = '(root)';
  select.appendChild(rootOpt);

  for (const label of flattenHierarchy(labelsMap)) {
    if (excludeUuids.has(label.uuid)) continue;
    const opt = document.createElement('option');
    opt.value = label.uuid;
    opt.textContent = `${'—'.repeat(label.depth)} ${label.name}`.trim();
    select.appendChild(opt);
  }

  select.value = currentParentUuid ?? '';
  return select;
}

/**
 * Toggle an inline "Move to…" row directly beneath `anchorRow`: a target
 * picker plus Move/Cancel buttons. Clicking the trigger again while the
 * picker for the same label is open closes it instead of opening a second one.
 *
 * @param {HTMLElement} anchorRow
 * @param {{label: object, labelsMap: Map<string,object>, indentLevel: number, onMove: (newParentUuid: string|null) => Promise<void>}} opts
 */
function toggleMovePicker(anchorRow, { label, labelsMap, indentLevel, onMove }) {
  const existing = anchorRow.nextElementSibling;
  if (existing?.dataset?.moveRowFor === label.uuid) {
    existing.remove();
    return;
  }
  // Only one picker open at a time within this row's parent container.
  anchorRow.parentElement?.querySelectorAll('[data-move-row-for]').forEach((el) => el.remove());

  const moveRow = document.createElement('div');
  moveRow.className = 'label-move-row';
  moveRow.dataset.moveRowFor = label.uuid;
  Object.assign(moveRow.style, {
    display: 'flex',
    alignItems: 'center',
    gap: '0.4em',
    padding: '3px 5px',
    paddingLeft: `${6 + (indentLevel + 1) * 14}px`,
  });

  const select = buildMoveTargetSelect(labelsMap, invalidMoveTargets(labelsMap, label.uuid), label.parentUuid ?? null);
  moveRow.appendChild(select);

  const applyBtn = document.createElement('button');
  applyBtn.type = 'button';
  applyBtn.className = 'label-move-apply';
  applyBtn.textContent = 'Move here';
  applyBtn.onclick = async () => {
    const newParentUuid = select.value || null;
    applyBtn.disabled = true;
    try {
      await onMove(newParentUuid);
    } catch (err) {
      toast(err.message, 5000, 'error');
      applyBtn.disabled = false;
    }
  };
  moveRow.appendChild(applyBtn);

  const cancelBtn = document.createElement('button');
  cancelBtn.type = 'button';
  cancelBtn.className = 'label-move-cancel';
  cancelBtn.textContent = 'Cancel';
  cancelBtn.onclick = () => moveRow.remove();
  moveRow.appendChild(cancelBtn);

  anchorRow.after(moveRow);
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
  row.dataset.testid = 'label-list-row';
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
  nameEl.dataset.testid = 'label-list-row-name';
  nameEl.textContent = label.name;
  nameEl.style.fontSize = '0.82em';

  row.appendChild(swatch);
  row.appendChild(nameEl);

  return { row, swatch, nameEl };
}

// ─── Edit mode ───────────────────────────────────────────────────────────────


function renderEditRows(byParent, container, parentUuid, indentLevel, onRerender, labelsMap) {
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

    const moveButton = {
      text: 'Move',
      emoji: '↕️',
      onClick: ({ row }) => {
        toggleMovePicker(row, {
          label,
          labelsMap,
          indentLevel,
          onMove: async (newParentUuid) => {
            await updateLabel(uuid, name, color, { parentUuid: newParentUuid });
            onRerender();
          },
        });
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
      leftButtons: [addChildButton, moveButton, deleteButton],
      rightButtons: [],
    });

    labelRow.dataset.uuid = uuid;
    labelRow.dataset.testid = 'label-list-row-edit';
    container.appendChild(labelRow);

    renderEditRows(byParent, container, uuid, indentLevel + 1, onRerender, labelsMap);
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
    row.dataset.testid = scopeValue === 'object' ? 'label-list-new-object-row' : 'label-list-new-image-row';
    inner.appendChild(row);
  }

  // Object Labels section
  // Scoped to a per-section labelsMap (not just a filtered byParent view) so
  // the Move picker's target list only ever offers same-scope labels — the
  // two sections are already rendered as separate trees, so this just keeps
  // "Move to…" consistent with what's on screen.
  const objLabelsMap = new Map([...labelsMap].filter(([, l]) => l.scope === 'object'));
  addSectionHeader('Object Labels');
  const byParentObj = buildHierarchy(objLabelsMap, null);
  renderEditRows(byParentObj, inner, null, 0, onRerender, objLabelsMap);
  addCreateRow('object');

  // Spacer between sections
  const spacer = document.createElement('div');
  spacer.style.height = '16px';
  inner.appendChild(spacer);

  // Image Labels section
  const imgLabelsMap = new Map([...labelsMap].filter(([, l]) => l.scope === 'image'));
  addSectionHeader('Image Labels');
  const byParentImg = buildHierarchy(imgLabelsMap, null);
  renderEditRows(byParentImg, inner, null, 0, onRerender, imgLabelsMap);
  addCreateRow('image');

  // Refresh button
  const refreshWrap = document.createElement('div');
  refreshWrap.className = 'label-refresh-wrap';
  const refreshBtn = document.createElement('button');
  refreshBtn.textContent = '🔄 Refresh';
  refreshBtn.className = 'label-refresh-btn';
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

// Option metadata for the two header preset menus. `needsTc` options only make
// sense when a training config is active (they reference "the training run").
const VISIBILITY_PRESET_OPTIONS = Object.freeze([
  { value: 'all',          label: 'Show all labels',                  desc: 'Every label stays visible.' },
  { value: 'training-run', label: 'Hide labels outside training run', desc: 'Show only labels in the active training run.', needsTc: true },
]);
const TARGET_PRESET_OPTIONS = Object.freeze([
  { value: 'all',          label: 'Everything',                desc: 'Apply-to-all affects every visible label.' },
  { value: 'training-run', label: 'Only active training run',  desc: 'Only labels in the active training run.', needsTc: true },
  { value: 'object',       label: 'Only object labels',        desc: 'Only bounding-box / object labels.' },
  { value: 'image',        label: 'Only image classification', desc: 'Only whole-image labels.' },
  { value: 'dynamic',      label: 'Only dynamic labels',       desc: 'Only dynamic labels.' },
  { value: 'none',         label: 'Nothing',                   desc: 'Clear the set — pick labels individually.' },
  // 'custom' is never picked from the menu — it's entered implicitly by toggling
  // rows. It's listed so the menu can show it as the checked/active option.
  { value: 'custom',       label: 'Custom (pick individually)', desc: 'Your hand-picked selection.' },
]);

/**
 * Build a radio-style preset menu for openAnchoredPopup. Each row shows a check
 * for the current value; clicking a row calls onPick(value).
 * @param {string} title
 * @param {Array<{value:string,label:string,desc:string}>} options
 * @param {string} currentValue
 * @param {(value:string)=>void} onPick
 * @returns {HTMLDivElement}
 */
function buildPresetMenu(title, options, currentValue, onPick) {
  const menu = document.createElement('div');
  menu.className = 'label-preset-menu';

  const hdr = document.createElement('div');
  hdr.textContent = title;
  hdr.className = 'label-preset-menu-header';
  menu.appendChild(hdr);

  for (const opt of options) {
    const row = document.createElement('div');
    row.dataset.value = opt.value;
    row.className = 'label-preset-option';
    row.addEventListener('mouseenter', () => { row.style.background = 'var(--pico-primary-background,rgba(255,255,255,.07))'; });
    row.addEventListener('mouseleave', () => { row.style.background = ''; });

    const check = document.createElement('span');
    check.textContent = opt.value === currentValue ? '✓' : '';
    check.className = 'label-preset-check';

    const txt = document.createElement('span');
    const t = document.createElement('div');
    t.textContent = opt.label;
    t.style.fontWeight = opt.value === currentValue ? '600' : '400';
    const d = document.createElement('div');
    d.textContent = opt.desc;
    d.className = 'label-preset-desc';
    txt.appendChild(t);
    txt.appendChild(d);

    row.appendChild(check);
    row.appendChild(txt);
    row.addEventListener('click', () => onPick(opt.value));
    menu.appendChild(row);
  }
  return menu;
}

async function mountSelectMode(container, { runtime, filter }) {
  // No overflow here — the parent `.sidebar-tab-pane.active` is the actual
  // scroll container (flex:1; overflow-y:auto). Giving this element its own
  // overflow:auto too used to create nested scroll boxes, which broke the
  // sticky header below: sticky anchors to the *nearest* scrolling ancestor,
  // and this element never itself grows a scrollbar (its height is
  // unconstrained), so the sticky child just rode along with the outer
  // pane's scroll instead of pinning to it.
  Object.assign(container.style, { padding: '4px' });

  // Pinned to the top of this scrollable panel (mirrors .inference-panel-actions)
  // so the header controls + bulk-op row stay reachable while the list scrolls.
  const stickyHeader = document.createElement('div');
  stickyHeader.className = 'label-panel-sticky-header';

  const header = document.createElement('div');
  Object.assign(header.style, {
    display: 'flex',
    alignItems: 'center',
    gap: '6px',
    padding: '4px 6px 6px',
  });
  const headerTitle = document.createElement('span');
  headerTitle.textContent = 'Labels';
  Object.assign(headerTitle.style, {
    fontWeight: 'bold',
    fontSize: '0.8em',
    textTransform: 'uppercase',
    letterSpacing: '0.05em',
    color: '#aaa',
  });
  header.appendChild(headerTitle);

  const tc = activeTrainingConfig.getActive();
  const visibilityPreset = activeTrainingConfig.getVisibilityPreset();
  const targetPreset = activeTrainingConfig.getTargetPreset();

  // Right-align the icon cluster (visibility / target menus + refresh).
  const headerSpacer = document.createElement('span');
  headerSpacer.style.marginLeft = 'auto';
  header.appendChild(headerSpacer);

  // A single shared popup for both menus — opening one closes the other.
  let _menuPopup = null;
  const closeMenu = () => { _menuPopup?.close(); _menuPopup = null; };

  function _makeMenuButton(icon, title, buildMenu) {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.textContent = icon;
    btn.title = title;
    btn.className = 'label-icon-btn';
    btn.addEventListener('click', () => {
      if (_menuPopup) { closeMenu(); return; }
      _menuPopup = openAnchoredPopup(btn, buildMenu());
    });
    return btn;
  }

  const visBtn = _makeMenuButton('👁', 'Label visibility', () =>
    buildPresetMenu(
      'Visibility',
      VISIBILITY_PRESET_OPTIONS.filter(o => !o.needsTc || tc),
      visibilityPreset,
      (v) => { closeMenu(); activeTrainingConfig.setVisibilityPreset(v); },
    ));
  header.appendChild(visBtn);

  const tgtBtn = _makeMenuButton('🎯', 'Apply-to-all target', () =>
    buildPresetMenu(
      'Apply-to-all target',
      // 'custom' only appears once the user has hand-picked a selection.
      TARGET_PRESET_OPTIONS.filter(o => (!o.needsTc || tc) && (o.value !== 'custom' || targetPreset === 'custom')),
      targetPreset,
      (v) => { closeMenu(); activeTrainingConfig.setTargetPreset(v); },
    ));
  header.appendChild(tgtBtn);

  const refreshBtn = document.createElement('button');
  refreshBtn.type = 'button';
  refreshBtn.textContent = '↻';
  refreshBtn.title = 'Refresh verified/unverified counts';
  refreshBtn.className = 'label-icon-btn label-icon-btn--muted';
  refreshBtn.addEventListener('click', async () => {
    refreshBtn.disabled = true;
    try {
      await activeTrainingConfig.refreshStats();
    } finally {
      refreshBtn.disabled = false;
    }
  });
  header.appendChild(refreshBtn);
  stickyHeader.appendChild(header);

  // Single global "Apply to all" bar. It acts on exactly the targeted set
  // (across every section), so the previous per-section buttons — and the bug
  // where the top section's "Mark all absent" reached past Image Classification
  // into Dynamic Labels — are gone by construction.
  const bulkRow = document.createElement('div');
  Object.assign(bulkRow.style, { padding: '0 6px 6px' });

  // "Apply to N targeted:" gets its own line — sharing a flex row with the
  // buttons let a long count force an awkward mid-button wrap in the narrow
  // sidebar.
  const bulkLabel = document.createElement('div');
  bulkLabel.className = 'label-bulk-caption';

  const bulkButtonsRow = document.createElement('div');
  Object.assign(bulkButtonsRow.style, { display: 'flex', gap: '6px', alignItems: 'center', flexWrap: 'wrap' });

  const markAbsentBtn = document.createElement('button');
  markAbsentBtn.textContent = 'Mark absent';
  markAbsentBtn.title = 'Mark every targeted label absent for this image';
  Object.assign(markAbsentBtn.style, { fontSize: '0.75em', padding: '2px 6px' });
  const resetAllBtn = document.createElement('button');
  resetAllBtn.textContent = 'Reset';
  resetAllBtn.title = 'Reset every targeted label to unreviewed for this image';
  Object.assign(resetAllBtn.style, { fontSize: '0.75em', padding: '2px 6px' });

  // Visual separator: Verify All acts on bbox tags (canvas state), not label
  // presence like Mark absent/Reset — kept in this row for one-glance access,
  // but set apart so it doesn't read as a third variant of the same action.
  const bulkSep = document.createElement('span');
  bulkSep.className = 'label-bulk-separator';

  const verifyAllBtn = document.createElement('button');
  verifyAllBtn.textContent = 'Verify all';
  verifyAllBtn.title = 'Mark every bbox on this image as verified';
  Object.assign(verifyAllBtn.style, { fontSize: '0.75em', padding: '2px 6px' });
  verifyAllBtn.addEventListener('click', () => {
    if (!runtime) return;
    verifyAllBboxes(runtime);
  });

  bulkButtonsRow.appendChild(markAbsentBtn);
  bulkButtonsRow.appendChild(resetAllBtn);
  bulkButtonsRow.appendChild(bulkSep);
  bulkButtonsRow.appendChild(verifyAllBtn);
  bulkRow.appendChild(bulkLabel);
  bulkRow.appendChild(bulkButtonsRow);
  stickyHeader.appendChild(bulkRow);
  container.appendChild(stickyHeader);

  const rawLabelsMap = await fetchLabels();
  const filteredMap = activeTrainingConfig.applyLabelFilter(rawLabelsMap);
  // Split by (scope, labelType): object-scope object labels / image-scope labels / dynamic labels.
  const regularMap = new Map([...filteredMap].filter(([, l]) => l.labelType !== 'dynamic' && l.scope !== 'image'));
  const imageClassMap = new Map([...filteredMap].filter(([, l]) => l.labelType !== 'dynamic' && l.scope === 'image'));
  const dynamicMap = new Map([...filteredMap].filter(([, l]) => l.labelType === 'dynamic'));
  const filterNotice = makeLabelFilterNotice(rawLabelsMap.size - regularMap.size - imageClassMap.size - dynamicMap.size);
  if (filterNotice) container.appendChild(filterNotice);
  // Labels outside the active TC — used only as the baseline for the
  // 'training-run' target preset (they stay visible but start untargeted).
  const excludedUuids = tc ? activeTrainingConfig.getExcludedUuids(rawLabelsMap) : new Set();
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
  // Re-entrancy guard for the CANVAS_ANNOTATIONS_SAVED flush below, matching the
  // `_flushInFlight` idiom in ui/inspector.js. A rapid double-save (or a save
  // racing a navigation auto-save) could otherwise put two setImageLabel calls
  // for the same key in flight concurrently.
  let _flushInFlight = false;

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

  // ── Apply-to-all targeting ──────────────────────────────────────────────
  // Two layers: a persisted preset provides the baseline; per-row overrides
  // (and the section all/none controls) refine it. Any manual toggle switches
  // the preset to 'custom' and persists the resulting selection to the session
  // (via applyCustomTargeting), so it survives remounts — e.g. the remount a
  // save triggers — instead of snapping back to the preset baseline.
  /** @type {Map<string, object>} uuid → label, for every rendered row */
  const labelByUuid = new Map();
  /** @type {Map<string, boolean>} per-row target overrides for this mount */
  const targetOverrides = new Map();
  /** @type {Map<string, () => void>} uuid → refresh that row's ◉/⊘ icon */
  const targetUpdaters = new Map();
  // The hand-picked selection backing a 'custom' preset, restored on remount.
  const customTargeted = activeTrainingConfig.getCustomTargeted();

  function baselineTargeted(label) {
    switch (targetPreset) {
      case 'none':         return false;
      case 'object':       return label.labelType !== 'dynamic' && label.scope !== 'image';
      case 'image':        return label.labelType !== 'dynamic' && label.scope === 'image';
      case 'dynamic':      return label.labelType === 'dynamic';
      case 'training-run': return !excludedUuids.has(label.uuid);
      case 'custom':       return customTargeted.has(label.uuid);
      case 'all':
      default:             return true;
    }
  }

  function effectiveTargeted(uuid) {
    if (targetOverrides.has(uuid)) return targetOverrides.get(uuid);
    const label = labelByUuid.get(uuid);
    return label ? baselineTargeted(label) : false;
  }

  function targetedUuids() {
    return [...labelByUuid.keys()].filter(u => effectiveTargeted(u));
  }

  function updateBulkLabel() {
    const n = targetedUuids().length;
    bulkLabel.textContent = `Apply to ${n} targeted:`;
    markAbsentBtn.disabled = n === 0;
    resetAllBtn.disabled = n === 0;
  }

  /** Set targeting for a set of rows and refresh their icons + the count. */
  function setTargetedMany(uuids, on) {
    uuids.forEach(u => { targetOverrides.set(u, on); targetUpdaters.get(u)?.(); });
    updateBulkLabel();
    // Persist as a hand-picked 'custom' selection so it survives the next
    // remount (does NOT publish, so this toggle won't itself trigger one).
    activeTrainingConfig.applyCustomTargeting(targetedUuids());
  }

  function renderSelectRows(parentUuid, indent, bp = byParent) {
    const children = bp.get(parentUuid ?? null) ?? [];
    for (const label of children) {
      labelByUuid.set(label.uuid, label);
      const { row, nameEl } = buildLabelRow(label, indent);
      row.className = 'label-panel-row';
      row.style.cursor = 'pointer';

      // Right-aligned row accessories: coverage badges + Apply-to-all target toggle.
      const rightWrap = document.createElement('span');
      rightWrap.className = 'label-row-right-wrap';

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
          : (stat.coLabelGapsCount ?? 0);

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
            const effectiveUuids = new Set([label.uuid, ...rollupChildUuids]);
            await showReviewModal({
              title: `${label.name} — needs review (${tc.name})`,
              initialViewFilter: 'bboxes',
              bboxFilterFn: (b) => effectiveUuids.has(b.labelUuid) && (b.tags || []).some(t => t.uuid === UNVERIFIED_TAG_UUID),
              fetchImages: async ({ afterUuid = null, afterSortValue = null, limit }) =>
                fetchTrainingConfigNeedsReview({
                  uuid: tc.uuid,
                  labelUuids: [label.uuid, ...rollupChildUuids],
                  afterUuid, afterSortValue, limit,
                }),
            });
          }
        }

        // Badge wrapper (appended into the shared right-aligned rightWrap)
        const badgeWrap = document.createElement('span');
        badgeWrap.className = 'label-badge-wrap';

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

        rightWrap.appendChild(badgeWrap);
      }

      // Apply-to-all target toggle — ◉ included / ⊘ excluded. Always rightmost.
      const tgtToggle = document.createElement('span');
      tgtToggle.className = 'label-target-toggle';
      function applyTargetIcon() {
        const on = effectiveTargeted(label.uuid);
        tgtToggle.textContent = on ? '◉' : '⊘';
        tgtToggle.style.color = on ? '#88aaff' : '#666';
        tgtToggle.title = on
          ? 'Included in Apply-to-all — click to exclude'
          : 'Excluded from Apply-to-all — click to include';
      }
      applyTargetIcon();
      targetUpdaters.set(label.uuid, applyTargetIcon);
      tgtToggle.addEventListener('click', (e) => {
        e.stopPropagation();
        setTargetedMany([label.uuid], !effectiveTargeted(label.uuid));
      });
      rightWrap.appendChild(tgtToggle);

      nameEl.after(rightWrap);

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

  markAbsentBtn.addEventListener('click', () => bulkOp(
    targetedUuids().filter(uuid => effectiveStatus(uuid) !== 'present'),
    'absent',
  ));

  resetAllBtn.addEventListener('click', () => bulkOp(
    targetedUuids().filter(uuid => serverStates.has(uuid)),
    'uncertain',
  ));

  /**
   * Render a labelled section: a header (title + [◉ all | ⊘ none] master
   * control) followed by its rows. The title lives in its own <div> so
   * callers/tests can still find it by exact text. Returns nothing; the master
   * control is wired to the UUIDs that appeared during this section's render.
   * @param {string} title
   * @param {Map<string|null, object[]>} bp
   * @param {{ color: string, italic?: boolean, rule?: boolean }} opts
   */
  function renderSection(title, bp, { color, italic = false, rule = false }) {
    if (rule) {
      const hr = document.createElement('hr');
      Object.assign(hr.style, { margin: '6px 4px', borderColor: '#444' });
      container.appendChild(hr);
    }

    const headerWrap = document.createElement('div');
    Object.assign(headerWrap.style, { display: 'flex', alignItems: 'center', gap: '6px', padding: '2px 6px 4px' });

    const titleEl = document.createElement('div');
    titleEl.textContent = title;
    Object.assign(titleEl.style, {
      fontWeight: 'bold', fontSize: '0.8em', textTransform: 'uppercase',
      letterSpacing: '0.05em', color, ...(italic ? { fontStyle: 'italic' } : {}),
    });
    headerWrap.appendChild(titleEl);

    const master = document.createElement('span');
    master.className = 'label-section-master';
    const allBtn = document.createElement('span');
    allBtn.className = 'label-section-all';
    allBtn.textContent = '◉ all';
    allBtn.title = 'Include every label in this section in Apply-to-all';
    const sep = document.createElement('span');
    sep.textContent = '|';
    sep.style.color = '#555';
    const noneBtn = document.createElement('span');
    noneBtn.className = 'label-section-none';
    noneBtn.textContent = '⊘ none';
    noneBtn.title = 'Exclude every label in this section from Apply-to-all';
    master.appendChild(allBtn);
    master.appendChild(sep);
    master.appendChild(noneBtn);
    headerWrap.appendChild(master);
    container.appendChild(headerWrap);

    // Capture which UUIDs this section contributes so the master targets them.
    const before = new Set(labelByUuid.keys());
    renderSelectRows(null, 0, bp);
    const sectionUuids = [...labelByUuid.keys()].filter(u => !before.has(u));
    allBtn.addEventListener('click', () => setTargetedMany(sectionUuids, true));
    noneBtn.addEventListener('click', () => setTargetedMany(sectionUuids, false));
  }

  // Object Labels are the primary section; Image Classification and Dynamic
  // Labels follow, each preceded by a divider. Empty sections are omitted.
  if (regularMap.size > 0) {
    renderSection('Object Labels', byParent, { color: '#aaa' });
  }
  if (imageClassMap.size > 0) {
    renderSection('Image Classification', byParentImageClass, { color: '#88aaff', rule: regularMap.size > 0 });
  }
  if (dynamicMap.size > 0) {
    renderSection('Dynamic Labels', byParentDynamic, {
      color: '#888', italic: true, rule: regularMap.size > 0 || imageClassMap.size > 0,
    });
  }

  updateBulkLabel();

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
  // Guarded by _flushInFlight (mirrors ui/inspector.js's identical handler): a
  // rapid double-save, or a save racing a navigation auto-save, would otherwise
  // put two setImageLabel calls for the same key in flight at once, which is
  // exactly the race that turns a get-then-insert upsert into a spurious
  // failure. Uses a while loop so entries queued *during* an in-flight flush
  // (the user toggled something while awaiting the network) are picked up on
  // the next iteration rather than silently dropped.
  const unsubSaved = events.subscribeToCanvas(
    runtime.id,
    EventTypes.CANVAS_ANNOTATIONS_SAVED,
    async () => {
      // Verified/unverified counts may have changed (presence marks and/or bboxes) — refresh
      // the cached coverage stats so badges elsewhere don't go stale until a manual refresh.
      activeTrainingConfig.refreshStats();
      if (!currentImagePath || !pendingFlush.size || _flushInFlight) return;
      _flushInFlight = true;
      try {
        while (currentImagePath && pendingFlush.size) {
          // Capture path and entries atomically; clear only after we know what succeeded.
          const pathAtSave = currentImagePath;
          const entries = [...pendingFlush];
          pendingFlush.clear();
          const results = await Promise.allSettled(
            entries.map(([uuid, status]) => setImageLabel(pathAtSave, uuid, status))
          );
          if (currentImagePath !== pathAtSave) break; // navigated away mid-flush; discard, matching IMAGE_LOADED_ONTO_CANVAS's own pendingFlush.clear()

          let skippedCount = 0;
          const failed = [];
          results.forEach((r, i) => {
            if (r.status === 'rejected') {
              console.error('Failed to persist image label:', r.reason);
              failed.push(entries[i]);
            } else if (r.value?.skipped) {
              skippedCount += 1;
            }
          });

          if (failed.length) {
            // Re-queue only entries that haven't been superseded by a newer user
            // interaction while the save was in flight.
            failed.forEach(([uuid, status]) => {
              if (!pendingFlush.has(uuid)) pendingFlush.set(uuid, status);
            });
            // Always show the error — data was lost whether or not we can retry.
            toast(`${failed.length} label state(s) failed to save`, 5000, 'error');
            break; // stop looping on persistent errors to avoid thrashing
          }

          if (skippedCount) {
            // A verified-bbox conflict is permanently blocked server-side — never
            // re-queue it (that would make it fire again on every subsequent save,
            // matching the "periodically" complaint) and never count it toward the
            // red "failed to save" toast, since nothing actually went wrong.
            toast(
              skippedCount === 1
                ? 'Label left unchanged: image has a verified bbox for it'
                : `${skippedCount} label state(s) left unchanged: verified bboxes exist`,
              4000,
              'info'
            );
          }
        }
      } finally {
        _flushInFlight = false;
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
      closeMenu();
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
  onLabelsChanged = () => {},
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
      cb.className = 'label-row-checkbox';

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
        rollUpBadge.className = 'label-toggle-badge';
        rollUpBadge.title = 'During training, attribute images from unchecked child labels to this label';

        rollUpToggle = document.createElement('input');
        rollUpToggle.type = 'checkbox';
        rollUpToggle.checked = rollUpUnchecked.has(label.uuid);
        rollUpToggle.className = 'label-toggle-checkbox';

        rollUpBadge.appendChild(rollUpToggle);
        rollUpBadge.appendChild(document.createTextNode('roll-up'));
        row.appendChild(rollUpBadge);
      }

      // "sub model" badge — only for labels with children
      let subToggle = null;
      if (showSubmodel) {
        const badge = document.createElement('label');
        badge.className = 'label-toggle-badge';
        badge.title = 'Use a separate sub-model for this label and its children';

        subToggle = document.createElement('input');
        subToggle.type = 'checkbox';
        subToggle.checked = submodels.has(label.uuid);
        subToggle.className = 'label-toggle-checkbox';

        badge.appendChild(subToggle);
        badge.appendChild(document.createTextNode('sub model'));
        row.appendChild(badge);
      }

      // "move" control — reachable regardless of parent/child status; keyed
      // entirely by label_uuid downstream (ModelConfigNode), so re-parenting
      // here never disturbs the selection/submodel/roll-up state above.
      const moveBtn = document.createElement('button');
      moveBtn.type = 'button';
      moveBtn.className = 'label-move-trigger';
      moveBtn.title = 'Move to a different parent';
      moveBtn.textContent = '↕️ Move';
      moveBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        toggleMovePicker(row, {
          label,
          labelsMap: displayMap,
          indentLevel: indent,
          onMove: async (newParentUuid) => {
            await updateLabel(label.uuid, label.name, label.color, { parentUuid: newParentUuid });
            await onLabelsChanged?.();
          },
        });
      });
      row.appendChild(moveBtn);

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
    hr.className = 'label-dynamic-hr';
    container.appendChild(hr);

    const dynHdr = document.createElement('p');
    dynHdr.className = 'panel-section-label';
    dynHdr.classList.add('label-dynamic-header');
    dynHdr.textContent = 'Dynamic Labels';
    container.appendChild(dynHdr);

    for (const label of dynamicMap.values()) {
      const { row, nameEl } = buildLabelRow(label, 0, { padding: '3px 6px' });

      const cb = document.createElement('input');
      cb.type = 'checkbox';
      cb.checked = selected.has(label.uuid);
      cb.className = 'label-row-checkbox';

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
 *   onLabelsChanged?: () => (void|Promise<void>), model-config mode; called after
 *     a re-parent ("Move to…") succeeds, so callers can refetch labels and
 *     rebuild both the tree and the sub-model panels derived from it.
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
    onLabelsChanged = () => {},
    scopeFilter = null,
  } = options;

  let _destroyFn = () => {};
  let _handle = null;
  // Re-entrancy guard: event handlers call mount() without awaiting, so two
  // synchronous events would interleave two async mounts and append the list
  // twice. Coalesce: one pending re-mount runs after the in-flight one.
  let _mounting = false;
  let _remountQueued = false;

  async function mount() {
    if (_mounting) {
      _remountQueued = true;
      return;
    }
    _mounting = true;
    try {
      do {
        _remountQueued = false;
        _destroyFn();
        _destroyFn = () => {};
        container.innerHTML = '';

        let handle;
        if (mode === 'edit') {
          handle = await mountEditMode(container, { filter, onRerender: mount });
        } else if (mode === 'model-config') {
          handle = await mountModelConfigMode(container, {
            labelsMap, initialSelected, initialSubmodels, initialRollUpUnchecked,
            defaultSpec, onSubmodelsChanged, onRollUpUncheckedChanged, onLabelsChanged, scopeFilter,
          });
        } else {
          handle = await mountSelectMode(container, { runtime, filter });
        }

        _handle = handle;
        _destroyFn = handle?.destroy ?? (() => {});
      } while (_remountQueued);
    } finally {
      _mounting = false;
    }
  }

  await mount();

  // Reactive subscriptions vary by mode.
  let _unsubMode = null;
  if (mode === 'select') {
    _unsubMode = events.subscribeMany([
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
