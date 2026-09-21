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
  fetchTrainingConfigImageGaps,
  fetchTrainingConfigImagesByLabelStatus,
  fetchTrainingConfigNeedsReview,
} from '/app-static/js/shared/api/trainingConfigs.js';
import { fetchImageLabels, setImageLabel } from '/app-static/js/shared/api/imageLabels.js';
import { VERIFIED_TAG_UUID, UNVERIFIED_TAG_UUID } from '/app-static/js/shared/api/review.js';
import { events, EventTypes } from '/app-static/js/shared/events/index.js';
import { prefs } from '/app-static/js/shared/prefs.js';
import { setTool } from '../app/image-tagging/canvas/tools.js';
import { verifyAllBboxes } from '../app/image-tagging/canvas/image.js';
import { registerCanvasAction } from '../app/image-tagging/canvasActions.js';
import { toast, actionNotice, makeLabelFilterNotice } from '/app-static/js/ui/utils/index.js';
import { previewSuppressionGroups, applySuppressionGroups } from '/app-static/js/shared/api/models.js';
import { openAnchoredPopup } from '/app-static/js/ui/utils/anchoredPopup.js';
import {
  ICONS,
  iconAction,
  makeColorWell,
  makeRails,
  makeSectionHeader,
  openRowEditor,
} from '/app-static/js/ui/utils/treeRow.js';

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
  moveRow.style.marginLeft = `${indentLevel * 14}px`;

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

/**
 * The label's Home Assistant publishing settings, as an inline panel under its row.
 *
 * A panel rather than two checkboxes in the row itself: the tree is dense and most of
 * these labels are annotation vocabulary that will never go near Home Assistant, so
 * putting the controls on every row would cost more attention than they are worth. It
 * mirrors `toggleMovePicker` exactly, which is the established shape for "a setting that
 * needs more than a click".
 *
 * The two flags are deliberately separate. Turning the label off removes its device
 * entirely; the second only controls the per-camera breakdown *within* that device,
 * which is the tier whose size is (labels × rules × cameras) and the one worth thinking
 * before enabling.
 */
function toggleHaPicker(anchorRow, { label, indentLevel, onSave }) {
  const existing = anchorRow.nextElementSibling;
  if (existing?.dataset?.haRowFor === label.uuid) {
    existing.remove();
    return;
  }
  anchorRow.parentElement?.querySelectorAll('[data-ha-row-for]').forEach((el) => el.remove());

  const haRow = document.createElement('div');
  haRow.className = 'label-move-row';
  haRow.dataset.haRowFor = label.uuid;
  haRow.dataset.testid = 'label-ha-panel';
  haRow.style.marginLeft = `${indentLevel * 14}px`;

  const check = (labelText, checked, testid, title) => {
    const wrap = document.createElement('label');
    wrap.className = 'label-ha-check';
    wrap.title = title;
    const box = document.createElement('input');
    box.type = 'checkbox';
    box.checked = checked;
    box.dataset.testid = testid;
    wrap.append(box, document.createTextNode(` ${labelText}`));
    return { wrap, box };
  };

  const publish = check(
    'Publish to Home Assistant',
    // `=== true`, not `!== false`: the flag is opt-in, so anything that is not an
    // explicit yes — including a label read back from before the column existed —
    // must render unticked rather than promising a device that is not there.
    label.publishToHa === true,
    'label-publish-to-ha',
    'Whether this label gets its own device in Home Assistant, with a "detected now" sensor. '
      + 'Off by default so the vocabulary does not become a wall of devices — tick the few you '
      + 'automate on. Off, the label is still detected, still recorded and still drives rule '
      + 'entities; it just has no device of its own.',
  );
  const perSource = check(
    'Per-camera breakdown',
    label.publishPerSource === true,
    'label-publish-per-source',
    'Splits this label per camera: a "detected now" sensor for each one, plus a counter per '
      + '(rule × camera). The presence sensors need no rules — they answer "is it in the '
      + 'garage right now" straight off the tracker. This is the tier that multiplies out '
      + '(labels × cameras), so it is off unless you want the breakdown specifically; the '
      + 'label\'s own device already reports where it was last seen.',
  );
  // The breakdown lives inside the device, so it cannot apply without one.
  const syncEnabled = () => {
    perSource.box.disabled = !publish.box.checked;
    perSource.wrap.classList.toggle('label-ha-check--muted', !publish.box.checked);
  };
  publish.box.addEventListener('change', syncEnabled);
  syncEnabled();

  haRow.append(publish.wrap, perSource.wrap);

  const applyBtn = document.createElement('button');
  applyBtn.type = 'button';
  applyBtn.className = 'label-move-apply';
  applyBtn.textContent = 'Save';
  applyBtn.onclick = async () => {
    applyBtn.disabled = true;
    try {
      await onSave({
        publishToHa: publish.box.checked,
        publishPerSource: publish.box.checked && perSource.box.checked,
      });
    } catch (err) {
      toast(err.message, 5000, 'error');
      applyBtn.disabled = false;
    }
  };
  haRow.appendChild(applyBtn);

  const cancelBtn = document.createElement('button');
  cancelBtn.type = 'button';
  cancelBtn.className = 'label-move-cancel';
  cancelBtn.textContent = 'Cancel';
  cancelBtn.onclick = () => haRow.remove();
  haRow.appendChild(cancelBtn);

  anchorRow.after(haRow);
}

const DEFAULT_EXCLUSIVE_VARIANTS_IOU_THRESHOLD = 0.85;

/**
 * A panel (mirrors `toggleHaPicker`) for the "Exclusive Variants" grouping used by
 * inference-time NMS — see the tooltip text below for the user-facing explanation.
 * The threshold field only appears once the checkbox is on: it is meaningless
 * otherwise, and showing it unconditionally on every label would just be another
 * number nobody but the handful of exclusive-variant parents ever touches.
 */
function toggleVariantsPicker(anchorRow, { label, indentLevel, onSave }) {
  const existing = anchorRow.nextElementSibling;
  if (existing?.dataset?.variantsRowFor === label.uuid) {
    existing.remove();
    return;
  }
  anchorRow.parentElement?.querySelectorAll('[data-variants-row-for]').forEach((el) => el.remove());

  const variantsRow = document.createElement('div');
  variantsRow.className = 'label-move-row';
  variantsRow.dataset.variantsRowFor = label.uuid;
  variantsRow.dataset.testid = 'label-variants-panel';
  variantsRow.style.marginLeft = `${indentLevel * 14}px`;

  const wrap = document.createElement('label');
  wrap.className = 'label-ha-check';
  wrap.title = 'On: a detection can only match one of this label\'s children at a time, so '
    + 'overlapping guesses for it collapse to the single best one. Off: children can '
    + 'legitimately appear together as separate objects, so overlapping detections are all kept.';
  const box = document.createElement('input');
  box.type = 'checkbox';
  box.checked = label.exclusiveVariants === true;
  box.dataset.testid = 'label-exclusive-variants';
  wrap.append(box, document.createTextNode(' Exclusive Variants'));

  const thresholdWrap = document.createElement('label');
  thresholdWrap.className = 'label-ha-check';
  thresholdWrap.title = 'How much two overlapping boxes must overlap before they are treated as '
    + 'one ambiguous detection and collapsed to the best guess. Higher = only near-total overlap '
    + 'collapses, so two separate objects standing close together are not wrongly merged.';
  const thresholdInput = document.createElement('input');
  thresholdInput.type = 'number';
  thresholdInput.min = '0';
  thresholdInput.max = '1';
  thresholdInput.step = '0.01';
  thresholdInput.dataset.testid = 'label-exclusive-variants-threshold';
  thresholdInput.value = String(
    typeof label.exclusiveVariantsIouThreshold === 'number'
      ? label.exclusiveVariantsIouThreshold
      : DEFAULT_EXCLUSIVE_VARIANTS_IOU_THRESHOLD,
  );
  thresholdWrap.append(document.createTextNode('Overlap threshold '), thresholdInput);

  const syncThresholdVisibility = () => { thresholdWrap.hidden = !box.checked; };
  box.addEventListener('change', syncThresholdVisibility);
  syncThresholdVisibility();

  variantsRow.append(wrap, thresholdWrap);

  const applyBtn = document.createElement('button');
  applyBtn.type = 'button';
  applyBtn.className = 'label-move-apply';
  applyBtn.textContent = 'Save';
  applyBtn.onclick = async () => {
    applyBtn.disabled = true;
    try {
      await onSave({
        exclusiveVariants: box.checked,
        exclusiveVariantsIouThreshold: box.checked ? parseFloat(thresholdInput.value) : null,
      });
    } catch (err) {
      toast(err.message, 5000, 'error');
      applyBtn.disabled = false;
    }
  };
  variantsRow.appendChild(applyBtn);

  const cancelBtn = document.createElement('button');
  cancelBtn.type = 'button';
  cancelBtn.className = 'label-move-cancel';
  cancelBtn.textContent = 'Cancel';
  cancelBtn.onclick = () => variantsRow.remove();
  variantsRow.appendChild(cancelBtn);

  anchorRow.after(variantsRow);
}

/**
 * After an Exclusive Variants save, check whether any already-trained models
 * still carry the old grouping and, if so, surface a dismissible notice
 * offering to apply the change to them right away (`Manager._reload_if_stale`
 * then picks the sidecar edit up on that model's next call — no restart).
 *
 * Best-effort: a preview failure (e.g. inference_server unreachable) is logged
 * rather than shown as an error — the label save itself already succeeded, and
 * this is only telling the user about a knock-on effect of it.
 */
async function notifySuppressionGroupsImpact(labelName) {
  let labelsMap;
  let impacted;
  try {
    labelsMap = await fetchLabels();
    impacted = await previewSuppressionGroups(labelsMap);
  } catch (err) {
    console.warn('Could not preview suppression-group impact:', err);
    return;
  }
  if (!impacted.length) return;

  actionNotice((el, close) => {
    const count = impacted.length;
    const header = document.createElement('div');
    header.textContent = `(!) "${labelName}" isn't applied to ${count} trained model${count === 1 ? '' : 's'} yet.`;
    header.style.marginBottom = '6px';
    el.appendChild(header);

    const list = document.createElement('div');
    list.style.cssText =
      'display:flex; flex-direction:column; gap:2px; margin-bottom:8px; max-height:140px; overflow-y:auto;';
    const checks = impacted.map(({ modelName, changedClasses, totalClasses }) => {
      const wrap = document.createElement('label');
      wrap.style.cssText = 'display:flex; align-items:center; gap:6px; font-size:13px;';
      const box = document.createElement('input');
      box.type = 'checkbox';
      box.checked = true;
      box.dataset.testid = 'suppression-impact-model';
      wrap.append(box, document.createTextNode(`${modelName} (${changedClasses}/${totalClasses} classes)`));
      list.appendChild(wrap);
      return { modelName, box };
    });
    el.appendChild(list);

    const actions = document.createElement('div');
    actions.style.cssText = 'display:flex; gap:8px;';

    const applyBtn = document.createElement('button');
    applyBtn.type = 'button';
    applyBtn.className = 'label-move-apply';
    applyBtn.textContent = 'Apply now';
    applyBtn.onclick = async () => {
      const chosen = checks.filter(c => c.box.checked).map(c => c.modelName);
      if (!chosen.length) return;
      applyBtn.disabled = true;
      try {
        await applySuppressionGroups(labelsMap, chosen);
        toast(
          `Updated ${chosen.length} model${chosen.length === 1 ? '' : 's'} — takes effect on their next call, no restart needed.`,
          5000,
          'success',
        );
        close();
      } catch (err) {
        toast(err.message, 5000, 'error');
        applyBtn.disabled = false;
      }
    };

    const laterBtn = document.createElement('button');
    laterBtn.type = 'button';
    laterBtn.className = 'label-move-cancel';
    laterBtn.textContent = 'Not now';
    laterBtn.onclick = close;

    actions.append(applyBtn, laterBtn);
    el.appendChild(actions);
  });
}

/**
 * A move (re-parent) can shift which model each label trains as part of — a
 * boundary label's sub-model gains or loses a class, an Exclusive Variants
 * group's membership changes, class_remap targets move. Unlike the
 * suppression-groups case, there's no cheap read-only preview for "which
 * trained models does this actually change and how" — that would mean
 * re-deriving training targets for every model config that references this
 * label's subtree and diffing the result, which isn't done here. So this is a
 * static advisory, not a computed impact list: it tells the user what CAN
 * change without claiming to know whether it did for this specific move.
 */
function notifyHierarchyMoveImpact(labelName) {
  actionNotice((el, close) => {
    const header = document.createElement('div');
    header.textContent = `(!) Moving "${labelName}" only affects future training runs.`;
    header.style.marginBottom = '6px';
    el.appendChild(header);

    const body = document.createElement('div');
    body.style.cssText = 'font-size:13px; margin-bottom:8px;';
    body.textContent =
      'Already-trained models keep using the class set and grouping they were trained with. '
      + 'Retrain a model (or, for an Exclusive Variants change specifically, use the notice that '
      + 'appears after that edit) to bring it up to date.';
    el.appendChild(body);

    const dismissBtn = document.createElement('button');
    dismissBtn.type = 'button';
    dismissBtn.className = 'label-move-cancel';
    dismissBtn.textContent = 'Got it';
    dismissBtn.onclick = close;
    el.appendChild(dismissBtn);
  });
}

// ─── Shared row builder ──────────────────────────────────────────────────────
//
// One compact row shape for every mode:
//
//     [rails][caret][swatch][name][kids?] ……… [mode extras][hover actions]
//
// Replaces the old createGenericRow + EditableField rows, which put bordered
// emoji buttons before the name (taller rows, no scannable name column).
// Actions are now icon-only, right-aligned and hover-revealed; depth is
// carried by hairline rails instead of raw padding.

// ── Collapse state ──────────────────────────────────────────────────────────
// Kept outside the component because every mutation re-mounts the whole list
// (mount() tears the container down), so component-local state would reset on
// the very next save. Persisted so a folded tree stays folded across reloads.

const COLLAPSE_PREF = 'ww_label_tree_collapsed_';
/** @type {Map<string, Set<string>>} tree key → collapsed label UUIDs */
const _collapsed = new Map();

/** @param {string} key @returns {Set<string>} */
function collapsedSet(key) {
  if (!_collapsed.has(key)) {
    const stored = prefs.read(COLLAPSE_PREF + key, []);
    _collapsed.set(key, new Set(Array.isArray(stored) ? stored : []));
  }
  return _collapsed.get(key);
}

/** @param {string} key @param {string} uuid */
function toggleCollapsed(key, uuid) {
  const set = collapsedSet(key);
  if (set.has(uuid)) set.delete(uuid);
  else set.add(uuid);
  prefs.write(COLLAPSE_PREF + key, [...set]);
}

/**
 * Collapse or expand every parent in `byParent` at once.
 * @param {string} key
 * @param {Map<string|null, object[]>} byParent
 * @param {boolean} collapse
 */
function setAllCollapsed(key, byParent, collapse) {
  const set = collapsedSet(key);
  for (const [parentUuid, children] of byParent) {
    if (parentUuid === null || !children.length) continue;
    if (collapse) set.add(parentUuid);
    else set.delete(parentUuid);
  }
  prefs.write(COLLAPSE_PREF + key, [...set]);
}

/** True when at least one parent in `byParent` is currently expanded. */
function anyExpanded(key, byParent) {
  const set = collapsedSet(key);
  for (const [parentUuid, children] of byParent) {
    if (parentUuid !== null && children.length && !set.has(parentUuid)) return true;
  }
  return false;
}

// ── Rails ────────────────────────────────────────────────────────────────────

/**
 * Derive the rail kinds for `label` by walking up to the root.
 *
 * Labels whose parent was filtered out of `labelsMap` (a different scope, a
 * dynamic label hidden from this tree) stop the walk there and render at the
 * depth they're actually drawn at, matching how buildHierarchy groups them.
 *
 * @param {object} label
 * @param {Map<string|null, object[]>} byParent
 * @param {Map<string, object>} labelsMap
 * @returns {string[]}
 */
function railKindsFor(label, byParent, labelsMap) {
  const chain = [];
  let cur = label;
  while (cur?.parentUuid && labelsMap.has(cur.parentUuid)) {
    chain.unshift(cur);
    cur = labelsMap.get(cur.parentUuid);
  }
  return chain.map((node, i) => {
    const siblings = byParent.get(node.parentUuid ?? null) ?? [];
    const hasNext = siblings.length > 0 && siblings[siblings.length - 1].uuid !== node.uuid;
    if (i === chain.length - 1) return hasNext ? 'elbow-cont' : 'elbow';
    return hasNext ? 'line' : 'blank';
  });
}

/** Total number of labels nested under `uuid`, at any depth. */
function countDescendants(uuid, byParent) {
  let total = 0;
  const queue = [uuid];
  while (queue.length > 0) {
    for (const child of byParent.get(queue.shift()) ?? []) {
      total += 1;
      queue.push(child.uuid);
    }
  }
  return total;
}

/**
 * Build the shared compact row. Callers add their own mode-specific controls by
 * inserting around the returned `swatch` / `nameEl` and appending to `row`.
 *
 * @param {object} label
 * @param {{
 *   byParent: Map<string|null, object[]>,
 *   labelsMap: Map<string, object>,
 *   collapseKey?: string|null,   omit to render a permanently-expanded tree
 *   onToggle?: (() => void)|null,
 *   showCaret?: boolean,
 *   flat?: boolean,              skip rails entirely (dynamic-label sections)
 * }} opts
 * @returns {{ row: HTMLDivElement, swatch: HTMLSpanElement, nameEl: HTMLSpanElement,
 *            collapsed: boolean, hasChildren: boolean }}
 */
export function buildLabelRow(label, {
  byParent,
  labelsMap,
  collapseKey = null,
  onToggle = null,
  showCaret = true,
  flat = false,
} = {}) {
  const row = document.createElement('div');
  row.className = 'lt-row';
  row.dataset.uuid = label.uuid;
  row.dataset.testid = 'label-list-row';

  row.appendChild(makeRails(flat ? [] : railKindsFor(label, byParent, labelsMap)));

  const hasChildren = (byParent.get(label.uuid) ?? []).length > 0;
  const collapsed = Boolean(collapseKey) && collapsedSet(collapseKey).has(label.uuid);

  if (showCaret) {
    const caret = document.createElement('button');
    caret.type = 'button';
    caret.className = `lt-caret${hasChildren && collapseKey ? '' : ' lt-caret--leaf'}`;
    caret.innerHTML = ICONS.caret;
    caret.setAttribute('aria-expanded', String(!collapsed));
    if (hasChildren && collapseKey) {
      caret.title = `${collapsed ? 'Expand' : 'Collapse'} ${label.name}`;
      caret.setAttribute('aria-label', caret.title);
      caret.addEventListener('click', (e) => { e.stopPropagation(); onToggle?.(); });
    } else {
      // Leaves keep the caret's width so names stay in one column, but must not
      // be reachable by keyboard or announced.
      caret.tabIndex = -1;
      caret.setAttribute('aria-hidden', 'true');
    }
    row.appendChild(caret);
  }

  const swatch = document.createElement('span');
  swatch.className = 'lt-swatch';
  swatch.style.background = label.color ?? '#888';
  row.appendChild(swatch);

  const nameEl = document.createElement('span');
  nameEl.className = 'lt-name';
  nameEl.dataset.testid = 'label-list-row-name';
  nameEl.textContent = label.name;
  row.appendChild(nameEl);

  if (collapsed && hasChildren) {
    const kids = document.createElement('span');
    kids.className = 'lt-kids';
    const n = countDescendants(label.uuid, byParent);
    kids.textContent = String(n);
    kids.title = `${n} nested label${n === 1 ? '' : 's'} hidden — expand to see them`;
    row.appendChild(kids);
  }

  return { row, swatch, nameEl, collapsed, hasChildren };
}

/**
 * Insert an inline create form directly beneath `anchorRow`, one level deeper.
 *
 * @param {HTMLElement} anchorRow
 * @param {{ parentUuid: string, railKinds: string[], onRerender: () => void }} opts
 */
function openInlineCreate(anchorRow, { parentUuid, railKinds, onRerender }) {
  // Only one create form open at a time within this tree.
  anchorRow.parentElement?.querySelectorAll('[data-create-row]').forEach((el) => el.remove());

  const row = document.createElement('div');
  row.className = 'lt-row';
  row.dataset.createRow = parentUuid;
  row.appendChild(makeRails(railKinds));

  // Occupy the caret column so the new name lines up with its future siblings.
  const spacer = document.createElement('span');
  spacer.className = 'lt-caret lt-caret--leaf';
  spacer.setAttribute('aria-hidden', 'true');
  row.appendChild(spacer);

  anchorRow.after(row);
  openRowEditor(row, {
    color: DEFAULT_COLOR,
    placeholder: 'New Label Name',
    onSave: async ({ name, color }) => {
      if (!name) { toast('Error: Name required', 5000, 'error'); return; }
      try {
        await createNewLabel(name, color, parentUuid);
        onRerender();
      } catch (err) {
        toast(err.message, 5000, 'error');
      }
    },
    onCancel: () => row.remove(),
  });
}

function renderEditRows(byParent, container, parentUuid, indentLevel, onRerender, labelsMap, collapseKey) {
  const children = byParent.get(parentUuid ?? null) ?? [];
  for (const label of children) {
    const { name, color, uuid } = label;

    const { row, swatch, nameEl, collapsed } = buildLabelRow(label, {
      byParent,
      labelsMap,
      collapseKey,
      onToggle: () => { toggleCollapsed(collapseKey, uuid); onRerender(); },
    });
    row.dataset.testid = 'label-list-row-edit';

    // The swatch *is* the colour picker — recolouring no longer requires
    // entering edit mode first. Committing the OS picker PATCHes immediately.
    const well = makeColorWell(color ?? DEFAULT_COLOR, async (newColor) => {
      try {
        await updateLabel(uuid, name, newColor);
        onRerender();
      } catch (err) {
        toast(err.message, 5000, 'error');
      }
    });
    swatch.replaceWith(well);

    const startRename = () => openRowEditor(row, {
      name,
      color: color ?? DEFAULT_COLOR,
      uuid,
      placeholder: 'Label name',
      onSave: async ({ name: newName, color: newColor }) => {
        if (!newName) { toast('Error: Name required', 5000, 'error'); return; }
        try {
          await updateLabel(uuid, newName, newColor);
          onRerender();
        } catch (err) {
          toast(err.message, 5000, 'error');
        }
      },
      onCancel: () => onRerender(),
    });

    nameEl.classList.add('lt-name--action');
    nameEl.title = 'Click to rename';
    nameEl.addEventListener('click', startRename);

    const acts = document.createElement('div');
    acts.className = 'lt-acts';
    acts.append(
      iconAction('addChild', 'Add Child', {
        onClick: () => {
          // A new child is always the last one, so its own rail is a plain
          // elbow; the ancestors above it keep whatever they already had.
          const parentRails = railKindsFor(label, byParent, labelsMap);
          const railKinds = [
            ...parentRails.map((k) => (k === 'elbow-cont' ? 'line' : k === 'elbow' ? 'blank' : k)),
            'elbow',
          ];
          // Adding to a collapsed parent expands it in state but doesn't
          // re-render — a re-render here would throw away the form we're about
          // to open. The row's caret catches up on the save that follows.
          if (collapsed) toggleCollapsed(collapseKey, uuid);
          openInlineCreate(row, { parentUuid: uuid, railKinds, onRerender });
        },
      }),
      iconAction('move', 'Move', {
        onClick: () => toggleMovePicker(row, {
          label,
          labelsMap,
          indentLevel,
          onMove: async (newParentUuid) => {
            await updateLabel(uuid, name, color, { parentUuid: newParentUuid });
            onRerender();
            notifyHierarchyMoveImpact(name);
          },
        }),
      }),
      iconAction('home', 'Home Assistant', {
        active: label.publishToHa === true,
        onClick: () => toggleHaPicker(row, {
          label,
          indentLevel,
          onSave: async (flags) => {
            await updateLabel(uuid, name, color, flags);
            onRerender();
          },
        }),
      }),
      iconAction('variants', 'Exclusive Variants', {
        active: label.exclusiveVariants === true,
        onClick: () => toggleVariantsPicker(row, {
          label,
          indentLevel,
          onSave: async (flags) => {
            await updateLabel(uuid, name, color, flags);
            onRerender();
            // Not awaited: this is a side notice about a save that already
            // succeeded, so it must not hold the panel open or the Save
            // button disabled while it round-trips to inference_server.
            notifySuppressionGroupsImpact(name);
          },
        }),
      }),
      iconAction('edit', 'Edit', { onClick: startRename }),
      iconAction('trash', 'Delete', {
        className: 'lt-act--danger',
        onClick: async () => {
          if (!window.confirm(`Delete label "${name}"?`)) return;
          try {
            await deleteLabel(uuid);
            onRerender();
          } catch (err) {
            toast(err.message, 5000, 'error');
          }
        },
      }),
    );
    row.appendChild(acts);
    container.appendChild(row);

    if (!collapsed) {
      renderEditRows(byParent, container, uuid, indentLevel + 1, onRerender, labelsMap, collapseKey);
    }
  }
}

async function mountEditMode(container, { filter, onRerender }) {
  const inner = document.createElement('div');
  inner.className = 'label-tree';
  container.appendChild(inner);

  const allLabels = await fetchLabels();
  const hideDynamic = (label) => label.labelType !== 'dynamic';
  const composedFilter = filter ? (label) => hideDynamic(label) && filter(label) : hideDynamic;
  const labelsMap = new Map([...allLabels].filter(([, l]) => composedFilter(l)));

  function addCreateRow(scopeValue) {
    const placeholder = scopeValue === 'object' ? 'New Object Label' : 'New Image Label';
    const row = document.createElement('div');
    row.className = 'lt-newrow';
    row.dataset.testid = scopeValue === 'object' ? 'label-list-new-object-row' : 'label-list-new-image-row';
    row.dataset.tour = scopeValue === 'object' ? 'config.labels.newObject' : 'config.labels.newImage';

    const well = makeColorWell(DEFAULT_COLOR);
    const input = document.createElement('input');
    input.type = 'text';
    input.placeholder = placeholder;
    input.setAttribute('aria-label', placeholder);

    const submit = async () => {
      const name = input.value.trim();
      if (!name) { toast('Error: Name required', 5000, 'error'); return; }
      try {
        await createNewLabel(name, well.getValue() || DEFAULT_COLOR, null, 'standard', scopeValue);
        onRerender();
      } catch (err) {
        toast(err.message, 5000, 'error');
      }
    };
    input.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') { e.preventDefault(); submit(); }
    });

    row.append(well, input, iconAction('add', 'Add label', { onClick: submit }));
    inner.appendChild(row);
  }

  /**
   * Render one scope section: header (with count + collapse-all) then its tree.
   *
   * Each section is scoped to its own labelsMap — not just a filtered byParent
   * view — so the Move picker's target list only ever offers same-scope labels.
   * The two sections are already separate trees on screen; this keeps "Move to…"
   * consistent with what's visible.
   */
  function addSection({ scope, title, collapseKey, extraControls = [] }) {
    const scopedMap = new Map([...labelsMap].filter(([, l]) => l.scope === scope));
    const byParent = buildHierarchy(scopedMap, null);

    const expanded = anyExpanded(collapseKey, byParent);
    const hasParents = [...byParent.keys()].some((k) => k !== null);
    const collapseAll = hasParents
      ? iconAction('caret', expanded ? 'Collapse all' : 'Expand all', {
        className: expanded ? '' : 'lt-act--collapsed',
        onClick: () => { setAllCollapsed(collapseKey, byParent, expanded); onRerender(); },
      })
      : null;

    inner.appendChild(makeSectionHeader(title, scopedMap.size, collapseAll, ...extraControls));
    renderEditRows(byParent, inner, null, 0, onRerender, scopedMap, collapseKey);
    if (scopedMap.size === 0) {
      const empty = document.createElement('div');
      empty.className = 'lt-empty';
      empty.textContent = `No ${title.toLowerCase()} yet`;
      inner.appendChild(empty);
    }
    addCreateRow(scope);
  }

  const refreshBtn = iconAction('refresh', 'Refresh', {
    onClick: () => { clearLabelsCache(); onRerender(); },
  });

  addSection({ scope: 'object', title: 'Object Labels', collapseKey: 'edit-object', extraControls: [refreshBtn] });
  addSection({ scope: 'image', title: 'Image Labels', collapseKey: 'edit-image' });

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

  // Standing count of what this image still owes the active run — the per-row
  // markers say which labels, this says how many are left without scrolling.
  const gapSummary = document.createElement('div');
  gapSummary.className = 'label-gap-summary';
  gapSummary.dataset.testid = 'label-gap-summary';
  gapSummary.hidden = true;
  stickyHeader.appendChild(gapSummary);

  container.appendChild(stickyHeader);

  const rawLabelsMap = await fetchLabels();
  const filteredMap = activeTrainingConfig.applyLabelFilter(rawLabelsMap);
  // Split by (scope, labelType): object-scope object labels / image-scope labels / dynamic labels.
  const regularMap = new Map([...filteredMap].filter(([, l]) => l.labelType !== 'dynamic' && l.scope !== 'image'));
  const imageClassMap = new Map([...filteredMap].filter(([, l]) => l.labelType !== 'dynamic' && l.scope === 'image'));
  const dynamicMap = new Map([...filteredMap].filter(([, l]) => l.labelType === 'dynamic'));
  const filterNotice = makeLabelFilterNotice(rawLabelsMap.size - regularMap.size - imageClassMap.size - dynamicMap.size);
  if (filterNotice) container.appendChild(filterNotice);
  // Rows live in their own element so remount() can rebuild them without
  // touching the sticky header (or the listeners wired to its buttons).
  const listEl = document.createElement('div');
  listEl.className = 'label-tree';
  container.appendChild(listEl);
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

  // ── Training-run gaps ───────────────────────────────────────────────────
  // What the active training run still needs from THIS image. The server owns
  // the definition (GET image-gaps == the coverage table's own notion of a gap,
  // rollups included), so this set is only ever fetched, never derived here.
  /** @type {Set<string>} label uuids the server reported as unresolved for this image */
  let gapUuids = new Set();
  let _gapSeq = 0;

  function effectiveStatus(uuid) {
    if (derivedPresent.has(uuid)) return 'present';
    return serverStates.get(uuid) ?? 'uncertain';
  }

  /**
   * Whether a gap the server reported is still open, given edits made since the
   * fetch (including unsaved ones).
   *
   * This mirrors the server's accounting rule (_co_label_accounted_for) but is
   * deliberately only allowed to *close* a gap the server already named — it
   * never invents one. So a drift between the two rules costs at most a stale
   * highlight until the next save re-fetches, never a false alarm.
   */
  function isOpenGap(uuid) {
    if (!gapUuids.has(uuid)) return false;
    const status = effectiveStatus(uuid);
    // Image-scope labels are settled by either verdict; object-scope ones need a
    // verified box (Present alone is not evidence) or an explicit Absent.
    // Read the scope from the full label map, not the rendered rows: a gap on a
    // label the current filter hides still counts toward the header summary.
    return rawLabelsMap.get(uuid)?.scope === 'image'
      ? status === 'uncertain'
      : !(derivedPresent.has(uuid) || status === 'absent');
  }

  /** Name of the run the gaps are judged against, for titles. */
  function _gapRunName() {
    return activeTrainingConfig.getActive()?.name ?? 'training run';
  }

  /** Header line: how many of this run's labels this image still owes. */
  function refreshGapSummary() {
    const open = [...gapUuids].filter(isOpenGap);
    gapSummary.hidden = open.length === 0;
    if (!open.length) return;
    const names = open.map(u => rawLabelsMap.get(u)?.name ?? u);
    gapSummary.textContent = `⚠ ${open.length} label${open.length === 1 ? '' : 's'} needed for "${_gapRunName()}"`;
    gapSummary.title = names.join(', ');
  }

  /** Re-read this image's gaps from the server. No-op without an active run. */
  async function loadGaps() {
    const tcNow = activeTrainingConfig.getActive();
    const imageUuid = runtime?.state?.image?.uuid;
    const seq = ++_gapSeq;
    if (!tcNow || !imageUuid) {
      gapUuids = new Set();
      refreshStatusButtons();
      return;
    }
    let gaps;
    try {
      gaps = await fetchTrainingConfigImageGaps(tcNow.uuid, [imageUuid]);
    } catch (err) {
      // A missing gap list must not blank out the working panel — leave the
      // last known set alone and let the next load try again.
      console.warn('[labelList] Failed to fetch training-run gaps:', err);
      return;
    }
    if (seq !== _gapSeq) return; // superseded by a newer image load
    gapUuids = new Set((gaps.get(imageUuid) ?? []).map(g => g.labelUuid));
    refreshStatusButtons();
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

  /**
   * Re-apply unsaved marks on top of a freshly fetched server snapshot.
   *
   * A server read is a snapshot from when the request was issued, so a mark
   * made while it was in flight is missing from it — writing the response
   * straight into `serverStates` would silently revert marks still queued in
   * `pendingFlush` for the next save (mark absent, see absent, watch it flip
   * back). `pendingFlush` is the authority for anything it holds: cleared on
   * image change and on successful flush, so an entry always means "asked for,
   * not yet told to the server".
   * @param {Map<string, 'present'|'absent'>} states
   */
  function applyPendingOver(states) {
    for (const [uuid, status] of pendingFlush) {
      if (status === 'uncertain') states.delete(uuid);
      else states.set(uuid, /** @type {'present'|'absent'} */ (status));
    }
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

  function renderSelectRows(parentUuid, indent, bp = byParent, opts = {}) {
    const { labelsMap: sectionMap = regularMap, collapseKey = null } = opts;
    const children = bp.get(parentUuid ?? null) ?? [];
    for (const label of children) {
      labelByUuid.set(label.uuid, label);
      const { row, swatch, nameEl, collapsed } = buildLabelRow(label, {
        byParent: bp,
        labelsMap: sectionMap,
        collapseKey,
        onToggle: () => { toggleCollapsed(collapseKey, label.uuid); remount(); },
      });
      row.classList.add('label-panel-row');
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
              gapsConfig: tc,
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
              gapsConfig: tc,
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
              gapsConfig: tc,
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
              gapsConfig: tc,
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
      // Named, not "the first button in the row" — the disclosure caret is now
      // also a <button> and sits ahead of this one.
      btn.className = 'label-status-btn';
      btn.type = 'button';
      Object.assign(btn.style, {
        flexShrink: '0',
        width: '22px',
        height: '20px',
        fontSize: '0.8em',
        border: '1px solid #444',
        borderRadius: '3px',
        padding: '0',
        // Pico puts `margin-bottom: var(--pico-spacing)` (1rem = 16px) on every
        // bare <button>. On a flex row that margin is part of the item's margin
        // box, so it sets the flex line's height: these rows measured 36px
        // against their own `min-height: 24px`, wasting ~12px on every label.
        // The explicit width/height above do NOT suppress it. Same trap as the
        // bare-<label> one in NOTES.md — this reset is load-bearing.
        margin: '0',
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

        // A label this image still owes the active training run. The marker sits
        // on the status button — the control that resolves it — so the eye lands
        // where the click goes, and it clears the moment the status settles.
        const isGap = isOpenGap(label.uuid);
        row.classList.toggle('label-panel-row--gap', isGap);
        btn.classList.toggle('label-status-btn--gap', isGap);
        if (isGap) {
          btn.title = label.scope === 'image'
            ? `Needed by "${_gapRunName()}" — mark it present or absent`
            : `Needed by "${_gapRunName()}" — verify a box or mark it absent`;
        }
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

      // Between the caret and the swatch, so the presence buttons line up in a
      // column instead of shifting right with each level of nesting.
      swatch.before(btn);

      row.addEventListener('click', () => {
        // Image-scope labels have no bboxes — don't arm the bbox draw tool.
        if (label.scope === 'image') return;
        _lastSelectedUuid = label.uuid;
        setTool(runtime, `bbox:${label.uuid}`);
      });

      listEl.appendChild(row);
      if (!collapsed) renderSelectRows(label.uuid, indent + 1, bp, opts);
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

  // Named so the hotkeys can fire the same code the buttons do — a second copy
  // of the "which uuids does this apply to" filter is exactly the sort of thing
  // that drifts once and then silently marks the wrong set absent.
  const doMarkAbsent = () => bulkOp(
    targetedUuids().filter(uuid => effectiveStatus(uuid) !== 'present'),
    'absent',
  );

  const doResetAll = () => bulkOp(
    targetedUuids().filter(uuid => serverStates.has(uuid)),
    'uncertain',
  );

  markAbsentBtn.addEventListener('click', doMarkAbsent);
  resetAllBtn.addEventListener('click', doResetAll);

  // Hotkey hooks. Both act on the panel's current targeting, which lives in this
  // closure, so hotkeys.js dispatches through the registry rather than
  // reimplementing the target set. Only registered when there is a runtime to
  // key them to — a panel with no canvas has nothing to mark dirty.
  let unregisterBulkActions = () => {};
  if (runtime) {
    const unregAbsent = registerCanvasAction(runtime.id, 'markAbsent', doMarkAbsent);
    const unregReset = registerCanvasAction(runtime.id, 'resetLabels', doResetAll);
    unregisterBulkActions = () => { unregAbsent(); unregReset(); };
  }

  /**
   * Render a labelled section: a header (title + [◉ all | ⊘ none] master
   * control) followed by its rows. The title lives in its own <div> so
   * callers/tests can still find it by exact text. Returns nothing; the master
   * control is wired to the UUIDs that appeared during this section's render.
   * @param {string} title
   * @param {Map<string|null, object[]>} bp
   * @param {{ color: string, italic?: boolean, labelsMap: Map<string,object>, collapseKey: string }} opts
   */
  function renderSection(title, bp, { color, italic = false, labelsMap: sectionMap, collapseKey }) {
    const expanded = anyExpanded(collapseKey, bp);
    const hasParents = [...bp.keys()].some((k) => k !== null);
    const collapseAll = hasParents
      ? iconAction('caret', expanded ? 'Collapse all' : 'Expand all', {
        className: expanded ? '' : 'lt-act--collapsed',
        onClick: () => { setAllCollapsed(collapseKey, bp, expanded); remount(); },
      })
      : null;

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
    master.append(allBtn, sep, noneBtn);

    const header = makeSectionHeader(title, null, collapseAll, master);
    header.style.color = color;
    if (italic) header.style.fontStyle = 'italic';
    listEl.appendChild(header);

    // Capture which UUIDs this section contributes so the master targets them.
    const before = new Set(labelByUuid.keys());
    renderSelectRows(null, 0, bp, { labelsMap: sectionMap, collapseKey });
    const sectionUuids = [...labelByUuid.keys()].filter(u => !before.has(u));
    allBtn.addEventListener('click', () => setTargetedMany(sectionUuids, true));
    noneBtn.addEventListener('click', () => setTargetedMany(sectionUuids, false));
  }

  /**
   * Redraw just the rows.
   *
   * Collapsing must NOT go through createLabelList's mount(): that re-runs this
   * whole function, and `pendingFlush` (presence changes not yet written to the
   * server) lives in this closure. Rebuilding only `listEl` keeps every piece of
   * unsaved state — pendingFlush, serverStates, targetOverrides — intact.
   */
  function remount() {
    listEl.innerHTML = '';
    labelByUuid.clear();
    rowUpdaters.clear();
    targetUpdaters.clear();
    renderSections();
    refreshStatusButtons();
    const tool = runtime?.tool ?? '';
    syncHighlight(tool.startsWith('bbox:') ? tool.slice(5) : null);
    updateBulkLabel();
  }

  // Object Labels are the primary section; Image Classification and Dynamic
  // Labels follow. Empty sections are omitted.
  function renderSections() {
    if (regularMap.size > 0) {
      renderSection('Object Labels', byParent, {
        color: '#aaa', labelsMap: regularMap, collapseKey: 'select-object',
      });
    }
    if (imageClassMap.size > 0) {
      renderSection('Image Classification', byParentImageClass, {
        color: '#88aaff', labelsMap: imageClassMap, collapseKey: 'select-image',
      });
    }
    if (dynamicMap.size > 0) {
      renderSection('Dynamic Labels', byParentDynamic, {
        color: '#888', italic: true, labelsMap: dynamicMap, collapseKey: 'select-dynamic',
      });
    }
  }

  renderSections();
  updateBulkLabel();
  // The panel can mount with an image already on the canvas — a tab switch, or
  // the remount a training-config publish triggers. Pick it up rather than
  // waiting for the next load event (see adoptImage).
  if (runtime?.state?.image?.image_path) {
    adoptImage(runtime.state.image.image_path);
  } else {
    loadGaps();
  }

  function refreshStatusButtons() {
    for (const [uuid, update] of rowUpdaters) {
      update(effectiveStatus(uuid));
    }
    refreshGapSummary();
  }

  function syncHighlight(activeLabelUuid) {
    container.querySelectorAll('.label-panel-row').forEach(row => {
      const isActive = row.dataset.uuid === activeLabelUuid;
      row.style.outline = isActive ? '2px solid #ff0033' : '';
      row.style.background = isActive ? '#ff003322' : '';
    });
  }

  /**
   * Adopt `path` as the panel's current image and load its server state.
   *
   * Shared by the IMAGE_LOADED_ONTO_CANVAS handler and the mount itself. The
   * mount call covers the case where an image is already on the canvas when
   * this panel mounts — a tab switch, or the remount a training-config publish
   * triggers — so label states and gaps are read for it straight away instead
   * of waiting for the next load event.
   */
  async function adoptImage(path) {
    // Discard any pending presence changes for the previous image.
    pendingFlush.clear();
    currentImagePath = path;
    // Gaps are fetched independently of the label-state fetch below, not after
    // it: they are separate requests answering separate questions, and chaining
    // them meant one slow (or stalled) label-state response silently suppressed
    // the gap highlight entirely.
    gapUuids = new Set();
    loadGaps();
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
    applyPendingOver(serverStates);
    derivedPresent = computeDerivedPresent(runtime?.state?.image?.bboxes);
    // Gaps were already requested above; this just repaints with the label
    // states that have now arrived (and whatever gaps landed meanwhile).
    refreshStatusButtons();
  }

  // Fetch server label states when a new image is loaded; discard any pending unsaved changes.
  const unsubImage = events.subscribeToCanvas(
    runtime.id,
    EventTypes.IMAGE_LOADED_ONTO_CANVAS,
    ({ path }) => adoptImage(path)
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
      // A save with no pending presence changes (boxes only) skips the flush
      // below but can still have closed gaps — resync before bailing out.
      if (!currentImagePath || !pendingFlush.size || _flushInFlight) { loadGaps(); return; }
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
      // Re-read gaps from the authority once the save has landed: boxes and
      // presence marks written just now may have closed some, and rollup rules
      // the local check doesn't model may have closed others.
      loadGaps();
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
        applyPendingOver(serverStates);
        derivedPresent = computeDerivedPresent(runtime?.state?.image?.bboxes);
        refreshStatusButtons();
      }
    } catch (err) {
      console.error('Failed to fetch initial image labels:', err);
    }
  }

  return {
    /**
     * Repaint the stats-derived parts (the ✓/⚠ badges) after a stats refresh,
     * WITHOUT rebuilding the panel.
     *
     * A full re-mount would tear down this closure, and `pendingFlush` —
     * presence marks the user has made but not saved yet — lives in it. Since
     * the stats/readiness fetch is fired off by every canvas save (and takes
     * seconds on a large database), a re-mount there both wiped those marks and
     * re-read label states behind the writes still in flight. Only the badge
     * numbers change on a stats update, so redraw the rows and keep the state.
     */
    onStatsUpdated() {
      _statsMap = _buildStatsMap();
      remount();
    },
    destroy() {
      closeMenu();
      unsubImage();
      unsubBboxes();
      unsubSaved();
      unsubTool();
      unregisterBulkActions();
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
  container.classList.add('label-tree');
  // Per-scope so folding the object tree doesn't fold the image one.
  const collapseKey = `modelconfig-${scopeFilter ?? 'all'}`;

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

      const { row, swatch, nameEl, collapsed } = buildLabelRow(label, {
        byParent,
        labelsMap: displayMap,
        collapseKey,
        onToggle: () => { toggleCollapsed(collapseKey, label.uuid); rerender(); },
      });

      const cb = document.createElement('input');
      cb.type = 'checkbox';
      cb.checked = selected.has(label.uuid);
      cb.className = 'label-row-checkbox';
      cb.setAttribute('aria-label', label.name);

      nameEl.classList.add('lt-name--action');
      nameEl.style.cursor = 'pointer';
      nameEl.addEventListener('click', () => {
        cb.checked = !cb.checked;
        cb.dispatchEvent(new Event('change'));
      });

      // Checkbox goes between the caret and the swatch, so the boxes form a
      // column of their own rather than shifting with depth.
      swatch.before(cb);

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

      // No re-parent control here. Editing the label hierarchy belongs on
      // Configuration › Labels & Tags — that is the one screen whose job is
      // editing labels, and having a second entry point meant a model config
      // could silently reshape the vocabulary every other screen depends on.

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
            // Enabling submodel auto-selects the label and seeds a starting spec. The
            // spec is now only a default the user can change to anything, including
            // its parent's — membership of this Map is what makes it a sub-model, and
            // it round-trips as the node's own isBoundary flag.
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

      if (isParent && !collapsed) appendRows(label.uuid, indent + 1);
    }
  }

  /**
   * Redraw in place. Collapsing can't go through createLabelList's mount(),
   * because that re-seeds the tree from the caller's `initialSelected` /
   * `initialSubmodels` snapshot and would throw away every checkbox the user
   * has ticked since the form opened. `selected` / `submodels` /
   * `rollUpUnchecked` are the live sets, so re-rendering from them is lossless.
   */
  function rerender() {
    container.innerHTML = '';
    render();
  }

  function render() {
    const emptyByParent = new Map();

    if (displayMap.size > 0) {
      const expanded = anyExpanded(collapseKey, byParent);
      const hasParents = childCountMap.size > 0;
      container.appendChild(makeSectionHeader(
        scopeFilter === 'image' ? 'Image Labels' : 'Object Labels',
        displayMap.size,
        hasParents
          ? iconAction('caret', expanded ? 'Collapse all' : 'Expand all', {
            className: expanded ? '' : 'lt-act--collapsed',
            onClick: () => { setAllCollapsed(collapseKey, byParent, expanded); rerender(); },
          })
          : null,
      ));
    }

    appendRows(null, 0);

    // ── Dynamic Labels section ──────────────────────────────────────────────
    // A flat list with checkboxes only — no sub-model or roll-up toggles, since
    // dynamic labels have no structured parent/child hierarchy to roll up.
    if (dynamicMap.size > 0) {
      container.appendChild(makeSectionHeader('Dynamic Labels', dynamicMap.size));

      for (const label of dynamicMap.values()) {
        const { row, swatch, nameEl } = buildLabelRow(label, {
          byParent: emptyByParent,
          labelsMap: dynamicMap,
          flat: true,
        });

        const cb = document.createElement('input');
        cb.type = 'checkbox';
        cb.checked = selected.has(label.uuid);
        cb.className = 'label-row-checkbox';
        cb.setAttribute('aria-label', label.name);

        nameEl.classList.add('lt-name--action');
        nameEl.style.cursor = 'pointer';
        nameEl.addEventListener('click', () => {
          cb.checked = !cb.checked;
          cb.dispatchEvent(new Event('change'));
        });

        swatch.before(cb);

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
  }

  render();

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
            defaultSpec, onSubmodelsChanged, onRollUpUncheckedChanged, scopeFilter,
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
      // `statsUpdated` means only the cached label stats/readiness changed — the
      // active run, its label set and the presets are all the same. Rebuilding
      // the whole panel for that destroys unsaved presence marks (they live in
      // mountSelectMode's closure) and re-reads label states behind writes that
      // may still be in flight, so take the light path when the handle offers it.
      [EventTypes.ACTIVE_TRAINING_CONFIG_CHANGED, (payload) => {
        // While a mount is in flight `_handle` is the outgoing panel; the one
        // being built reads the fresh cached stats itself, so drop this.
        if (payload?.statsUpdated && _mounting) return;
        if (payload?.statsUpdated && _handle?.onStatsUpdated) _handle.onStatsUpdated();
        else mount();
      }],
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
