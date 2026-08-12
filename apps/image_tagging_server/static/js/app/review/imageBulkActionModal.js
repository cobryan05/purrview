/**
 * Generic "apply an action to a set of images" modal — extracted out of
 * review/main.js so it can be reused outside the review grid's own toolbar
 * (e.g. runClassifierOnResults.js's classifier-match confirmation step).
 *
 * Supports the full image-level action set: label status (setPresent/setAbsent/
 * setUncertain), tags (addTags/removeTags/setTagAbsent/setTagUncertain), and the
 * two removal actions (delete / removeFromDb).
 */

import { deleteImage, deleteImageDbOnly } from '/app-static/js/shared/api/images.js';
import { showConfirmModal } from '/app-static/js/ui/utils/confirmModal.js';
import { fetchTags } from '/app-static/js/shared/api/tags.js';
import { filterVisibleTags } from '/app-static/js/shared/activeModel.js';
import { activeTrainingConfig } from '/app-static/js/shared/activeTrainingConfig.js';
import { bulkApplyImagesByUuids } from '/app-static/js/shared/api/review.js';
import { flattenHierarchy } from '/app-static/js/ui/labelList.js';
import { Logger, toast } from '/app-static/js/ui/utils/logging.js';
import { acquireModalLayer, applyModalCascade, registerEscHandler, unregisterEscHandler } from '/app-static/js/ui/utils/modalZIndex.js';
import { trapModalFocus } from '/app-static/js/ui/utils/modalShell.js';
import { MultiSelectField } from '/app-static/js/ui/utils/fields/MultiSelectField.js';
import { createMultiSelectDropdown } from '/app-static/js/ui/utils/createMultiSelectDropdown.js';

export const TAG_ACTIONS = ['addTags', 'removeTags', 'setTagAbsent', 'setTagUncertain'];
export const LABEL_ACTIONS = ['setPresent', 'setAbsent', 'setUncertain'];
/**
 * Actions that make the image disappear from the review set. Both drop the DB
 * record (and its bboxes, tags and sidecar); they differ only in whether the
 * image file on disk goes with it. Callers treat them alike for card staleness
 * and IMAGE_DELETED bookkeeping, so they're grouped rather than special-cased
 * one at a time.
 */
export const REMOVAL_ACTIONS = ['delete', 'removeFromDb'];

export function isRemovalAction(action) {
  return REMOVAL_ACTIONS.includes(action);
}

/**
 * Confirm-modal for a removal action, with copy that spells out the disk-file
 * consequence — the only difference between the two, and the one the user has
 * to get right.
 *
 * @param {string} action  'delete' | 'removeFromDb'
 * @param {number} count   how many images the action will hit
 * @param {string} [scopeNote]  extra clause naming the scope, e.g.
 *   'from the full candidate list' or 'matching the current filter, across all pages'
 * @returns {Promise<boolean>}
 */
export function confirmRemovalAction(action, count, scopeNote = '') {
  const n = count.toLocaleString();
  const noun = count === 1 ? 'image' : 'images';
  const scope = scopeNote ? ` ${scopeNote}` : '';
  return action === 'delete'
    ? showConfirmModal({
        title: `Delete ${n} ${noun}`,
        message: `Permanently delete ${n} ${noun}${scope}? This removes the files and all their annotations from disk. This cannot be undone.`,
        confirmText: 'Delete',
        danger: true,
      })
    : showConfirmModal({
        title: `Remove ${n} ${noun} from Database`,
        message: `Remove ${n} ${noun}${scope} from the database? Their bboxes, tags and label status are deleted `
          + 'and the JSON sidecars are removed, but the image files stay on disk. This cannot be undone.',
        confirmText: 'Remove from DB',
        danger: true,
      });
}

/** Per-image API call behind a removal action. */
export function removeSingleImage(action, imageUuid) {
  return action === 'delete' ? deleteImage(imageUuid) : deleteImageDbOnly(imageUuid);
}

/** Past-tense verb for status/toast copy after a removal action ran. */
export function removalVerb(action) {
  return action === 'delete' ? 'Deleted' : 'Removed from DB';
}

/**
 * @param {Array<{img: object}>} entries
 * @param {Map<string, object>} labels
 * @param {(action: string, labelUuids: string[], tagUuids: string[]) => Promise<void>} onConfirm
 *   `labelUuids` is a list — label actions can target several labels at once —
 *   and is empty for tag and delete actions.
 * @param {string|null} [titleText]
 * @param {object} [opts]
 * @param {HTMLElement} [opts.previewNode] — inserted directly under the title.
 *   Callers use it to show a thumbnail preview of the affected range; the node
 *   may still be populating asynchronously when the modal opens.
 */
export async function showImageBulkActionModal(entries, labels, onConfirm, titleText = null, opts = {}) {
  const tags = await fetchTags().catch(() => new Map());

  const overlay = document.createElement('div');
  overlay.className = 'bulk-modal-overlay';
  const _layer = acquireModalLayer({ dialog: true, overlay });
  overlay.style.zIndex = _layer.zIndex;

  const dialog = document.createElement('div');
  // max-height/overflow are load-bearing, not cosmetic: #bottom-bars is fixed
  // ABOVE the modal z-index band, so a dialog tall enough to reach the bottom
  // of the viewport has its Apply row covered by the active-model bar and
  // becomes unclickable. Bounding the height keeps the footer clear of it.
  dialog.style.cssText = `background:#222;border:1px solid #555;border-radius:6px;
    padding:1.5em;min-width:380px;display:flex;flex-direction:column;gap:0.8em;
    max-height:85vh;overflow-y:auto;`;

  const title = document.createElement('strong');
  title.textContent = titleText ?? `Apply to ${entries.length} selected image${entries.length === 1 ? '' : 's'}`;

  const actions = [
    { value: 'setPresent', label: 'Mark label Present (positive training example)' },
    { value: 'setAbsent', label: 'Mark label Absent (confirmed not present)' },
    { value: 'setUncertain', label: 'Clear label status → Unreviewed' },
    { value: 'addTags', label: 'Add Tags' },
    { value: 'removeTags', label: 'Remove Tags' },
    { value: 'setTagAbsent', label: 'Mark Tag Verified Absent' },
    { value: 'setTagUncertain', label: 'Clear Tag Verification → Uncertain' },
    { value: 'delete', label: 'Delete Images (removes the files from disk)' },
    { value: 'removeFromDb', label: 'Remove Images from Database (keeps the files on disk)' },
  ];

  const radioGroup = document.createElement('div');
  radioGroup.style.cssText = 'display:flex;flex-direction:column;gap:0.4em;';
  let selectedAction = 'setPresent';

  for (const action of actions) {
    const lbl = document.createElement('label');
    lbl.style.cssText = 'display:flex;align-items:center;gap:0.5em;cursor:pointer;';
    const radio = document.createElement('input');
    radio.type = 'radio';
    radio.name = 'image-bulk-action';
    radio.value = action.value;
    if (action.value === 'setPresent') radio.checked = true;
    radio.addEventListener('change', () => {
      selectedAction = action.value;
      labelPickerRow.style.display = LABEL_ACTIONS.includes(selectedAction) ? 'flex' : 'none';
      tagPickerRow.style.display = TAG_ACTIONS.includes(selectedAction) ? 'flex' : 'none';
    });
    lbl.appendChild(radio);
    lbl.appendChild(document.createTextNode(action.label));
    radioGroup.appendChild(lbl);
  }

  // ── Label picker — multi-select ───────────────────────────────────────────
  // A label action applies to every checked label in one pass, so "mark Absent
  // for the six labels this training run cares about" is one trip through the
  // modal rather than six. Labels in the active training run are badged, since
  // that is usually the set the user means to act on.
  const trainingLabelUuids = activeTrainingConfig.getLabelUuids();
  const activeConfig = activeTrainingConfig.getActive();
  const labelItems = flattenHierarchy(labels).map(lbl => ({
    uuid: lbl.uuid,
    name: lbl.name,
    color: lbl.color || null,
    depth: lbl.depth,
    parentUuid: lbl.parentUuid ?? null,
    badge: trainingLabelUuids.has(lbl.uuid)
      ? { text: 'training', className: 'ms-item__badge--training', title: `In the active training run${activeConfig ? `: ${activeConfig.name}` : ''}` }
      : null,
  }));

  let selectedLabelUuids = [];
  const labelDropdown = createMultiSelectDropdown({
    label: 'Labels',
    items: labelItems,
    defaultSelected: [],
    showModeToggle: false,
    allowExclude: false,
    searchable: true,
    inline: true,
    emptyText: 'No labels defined yet',
    emptySummary: 'None selected',
    onChange: (selected) => { selectedLabelUuids = selected; },
  });

  const labelPickerRow = document.createElement('div');
  labelPickerRow.className = 'bulk-modal-picker bulk-modal-picker--labels';
  labelPickerRow.style.cssText = 'display:flex;flex-direction:column;gap:0.3em;padding:0.3em 0;';
  const labelPickerHeading = document.createElement('span');
  labelPickerHeading.style.cssText = 'font-size:0.85em;color:var(--text-muted,#aaa);';
  labelPickerHeading.textContent = trainingLabelUuids.size > 0
    ? 'Labels (badged ones are in the active training run)'
    : 'Labels';
  labelPickerRow.append(labelPickerHeading, labelDropdown.element);
  // Nothing is pre-selected on purpose: these actions rewrite label state across
  // every image in the range, so the user has to say which labels — a helpful
  // default here would be a silently destructive one.

  // Tag picker (shown only for addTags / removeTags) — same picker as the bbox bulk modal.
  const tagOptions = filterVisibleTags(tags.values())
    .map(tag => ({
      key: tag.uuid,
      text: (tag.icon ? tag.icon + ' ' : '') + tag.name,
      color: tag.color || '#cccccc',
    }));
  const tagField = new MultiSelectField({ label: 'Tags', value: [], options: tagOptions, emptyText: 'No tags defined yet', inline: true });
  const tagPickerRow = document.createElement('div');
  tagPickerRow.className = 'bulk-modal-picker bulk-modal-picker--tags';
  tagPickerRow.style.cssText = 'display:none;flex-direction:column;gap:0.3em;padding:0.3em 0;';
  const tagPickerHeading = document.createElement('span');
  tagPickerHeading.style.cssText = 'font-size:0.85em;color:var(--text-muted,#aaa);';
  tagPickerHeading.textContent = 'Tags';
  tagPickerRow.append(tagPickerHeading, await tagField.renderEdit());

  const btnRow = document.createElement('div');
  btnRow.style.cssText = 'display:flex;gap:0.5em;justify-content:flex-end;';

  let releaseFocusTrap = null;
  function closeImageBulk() {
    unregisterEscHandler(closeImageBulk);
    labelDropdown.destroy();
    tagField.destroy();
    _layer.release();
    releaseFocusTrap?.();
    overlay.remove();
  }

  const cancelBtn = document.createElement('button');
  cancelBtn.textContent = 'Cancel';
  cancelBtn.addEventListener('click', closeImageBulk);

  const okBtn = document.createElement('button');
  okBtn.textContent = 'Apply';
  okBtn.style.cssText = 'background:#ff003322;border:1px solid #ff0033;border-radius:3px;padding:0.2em 0.8em;cursor:pointer;';
  okBtn.addEventListener('click', async () => {
    const isTagAction = TAG_ACTIONS.includes(selectedAction);
    const isLabelAction = LABEL_ACTIONS.includes(selectedAction);
    const selectedTagUuids = isTagAction ? tagField.getValue() : [];
    if (isTagAction && selectedTagUuids.length === 0) {
      toast('Select at least one tag', 'error');
      return;
    }
    if (isLabelAction && selectedLabelUuids.length === 0) {
      toast('Select at least one label', 'error');
      return;
    }
    closeImageBulk();
    await onConfirm(selectedAction, isLabelAction ? [...selectedLabelUuids] : [], selectedTagUuids);
  });

  btnRow.appendChild(cancelBtn);
  btnRow.appendChild(okBtn);
  dialog.append(title);
  if (opts.previewNode) {
    // Two thumbnail groups plus the separator need more room than the
    // action-picker-only default (matches .bulk-modal-dialog--wide).
    dialog.style.minWidth = '420px';
    dialog.style.maxWidth = '600px';
    dialog.appendChild(opts.previewNode);
  }
  dialog.append(radioGroup, labelPickerRow, tagPickerRow, btnRow);
  overlay.appendChild(dialog);
  document.body.appendChild(overlay);
  applyModalCascade(dialog, _layer.depth);
  registerEscHandler(closeImageBulk);

  overlay.addEventListener('click', e => { if (e.target === overlay) closeImageBulk(); });
  releaseFocusTrap = trapModalFocus(dialog);
}

/**
 * @param {string} action
 * @param {Array<{img: object}>} entries
 * @param {string[]} labelUuids  every label a label-action should be applied to
 * @param {string[]} [tagUuids]
 */
export async function executeBulkImageAction(action, entries, labelUuids = [], tagUuids = []) {
  let failCount = 0;
  let skippedCount = 0;
  const succeededEntries = [];

  if (TAG_ACTIONS.includes(action)) {
    // Batch all images into a single server-side pass instead of one HTTP
    // round-trip per (image, tag) pair.
    try {
      const { failCount: serverFailCount } = await bulkApplyImagesByUuids({
        imageUuids: entries.map(e => e.img.uuid),
        action,
        tagUuids,
      });
      failCount = serverFailCount;
      for (const entry of entries) {
        const { img } = entry;
        const currentTags = img.tags ?? [];
        // img.tags only ever lists PRESENT tags — addTags is the only action
        // that adds to it; everything else (remove / verified-absent / clear
        // verification) means "not in the list".
        img.tags = action === 'addTags'
          ? [...currentTags.filter(t => !tagUuids.includes(t.uuid)), ...tagUuids.map(uuid => ({ uuid, icon: null }))]
          : currentTags.filter(t => !tagUuids.includes(t.uuid));
        succeededEntries.push(entry);
      }
    } catch (err) {
      Logger.error(`Bulk image tag action failed: ${err.message}`);
      failCount = entries.length;
    }
    return { failCount, skippedCount, succeededEntries };
  }

  // Label-status actions (setPresent/setAbsent/setUncertain) — batch through
  // the same server-side bulk endpoint as tag actions above, instead of one
  // HTTP round-trip per image (setImageLabel). The action names line up
  // 1:1 with the backend's bulk-apply action dispatch, so no status-string
  // translation is needed here anymore.
  //
  // Note: like the tag-action branch above, the bulk endpoint only reports
  // aggregate counts, not a per-image breakdown — every entry is optimistically
  // treated as "succeeded" for card-staleness purposes. skippedCount (setAbsent
  // refused because a verified bbox already contradicts it) is surfaced
  // separately so callers can report it distinctly from a real failure.
  try {
    const { failCount: serverFailCount, skippedCount: serverSkippedCount } = await bulkApplyImagesByUuids({
      imageUuids: entries.map(e => e.img.uuid),
      action,
      labelUuids,
    });
    failCount = serverFailCount;
    skippedCount = serverSkippedCount;
    succeededEntries.push(...entries);
  } catch (err) {
    Logger.error(`Bulk image label action failed: ${err.message}`);
    failCount = entries.length;
  }

  return { failCount, skippedCount, succeededEntries };
}
