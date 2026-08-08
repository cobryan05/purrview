/**
 * Generic "apply an action to a set of images" modal — extracted out of
 * review/main.js so it can be reused outside the review grid's own toolbar
 * (e.g. runClassifierOnResults.js's classifier-match confirmation step).
 *
 * Supports the full image-level action set: label status (setPresent/setAbsent/
 * setUncertain), tags (addTags/removeTags/setTagAbsent/setTagUncertain), and delete.
 */

import { fetchTags } from '/app-static/js/shared/api/tags.js';
import { filterVisibleTags } from '/app-static/js/shared/activeModel.js';
import { activeTrainingConfig } from '/app-static/js/shared/activeTrainingConfig.js';
import { bulkApplyImagesByUuids } from '/app-static/js/shared/api/review.js';
import { flattenHierarchy } from '/app-static/js/ui/labelList.js';
import { Logger, toast } from '/app-static/js/ui/utils/logging.js';
import { acquireModalLayer, applyModalCascade, registerEscHandler, unregisterEscHandler } from '/app-static/js/ui/utils/modalZIndex.js';
import { MultiSelectField } from '/app-static/js/ui/utils/fields/MultiSelectField.js';
import { createMultiSelectDropdown } from '/app-static/js/ui/utils/createMultiSelectDropdown.js';

export const TAG_ACTIONS = ['addTags', 'removeTags', 'setTagAbsent', 'setTagUncertain'];
export const LABEL_ACTIONS = ['setPresent', 'setAbsent', 'setUncertain'];

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

  const _layer = acquireModalLayer({ dialog: true });
  const overlay = document.createElement('div');
  overlay.style.cssText = `position:fixed;inset:0;background:#0008;display:flex;
    align-items:center;justify-content:center;z-index:${_layer.zIndex};`;

  const dialog = document.createElement('div');
  dialog.style.cssText = `background:#222;border:1px solid #555;border-radius:6px;
    padding:1.5em;min-width:380px;display:flex;flex-direction:column;gap:0.8em;`;

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
    { value: 'delete', label: 'Delete Images' },
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

  function closeImageBulk() {
    unregisterEscHandler(closeImageBulk);
    labelDropdown.destroy();
    tagField.destroy();
    _layer.release();
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
