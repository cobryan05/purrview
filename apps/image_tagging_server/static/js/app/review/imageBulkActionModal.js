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
import { bulkApplyImagesByUuids } from '/app-static/js/shared/api/review.js';
import { setImageLabel } from '/app-static/js/shared/api/imageLabels.js';
import { Logger, toast } from '/app-static/js/ui/utils/logging.js';
import { nextModalZIndex, applyModalCascade, registerEscHandler, unregisterEscHandler } from '/app-static/js/ui/utils/modalZIndex.js';
import { CheckboxField } from '/app-static/js/ui/utils/fields/CheckboxField.js';

export const TAG_ACTIONS = ['addTags', 'removeTags', 'setTagAbsent', 'setTagUncertain'];

export async function showImageBulkActionModal(entries, labels, onConfirm, titleText = null) {
  const tags = await fetchTags().catch(() => new Map());

  const overlay = document.createElement('div');
  overlay.style.cssText = `position:fixed;inset:0;background:#0008;display:flex;
    align-items:center;justify-content:center;z-index:${nextModalZIndex()};`;

  const dialog = document.createElement('div');
  dialog.style.cssText = `background:#222;border:1px solid #555;border-radius:6px;
    padding:1.5em;min-width:300px;display:flex;flex-direction:column;gap:0.8em;`;

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
      labelPickerRow.style.display = (selectedAction === 'setPresent' || selectedAction === 'setAbsent' || selectedAction === 'setUncertain') ? '' : 'none';
      tagPickerRow.style.display = TAG_ACTIONS.includes(selectedAction) ? 'flex' : 'none';
    });
    lbl.appendChild(radio);
    lbl.appendChild(document.createTextNode(action.label));
    radioGroup.appendChild(lbl);
  }

  const labelPickerRow = document.createElement('div');
  const labelPicker = document.createElement('select');
  for (const lbl of labels.values()) {
    const opt = document.createElement('option');
    opt.value = lbl.uuid;
    opt.textContent = lbl.name;
    labelPicker.appendChild(opt);
  }
  labelPickerRow.appendChild(document.createTextNode('Label: '));
  labelPickerRow.appendChild(labelPicker);

  // Tag picker (shown only for addTags / removeTags) — same picker as the bbox bulk modal.
  const tagOptions = filterVisibleTags(tags.values())
    .map(tag => ({
      key: tag.uuid,
      text: (tag.icon ? tag.icon + ' ' : '') + tag.name,
      color: tag.color || '#cccccc',
    }));
  const tagCheckboxField = new CheckboxField({ value: [], options: tagOptions });
  const tagPickerRow = document.createElement('div');
  tagPickerRow.style.cssText = 'max-height:180px;overflow-y:auto;display:none;flex-direction:column;gap:0.3em;padding:0.3em 0;';
  tagPickerRow.appendChild(await tagCheckboxField.renderEdit());

  const btnRow = document.createElement('div');
  btnRow.style.cssText = 'display:flex;gap:0.5em;justify-content:flex-end;';

  function closeImageBulk() { unregisterEscHandler(closeImageBulk); overlay.remove(); }

  const cancelBtn = document.createElement('button');
  cancelBtn.textContent = 'Cancel';
  cancelBtn.addEventListener('click', closeImageBulk);

  const okBtn = document.createElement('button');
  okBtn.textContent = 'Apply';
  okBtn.style.cssText = 'background:#ff003322;border:1px solid #ff0033;border-radius:3px;padding:0.2em 0.8em;cursor:pointer;';
  okBtn.addEventListener('click', async () => {
    const isTagAction = TAG_ACTIONS.includes(selectedAction);
    const selectedTagUuids = isTagAction ? tagCheckboxField.getValue() : [];
    if (isTagAction && selectedTagUuids.length === 0) {
      toast('Select at least one tag', 'error');
      return;
    }
    closeImageBulk();
    await onConfirm(selectedAction, labelPicker.value, selectedTagUuids);
  });

  btnRow.appendChild(cancelBtn);
  btnRow.appendChild(okBtn);
  dialog.append(title, radioGroup, labelPickerRow, tagPickerRow, btnRow);
  overlay.appendChild(dialog);
  document.body.appendChild(overlay);
  applyModalCascade(dialog);
  registerEscHandler(closeImageBulk);

  overlay.addEventListener('click', e => { if (e.target === overlay) closeImageBulk(); });
}

export async function executeBulkImageAction(action, entries, labelUuid, tagUuids = []) {
  let failCount = 0;
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
    return { failCount, succeededEntries };
  }

  for (const entry of entries) {
    const { img } = entry;
    try {
      const status = action === 'setPresent' ? 'present' : action === 'setAbsent' ? 'absent' : 'uncertain';
      await setImageLabel(img.filename, labelUuid, status);
      succeededEntries.push(entry);
    } catch (err) {
      Logger.error(`Image label action failed for ${img.uuid}: ${err.message}`);
      failCount++;
    }
  }

  return { failCount, succeededEntries };
}
