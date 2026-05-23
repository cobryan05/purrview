import { fetchLabels } from '/app-static/js/shared/api/labels.js';
import {
  fetchReviewQueue,
  updateImageBboxes,
  updateBbox,
  deleteBbox,
  swapVerificationTag,
  getBboxThumbnailUrl,
  invalidateBboxThumbnails,
  UNVERIFIED_TAG_UUID,
  VERIFIED_TAG_UUID,
  FALSE_POS_TAG_UUID,
} from '/app-static/js/shared/api/review.js';
import { events, EventTypes } from '/app-static/js/shared/events/index.js';
import { Logger } from '/app-static/js/ui/utils/logging.js';

const PAGE_SIZE = 50;

/**
 * Initialize the Review tab. Called once on first tab focus.
 */
export async function init() {
  const root = document.getElementById('review-root');
  if (!root) return;
  root.innerHTML = '';

  const labels = await fetchLabels().catch(() => new Map());

  // ── Toolbar ───────────────────────────────────────────────────────
  const toolbar = document.createElement('div');
  toolbar.className = 'review-toolbar';

  // Label filter
  const labelSelect = document.createElement('select');
  labelSelect.title = 'Filter by label';
  const allOption = document.createElement('option');
  allOption.value = '';
  allOption.textContent = '— All labels —';
  labelSelect.appendChild(allOption);
  for (const lbl of labels.values()) {
    const opt = document.createElement('option');
    opt.value = lbl.uuid;
    opt.textContent = lbl.name;
    labelSelect.appendChild(opt);
  }

  // Select-all checkbox
  const selectAllLabel = document.createElement('label');
  selectAllLabel.style.cssText = 'display:flex;align-items:center;gap:0.3em;cursor:pointer;white-space:nowrap;';
  const selectAllCb = document.createElement('input');
  selectAllCb.type = 'checkbox';
  selectAllLabel.appendChild(selectAllCb);
  selectAllLabel.appendChild(document.createTextNode('Select all'));

  // Bulk Apply button
  const bulkBtn = document.createElement('button');
  bulkBtn.textContent = 'Bulk Apply (0)';
  bulkBtn.disabled = true;
  bulkBtn.style.cssText = 'padding:0.3em 0.8em;cursor:pointer;';

  // Refresh
  const refreshBtn = document.createElement('button');
  refreshBtn.textContent = 'Refresh';
  refreshBtn.style.cssText = 'padding:0.3em 0.8em;cursor:pointer;margin-left:auto;';

  // Pagination
  const pageInfo = document.createElement('span');
  pageInfo.style.cssText = 'color:#aaa;font-size:0.9em;white-space:nowrap;';
  const prevBtn = document.createElement('button');
  prevBtn.textContent = '‹ Prev';
  prevBtn.style.padding = '0.3em 0.6em';
  const nextBtn = document.createElement('button');
  nextBtn.textContent = 'Next ›';
  nextBtn.style.padding = '0.3em 0.6em';

  // Status message
  const statusMsg = document.createElement('span');
  statusMsg.style.cssText = 'color:#aaa;font-size:0.9em;';

  toolbar.appendChild(labelSelect);
  toolbar.appendChild(selectAllLabel);
  toolbar.appendChild(bulkBtn);
  toolbar.appendChild(statusMsg);
  toolbar.appendChild(prevBtn);
  toolbar.appendChild(pageInfo);
  toolbar.appendChild(nextBtn);
  toolbar.appendChild(refreshBtn);
  root.appendChild(toolbar);

  // ── Grid container ────────────────────────────────────────────────
  const grid = document.createElement('div');
  grid.className = 'review-grid';
  root.appendChild(grid);

  // ── State ─────────────────────────────────────────────────────────
  let _allBboxEntries = []; // flat list of { bbox, img } for current page
  let _selected = new Set(); // selected bbox UUIDs
  let _offset = 0;
  let _rendering = false;
  let _hasMore = false;

  function updateBulkBtn() {
    bulkBtn.textContent = `Bulk Apply (${_selected.size})`;
    bulkBtn.disabled = _selected.size === 0;
    selectAllCb.checked = _selected.size > 0 && _selected.size === _allBboxEntries.length;
    selectAllCb.indeterminate = _selected.size > 0 && _selected.size < _allBboxEntries.length;
  }

  // ── Render ────────────────────────────────────────────────────────
  async function render() {
    if (_rendering) return;
    _rendering = true;
    refreshBtn.disabled = true;
    prevBtn.disabled = true;
    nextBtn.disabled = true;
    statusMsg.textContent = 'Loading…';
    grid.innerHTML = '';
    _selected.clear();
    _allBboxEntries = [];
    updateBulkBtn();

    let images;
    try {
      images = await fetchReviewQueue({
        tagUuid: UNVERIFIED_TAG_UUID,
        labelUuid: labelSelect.value || undefined,
        limit: PAGE_SIZE,
        offset: _offset,
      });
    } catch (err) {
      statusMsg.textContent = `Error: ${err.message}`;
      refreshBtn.disabled = false;
      _rendering = false;
      return;
    }

    _hasMore = images.length === PAGE_SIZE;

    // Flatten images → unverified bboxes
    for (const img of images) {
      const unverified = (img.bboxes || []).filter(
        b => (b.tags || []).some(t => t.uuid === UNVERIFIED_TAG_UUID)
      );
      for (const bbox of unverified) {
        _allBboxEntries.push({ bbox, img });
      }
    }

    const total = _allBboxEntries.length;
    statusMsg.textContent = total === 0
      ? 'No unverified annotations'
      : `${total} unverified bbox${total === 1 ? '' : 'es'} in ${images.length} image${images.length === 1 ? '' : 's'}`;

    pageInfo.textContent = `Page ${Math.floor(_offset / PAGE_SIZE) + 1}`;
    prevBtn.disabled = _offset === 0;
    nextBtn.disabled = !_hasMore;

    if (total === 0) {
      const empty = document.createElement('p');
      empty.textContent = _hasMore
        ? 'No unverified annotations on this page.'
        : 'No unverified annotations remaining.';
      empty.style.color = '#aaa';
      grid.appendChild(empty);
      refreshBtn.disabled = false;
      _rendering = false;
      return;
    }

    for (const { bbox, img } of _allBboxEntries) {
      grid.appendChild(buildCard(bbox, img, labels));
    }

    refreshBtn.disabled = false;
    _rendering = false;
  }

  // ── Card builder ──────────────────────────────────────────────────
  function buildCard(bbox, img, labels) {
    const label = bbox.labelUuid ? labels.get(bbox.labelUuid) : null;
    const labelName = label?.name ?? '(unknown)';
    const labelColor = label?.color ?? '#888';
    const confidence = bbox.metadataJson?.confidence
      ? `${(parseFloat(bbox.metadataJson.confidence) * 100).toFixed(0)}%`
      : '';

    const card = document.createElement('div');
    card.className = 'review-card';
    card.dataset.bboxUuid = bbox.uuid;

    const cb = document.createElement('input');
    cb.type = 'checkbox';
    cb.addEventListener('change', () => {
      if (cb.checked) _selected.add(bbox.uuid);
      else _selected.delete(bbox.uuid);
      card.classList.toggle('selected', cb.checked);
      updateBulkBtn();
    });

    const thumbImg = document.createElement('img');
    thumbImg.className = 'thumb';
    thumbImg.alt = labelName;
    thumbImg.src = getBboxThumbnailUrl(img.filename, bbox.uuid, 300);
    thumbImg.addEventListener('click', () => {
      events.publish(EventTypes.LOAD_IMAGE_ONTO_CANVAS, { path: img.filename, showCanvas: true });
      events.publish(EventTypes.SHOW_CANVAS);
    });

    const info = document.createElement('div');
    info.className = 'card-info';

    const chip = document.createElement('span');
    chip.className = 'label-chip';
    chip.textContent = labelName;
    chip.style.cssText = `background:${labelColor}33;border:1px solid ${labelColor};`;

    const meta = document.createElement('div');
    meta.style.cssText = 'margin-top:0.2em;color:#aaa;font-size:0.75em;';
    const shortPath = img.filename.split('/').slice(-2).join('/');
    meta.textContent = confidence ? `${confidence} · ${shortPath}` : shortPath;

    info.appendChild(chip);
    info.appendChild(meta);
    card.appendChild(cb);
    card.appendChild(thumbImg);
    card.appendChild(info);
    return card;
  }

  // ── Select all ────────────────────────────────────────────────────
  selectAllCb.addEventListener('change', () => {
    const cards = grid.querySelectorAll('.review-card');
    for (const { bbox } of _allBboxEntries) {
      if (selectAllCb.checked) _selected.add(bbox.uuid);
      else _selected.delete(bbox.uuid);
    }
    cards.forEach(c => {
      const cb = c.querySelector('input[type=checkbox]');
      if (cb) cb.checked = selectAllCb.checked;
      c.classList.toggle('selected', selectAllCb.checked);
    });
    updateBulkBtn();
  });

  // ── Bulk Apply ────────────────────────────────────────────────────
  bulkBtn.addEventListener('click', () => {
    if (_selected.size === 0) return;
    const selectedEntries = _allBboxEntries.filter(e => _selected.has(e.bbox.uuid));
    showBulkActionModal(selectedEntries, labels, async (action, newLabelUuid) => {
      const failCount = await executeBulkAction(action, selectedEntries, newLabelUuid);
      _offset = 0;
      await render();
      if (failCount > 0) {
        statusMsg.textContent += ` (${failCount} action${failCount === 1 ? '' : 's'} failed — check console)`;
      }
    });
  });

  // ── Pagination ────────────────────────────────────────────────────
  prevBtn.addEventListener('click', () => {
    _offset = Math.max(0, _offset - PAGE_SIZE);
    render();
  });
  nextBtn.addEventListener('click', () => {
    _offset += PAGE_SIZE;
    render();
  });

  refreshBtn.addEventListener('click', () => { _offset = 0; render(); });
  labelSelect.addEventListener('change', () => { _offset = 0; render(); });

  render();
}

// ── Bulk action execution ─────────────────────────────────────────────

async function executeBulkAction(action, entries, newLabelUuid) {
  const toInvalidate = [];
  let failCount = 0;

  for (const { bbox } of entries) {
    try {
      if (action === 'delete') {
        await deleteBbox(bbox.uuid);
      } else {
        // swapVerificationTag keeps all non-verification tags, replacing only
        // the verification slot — so other custom tags on the bbox are preserved.
        const currentTagUuids = (bbox.tags || []).map(t => t.uuid);
        const newTagUuid = action === 'fp' ? FALSE_POS_TAG_UUID : VERIFIED_TAG_UUID;
        const tagUuids = swapVerificationTag(currentTagUuids, newTagUuid);
        const labelUuid = action === 'relabel' && newLabelUuid
          ? newLabelUuid
          : (bbox.labelUuid ?? bbox.label?.uuid);
        await updateBbox(bbox.uuid, { labelUuid, tagUuids });
      }
      toInvalidate.push(bbox.uuid);
    } catch (err) {
      Logger.error(`Bulk action failed for bbox ${bbox.uuid}: ${err.message}`);
      failCount++;
    }
  }

  if (toInvalidate.length) await invalidateBboxThumbnails(toInvalidate);
  return failCount;
}

// ── Bulk action modal ─────────────────────────────────────────────────

function showBulkActionModal(entries, labels, onConfirm) {
  const overlay = document.createElement('div');
  overlay.style.cssText = `position:fixed;inset:0;background:#0008;display:flex;
    align-items:center;justify-content:center;z-index:1000;`;

  const dialog = document.createElement('div');
  dialog.style.cssText = `background:#222;border:1px solid #555;border-radius:6px;
    padding:1.5em;min-width:300px;display:flex;flex-direction:column;gap:0.8em;`;

  const title = document.createElement('strong');
  title.textContent = `Apply to ${entries.length} selected bbox${entries.length === 1 ? '' : 'es'}`;

  // Action radio buttons
  const actions = [
    { value: 'verify', label: 'Verify / Accept' },
    { value: 'fp', label: 'Mark False Positive' },
    { value: 'relabel', label: 'Change Label (and verify)' },
    { value: 'delete', label: 'Delete Bboxes' },
  ];

  const radioGroup = document.createElement('div');
  radioGroup.style.cssText = 'display:flex;flex-direction:column;gap:0.4em;';
  let selectedAction = 'verify';

  for (const action of actions) {
    const lbl = document.createElement('label');
    lbl.style.cssText = 'display:flex;align-items:center;gap:0.5em;cursor:pointer;';
    const radio = document.createElement('input');
    radio.type = 'radio';
    radio.name = 'bulk-action';
    radio.value = action.value;
    if (action.value === 'verify') radio.checked = true;
    radio.addEventListener('change', () => {
      selectedAction = action.value;
      labelPickerRow.style.display = selectedAction === 'relabel' ? '' : 'none';
    });
    lbl.appendChild(radio);
    lbl.appendChild(document.createTextNode(action.label));
    radioGroup.appendChild(lbl);
  }

  // Label picker (shown only for relabel)
  const labelPickerRow = document.createElement('div');
  labelPickerRow.style.display = 'none';
  const labelPicker = document.createElement('select');
  for (const lbl of labels.values()) {
    const opt = document.createElement('option');
    opt.value = lbl.uuid;
    opt.textContent = lbl.name;
    labelPicker.appendChild(opt);
  }
  labelPickerRow.appendChild(document.createTextNode('New label: '));
  labelPickerRow.appendChild(labelPicker);

  const btnRow = document.createElement('div');
  btnRow.style.cssText = 'display:flex;gap:0.5em;justify-content:flex-end;';

  const cancelBtn = document.createElement('button');
  cancelBtn.textContent = 'Cancel';
  cancelBtn.addEventListener('click', () => overlay.remove());

  const okBtn = document.createElement('button');
  okBtn.textContent = 'Apply';
  okBtn.style.cssText = 'background:#ff003322;border:1px solid #ff0033;border-radius:3px;padding:0.2em 0.8em;cursor:pointer;';
  okBtn.addEventListener('click', async () => {
    overlay.remove();
    const newLabelUuid = selectedAction === 'relabel' ? labelPicker.value : null;
    await onConfirm(selectedAction, newLabelUuid);
  });

  btnRow.appendChild(cancelBtn);
  btnRow.appendChild(okBtn);
  dialog.append(title, radioGroup, labelPickerRow, btnRow);
  overlay.appendChild(dialog);
  document.body.appendChild(overlay);

  overlay.addEventListener('click', e => { if (e.target === overlay) overlay.remove(); });
}

// ── Shared helpers ─────────────────────────────────────────────────────

function bboxToPayload(bbox) {
  return {
    uuid: bbox.uuid,
    labelUuid: bbox.labelUuid ?? bbox.label?.uuid,
    x: bbox.x,
    y: bbox.y,
    width: bbox.width,
    height: bbox.height,
    tagUuids: (bbox.tags || []).map(t => t.uuid),
    extra: bbox.metadataJson ?? {},
  };
}

async function applyTagSwap(imagePath, allBboxes, targetBboxUuid, newTagUuid) {
  const updated = allBboxes.map(b => {
    const payload = bboxToPayload(b);
    if (b.uuid === targetBboxUuid) {
      payload.tagUuids = swapVerificationTag(payload.tagUuids, newTagUuid);
    }
    return payload;
  });
  await updateImageBboxes(imagePath, updated);
}

async function applyTagSwapAndRelabel(imagePath, allBboxes, targetBboxUuid, newTagUuid, newLabelUuid) {
  const updated = allBboxes.map(b => {
    const payload = bboxToPayload(b);
    if (b.uuid === targetBboxUuid) {
      payload.tagUuids = swapVerificationTag(payload.tagUuids, newTagUuid);
      payload.labelUuid = newLabelUuid;
    }
    return payload;
  });
  await updateImageBboxes(imagePath, updated);
}
