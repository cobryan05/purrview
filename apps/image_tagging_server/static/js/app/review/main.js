import { fetchLabels } from '/app-static/js/shared/api/labels.js';
import { fetchTags } from '/app-static/js/shared/api/tags.js';
import { createFilterPanel } from '/app-static/js/app/review/filters.js';
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
import { Logger } from '/app-static/js/ui/utils/logging.js';
import { CheckboxField } from '/app-static/js/ui/utils/fields/CheckboxField.js';
import { createCanvasInstance } from '/app-static/js/app/image-tagging/canvas/instance.js';
import { saveAnnotations } from '/app-static/js/app/image-tagging/canvas/image.js';
import { init as initToolbar } from '/app-static/js/app/image-tagging/toolbar.js';
import { initLabelPanel } from '/app-static/js/app/image-tagging/canvas/labelPanel.js';
import { initInspector } from '/app-static/js/ui/inspector.js';

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

  // Filter toggle
  const filterToggleBtn = document.createElement('button');
  filterToggleBtn.textContent = '☰ Filters';
  filterToggleBtn.title = 'Toggle search filters';
  filterToggleBtn.style.cssText = 'padding:0.3em 0.8em;cursor:pointer;';

  // Label filter (quick bbox-label filter in toolbar)
  const labelSelect = document.createElement('select');
  labelSelect.title = 'Filter by bbox label';
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

  toolbar.appendChild(filterToggleBtn);
  toolbar.appendChild(labelSelect);
  toolbar.appendChild(selectAllLabel);
  toolbar.appendChild(bulkBtn);
  toolbar.appendChild(statusMsg);
  toolbar.appendChild(prevBtn);
  toolbar.appendChild(pageInfo);
  toolbar.appendChild(nextBtn);
  toolbar.appendChild(refreshBtn);
  root.appendChild(toolbar);

  // ── Filter panel ──────────────────────────────────────────────────
  const tags = await fetchTags().catch(() => new Map());

  const filterPanel = createFilterPanel(labels, tags, () => {
    filterToggleBtn.classList.toggle('filter-active', !filterPanel.isDefault());
    _pageCursors = [null];
    _lastCursor = null;
    render();
  });
  filterPanel.element.hidden = true;
  root.appendChild(filterPanel.element);

  filterToggleBtn.addEventListener('click', () => {
    filterPanel.element.hidden = !filterPanel.element.hidden;
  });

  // ── Grid container ────────────────────────────────────────────────
  const grid = document.createElement('div');
  grid.className = 'review-grid';
  root.appendChild(grid);

  // ── State ─────────────────────────────────────────────────────────
  let _allBboxEntries = []; // flat list of { bbox, img } for current page
  let _selected = new Set(); // selected bbox UUIDs
  let _staleUuids = new Set(); // bbox UUIDs modified on this page (for status display)
  let _deletedUuids = new Set(); // bbox UUIDs deleted this page (disabled/unselectable)
  let _pageCursors = [null]; // cursor stack: null = first page, {uuid, sortValue} = keyset
  let _lastCursor = null; // {uuid: string, sortValue: string|null} of last image on current page
  let _lastClickedIndex = -1; // index in _allBboxEntries of the last checkbox click (for shift-select)
  let _rendering = false;
  let _hasMore = false;

  function updateBulkBtn() {
    const activeCount = _allBboxEntries.filter(e => !_deletedUuids.has(e.bbox.uuid)).length;
    bulkBtn.textContent = `Bulk Apply (${_selected.size})`;
    bulkBtn.disabled = _selected.size === 0;
    selectAllCb.checked = _selected.size > 0 && _selected.size === activeCount;
    selectAllCb.indeterminate = _selected.size > 0 && _selected.size < activeCount;
  }

  function updateStatusMsg() {
    const total = _allBboxEntries.length;
    const stale = _staleUuids.size;
    const fresh = total - stale;
    if (total === 0) {
      statusMsg.textContent = 'No matching annotations on this page';
    } else if (stale === 0) {
      statusMsg.textContent = `${total} bbox${total === 1 ? '' : 'es'} on this page`;
    } else {
      statusMsg.textContent = `${fresh} remaining · ${stale} modified`;
    }
  }

  function markCardStale(card, state) {
    card.querySelector('.card-state-badge')?.remove();
    card.classList.remove('stale-verified', 'stale-false-positive', 'stale-relabeled', 'stale-deleted', 'stale-tagged');
    card.classList.add(`stale-${state}`);
    const stateLabels = {
      verified: '✓ Verified',
      'false-positive': '✗ False Pos.',
      relabeled: '⤴ Relabeled',
      deleted: '🗑 Deleted',
      tagged: '🏷 Tagged',
    };
    const badge = document.createElement('div');
    badge.className = `card-state-badge state-${state}`;
    badge.textContent = stateLabels[state] ?? state;
    card.querySelector('.card-info')?.prepend(badge);
    const uuid = card.dataset.bboxUuid;
    _staleUuids.add(uuid);
    if (state === 'deleted') {
      const cb = card.querySelector('input[type=checkbox]');
      if (cb) { cb.checked = false; cb.disabled = true; }
      _selected.delete(uuid);
      _deletedUuids.add(uuid);
    }
  }

  // ── Render ────────────────────────────────────────────────────────
  async function render() {
    if (_rendering) return;
    _rendering = true;

    // currentCursor tracks the effective fetch cursor, which may advance past
    // server pages that contain zero unverified bboxes (all already verified).
    // We update the top of _pageCursors so Prev navigates correctly.
    const MAX_AUTO_SKIP = 10;
    let autoSkipped = 0;

    fetchLoop: while (true) {
      refreshBtn.disabled = true;
      prevBtn.disabled = true;
      nextBtn.disabled = true;
      statusMsg.textContent = 'Loading…';
      grid.innerHTML = '';
      _selected.clear();
      _staleUuids = new Set();
      _deletedUuids = new Set();
      _allBboxEntries = [];
      _lastCursor = null;
      _lastClickedIndex = -1;
      updateBulkBtn();

      const fs = filterPanel.getState();
      const cursor = _pageCursors[_pageCursors.length - 1];

      let images;
      try {
        images = await fetchReviewQueue({
          tagUuids: fs.showAllBboxes ? [] : fs.tagUuids,
          showAllBboxes: fs.showAllBboxes,
          labelUuid: labelSelect.value || undefined,
          sortBy: fs.sortBy,
          sortOrder: fs.sortOrder,
          filenameSearch: fs.filenameSearch || undefined,
          dateFrom: fs.dateFrom || undefined,
          dateTo: fs.dateTo || undefined,
          afterUuid: cursor?.uuid ?? null,
          afterSortValue: cursor?.sortValue ?? null,
          imageLabelFilters: fs.imageLabelFilters,
          limit: PAGE_SIZE,
        });
      } catch (err) {
        statusMsg.textContent = `Error: ${err.message}`;
        refreshBtn.disabled = false;
        _rendering = false;
        return;
      }

      _hasMore = images.length === PAGE_SIZE;

      // Update the keyset cursor for the Next button.
      if (images.length > 0) {
        const last = images[images.length - 1];
        _lastCursor = {
          uuid: last.uuid,
          sortValue: fs.sortBy === 'uuid' ? null
            : fs.sortBy === 'updated_at' ? (last.updatedAt ?? null)
            : (last.createdAt ?? null),
        };
      }

      // Flatten images → matching bboxes.
      // When show_all_bboxes, every bbox is shown; otherwise filter client-side
      // to only bboxes that carry one of the selected tags (server may return extras
      // if a page contains images with mixed bbox states).
      const tagSet = new Set(fs.tagUuids);
      for (const img of images) {
        const matching = (img.bboxes || []).filter(b =>
          fs.showAllBboxes || (b.tags || []).some(t => tagSet.has(t.uuid))
        );
        for (const bbox of matching) {
          _allBboxEntries.push({ bbox, img });
        }
      }

      // If the server returned a full page of images but the client-side tag filter
      // found nothing on any of them, advance the cursor automatically rather than
      // showing an empty page.
      if (_allBboxEntries.length === 0 && _hasMore && autoSkipped < MAX_AUTO_SKIP) {
        autoSkipped++;
        // Advance the top of the cursor stack in place so Prev still navigates correctly.
        _pageCursors[_pageCursors.length - 1] = _lastCursor;
        continue fetchLoop;
      }

      break fetchLoop;
    }

    updateStatusMsg();
    pageInfo.textContent = `Page ${_pageCursors.length}`;
    prevBtn.disabled = _pageCursors.length <= 1;
    nextBtn.disabled = !_hasMore;

    if (_allBboxEntries.length === 0) {
      const empty = document.createElement('p');
      empty.textContent = _hasMore
        ? 'No matching annotations on this page.'
        : 'No matching annotations found.';
      empty.style.color = '#aaa';
      grid.appendChild(empty);
      refreshBtn.disabled = false;
      _rendering = false;
      return;
    }

    _allBboxEntries.forEach(({ bbox, img }, index) => {
      grid.appendChild(buildCard(bbox, img, labels, index));
    });

    refreshBtn.disabled = false;
    _rendering = false;
  }

  // ── Card builder ──────────────────────────────────────────────────
  function buildCard(bbox, img, labels, index) {
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

    function applySelect(shiftKey, newChecked) {
      cb.checked = newChecked;
      if (shiftKey && _lastClickedIndex !== -1 && _lastClickedIndex !== index) {
        const lo = Math.min(_lastClickedIndex, index);
        const hi = Math.max(_lastClickedIndex, index);
        for (let i = lo; i <= hi; i++) {
          const entry = _allBboxEntries[i];
          if (!entry || _deletedUuids.has(entry.bbox.uuid)) continue;
          const c = grid.querySelector(`[data-bbox-uuid="${entry.bbox.uuid}"]`);
          const cCb = c?.querySelector('input[type=checkbox]');
          if (!cCb || cCb.disabled) continue;
          cCb.checked = newChecked;
          if (newChecked) _selected.add(entry.bbox.uuid);
          else _selected.delete(entry.bbox.uuid);
          c.classList.toggle('selected', newChecked);
        }
      } else {
        if (newChecked) _selected.add(bbox.uuid);
        else _selected.delete(bbox.uuid);
        card.classList.toggle('selected', newChecked);
      }
      _lastClickedIndex = index;
      updateBulkBtn();
    }

    // Use 'click' (not 'change') so we can read e.shiftKey for range selection.
    // By the time 'click' fires, cb.checked already reflects the new toggled state.
    cb.addEventListener('click', (e) => applySelect(e.shiftKey, cb.checked));

    let _clickTimer = null;
    card.addEventListener('click', (e) => {
      if (e.target === cb || cb.disabled) return;
      if (e.shiftKey) e.preventDefault(); // prevent text selection from shift-click
      clearTimeout(_clickTimer);
      const shiftKey = e.shiftKey;
      const newChecked = !cb.checked;
      _clickTimer = setTimeout(() => applySelect(shiftKey, newChecked), 220);
    });

    const thumbImg = document.createElement('img');
    thumbImg.className = 'thumb';
    thumbImg.alt = labelName;
    thumbImg.src = getBboxThumbnailUrl(img.filename, bbox.uuid, 300);
    thumbImg.title = 'Double-click to edit in canvas';
    thumbImg.addEventListener('dblclick', () => {
      clearTimeout(_clickTimer);
      showCanvasModal(bbox, img, labels, async (updatedUIBbox) => {
        const card = grid.querySelector(`[data-bbox-uuid="${bbox.uuid}"]`);
        if (!card) return;

        if (updatedUIBbox === undefined) {
          // Bbox was deleted inside the canvas editor
          markCardStale(card, 'deleted');
        } else {
          // Force thumbnail re-download by cache-busting the URL
          const thumb = card.querySelector('.thumb');
          if (thumb) thumb.src = getBboxThumbnailUrl(img.filename, bbox.uuid, 300) + `&t=${Date.now()}`;

          // Update label chip to reflect the saved state (including the label-less case)
          const chip = card.querySelector('.label-chip');
          if (chip) {
            if (updatedUIBbox.label) {
              chip.textContent = updatedUIBbox.label.text;
              chip.style.cssText = `background:${updatedUIBbox.label.color}33;border:1px solid ${updatedUIBbox.label.color};`;
            } else {
              chip.textContent = '(unknown)';
              chip.style.cssText = 'background:#88888833;border:1px solid #888;';
            }
          }

          // If it's no longer unverified, mark it stale with the appropriate state
          const stillUnverified = (updatedUIBbox.tags ?? []).some(t => t.uuid === UNVERIFIED_TAG_UUID);
          if (!stillUnverified) {
            const isFP = (updatedUIBbox.tags ?? []).some(t => t.uuid === FALSE_POS_TAG_UUID);
            markCardStale(card, isFP ? 'false-positive' : 'verified');
          }
        }

        updateBulkBtn();
        updateStatusMsg();
      });
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
      if (_deletedUuids.has(bbox.uuid)) continue; // deleted cards can't be selected
      if (selectAllCb.checked) _selected.add(bbox.uuid);
      else _selected.delete(bbox.uuid);
    }
    cards.forEach(c => {
      if (_deletedUuids.has(c.dataset.bboxUuid)) return; // deleted cards stay disabled
      const cb = c.querySelector('input[type=checkbox]');
      if (cb) cb.checked = selectAllCb.checked;
      c.classList.toggle('selected', selectAllCb.checked);
    });
    // Clear the shift-select anchor so a subsequent shift-click doesn't range
    // from a stale individual-click position.
    _lastClickedIndex = -1;
    updateBulkBtn();
  });

  // ── Bulk Apply ────────────────────────────────────────────────────
  bulkBtn.addEventListener('click', async () => {
    if (_selected.size === 0) return;
    const selectedEntries = _allBboxEntries.filter(e => _selected.has(e.bbox.uuid));
    await showBulkActionModal(selectedEntries, labels, async (action, newLabelUuid, selectedTagUuids) => {
      const { failCount, succeededEntries } = await executeBulkAction(action, selectedEntries, newLabelUuid, selectedTagUuids);

      for (const { bbox, img } of succeededEntries) {
        const card = grid.querySelector(`[data-bbox-uuid="${bbox.uuid}"]`);
        if (!card) continue;

        if (action === 'delete') {
          markCardStale(card, 'deleted');
        } else {
          const thumb = card.querySelector('.thumb');
          if (thumb) thumb.src = getBboxThumbnailUrl(img.filename, bbox.uuid, 300) + `&t=${Date.now()}`;

          if (action === 'relabel' && newLabelUuid) {
            const label = labels.get(newLabelUuid);
            if (label) {
              const chip = card.querySelector('.label-chip');
              if (chip) {
                chip.textContent = label.name;
                chip.style.cssText = `background:${label.color}33;border:1px solid ${label.color};`;
              }
            }
            markCardStale(card, 'relabeled');
          } else if (action === 'addTags' || action === 'removeTags') {
            // If a verification tag was bulk-applied, surface the right stale state.
            const tagSet = new Set(selectedTagUuids);
            const staleState = action === 'addTags' && tagSet.has(FALSE_POS_TAG_UUID) ? 'false-positive'
              : action === 'addTags' && tagSet.has(VERIFIED_TAG_UUID) ? 'verified'
              : 'tagged';
            markCardStale(card, staleState);
          } else {
            markCardStale(card, action === 'fp' ? 'false-positive' : 'verified');
          }
        }
      }

      updateBulkBtn();
      updateStatusMsg();
      if (failCount > 0) {
        statusMsg.textContent += ` (${failCount} action${failCount === 1 ? '' : 's'} failed — check console)`;
      }
    });
  });

  // ── Pagination ────────────────────────────────────────────────────
  prevBtn.addEventListener('click', () => {
    if (_pageCursors.length > 1) _pageCursors.pop();
    render();
  });
  nextBtn.addEventListener('click', () => {
    _pageCursors.push(_lastCursor);
    render();
  });

  function resetAndRender() { _pageCursors = [null]; _lastCursor = null; render(); }
  refreshBtn.addEventListener('click', resetAndRender);
  labelSelect.addEventListener('change', resetAndRender);

  render();
}

// ── Bulk action execution ─────────────────────────────────────────────

async function executeBulkAction(action, entries, newLabelUuid, selectedTagUuids = []) {
  const toInvalidate = [];
  const succeededEntries = [];
  let failCount = 0;

  for (const entry of entries) {
    const { bbox } = entry;
    try {
      if (action === 'delete') {
        await deleteBbox(bbox.uuid);
      } else if (action === 'addTags') {
        const currentTagUuids = (bbox.tags || []).map(t => t.uuid);
        // Union: merge selected tags into existing, dedup
        const tagUuids = [...new Set([...currentTagUuids, ...selectedTagUuids])];
        const labelUuid = bbox.labelUuid ?? bbox.label?.uuid;
        await updateBbox(bbox.uuid, { labelUuid, tagUuids });
      } else if (action === 'removeTags') {
        const currentTagUuids = (bbox.tags || []).map(t => t.uuid);
        const removeSet = new Set(selectedTagUuids);
        const tagUuids = currentTagUuids.filter(u => !removeSet.has(u));
        const labelUuid = bbox.labelUuid ?? bbox.label?.uuid;
        await updateBbox(bbox.uuid, { labelUuid, tagUuids });
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
      succeededEntries.push(entry);
    } catch (err) {
      Logger.error(`Bulk action failed for bbox ${bbox.uuid}: ${err.message}`);
      failCount++;
    }
  }

  if (toInvalidate.length) await invalidateBboxThumbnails(toInvalidate);
  return { failCount, succeededEntries };
}

// ── Canvas edit modal ─────────────────────────────────────────────────

let _modalOpen = false;

/**
 * Snapshot the current bbox map so we can detect which bboxes were modified.
 * @param {Map<string, import('@domain_image').UIBbox>} bboxMap
 * @returns {Map<string, {x:number, y:number, width:number, height:number, labelUuid:string|null, tagUuids:string}>}
 */
function snapshotBboxes(bboxMap) {
  const snap = new Map();
  for (const [uuid, bbox] of bboxMap.entries()) {
    snap.set(uuid, {
      x: bbox.x, y: bbox.y,
      width: bbox.width, height: bbox.height,
      labelUuid: bbox.label?.uuid ?? null,
      tagUuids: (bbox.tags ?? []).map(t => t.uuid).sort().join(','),
    });
  }
  return snap;
}

/**
 * Return the set of bbox UUIDs that differ from the snapshot.
 * New bboxes (not present in snapshot) are always included.
 * @param {Map<string, import('@domain_image').UIBbox>} bboxMap
 * @param {ReturnType<typeof snapshotBboxes>} snapshot
 * @returns {Set<string>}
 */
function getModifiedBboxUuids(bboxMap, snapshot) {
  const modified = new Set();
  for (const [uuid, bbox] of bboxMap.entries()) {
    const orig = snapshot.get(uuid);
    if (!orig) {
      modified.add(uuid); // newly drawn bbox
    } else {
      const tagStr = (bbox.tags ?? []).map(t => t.uuid).sort().join(',');
      if (
        orig.x !== bbox.x || orig.y !== bbox.y ||
        orig.width !== bbox.width || orig.height !== bbox.height ||
        orig.labelUuid !== (bbox.label?.uuid ?? null) ||
        orig.tagUuids !== tagStr
      ) {
        modified.add(uuid);
      }
    }
  }
  return modified;
}

/**
 * Open a full canvas editor modal for a single bbox.
 * @param {object} bbox  The bbox object from the review queue
 * @param {object} img   The parent image object
 * @param {Map} labels   Labels map (uuid → { name, color })
 * @param {(updatedBbox?: object) => Promise<void>} onSaved  Called after a successful save with
 *   the updated UIBBox (undefined if the bbox was deleted inside the editor)
 */
async function showCanvasModal(bbox, img, labels, onSaved) {
  if (_modalOpen) return;
  _modalOpen = true;
  const shortPath = img.filename.split('/').slice(-2).join('/');

  // ── Build DOM ─────────────────────────────────────────────────────
  const overlay = document.createElement('div');
  overlay.className = 'canvas-modal-overlay';

  const dialog = document.createElement('div');
  dialog.className = 'canvas-modal-dialog';

  // Header
  const header = document.createElement('div');
  header.className = 'canvas-modal-header';
  const title = document.createElement('span');
  title.className = 'canvas-modal-title';
  title.textContent = `Edit: ${shortPath}`;
  const closeBtn = document.createElement('button');
  closeBtn.className = 'canvas-modal-close';
  closeBtn.textContent = '✕';
  closeBtn.title = 'Close without saving';
  header.append(title, closeBtn);

  // Toolbar
  const toolbarEl = document.createElement('div');
  toolbarEl.className = 'canvas-modal-toolbar';

  // Content area: label panel (left) + canvas (right)
  const contentArea = document.createElement('div');
  contentArea.className = 'canvas-modal-content';

  const labelPanelEl = document.createElement('div');
  labelPanelEl.className = 'canvas-modal-label-panel';

  const canvasWrap = document.createElement('div');
  canvasWrap.className = 'canvas-modal-canvas';

  const inspectorPanel = document.createElement('div');
  inspectorPanel.className = 'canvas-modal-inspector';
  const inspectorHint = document.createElement('div');
  inspectorHint.className = 'canvas-modal-inspector-hint';
  inspectorHint.textContent = 'Double-click a box to edit its tags & label';
  inspectorPanel.appendChild(inspectorHint);

  contentArea.append(labelPanelEl, canvasWrap, inspectorPanel);

  // Footer
  const footer = document.createElement('div');
  footer.className = 'canvas-modal-footer';

  const markVerifiedLabel = document.createElement('label');
  markVerifiedLabel.style.cssText = 'display:flex;align-items:center;gap:0.4em;color:#aaa;font-size:0.85em;cursor:pointer;user-select:none;margin-right:auto;';
  const markVerifiedCheckbox = document.createElement('input');
  markVerifiedCheckbox.type = 'checkbox';
  markVerifiedCheckbox.checked = true;
  markVerifiedLabel.append(markVerifiedCheckbox, 'Mark modified boxes as verified');

  const saveBtn = document.createElement('button');
  saveBtn.textContent = 'Save & Close';
  saveBtn.style.cssText = 'background:#ff003322;border:1px solid #ff0033;border-radius:3px;padding:0.3em 1em;cursor:pointer;color:#ff6666;';
  footer.append(markVerifiedLabel, saveBtn);

  dialog.append(header, toolbarEl, contentArea, footer);
  overlay.appendChild(dialog);
  document.body.appendChild(overlay);

  // ── Create canvas instance ─────────────────────────────────────────
  let instance;
  let toolbarHandle = { destroy: () => {} };
  let labelHandle   = { destroy: () => {} };
  let unsubInspector = () => {};

  try {
    instance = await createCanvasInstance(canvasWrap, { tool: 'select' });
    await instance.loadImage(img.filename);
  } catch (err) {
    Logger.error('Canvas modal: failed to load image', err);
    instance?.destroy();
    overlay.remove();
    _modalOpen = false;
    return;
  }

  // Snapshot bbox state so we can detect which ones were modified at save time
  const initialBboxSnapshot = snapshotBboxes(instance.runtime.state.image.bboxes);

  // ── Wire shared toolbar (select / bbox / delete) ───────────────────
  toolbarHandle = initToolbar(instance.runtime, toolbarEl, {
    buttons: ['select', 'bbox', 'delete'],
  });

  // ── Wire label panel ───────────────────────────────────────────────
  labelHandle = await initLabelPanel(instance.runtime, labelPanelEl);

  // ── Wire inspector panel (double-click a bbox to edit tags/label) ──
  unsubInspector = initInspector(instance.runtime, { container: inspectorPanel, openTab: false });

  // Highlight the target bbox
  const layer = instance.runtime.canvas.layer;
  const targetView = layer.findOne('#' + bbox.uuid);
  if (targetView) {
    instance.runtime.canvas.transformer.nodes([targetView]);
    layer.draw();
  }

  // ── Close / Save ───────────────────────────────────────────────────
  function closeModal() {
    unsubInspector();
    toolbarHandle.destroy();
    labelHandle.destroy();
    instance.destroy();
    overlay.remove();
    _modalOpen = false;
  }

  // Use closeBtn.disabled as the "save in flight" sentinel so the backdrop
  // handler doesn't need a separate flag.
  closeBtn.addEventListener('click', closeModal);
  overlay.addEventListener('click', e => {
    if (e.target === overlay && !closeBtn.disabled) closeModal();
  });

  saveBtn.addEventListener('click', async () => {
    saveBtn.disabled = true;
    closeBtn.disabled = true; // block close while save is in-flight
    saveBtn.textContent = 'Saving…';

    // If checked, swap UNVERIFIED → VERIFIED on any bbox that was changed.
    // We snapshot the original tags so we can revert if the network call fails —
    // otherwise the user can't retry cleanly (or uncheck the option and retry).
    let tagRollback = null;
    if (markVerifiedCheckbox.checked) {
      const VERIFICATION_UUIDS = new Set([UNVERIFIED_TAG_UUID, VERIFIED_TAG_UUID, FALSE_POS_TAG_UUID]);
      const modifiedUuids = getModifiedBboxUuids(instance.runtime.state.image.bboxes, initialBboxSnapshot);
      tagRollback = new Map();
      for (const [, b] of instance.runtime.state.image.bboxes.entries()) {
        if (modifiedUuids.has(b.uuid)) {
          tagRollback.set(b.uuid, b.tags); // save original so we can roll back on failure
          const nonVerifTags = (b.tags ?? []).filter(t => !VERIFICATION_UUIDS.has(t.uuid));
          b.tags = [...nonVerifTags, { uuid: VERIFIED_TAG_UUID, icon: null }];
        }
      }
    }

    let saved = false;
    try {
      saved = await saveAnnotations(instance.runtime);
      if (saved) {
        const allUuids = [...instance.runtime.state.image.bboxes.keys()];
        await invalidateBboxThumbnails([...new Set([...allUuids, bbox.uuid])]);
      }
    } catch (err) {
      Logger.error('Canvas modal: save failed', err);
    }
    if (saved) {
      // Read the bbox's updated state from the canvas runtime before destroying it.
      // Returns undefined if the user deleted this bbox inside the editor.
      const updatedUIBbox = instance.runtime.state.image.bboxes.get(bbox.uuid);
      closeModal();
      await onSaved(updatedUIBbox);
    } else {
      // Revert auto-verify tag mutations so the user can retry with a clean state
      // (including if they uncheck "Mark modified boxes as verified" before retrying).
      if (tagRollback) {
        for (const [uuid, originalTags] of tagRollback) {
          const b = instance.runtime.state.image.bboxes.get(uuid);
          if (b) b.tags = originalTags;
        }
      }
      saveBtn.disabled = false;
      closeBtn.disabled = false; // re-enable close on failure
      saveBtn.textContent = 'Save & Close';
    }
  });
}

// ── Bulk action modal ─────────────────────────────────────────────────

async function showBulkActionModal(entries, labels, onConfirm) {
  // Fetch fresh tags so recently-created tags always appear in the picker.
  const tags = await fetchTags().catch(() => new Map());
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
    { value: 'addTags', label: 'Add Tags' },
    { value: 'removeTags', label: 'Remove Tags' },
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
      tagPickerRow.style.display = (selectedAction === 'addTags' || selectedAction === 'removeTags') ? 'flex' : 'none';
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

  // Tag picker (shown only for addTags / removeTags)
  const tagOptions = Array.from(tags.values())
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

  const cancelBtn = document.createElement('button');
  cancelBtn.textContent = 'Cancel';
  cancelBtn.addEventListener('click', () => overlay.remove());

  const okBtn = document.createElement('button');
  okBtn.textContent = 'Apply';
  okBtn.style.cssText = 'background:#ff003322;border:1px solid #ff0033;border-radius:3px;padding:0.2em 0.8em;cursor:pointer;';
  okBtn.addEventListener('click', async () => {
    overlay.remove();
    const newLabelUuid = selectedAction === 'relabel' ? labelPicker.value : null;
    const selectedTagUuids = (selectedAction === 'addTags' || selectedAction === 'removeTags')
      ? tagCheckboxField.getValue()
      : [];
    await onConfirm(selectedAction, newLabelUuid, selectedTagUuids);
  });

  btnRow.appendChild(cancelBtn);
  btnRow.appendChild(okBtn);
  dialog.append(title, radioGroup, labelPickerRow, tagPickerRow, btnRow);
  overlay.appendChild(dialog);
  document.body.appendChild(overlay);

  overlay.addEventListener('click', e => { if (e.target === overlay) overlay.remove(); });
}
