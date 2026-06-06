import { fetchLabels } from '/app-static/js/shared/api/labels.js';
import { fetchTags } from '/app-static/js/shared/api/tags.js';
import { createFilterPanel } from '/app-static/js/app/review/filters.js';
import {
  fetchReviewQueue,
  updateImageBboxes,
  updateBbox,
  deleteBbox,
  getBboxThumbnailUrl,
  getImageThumbnailUrl,
  invalidateBboxThumbnails,
  UNVERIFIED_TAG_UUID,
} from '/app-static/js/shared/api/review.js';
import { setImageLabel } from '/app-static/js/shared/api/imageLabels.js';
import { Logger, toast } from '/app-static/js/ui/utils/logging.js';
import { showUnsavedChangesModal } from '/app-static/js/ui/utils/confirmModal.js';
import { CheckboxField } from '/app-static/js/ui/utils/fields/CheckboxField.js';
import { createCanvasInstance } from '/app-static/js/app/image-tagging/canvas/instance.js';
import { saveAnnotations } from '/app-static/js/app/image-tagging/canvas/image.js';
import { events, EventTypes } from '/app-static/js/shared/events/index.js';
import { init as initToolbar } from '/app-static/js/app/image-tagging/toolbar.js';
import { initLabelPanel } from '/app-static/js/app/image-tagging/canvas/labelPanel.js';
import { initInspector } from '/app-static/js/ui/inspector.js';
import { createHoverPreview } from '/app-static/js/ui/thumbnailHoverPreview.js';
import { createEntryView } from '/app-static/js/ui/entryView.js';


const DEFAULT_PAGE_SIZE = 50;
const MAX_ENTRIES = 500; // hard cap on total cards per page

/**
 * Loads thumbnail images with a concurrency cap and cancellation support.
 * Uses fetch() + blob URLs so in-flight requests can be truly aborted when
 * the user navigates to a new page before thumbnails finish loading.
 */
class ThumbnailLoader {
  constructor({ concurrency = 6 } = {}) {
    this._concurrency = concurrency;
    this._queue = [];
    this._active = 0;
    this._controllers = new Set();
  }

  /** Abort all in-flight requests and drop all queued work. Safe to call multiple times. */
  reset() {
    this._queue = [];
    for (const ctrl of this._controllers) ctrl.abort();
    this._controllers.clear();
    this._active = 0;
  }

  /** Enqueue a thumbnail load. Sets imgEl.src (as a blob URL) when a slot is available. */
  load(imgEl, url) {
    this._queue.push({ imgEl, url });
    this._pump();
  }

  _pump() {
    while (this._active < this._concurrency && this._queue.length > 0) {
      const { imgEl, url } = this._queue.shift();
      this._active++;
      const ctrl = new AbortController();
      this._controllers.add(ctrl);
      fetch(url, { signal: ctrl.signal })
        .then(r => {
          if (!r.ok) throw new Error(`HTTP ${r.status}`);
          return r.blob();
        })
        .then(blob => {
          const objUrl = URL.createObjectURL(blob);
          imgEl.src = objUrl;
          const revoke = () => URL.revokeObjectURL(objUrl);
          imgEl.addEventListener('load', revoke, { once: true });
          imgEl.addEventListener('error', revoke, { once: true });
        })
        .catch(err => {
          if (err.name !== 'AbortError') {
            imgEl.alt = 'Load failed';
          }
        })
        .finally(() => {
          this._controllers.delete(ctrl);
          this._active--;
          this._pump();
        });
    }
  }
}

function makeSelect(className, options, defaultValue) {
  const sel = document.createElement('select');
  sel.className = className;
  for (const [value, text] of options) {
    const opt = document.createElement('option');
    opt.value = value;
    opt.textContent = text;
    sel.appendChild(opt);
  }
  sel.value = defaultValue;
  return sel;
}

/**
 * Create a self-contained review panel inside `container`.
 * Works both standalone (review tab) and embedded inside a modal.
 *
 * @param {HTMLElement} container - The element to build the panel inside. Will be cleared.
 * @param {{
 *   fetchImages?: (params: {
 *     sortBy: string, sortOrder: string,
 *     afterUuid: string|null, afterSortValue: string|null, limit: number
 *   }) => Promise<object[]>
 * }} options
 *   fetchImages — optional custom data-fetcher. When provided the tag/date/filename
 *   filter panel is hidden and this function is called instead of fetchReviewQueue.
 *   Receives sort and pagination params; returns an images array with the same shape
 *   as fetchReviewQueue. All bboxes in returned images are shown (no client-side tag
 *   filtering). When omitted the full filter panel is shown and fetchReviewQueue is used.
 *
 * @returns {Promise<{ destroy: () => void }>}
 */
export async function createReviewPanel(container, { fetchImages = null } = {}) {
  container.innerHTML = '';

  const labelsPromise = fetchLabels().catch(() => new Map());
  const tagsPromise = !fetchImages ? fetchTags().catch(() => new Map()) : null;
  const labels = await labelsPromise;

  // ── Toolbar ───────────────────────────────────────────────────────
  // Built as logical groups (search/filter · view mode · selection · pagination ·
  // status) separated by thin rules, so adding/removing a control doesn't
  // reflow the whole bar — each cluster wraps and aligns independently.
  const toolbar = document.createElement('div');
  toolbar.className = 'review-toolbar';

  function makeToolbarGroup(...children) {
    const group = document.createElement('div');
    group.className = 'review-toolbar__group';
    group.append(...children);
    return group;
  }

  function makeToolbarSep() {
    const sep = document.createElement('span');
    sep.className = 'review-toolbar__sep';
    sep.setAttribute('aria-hidden', 'true');
    return sep;
  }

  // Filter toggle — only visible in full-filter mode
  const filterToggleBtn = document.createElement('button');
  filterToggleBtn.className = 'review-toolbar__btn';
  filterToggleBtn.textContent = '☰ Filters';
  filterToggleBtn.title = 'Toggle search filters';
  if (fetchImages) filterToggleBtn.hidden = true;

  // Sort controls — always visible in toolbar (moved from filter panel)
  const sortBySelect = makeSelect('filter-sort-by', [['created_at', 'Date created'], ['updated_at', 'Date modified']], 'created_at');
  const sortOrderSelect = makeSelect('filter-sort-dir', [['desc', 'Newest first'], ['asc', 'Oldest first']], 'desc');

  // ── View filter toggle (All / Images / Bboxes) ───────────────────
  const viewToggle = document.createElement('div');
  viewToggle.className = 'review-view-toggle';
  const viewBtns = [
    { key: 'all', label: 'All' },
    { key: 'images', label: 'Images' },
    { key: 'bboxes', label: 'Bboxes' },
  ];
  for (const { key, label } of viewBtns) {
    const btn = document.createElement('button');
    btn.className = 'view-btn' + (key === 'all' ? ' active' : '');
    btn.dataset.viewKey = key;
    btn.textContent = label;
    btn.addEventListener('click', () => {
      if (_viewFilter === key) return;
      _viewFilter = key;
      viewToggle.querySelectorAll('.view-btn').forEach(b =>
        b.classList.toggle('active', b.dataset.viewKey === key));
      _selected.clear();
      _selectionKind = null;
      renderFromEntries();
    });
    viewToggle.appendChild(btn);
  }

  // ── Per-page selector ────────────────────────────────────────────
  const pageSizeSelect = makeSelect('filter-page-size',
    [['10', '10 imgs/page'], ['25', '25 imgs/page'], ['50', '50 imgs/page'], ['100', '100 imgs/page']],
    String(DEFAULT_PAGE_SIZE));

  // Select-all checkbox
  const selectAllLabel = document.createElement('label');
  selectAllLabel.className = 'review-toolbar__select-all';
  const selectAllCb = document.createElement('input');
  selectAllCb.type = 'checkbox';
  selectAllLabel.appendChild(selectAllCb);
  selectAllLabel.appendChild(document.createTextNode('Select all'));

  // Bulk Apply button
  const bulkBtn = document.createElement('button');
  bulkBtn.className = 'review-toolbar__btn review-toolbar__btn--primary';
  bulkBtn.textContent = 'Bulk Apply (0)';
  bulkBtn.disabled = true;

  // Refresh
  const refreshBtn = document.createElement('button');
  refreshBtn.className = 'review-toolbar__btn';
  refreshBtn.textContent = 'Refresh';

  // Pagination
  const pageInfo = document.createElement('span');
  pageInfo.className = 'review-toolbar__page-info';
  const prevBtn = document.createElement('button');
  prevBtn.className = 'review-toolbar__btn';
  prevBtn.textContent = '‹ Prev';
  const nextBtn = document.createElement('button');
  nextBtn.className = 'review-toolbar__btn';
  nextBtn.textContent = 'Next ›';

  // Status message
  const statusMsg = document.createElement('span');
  statusMsg.className = 'review-toolbar__status';

  // ── Entry view (grid/list scaffold) ───────────────────────────────
  const entryView = createEntryView({
    storageKey: 'review-view',
    defaultView: 'grid',
    showToggle: true,
    listClass: 'review-list',
    gridClass: 'review-grid',
    renderCell: (entry, i) => entry.kind === 'image'
      ? buildImageCard(entry.img, labels, i)
      : buildCard(entry.bbox, entry.img, labels, i),
    renderRow: (entry, i) => entry.kind === 'image'
      ? buildImageRow(entry.img, labels, i)
      : buildBboxRow(entry.bbox, entry.img, labels, i),
  });
  // `grid` aliases the live container holding cards/rows — selection, stale-marking,
  // group-highlight, overlay refresh, bulk apply and nav-context queries all go through it.
  const grid = entryView.layout;

  // ── Query bar: filter · sort · view mode · page size · refresh ────
  // Controls that change *what* is shown. Stays compact so it never wraps.
  toolbar.append(
    makeToolbarGroup(filterToggleBtn, sortBySelect, sortOrderSelect),
    makeToolbarSep(),
    makeToolbarGroup(viewToggle, entryView.element, pageSizeSelect),
    makeToolbarGroup(refreshBtn),
  );
  container.appendChild(toolbar);

  // ── Action bar: sits directly above the results ───────────────────
  // Controls that act on the current page (select, bulk-apply) and navigate
  // between pages (prev/next). Placing them flush against the results makes
  // the spatial relationship obvious and eliminates the "empty space" gap.
  const actionBar = document.createElement('div');
  actionBar.className = 'review-action-bar';
  actionBar.append(
    makeToolbarGroup(selectAllLabel, bulkBtn),
    makeToolbarGroup(prevBtn, pageInfo, nextBtn),
    makeToolbarGroup(statusMsg),
  );
  container.appendChild(actionBar);

  // ── Filter panel (full mode only) ─────────────────────────────────
  // Inserted between the query toolbar and the action bar.
  let filterPanel = null;
  if (!fetchImages) {
    const tags = await tagsPromise;

    function updateFilterBtn() {
      const n = filterPanel.getActiveFilterCount();
      filterToggleBtn.textContent = n > 0 ? `☰ Filters (${n})` : '☰ Filters';
      filterToggleBtn.classList.toggle('filter-active', !filterPanel.isDefault());
    }

    filterPanel = createFilterPanel(labels, tags, () => {
      updateFilterBtn();
      _pageCursors = [null];
      _lastCursor = null;
      render();
    });
    container.insertBefore(filterPanel.element, actionBar);
    updateFilterBtn(); // reflect initial state (Unverified checked → shows "(1)")

    filterToggleBtn.addEventListener('click', () => {
      filterPanel.element.hidden = !filterPanel.element.hidden;
    });
  }

  container.appendChild(entryView.layout);

  // ── Hover preview ─────────────────────────────────────────────────
  const _imageCache = new Map();
  const hoverPreview = createHoverPreview({ imageCache: _imageCache });
  const _thumbLoader = new ThumbnailLoader();

  // ── State ─────────────────────────────────────────────────────────
  // _allEntries holds ALL entries for the current page (both image and bbox).
  // _viewFilter controls which subset is rendered to DOM.
  let _allEntries = [];       // Array<{ kind: 'bbox', bbox, img } | { kind: 'image', img }>
  let _viewFilter = 'all';    // 'all' | 'images' | 'bboxes'
  let _selectionKind = null;  // 'bbox' | 'image' | null — type of currently selected cards
  let _pageSize = DEFAULT_PAGE_SIZE;
  let _imageCardRefreshers = new Map(); // img.uuid → () => void — redraws bbox overlay

  let _selected = new Set();      // selected IDs (bbox.uuid or img.uuid depending on kind)
  let _staleUuids = new Set();    // entry IDs modified on this page (for status display)
  let _deletedUuids = new Set();  // entry IDs deleted this page (disabled/unselectable)
  let _pageCursors = [null];
  let _lastCursor = null;
  let _lastClickedIndex = -1;     // index in _visibleEntries of the last checkbox click
  let _rendering = false;
  let _hasMore = false;
  let _totalCount = 0;

  // _visibleEntries is the filtered subset rendered to DOM (updated by renderFromEntries)
  let _visibleEntries = [];

  const entryId = e => e.kind === 'image' ? e.img.uuid : e.bbox.uuid;

  function updateBulkBtn() {
    // Count only active entries of the current selection kind so the select-all
    // checkbox reaches `checked=true` (not `indeterminate`) when every entry of
    // the selected type is chosen — even if entries of the other type exist.
    const activeCount = _visibleEntries.filter(e =>
      !_deletedUuids.has(entryId(e)) &&
      (_selectionKind === null || e.kind === _selectionKind)
    ).length;
    bulkBtn.textContent = `Bulk Apply (${_selected.size})`;
    bulkBtn.disabled = _selected.size === 0;
    selectAllCb.checked = _selected.size > 0 && _selected.size === activeCount;
    selectAllCb.indeterminate = _selected.size > 0 && _selected.size < activeCount;
  }

  function updateStatusMsg() {
    const stale = _staleUuids.size;

    if (_viewFilter === 'images' || _viewFilter === 'bboxes') {
      const singular = _viewFilter === 'images' ? 'image' : 'bbox';
      const plural   = _viewFilter === 'images' ? 'images' : 'bboxes';
      const total = _visibleEntries.length;
      const unit  = total === 1 ? singular : plural;
      if (total === 0) {
        statusMsg.textContent = `No matching ${plural} on this page`;
      } else if (stale === 0) {
        statusMsg.textContent = `${total} ${unit} on this page`;
      } else {
        const fresh = total - stale;
        statusMsg.textContent = `${fresh} remaining · ${stale} modified`;
      }
    } else {
      // 'all' view — count images and bboxes separately
      let imgCount = 0, bboxCount = 0;
      for (const e of _visibleEntries) {
        if (e.kind === 'image') imgCount++;
        else bboxCount++;
      }
      if (imgCount === 0 && bboxCount === 0) {
        statusMsg.textContent = 'No matching annotations on this page';
      } else if (stale === 0) {
        const imgPart  = `${imgCount} ${imgCount === 1 ? 'image' : 'images'}`;
        const bboxPart = `${bboxCount} ${bboxCount === 1 ? 'bbox' : 'bboxes'}`;
        statusMsg.textContent = `${bboxPart} across ${imgPart} on this page`;
      } else {
        statusMsg.textContent = `${stale} modified on this page`;
      }
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
    const id = card.dataset.bboxUuid ?? card.dataset.imageUuid;
    _staleUuids.add(id);
    if (state === 'deleted') {
      const cb = card.querySelector('input[type=checkbox]');
      if (cb) { cb.checked = false; cb.disabled = true; }
      _selected.delete(id);
      _deletedUuids.add(id);
    }
  }

  function _refreshBboxCard(card, uiBbox, imgFilename, bboxUuid) {
    const thumb = card.querySelector('.thumb');
    if (thumb) thumb.src = getBboxThumbnailUrl(imgFilename, bboxUuid, 300) + `&t=${Date.now()}`;
    const chip = card.querySelector('.label-chip');
    if (chip) {
      if (uiBbox.label) {
        chip.textContent = uiBbox.label.text;
        chip.style.cssText = `background:${uiBbox.label.color}33;border:1px solid ${uiBbox.label.color};`;
      } else {
        chip.textContent = '(unknown)';
        chip.style.cssText = 'background:#88888833;border:1px solid #888;';
      }
    }
    const stillUnverified = (uiBbox.tags ?? []).some(t => t.uuid === UNVERIFIED_TAG_UUID);
    if (!stillUnverified) markCardStale(card, 'verified');
  }

  function updateTypeFading() {
    grid.querySelectorAll('.review-card').forEach(card => {
      const shouldFade = _selectionKind !== null && card.dataset.kind !== _selectionKind;
      card.classList.toggle('type-faded', shouldFade);
    });
  }

  function clearSelection() {
    _selected.clear();
    _selectionKind = null;
    selectAllCb.checked = false;
    selectAllCb.indeterminate = false;
    grid.querySelectorAll('.review-card').forEach(c => {
      c.classList.remove('selected');
      const cb = c.querySelector('input[type=checkbox]');
      if (cb && !cb.disabled) cb.checked = false;
    });
    updateBulkBtn();
    updateTypeFading();
  }

  // ── Card hover utilities (grouping highlight + preview) ───────────
  function attachGroupHighlight(card, imageUuid) {
    card.addEventListener('mouseenter', () => {
      // Bold highlight on the hovered card itself
      card.classList.add(card.dataset.kind === 'image' ? 'group-highlight-self-image' : 'group-highlight-self-bbox');
      // Softer highlight on related cards
      grid.querySelectorAll(`[data-image-uuid="${imageUuid}"]`).forEach(c => {
        if (c === card) return;
        c.classList.add(c.dataset.kind === 'image' ? 'group-highlight-image' : 'group-highlight-bbox');
      });
    });
    card.addEventListener('mouseleave', () => {
      card.classList.remove('group-highlight-self-image', 'group-highlight-self-bbox');
      grid.querySelectorAll('.group-highlight-image,.group-highlight-bbox').forEach(c =>
        c.classList.remove('group-highlight-image', 'group-highlight-bbox'));
    });
  }

  // ── Shared applySelect logic ──────────────────────────────────────
  function applySelect(cb, card, id, entryKind, index, shiftKey, newChecked) {
    // Block if selection type is locked to a different kind
    if (newChecked && _selectionKind !== null && entryKind !== _selectionKind) {
      toast('You can only select one type at a time. Deselect current items first.');
      cb.checked = false;
      return;
    }

    cb.checked = newChecked;

    if (shiftKey && _lastClickedIndex !== -1 && _lastClickedIndex !== index) {
      const lo = Math.min(_lastClickedIndex, index);
      const hi = Math.max(_lastClickedIndex, index);
      for (let i = lo; i <= hi; i++) {
        const entry = _visibleEntries[i];
        if (!entry) continue;
        const eid = entryId(entry);
        if (_deletedUuids.has(eid)) continue;
        // Only select entries of the same kind
        if (entry.kind !== entryKind) continue;
        const selector = entry.kind === 'image'
          ? `[data-image-uuid="${eid}"][data-kind="image"]`
          : `[data-bbox-uuid="${eid}"]`;
        const c = grid.querySelector(selector);
        const cCb = c?.querySelector('input[type=checkbox]');
        if (!cCb || cCb.disabled) continue;
        cCb.checked = newChecked;
        if (newChecked) _selected.add(eid);
        else _selected.delete(eid);
        c.classList.toggle('selected', newChecked);
      }
    } else {
      if (newChecked) _selected.add(id);
      else _selected.delete(id);
      card.classList.toggle('selected', newChecked);
    }

    _lastClickedIndex = index;

    // Update selectionKind
    if (_selected.size === 0) {
      _selectionKind = null;
    } else if (_selectionKind === null) {
      _selectionKind = entryKind;
    }

    updateBulkBtn();
    updateTypeFading();
  }

  // ── Render filtered entries to grid ──────────────────────────────
  function renderFromEntries() {
    _selected.clear();
    _staleUuids = new Set();
    _deletedUuids = new Set();
    _lastClickedIndex = -1;
    _selectionKind = null;

    // Filter entries by view mode
    _visibleEntries = _allEntries.filter(e => {
      if (_viewFilter === 'images') return e.kind === 'image';
      if (_viewFilter === 'bboxes') return e.kind === 'bbox';
      return true; // 'all'
    });

    updateBulkBtn();

    if (_visibleEntries.length === 0) {
      entryView.setEmptyText(_hasMore
        ? 'No matching annotations on this page.'
        : 'No matching annotations found.');
      entryView.setEntries([]);
      updateStatusMsg();
      return;
    }

    entryView.setEmptyText('');
    entryView.setEntries(_visibleEntries);

    updateStatusMsg();
  }

  // ── Render (fetch + display) ──────────────────────────────────────
  async function render() {
    if (_rendering) return;
    _rendering = true;

    const MAX_AUTO_SKIP = 10;
    let autoSkipped = 0;

    fetchLoop: while (true) {
      refreshBtn.disabled = true;
      prevBtn.disabled = true;
      nextBtn.disabled = true;
      statusMsg.textContent = 'Loading…';
      _thumbLoader.reset();
      grid.innerHTML = '';
      _selected.clear();
      _staleUuids = new Set();
      _deletedUuids = new Set();
      _allEntries = [];
      _visibleEntries = [];
      _selectionKind = null;
      _lastCursor = null;
      _lastClickedIndex = -1;
      for (const { objectUrl } of _imageCache.values()) URL.revokeObjectURL(objectUrl);
      _imageCache.clear();
      _imageCardRefreshers.clear();
      updateBulkBtn();

      const sortBy = sortBySelect.value;
      const sortOrder = sortOrderSelect.value;
      const cursor = _pageCursors[_pageCursors.length - 1];

      const fs = !fetchImages ? filterPanel.getState() : null;
      let images;
      let hasMore = false;
      let totalCount = 0;
      try {
        if (fetchImages) {
          images = await fetchImages({
            sortBy,
            sortOrder,
            afterUuid: cursor?.uuid ?? null,
            afterSortValue: cursor?.sortValue ?? null,
            limit: _pageSize,
          });
          // fetchImages callers don't support has_more; fall back to length heuristic.
          hasMore = images.length === _pageSize;
        } else {
          ({ images, hasMore, totalCount } = await fetchReviewQueue({
            tagUuids: fs.showAllBboxes ? [] : fs.tagUuids,
            showAllBboxes: fs.showAllBboxes,
            labelUuid: fs.labelUuid || undefined,
            sortBy,
            sortOrder,
            filenameSearch: fs.filenameSearch || undefined,
            dateFrom: fs.dateFrom || undefined,
            dateTo: fs.dateTo || undefined,
            afterUuid: cursor?.uuid ?? null,
            afterSortValue: cursor?.sortValue ?? null,
            imageLabelFilters: fs.imageLabelFilters,
            limit: _pageSize,
          }));
        }
      } catch (err) {
        statusMsg.textContent = `Error: ${err.message}`;
        refreshBtn.disabled = false;
        prevBtn.disabled = _pageCursors.length <= 1;
        nextBtn.disabled = true;
        _rendering = false;
        return;
      }

      _hasMore = hasMore;
      _totalCount = totalCount;

      if (images.length > 0) {
        const last = images[images.length - 1];
        _lastCursor = {
          uuid: last.uuid,
          sortValue: sortBy === 'uuid' ? null
            : sortBy === 'updated_at' ? (last.updatedAt ?? null)
            : (last.createdAt ?? null),
        };
      }

      // Build _allEntries: for each image, one ImageEntry then its matching BboxEntries.
      const showAll = fetchImages || fs?.showAllBboxes;
      const tagSet = (fs && !showAll) ? new Set(fs.tagUuids) : null;

      for (const img of images) {
        const matching = (img.bboxes || []).filter(b =>
          showAll || (b.tags || []).some(t => tagSet.has(t.uuid))
        );
        // Always add an image-level entry
        _allEntries.push({ kind: 'image', img });
        // Add bbox entries for each matching bbox
        for (const bbox of matching) {
          _allEntries.push({ kind: 'bbox', bbox, img });
        }
      }

      // Hard cap: truncate total entries to prevent runaway card counts
      if (_allEntries.length > MAX_ENTRIES) {
        _allEntries = _allEntries.slice(0, MAX_ENTRIES);
        _hasMore = true; // there are more, user must paginate
      }

      // Auto-skip empty pages (when client-side tag filter leaves nothing)
      const bboxEntries = _allEntries.filter(e => e.kind === 'bbox');
      if (bboxEntries.length === 0 && _hasMore && autoSkipped < MAX_AUTO_SKIP) {
        autoSkipped++;
        _pageCursors[_pageCursors.length - 1] = _lastCursor;
        continue fetchLoop;
      }

      break fetchLoop;
    }

    const totalPages = _totalCount > 0 ? Math.ceil(_totalCount / _pageSize) : null;
    const pageLabel = totalPages ? `Page ${_pageCursors.length} of ${totalPages}` : `Page ${_pageCursors.length}`;
    pageInfo.textContent = pageLabel;
    prevBtn.disabled = _pageCursors.length <= 1;
    nextBtn.disabled = !_hasMore;

    renderFromEntries();

    refreshBtn.disabled = false;
    _rendering = false;
  }

  // ── Nav context builder for canvas modal ─────────────────────────
  // Builds a deduplicated list of images from the current visible entries,
  // finds the start index for the given image UUID, and attaches an onClose
  // callback that scrolls to + briefly highlights the last-viewed card.
  function _buildNavContext(targetImgUuid) {
    const seen = new Set();
    const navItems = [];
    for (const entry of _visibleEntries) {
      const uuid = entry.img.uuid;
      if (seen.has(uuid)) continue;
      seen.add(uuid);
      navItems.push({ img: entry.img, bbox: entry.bbox ?? entry.img.bboxes?.[0] ?? null });
    }
    const foundIndex = navItems.findIndex(item => item.img.uuid === targetImgUuid);
    const startIndex = foundIndex >= 0 ? foundIndex : 0;
    return {
      items: navItems,
      startIndex,
      onSaved(savedImg, savedBboxesMap) {
        if (!savedImg || !savedBboxesMap) return;
        for (const [bboxUuid, uiBbox] of savedBboxesMap) {
          const card = grid.querySelector(`[data-bbox-uuid="${bboxUuid}"]`);
          if (card) _refreshBboxCard(card, uiBbox, savedImg.filename, bboxUuid);
        }
      },
      onClose(closedImg) {
        if (!closedImg) return;
        const card = grid.querySelector(`[data-image-uuid="${closedImg.uuid}"]`);
        if (!card) return;
        card.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        card.classList.remove('recently-viewed');
        void card.offsetWidth; // force reflow to restart the animation
        card.classList.add('recently-viewed');
        card.addEventListener('animationend', () => card.classList.remove('recently-viewed'), { once: true });
      },
    };
  }

  // ── Shared card-interaction helpers ──────────────────────────────

  /**
   * Attach the checkbox-click and 220ms debounced row-click listeners used by
   * all four card/row builders. Returns a `clearTimer()` function for dblclick
   * handlers to cancel a pending single-click before opening the canvas modal.
   */
  function attachSelectListeners(el, cb, id, kind, index) {
    cb.addEventListener('click', (e) => {
      applySelect(cb, el, id, kind, index, e.shiftKey, cb.checked);
    });
    let _timer = null;
    el.addEventListener('click', (e) => {
      if (e.target === cb || cb.disabled) return;
      if (e.shiftKey) e.preventDefault();
      clearTimeout(_timer);
      const shiftKey = e.shiftKey;
      const newChecked = !cb.checked;
      _timer = setTimeout(() => {
        if (!el.isConnected) return;
        applySelect(cb, el, id, kind, index, shiftKey, newChecked);
      }, 220);
    });
    return () => clearTimeout(_timer);
  }

  /**
   * Populate `container` with deduplicated label chips for `img.bboxes`.
   * Falls back to an "unlabeled" chip when boxes exist but none have a label.
   */
  function appendLabelChips(container, bboxes, labels) {
    const seen = new Set();
    for (const bbox of bboxes) {
      const luuid = bbox.labelUuid;
      if (!luuid || seen.has(luuid)) continue;
      seen.add(luuid);
      const lbl = labels.get(luuid);
      if (!lbl) continue;
      const chip = document.createElement('span');
      chip.className = 'label-chip';
      chip.textContent = lbl.name;
      chip.style.cssText = `background:${lbl.color}33;border:1px solid ${lbl.color};`;
      container.appendChild(chip);
    }
    if (seen.size === 0 && bboxes.length > 0) {
      const chip = document.createElement('span');
      chip.className = 'label-chip';
      chip.textContent = 'unlabeled';
      chip.style.cssText = 'background:#88888833;border:1px solid #888;';
      container.appendChild(chip);
    }
  }

  // ── Bbox card builder ─────────────────────────────────────────────
  function buildCard(bbox, img, labels, index) {
    // Prefer the label embedded in the API response (always present and current);
    // fall back to the labels Map only when the response omits it.
    const label = bbox.label ?? (bbox.labelUuid ? labels.get(bbox.labelUuid) : null);
    const labelName = label?.name ?? '(unknown)';
    const labelColor = label?.color ?? '#888';
    const isVerified = !(bbox.tags ?? []).some(t => t.uuid === UNVERIFIED_TAG_UUID);
    const confidence = bbox.metadataJson?.confidence
      ? `${(parseFloat(bbox.metadataJson.confidence) * 100).toFixed(0)}%`
      : '';

    const card = document.createElement('div');
    card.className = 'review-card';
    card.dataset.bboxUuid = bbox.uuid;
    card.dataset.imageUuid = img.uuid;
    card.dataset.kind = 'bbox';

    const cb = document.createElement('input');
    cb.type = 'checkbox';
    const clearTimer = attachSelectListeners(card, cb, bbox.uuid, 'bbox', index);

    const thumbImg = document.createElement('img');
    thumbImg.className = 'thumb';
    thumbImg.alt = labelName;
    thumbImg.title = 'Double-click to edit in canvas';
    _thumbLoader.load(thumbImg, getBboxThumbnailUrl(img.filename, bbox.uuid, 300));
    thumbImg.addEventListener('dblclick', () => {
      clearTimer();
      const navCtx = _buildNavContext(img.uuid);
      showCanvasModal(bbox, img, labels, async (updatedUIBbox, currentImg) => {
        if (currentImg?.uuid === img.uuid) {
          // Saved from the originally-opened image — update the specific bbox card.
          const card = grid.querySelector(`[data-bbox-uuid="${bbox.uuid}"]`);
          if (card) {
            if (updatedUIBbox === undefined) {
              markCardStale(card, 'deleted');
            } else {
              _refreshBboxCard(card, updatedUIBbox, img.filename, bbox.uuid);
            }
          }
        }
        updateBulkBtn();
        updateStatusMsg();
      }, navCtx);
    });

    const info = document.createElement('div');
    info.className = 'card-info';

    const chip = document.createElement('span');
    chip.className = 'label-chip';
    chip.textContent = labelName;
    chip.style.cssText = `background:${labelColor}33;border:1px solid ${labelColor};`;

    const verifyChip = document.createElement('span');
    verifyChip.className = 'verify-chip';
    verifyChip.textContent = isVerified ? '✓ Verified' : '◌ Unverified';
    verifyChip.classList.toggle('verify-chip--verified', isVerified);
    verifyChip.title = isVerified ? 'Verified' : 'Not yet verified';

    const meta = document.createElement('div');
    meta.style.cssText = 'margin-top:0.2em;color:#aaa;font-size:0.75em;';
    const shortPath = img.filename.split('/').slice(-2).join('/');
    meta.textContent = confidence ? `${confidence} · ${shortPath}` : shortPath;

    info.appendChild(chip);
    info.appendChild(verifyChip);
    info.appendChild(meta);
    card.appendChild(cb);
    card.appendChild(thumbImg);
    card.appendChild(info);

    attachGroupHighlight(card, img.uuid);
    hoverPreview.attach(card, {
      imagePath: img.filename,
      imageUuid: img.uuid,
      allBboxes: img.bboxes,
      hoveredBboxUuid: bbox.uuid,
    });

    return card;
  }

  // ── Image card builder ────────────────────────────────────────────
  function buildImageCard(img, labels, index) {
    const card = document.createElement('div');
    card.className = 'review-card review-card--image';
    card.dataset.imageUuid = img.uuid;
    card.dataset.kind = 'image';

    const cb = document.createElement('input');
    cb.type = 'checkbox';
    const clearTimer = attachSelectListeners(card, cb, img.uuid, 'image', index);

    // Double-click thumbnail → open canvas modal on first available bbox
    const thumbImg = document.createElement('img');
    thumbImg.className = 'thumb';
    thumbImg.alt = img.filename.split('/').pop();
    thumbImg.title = 'Double-click to edit in canvas';
    _thumbLoader.load(thumbImg, getImageThumbnailUrl(img.filename, 200));
    thumbImg.addEventListener('dblclick', () => {
      clearTimer();
      const firstBbox = (img.bboxes ?? [])[0];
      if (firstBbox) {
        const navCtx = _buildNavContext(img.uuid);
        showCanvasModal(firstBbox, img, labels, async (_updatedUIBbox, _currentImg) => {
          updateBulkBtn();
          updateStatusMsg();
        }, navCtx);
      }
    });

    // Bbox canvas overlay — draws normalized rects over the thumbnail
    const thumbCanvas = document.createElement('canvas');
    thumbCanvas.className = 'thumb-overlay';
    const thumbCtx = thumbCanvas.getContext('2d');

    function drawBboxOverlay() {
      const natW = thumbImg.naturalWidth;
      const natH = thumbImg.naturalHeight;
      if (!natW || !natH) return;
      const cssW = thumbImg.clientWidth;
      const cssH = thumbImg.clientHeight;
      if (!cssW || !cssH) return;

      thumbCanvas.width = cssW;
      thumbCanvas.height = cssH;

      // The img uses object-fit:contain, so compute the letterbox offsets.
      const scale = Math.min(cssW / natW, cssH / natH);
      const dispW = natW * scale;
      const dispH = natH * scale;
      const offsetX = (cssW - dispW) / 2;
      const offsetY = (cssH - dispH) / 2;

      thumbCtx.clearRect(0, 0, cssW, cssH);
      thumbCtx.lineWidth = Math.max(1, Math.round(cssW / 150));
      for (const bbox of (img.bboxes ?? [])) {
        thumbCtx.strokeStyle = bbox.label?.color ?? '#ffdd00';
        thumbCtx.strokeRect(
          offsetX + bbox.x * dispW,
          offsetY + bbox.y * dispH,
          bbox.width * dispW,
          bbox.height * dispH,
        );
      }
    }

    if (thumbImg.complete && thumbImg.naturalWidth) {
      drawBboxOverlay();
    } else {
      thumbImg.onload = drawBboxOverlay;
    }

    // Register so bulk bbox actions can trigger a redraw
    _imageCardRefreshers.set(img.uuid, drawBboxOverlay);

    // Bbox count badge
    const countBadge = document.createElement('div');
    countBadge.className = 'bbox-count-badge';
    const bc = (img.bboxes ?? []).length;
    countBadge.textContent = `${bc} box${bc !== 1 ? 'es' : ''}`;

    const thumbWrap = document.createElement('div');
    thumbWrap.style.cssText = 'position:relative;';
    thumbWrap.appendChild(thumbImg);
    thumbWrap.appendChild(thumbCanvas);
    thumbWrap.appendChild(countBadge);

    const info = document.createElement('div');
    info.className = 'card-info';

    // Deduplicated label chips
    const bboxes = img.bboxes ?? [];
    const chipsRow = document.createElement('div');
    chipsRow.style.cssText = 'display:flex;flex-wrap:wrap;gap:2px;margin-bottom:0.2em;';
    appendLabelChips(chipsRow, bboxes, labels);

    // Verification status line (verified / unverified counts).
    const statusRow = document.createElement('div');
    statusRow.style.cssText = 'color:#aaa;font-size:0.75em;margin-bottom:0.15em;';
    if (bboxes.length > 0) {
      let unverified = 0;
      for (const bbox of bboxes) {
        if ((bbox.tags ?? []).some(t => t.uuid === UNVERIFIED_TAG_UUID)) unverified++;
      }
      const verified = bboxes.length - unverified;
      const parts = [];
      if (verified > 0) parts.push(`✓ ${verified}`);
      if (unverified > 0) parts.push(`? ${unverified}`);
      statusRow.textContent = parts.join('  ');
    }

    const meta = document.createElement('div');
    meta.style.cssText = 'color:#aaa;font-size:0.75em;';
    meta.textContent = img.filename.split('/').slice(-2).join('/');

    info.appendChild(chipsRow);
    if (bboxes.length > 0) info.appendChild(statusRow);
    info.appendChild(meta);
    card.appendChild(cb);
    card.appendChild(thumbWrap);
    card.appendChild(info);

    attachGroupHighlight(card, img.uuid);
    hoverPreview.attach(card, {
      imagePath: img.filename,
      imageUuid: img.uuid,
      allBboxes: img.bboxes,
      hoveredBboxUuid: null,
    });

    return card;
  }

  // ── Bbox list-row builder (compact card for list view) ────────────
  function buildBboxRow(bbox, img, labels, index) {
    const label = bbox.label ?? (bbox.labelUuid ? labels.get(bbox.labelUuid) : null);
    const labelName = label?.name ?? '(unknown)';
    const labelColor = label?.color ?? '#888';
    const isVerified = !(bbox.tags ?? []).some(t => t.uuid === UNVERIFIED_TAG_UUID);
    const confidence = bbox.metadataJson?.confidence
      ? `${(parseFloat(bbox.metadataJson.confidence) * 100).toFixed(0)}%`
      : '';

    const row = document.createElement('li');
    row.className = 'review-row review-card';
    row.dataset.bboxUuid = bbox.uuid;
    row.dataset.imageUuid = img.uuid;
    row.dataset.kind = 'bbox';

    const cb = document.createElement('input');
    cb.type = 'checkbox';
    const clearTimer = attachSelectListeners(row, cb, bbox.uuid, 'bbox', index);

    const thumbImg = document.createElement('img');
    thumbImg.className = 'thumb';
    thumbImg.alt = labelName;
    thumbImg.title = 'Double-click to edit in canvas';
    _thumbLoader.load(thumbImg, getBboxThumbnailUrl(img.filename, bbox.uuid, 96));
    thumbImg.addEventListener('dblclick', () => {
      clearTimer();
      const navCtx = _buildNavContext(img.uuid);
      showCanvasModal(bbox, img, labels, async (updatedUIBbox, currentImg) => {
        if (currentImg?.uuid === img.uuid) {
          const card = grid.querySelector(`[data-bbox-uuid="${bbox.uuid}"]`);
          if (card) {
            if (updatedUIBbox === undefined) {
              markCardStale(card, 'deleted');
            } else {
              _refreshBboxCard(card, updatedUIBbox, img.filename, bbox.uuid);
            }
          }
        }
        updateBulkBtn();
        updateStatusMsg();
      }, navCtx);
    });

    const info = document.createElement('div');
    info.className = 'card-info';

    const chip = document.createElement('span');
    chip.className = 'label-chip';
    chip.textContent = labelName;
    chip.style.cssText = `background:${labelColor}33;border:1px solid ${labelColor};`;

    const verifyChip = document.createElement('span');
    verifyChip.className = 'verify-chip';
    verifyChip.textContent = isVerified ? '✓ Verified' : '◌ Unverified';
    verifyChip.classList.toggle('verify-chip--verified', isVerified);
    verifyChip.title = isVerified ? 'Verified' : 'Not yet verified';

    const meta = document.createElement('span');
    meta.className = 'review-row__meta';
    const shortPath = img.filename.split('/').slice(-2).join('/');
    meta.textContent = confidence ? `${confidence} · ${shortPath}` : shortPath;

    info.appendChild(chip);
    info.appendChild(verifyChip);
    info.appendChild(meta);
    row.appendChild(cb);
    row.appendChild(thumbImg);
    row.appendChild(info);

    attachGroupHighlight(row, img.uuid);
    hoverPreview.attach(row, {
      imagePath: img.filename,
      imageUuid: img.uuid,
      allBboxes: img.bboxes,
      hoveredBboxUuid: bbox.uuid,
    });

    return row;
  }

  // ── Image list-row builder (compact card for list view) ───────────
  function buildImageRow(img, labels, index) {
    const row = document.createElement('li');
    row.className = 'review-row review-card review-card--image';
    row.dataset.imageUuid = img.uuid;
    row.dataset.kind = 'image';

    const cb = document.createElement('input');
    cb.type = 'checkbox';
    const clearTimer = attachSelectListeners(row, cb, img.uuid, 'image', index);

    // Double-click thumbnail → open canvas modal on first available bbox
    const thumbImg = document.createElement('img');
    thumbImg.className = 'thumb';
    thumbImg.alt = img.filename.split('/').pop();
    thumbImg.title = 'Double-click to edit in canvas';
    _thumbLoader.load(thumbImg, getImageThumbnailUrl(img.filename, 96));
    thumbImg.addEventListener('dblclick', () => {
      clearTimer();
      const firstBbox = (img.bboxes ?? [])[0];
      if (firstBbox) {
        const navCtx = _buildNavContext(img.uuid);
        showCanvasModal(firstBbox, img, labels, async (_updatedUIBbox, _currentImg) => {
          updateBulkBtn();
          updateStatusMsg();
        }, navCtx);
      }
    });

    // Bbox count badge — list rows skip the per-row overlay canvas to stay light
    // (hover preview still shows annotations), so this row is NOT registered in
    // _imageCardRefreshers; bulk overlay redraws simply no-op for it.
    const countBadge = document.createElement('span');
    countBadge.className = 'bbox-count-badge';
    const bc = (img.bboxes ?? []).length;
    countBadge.textContent = `${bc} box${bc !== 1 ? 'es' : ''}`;

    const info = document.createElement('div');
    info.className = 'card-info';

    // Deduplicated label chips
    const bboxes = img.bboxes ?? [];
    const chipsRow = document.createElement('span');
    chipsRow.className = 'review-row__chips';
    appendLabelChips(chipsRow, bboxes, labels);

    const meta = document.createElement('span');
    meta.className = 'review-row__meta';
    meta.textContent = img.filename.split('/').slice(-2).join('/');

    info.appendChild(chipsRow);
    info.appendChild(meta);
    row.appendChild(cb);
    row.appendChild(thumbImg);
    row.appendChild(countBadge);
    row.appendChild(info);

    attachGroupHighlight(row, img.uuid);
    hoverPreview.attach(row, {
      imagePath: img.filename,
      imageUuid: img.uuid,
      allBboxes: img.bboxes,
      hoveredBboxUuid: null,
    });

    return row;
  }

  // ── Select all ────────────────────────────────────────────────────
  selectAllCb.addEventListener('change', () => {
    if (!selectAllCb.checked) {
      // Unchecked → clear everything
      clearSelection();
      _lastClickedIndex = -1;
      return;
    }

    // Determine kind: use current selection's kind, or the kind of the first
    // visible non-deleted entry (so selecting nothing then hitting select-all
    // anchors to whatever type comes first on the page).
    const firstEntry = _visibleEntries.find(e => !_deletedUuids.has(entryId(e)));
    const targetKind = _selectionKind ?? firstEntry?.kind ?? null;
    if (!targetKind) {
      selectAllCb.checked = false;
      return;
    }

    _selectionKind = targetKind;
    for (const entry of _visibleEntries) {
      const eid = entryId(entry);
      if (_deletedUuids.has(eid)) continue;
      if (entry.kind !== targetKind) continue;
      _selected.add(eid);
    }

    grid.querySelectorAll('.review-card').forEach(c => {
      const eid = c.dataset.bboxUuid ?? c.dataset.imageUuid;
      if (_deletedUuids.has(eid)) return;
      if (c.dataset.kind !== targetKind) return;
      const cb = c.querySelector('input[type=checkbox]');
      if (cb && !cb.disabled) cb.checked = true;
      c.classList.add('selected');
    });

    _lastClickedIndex = -1;
    updateBulkBtn();
    updateTypeFading();
  });

  // ── Bulk Apply ────────────────────────────────────────────────────
  bulkBtn.addEventListener('click', async () => {
    if (_selected.size === 0) return;
    const selectedEntries = _visibleEntries.filter(e => _selected.has(entryId(e)));

    if (_selectionKind === 'image') {
      await showImageBulkActionModal(selectedEntries, labels, async (action, labelUuid) => {
        const { failCount, succeededEntries } = await executeBulkImageAction(action, selectedEntries, labelUuid);
        for (const { img } of succeededEntries) {
          const card = grid.querySelector(`[data-image-uuid="${img.uuid}"][data-kind="image"]`);
          if (card) markCardStale(card, 'tagged');
        }
        clearSelection();
        updateStatusMsg();
        if (failCount > 0) {
          statusMsg.textContent += ` (${failCount} action${failCount === 1 ? '' : 's'} failed — check console)`;
        }
      });
    } else {
      await showBulkActionModal(selectedEntries, labels, async (action, newLabelUuid, selectedTagUuids) => {
        const { failCount, succeededEntries } = await executeBulkAction(action, selectedEntries, newLabelUuid, selectedTagUuids);

        for (const { bbox, img } of succeededEntries) {
          const card = grid.querySelector(`[data-bbox-uuid="${bbox.uuid}"]`);
          if (!card) continue;

          if (action === 'delete') {
            markCardStale(card, 'deleted');
            // Remove from in-memory bboxes so image card overlay updates
            const idx = img.bboxes?.findIndex(b => b.uuid === bbox.uuid) ?? -1;
            if (idx !== -1) img.bboxes.splice(idx, 1);
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
                // Update in-memory bbox label so image card overlay reflects new color
                const bboxInImg = img.bboxes?.find(b => b.uuid === bbox.uuid);
                if (bboxInImg) { bboxInImg.labelUuid = newLabelUuid; bboxInImg.label = label; }
              }
              markCardStale(card, 'relabeled');
            } else if (action === 'addTags' || action === 'removeTags') {
              markCardStale(card, 'tagged');
            } else {
              markCardStale(card, 'relabeled');
            }
          }
          // Redraw bbox overlay on the related image card (if visible)
          _imageCardRefreshers.get(img.uuid)?.();
        }

        clearSelection();
        updateStatusMsg();
        if (failCount > 0) {
          statusMsg.textContent += ` (${failCount} action${failCount === 1 ? '' : 's'} failed — check console)`;
        }
      });
    }
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
  sortBySelect.addEventListener('change', resetAndRender);
  sortOrderSelect.addEventListener('change', resetAndRender);
  pageSizeSelect.addEventListener('change', () => {
    _pageSize = parseInt(pageSizeSelect.value, 10) || DEFAULT_PAGE_SIZE;
    resetAndRender();
  });

  render();

  return {
    destroy() {
      _thumbLoader.reset();
      hoverPreview.destroy();
      entryView.destroy();
      container.innerHTML = '';
    },
    filterPanel,
  };
}

/** Module-level reference to the active filter panel for the Review tab. */
let _activeFilterPanel = null;
/** Queued filter state to apply as soon as the panel is ready. */
let _pendingFilterState = null;

/**
 * Apply filter state now if the panel is ready, or queue it to be applied
 * once init() completes. Eliminates the rAF/setTimeout timing race in main.js.
 */
export function applyFilterWhenReady(state) {
  if (_activeFilterPanel) {
    _activeFilterPanel.setState(state);
  } else {
    _pendingFilterState = state;
  }
}

/**
 * Initialize the Review tab. Called once on first tab focus.
 */
export async function init() {
  const root = document.getElementById('review-root');
  if (!root) return;
  const panel = await createReviewPanel(root);
  _activeFilterPanel = panel.filterPanel ?? null;
  if (_pendingFilterState && _activeFilterPanel) {
    _activeFilterPanel.setState(_pendingFilterState);
    _pendingFilterState = null;
  }
}

/**
 * Open a modal window containing a full review panel.
 *
 * @param {{
 *   title?: string,
 *   fetchImages?: (params: object) => Promise<object[]>
 * }} options
 *   title      — shown in the modal header bar.
 *   fetchImages — passed straight through to createReviewPanel; see its docs.
 */
export async function showReviewModal({ title = 'Browse Images', fetchImages } = {}) {
  const overlay = document.createElement('div');
  overlay.style.cssText = 'position:fixed;inset:0;background:#0009;display:flex;align-items:center;justify-content:center;z-index:900;';

  const dialog = document.createElement('div');
  dialog.style.cssText = 'background:var(--pico-background-color,#121212);border:1px solid #555;border-radius:6px;width:92vw;height:90vh;display:flex;flex-direction:column;overflow:hidden;';

  // Header
  const header = document.createElement('div');
  header.style.cssText = 'display:flex;align-items:center;padding:0.5em 1em;border-bottom:1px solid #444;gap:0.8em;flex-shrink:0;';

  const titleEl = document.createElement('strong');
  titleEl.textContent = title;

  const closeBtn = document.createElement('button');
  closeBtn.textContent = '✕';
  closeBtn.title = 'Close';
  closeBtn.style.cssText = 'margin-left:auto;cursor:pointer;padding:0.2em 0.6em;';

  header.append(titleEl, closeBtn);

  // Scrollable content area for the panel
  const content = document.createElement('div');
  content.style.cssText = 'flex:1;overflow:auto;display:flex;flex-direction:column;min-height:0;';

  dialog.append(header, content);
  overlay.appendChild(dialog);
  document.body.appendChild(overlay);

  function close() { overlay.remove(); }
  closeBtn.addEventListener('click', close);
  overlay.addEventListener('click', (e) => { if (e.target === overlay) close(); });

  await createReviewPanel(content, { fetchImages });
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
        // relabel: change the label, strip UNVERIFIED only when a real label is being
        // assigned — if newLabelUuid is empty/null (e.g. no labels exist in the system)
        // keep UNVERIFIED so the bbox stays visible in the review queue.
        const shouldStripUnverified = !!newLabelUuid;
        const currentTagUuids = (bbox.tags || []).map(t => t.uuid)
          .filter(u => !shouldStripUnverified || u !== UNVERIFIED_TAG_UUID);
        const labelUuid = newLabelUuid ?? (bbox.labelUuid ?? bbox.label?.uuid);
        await updateBbox(bbox.uuid, { labelUuid, tagUuids: currentTagUuids });
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
 * Open a full canvas editor modal for a single bbox.
 *
 * @param {object} bbox    The bbox to open (highlighted in the canvas)
 * @param {object} img     The parent image object
 * @param {Map}    labels  Labels map (uuid → { name, color })
 * @param {(updatedBbox: object|undefined, currentImg: object) => Promise<void>} onSaved
 *   Called after a successful save. `updatedBbox` is the updated UIBBox for the
 *   initially-opened bbox (undefined if deleted). `currentImg` is the image that
 *   was on-screen when Save & Close was pressed (may differ from `img` after nav).
 * @param {{ items: {img,bbox}[], startIndex: number, onClose?: (img) => void }|null} [navContext]
 *   Optional navigation context. When provided, the [ ] hotkeys navigate through
 *   `items`; `onClose` is called after the modal closes with the last-viewed image.
 */
async function showCanvasModal(bbox, img, labels, onSaved, navContext = null) {
  if (_modalOpen) return;
  _modalOpen = true;

  // ── Navigation state ──────────────────────────────────────────────
  let _navIndex = navContext?.startIndex ?? 0;
  let _currentImg = img;
  let _currentBbox = bbox;

  // ── Build DOM ─────────────────────────────────────────────────────
  const overlay = document.createElement('div');
  overlay.className = 'canvas-modal-overlay';

  const dialog = document.createElement('div');
  dialog.className = 'canvas-modal-dialog';

  // Header
  const header = document.createElement('div');
  header.className = 'canvas-modal-header';
  const titleEl = document.createElement('span');
  titleEl.className = 'canvas-modal-title';
  titleEl.textContent = `Edit: ${img.filename.split('/').slice(-2).join('/')}`;
  const closeBtn = document.createElement('button');
  closeBtn.className = 'canvas-modal-close';
  closeBtn.textContent = '✕';
  closeBtn.title = 'Close';
  header.append(titleEl, closeBtn);

  // Toolbar
  const toolbarEl = document.createElement('div');
  toolbarEl.className = 'canvas-modal-toolbar';

  // Content area: label panel (left) + canvas (right) + inspector
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

  // Footer — Save & Close starts dimmed (no unsaved changes yet)
  const footer = document.createElement('div');
  footer.className = 'canvas-modal-footer';

  const cancelBtn = document.createElement('button');
  cancelBtn.textContent = 'Cancel';
  cancelBtn.style.cssText = 'border:1px solid #555;border-radius:3px;padding:0.3em 1em;cursor:pointer;color:#aaa;margin-left:auto;';

  const saveBtn = document.createElement('button');
  saveBtn.textContent = 'Save & Close';
  saveBtn.style.cssText = 'background:#33333322;border:1px solid #555;border-radius:3px;padding:0.3em 1em;cursor:pointer;color:#888;';
  footer.append(cancelBtn, saveBtn);

  dialog.append(header, toolbarEl, contentArea, footer);
  overlay.appendChild(dialog);
  document.body.appendChild(overlay);

  // ── Create canvas instance ─────────────────────────────────────────
  let instance;
  let toolbarHandle  = { destroy: () => {} };
  let labelHandle    = { destroy: () => {} };
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

  // ── Wire shared toolbar (select / bbox / delete) ───────────────────
  toolbarHandle = initToolbar(instance.runtime, toolbarEl, {
    buttons: ['undo', 'redo', 'select', 'bbox', 'delete'],
  });

  // ── Wire label panel ───────────────────────────────────────────────
  labelHandle = await initLabelPanel(instance.runtime, labelPanelEl);

  // ── Wire inspector panel (double-click a bbox to edit tags/label) ──
  unsubInspector = initInspector(instance.runtime, { container: inspectorPanel, openTab: false });

  // Highlight the initially-opened bbox
  function _highlightBbox(bboxUuid) {
    const layer = instance.runtime.canvas.layer;
    const view = layer.findOne('#' + bboxUuid);
    if (view) {
      instance.runtime.canvas.transformer.nodes([view]);
      layer.draw();
    }
  }
  _highlightBbox(bbox.uuid);

  // ── Save button state: highlight only when there are unsaved changes ──
  function _updateSaveBtn() {
    if (saveBtn.disabled) return; // don't override while saving
    const dirty = instance.runtime.isDirty;
    saveBtn.style.cssText = dirty
      ? 'background:#ff003322;border:1px solid #ff0033;border-radius:3px;padding:0.3em 1em;cursor:pointer;color:#ff6666;'
      : 'background:#33333322;border:1px solid #555;border-radius:3px;padding:0.3em 1em;cursor:pointer;color:#888;';
  }
  const unsubDirty = events.subscribeToCanvas(
    instance.runtime.id, EventTypes.CANVAS_BBOXES_CHANGED, _updateSaveBtn,
  );

  // ── Navigation: wire [ ] hotkeys to navigate within the results page ──
  let unsubNavLoad = null;
  if (navContext?.items?.length) {
    // Duck-typed FileNavigation used by files.js's next_image / prev_image.
    // Those functions also call guardUnsavedChanges before advancing, so the
    // "are you sure?" dialog is handled for free on nav-hotkey presses.
    let _lastBoundaryToast = 0;
    const navObj = {
      hasNext() {
        const ok = _navIndex < navContext.items.length - 1;
        if (!ok) {
          const now = Date.now();
          if (now - _lastBoundaryToast > 1200) {
            _lastBoundaryToast = now;
            toast('End of results page — go to the next page for more', 3000, 'warning');
          }
        }
        return ok;
      },
      hasPrev() {
        const ok = _navIndex > 0;
        if (!ok) {
          const now = Date.now();
          if (now - _lastBoundaryToast > 1200) {
            _lastBoundaryToast = now;
            toast('Beginning of results page — go to the previous page for more', 3000, 'warning');
          }
        }
        return ok;
      },
      next() { _navIndex = Math.min(_navIndex + 1, navContext.items.length - 1); },
      prev() { _navIndex = Math.max(_navIndex - 1, 0); },
      currentPath() { return navContext.items[_navIndex]?.img.filename ?? null; },
    };
    instance.runtime.setFileNavigation(navObj);

    // When files.js publishes LOAD_IMAGE_ONTO_CANVAS for our runtime, load the
    // new image into this same modal canvas.
    unsubNavLoad = events.subscribeToCanvas(
      instance.runtime.id, EventTypes.LOAD_IMAGE_ONTO_CANVAS,
      async () => {
        const item = navContext.items[_navIndex];
        if (!item) return;
        // For the start image, prefer the originally-opened bbox so the highlight
        // matches the card the user double-clicked, not the first occurrence in navItems.
        const targetBbox = (_navIndex === navContext.startIndex) ? bbox : (item.bbox ?? null);
        titleEl.textContent = `Edit: ${item.img.filename.split('/').slice(-2).join('/')}`;
        try {
          await instance.loadImage(item.img.filename);
        } catch (err) {
          Logger.error('Canvas modal: failed to load image during navigation', err);
          toast('Failed to load image', 3000, 'error');
          titleEl.textContent = `Edit: ${_currentImg.filename.split('/').slice(-2).join('/')}`;
          return; // Don't update state — canvas and _currentImg stay in sync
        }
        _currentImg = item.img;
        _currentBbox = targetBbox;
        if (_currentBbox) _highlightBbox(_currentBbox.uuid);
      },
    );
  }

  // ── Teardown ───────────────────────────────────────────────────────
  function closeModal() {
    unsubDirty();
    if (unsubNavLoad) unsubNavLoad();
    unsubInspector();
    toolbarHandle.destroy();
    labelHandle.destroy();
    instance.destroy();
    overlay.remove();
    _modalOpen = false;
  }

  // ── Shared save-then-close path ────────────────────────────────────
  async function _doSaveAndClose() {
    const currentImg  = _currentImg;
    const currentBbox = _currentBbox;
    const allUuids = [...new Set([
      ...instance.runtime.state.image.bboxes.keys(),
      bbox.uuid, // always include the originally-opened bbox
    ])];
    await invalidateBboxThumbnails(allUuids);
    const currentBboxesMap = new Map(instance.runtime.state.image.bboxes);
    const updatedUIBbox = currentBboxesMap.get(currentBbox?.uuid ?? bbox.uuid);
    closeModal();
    if (currentImg?.uuid !== img.uuid) {
      // Navigated to a different image before saving — let navContext refresh its cards.
      navContext?.onSaved?.(currentImg, currentBboxesMap);
    }
    navContext?.onClose?.(currentImg);
    await onSaved(updatedUIBbox, currentImg);
  }

  // ── Close guard: prompt when there are unsaved changes ────────────
  async function tryClose() {
    if (instance.runtime.isDirty) {
      const choice = await showUnsavedChangesModal();
      if (choice === 'cancel') return;
      if (choice === 'save') {
        saveBtn.disabled = true;
        cancelBtn.disabled = true;
        closeBtn.disabled = true;
        const saved = await saveAnnotations(instance.runtime);
        if (!saved) {
          saveBtn.disabled = false;
          cancelBtn.disabled = false;
          closeBtn.disabled = false;
          _updateSaveBtn();
          return; // save failed — keep modal open
        }
        await _doSaveAndClose();
        return;
      }
      // 'discard': fall through to close without saving
    }
    const currentImg = _currentImg;
    closeModal();
    navContext?.onClose?.(currentImg);
  }

  closeBtn.addEventListener('click', tryClose);
  cancelBtn.addEventListener('click', tryClose);
  overlay.addEventListener('click', e => {
    if (e.target === overlay && !closeBtn.disabled) tryClose();
  });

  // ── Save & Close button ────────────────────────────────────────────
  saveBtn.addEventListener('click', async () => {
    if (!instance.runtime.isDirty) return; // nothing to save — button is dimmed
    saveBtn.disabled = true;
    cancelBtn.disabled = true;
    closeBtn.disabled = true;
    saveBtn.textContent = 'Saving…';

    let saved = false;
    try {
      saved = await saveAnnotations(instance.runtime);
    } catch (err) {
      Logger.error('Canvas modal: save failed', err);
    }

    if (saved) {
      await _doSaveAndClose();
    } else {
      saveBtn.disabled = false;
      cancelBtn.disabled = false;
      closeBtn.disabled = false;
      _updateSaveBtn();
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
    { value: 'relabel', label: 'Change Label' },
    { value: 'delete', label: 'Delete Bboxes' },
    { value: 'addTags', label: 'Add Tags' },
    { value: 'removeTags', label: 'Remove Tags' },
  ];

  const radioGroup = document.createElement('div');
  radioGroup.style.cssText = 'display:flex;flex-direction:column;gap:0.4em;';
  let selectedAction = 'relabel';

  for (const action of actions) {
    const lbl = document.createElement('label');
    lbl.style.cssText = 'display:flex;align-items:center;gap:0.5em;cursor:pointer;';
    const radio = document.createElement('input');
    radio.type = 'radio';
    radio.name = 'bulk-action';
    radio.value = action.value;
    if (action.value === 'relabel') radio.checked = true;
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

async function showImageBulkActionModal(entries, labels, onConfirm) {
  const overlay = document.createElement('div');
  overlay.style.cssText = `position:fixed;inset:0;background:#0008;display:flex;
    align-items:center;justify-content:center;z-index:1000;`;

  const dialog = document.createElement('div');
  dialog.style.cssText = `background:#222;border:1px solid #555;border-radius:6px;
    padding:1.5em;min-width:300px;display:flex;flex-direction:column;gap:0.8em;`;

  const title = document.createElement('strong');
  title.textContent = `Apply to ${entries.length} selected image${entries.length === 1 ? '' : 's'}`;

  const actions = [
    { value: 'setAbsent', label: 'Mark label Absent (negative label)' },
    { value: 'setUncertain', label: 'Clear Absent → Uncertain' },
  ];

  const radioGroup = document.createElement('div');
  radioGroup.style.cssText = 'display:flex;flex-direction:column;gap:0.4em;';
  let selectedAction = 'setAbsent';

  for (const action of actions) {
    const lbl = document.createElement('label');
    lbl.style.cssText = 'display:flex;align-items:center;gap:0.5em;cursor:pointer;';
    const radio = document.createElement('input');
    radio.type = 'radio';
    radio.name = 'image-bulk-action';
    radio.value = action.value;
    if (action.value === 'setAbsent') radio.checked = true;
    radio.addEventListener('change', () => { selectedAction = action.value; });
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
    await onConfirm(selectedAction, labelPicker.value);
  });

  btnRow.appendChild(cancelBtn);
  btnRow.appendChild(okBtn);
  dialog.append(title, radioGroup, labelPickerRow, btnRow);
  overlay.appendChild(dialog);
  document.body.appendChild(overlay);

  overlay.addEventListener('click', e => { if (e.target === overlay) overlay.remove(); });
}

async function executeBulkImageAction(action, entries, labelUuid) {
  const status = action === 'setAbsent' ? 'absent' : 'uncertain';
  let failCount = 0;
  const succeededEntries = [];

  for (const entry of entries) {
    const { img } = entry;
    try {
      await setImageLabel(img.filename, labelUuid, status);
      succeededEntries.push(entry);
    } catch (err) {
      Logger.error(`Image label action failed for ${img.uuid}: ${err.message}`);
      failCount++;
    }
  }

  return { failCount, succeededEntries };
}
