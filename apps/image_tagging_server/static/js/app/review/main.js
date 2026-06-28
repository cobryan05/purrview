import { fetchLabels } from '/app-static/js/shared/api/labels.js';
import { createDynamicLabelPicker } from '/app-static/js/ui/utils/dynamicLabelPicker.js';
import { fetchTags } from '/app-static/js/shared/api/tags.js';
import { filterVisibleTags } from '/app-static/js/shared/activeModel.js';
import { createFilterPanel } from '/app-static/js/app/review/filters.js';
import { expandWithDescendants, createLabelList } from '/app-static/js/ui/labelList.js';
import {
  fetchReviewQueue,
  bulkApplyAll,
  bulkApplyBboxesByUuids,
  bulkApplyImagesByUuids,
  updateImageBboxes,
  updateBbox,
  deleteBbox,
  getBboxThumbnailUrl,
  getImageThumbnailUrl,
  invalidateBboxThumbnails,
  UNVERIFIED_TAG_UUID,
  VERIFIED_TAG_UUID,
  FALSE_POS_TAG_UUID,
  REJECTION_TAG_UUID,
  VERIFICATION_TAG_UUIDS,
  swapVerificationTag,
} from '/app-static/js/shared/api/review.js';
import { fetchImageLabels, setImageLabel } from '/app-static/js/shared/api/imageLabels.js';
import { setImageTag, setImageTagStatus } from '/app-static/js/shared/api/imageTags.js';
import { attachContext } from '/app-static/js/ui/contextMenu/index.js';
import { getPrimaryRuntimeId } from '/app-static/js/app/image-tagging/runtime.js';
import { Logger, toast } from '/app-static/js/ui/utils/logging.js';
import { showUnsavedChangesModal, showExternalChangesModal, showConfirmModal } from '/app-static/js/ui/utils/confirmModal.js';
import { deleteImage } from '/app-static/js/shared/api/images.js';
import { rerenderBboxViews } from '/app-static/js/app/image-tagging/canvas/manager.js';
import { nextModalZIndex, applyModalCascade, registerEscHandler, unregisterEscHandler } from '/app-static/js/ui/utils/modalZIndex.js';
import { CheckboxField } from '/app-static/js/ui/utils/fields/CheckboxField.js';
import { createCanvasInstance } from '/app-static/js/app/image-tagging/canvas/instance.js';
import { saveAnnotations } from '/app-static/js/app/image-tagging/canvas/image.js';
import { initSplitResize } from '/app-static/js/ui/utils/splitResize.js';
import { events, EventTypes } from '/app-static/js/shared/events/index.js';
import { init as initToolbar } from '/app-static/js/app/image-tagging/toolbar.js';
import { createTagPanel } from '/app-static/js/ui/inspector.js';
import { setupSidebarTabs } from '/app-static/js/app/utils/tabs.js';
import { parseToolUuid } from '/app-static/js/app/image-tagging/canvas/tools.js';
import { createHoverPreview } from '/app-static/js/ui/thumbnailHoverPreview.js';
import { createEntryView } from '/app-static/js/ui/entryView.js';
import { reviewStateToDatasetForm, datasetFormToReviewState } from '/app-static/js/shared/filterUtils.js';
import { openDissimilarGlobalModal } from '/app-static/js/ui/dissimilarSearch.js';


const DEFAULT_PAGE_SIZE = 250;
const MAX_ENTRIES = 1000; // hard cap on total cards per page

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
    this._observer = new IntersectionObserver(entries => {
      let any = false;
      for (const entry of entries) {
        if (entry.isIntersecting) {
          this._observer.unobserve(entry.target);
          if (this._prioritize(entry.target)) any = true;
        }
      }
      if (any) this._pump();
    }, { rootMargin: '200px', threshold: 0 });
  }

  /** Abort all in-flight requests and drop all queued work. Safe to call multiple times. */
  reset() {
    this._queue = [];
    this._observer.disconnect();
    for (const ctrl of this._controllers) ctrl.abort();
    this._controllers.clear();
    this._active = 0;
  }

  /** Enqueue a thumbnail load. Sets imgEl.src (as a blob URL) when a slot is available. */
  load(imgEl, url) {
    this._queue.push({ imgEl, url });
    this._observer.observe(imgEl);
    this._pump();
  }

  /** Move imgEl to the front of the queue if it is still pending. Returns true if moved. */
  _prioritize(imgEl) {
    const idx = this._queue.findIndex(item => item.imgEl === imgEl);
    if (idx <= 0) return false; // not found or already at front
    const [item] = this._queue.splice(idx, 1);
    this._queue.unshift(item);
    return true;
  }

  _pump() {
    while (this._active < this._concurrency && this._queue.length > 0) {
      const { imgEl, url } = this._queue.shift();
      this._observer.unobserve(imgEl);
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
/** Build and return a verify-status chip span for a bbox's tag UUIDs. */
function buildVerifyChip(tagUuids) {
  const isFalsePos = tagUuids.includes(FALSE_POS_TAG_UUID);
  const isVerified = tagUuids.includes(VERIFIED_TAG_UUID);
  const chip = document.createElement('span');
  chip.className = 'verify-chip';
  chip.textContent = isFalsePos ? '✗ False Pos.' : isVerified ? '✓ Verified' : '◌ Unverified';
  chip.classList.toggle('verify-chip--verified', isVerified);
  chip.classList.toggle('verify-chip--false-pos', isFalsePos);
  chip.title = isFalsePos ? 'False Positive' : isVerified ? 'Verified' : 'Not yet verified';
  return chip;
}

export async function createReviewPanel(container, { fetchImages = null, datasetActions = true, referenceBboxUuids = new Set(), referenceImageUuids = new Set(), initialViewFilter = 'images', toolbarControls = null, bboxFilterFn = null } = {}) {
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

  // Sort controls — hidden when a custom fetchImages is provided (caller controls ordering)
  const sortLabel = document.createElement('span');
  sortLabel.className = 'review-toolbar__sort-label';
  sortLabel.textContent = 'Sort:';
  const sortBySelect = makeSelect('filter-sort-by', [['created_at', 'Date created'], ['updated_at', 'Date modified']], 'created_at');
  const sortOrderSelect = makeSelect('filter-sort-dir', [['desc', 'Newest first'], ['asc', 'Oldest first']], 'desc');
  if (fetchImages) { sortLabel.hidden = true; sortBySelect.hidden = true; sortOrderSelect.hidden = true; }

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
    btn.className = 'view-btn' + (key === initialViewFilter ? ' active' : '');
    btn.dataset.viewKey = key;
    btn.textContent = label;
    btn.addEventListener('click', () => {
      if (_viewFilter === key) return;
      _viewFilter = key;
      viewToggle.querySelectorAll('.view-btn').forEach(b =>
        b.classList.toggle('active', b.dataset.viewKey === key));
      _selected.clear();
      _selectionKind = null;
      // Full re-fetch so requireBboxes takes effect (bboxes view filters
      // out empty-bbox images server-side for correct pagination).
      _pageCursors.length = 1;
      render();
    });
    viewToggle.appendChild(btn);
  }

  // ── Per-page selector ────────────────────────────────────────────
  const pageSizeSelect = makeSelect('filter-page-size',
    [['10', '10 imgs/pg'], ['25', '25 imgs/pg'], ['50', '50 imgs/pg'], ['100', '100 imgs/pg'], ['250', '250 imgs/pg'], ['500', '500 imgs/pg'], ['1000', '1000 imgs/pg']],
    String(DEFAULT_PAGE_SIZE));

  // Select-all checkbox
  const selectAllLabel = document.createElement('label');
  selectAllLabel.className = 'review-toolbar__select-all';
  const selectAllCb = document.createElement('input');
  selectAllCb.type = 'checkbox';
  selectAllLabel.appendChild(selectAllCb);
  selectAllLabel.appendChild(document.createTextNode('Select all'));

  // Bulk Apply button (selected items on current page)
  const bulkBtn = document.createElement('button');
  bulkBtn.className = 'review-toolbar__btn review-toolbar__btn--primary';
  bulkBtn.textContent = 'Bulk Apply (0)';
  bulkBtn.disabled = true;

  // Bulk Apply All button — all matching results across all pages (filter mode),
  // or the full candidate list (custom fetchImages mode, e.g. similarity search).
  const bulkAllBtn = document.createElement('button');
  bulkAllBtn.className = 'review-toolbar__btn review-toolbar__btn--primary';
  bulkAllBtn.textContent = 'Apply to All';
  bulkAllBtn.disabled = true;

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
    makeToolbarGroup(filterToggleBtn, sortLabel, sortBySelect, sortOrderSelect),
    makeToolbarSep(),
    makeToolbarGroup(viewToggle, entryView.element, pageSizeSelect),
    ...(toolbarControls ? [makeToolbarSep(), makeToolbarGroup(toolbarControls)] : []),
    makeToolbarGroup(refreshBtn),
  );

  // ── Action bar: select-all, bulk-apply, prev/next, status ─────────
  const actionBar = document.createElement('div');
  actionBar.className = 'review-action-bar';
  actionBar.append(
    makeToolbarGroup(selectAllLabel, bulkBtn, bulkAllBtn),
    makeToolbarGroup(prevBtn, pageInfo, nextBtn),
    makeToolbarGroup(statusMsg),
  );

  // ── Sticky top wrapper ────────────────────────────────────────────
  // Pins toolbar + action bar (and the expandable filter panel) to the top
  // of the scroll container. The filter panel lives here so expanding it
  // never requires scrolling back up to find the filter controls.
  const stickyTop = document.createElement('div');
  stickyTop.className = 'review-sticky-top';
  stickyTop.append(toolbar, actionBar);
  container.appendChild(stickyTop);
  container.appendChild(entryView.layout);

  // ── Filter panel (full mode only) ─────────────────────────────────
  // Lives inside stickyTop so it expands in-place and never scrolls away.
  // Starts hidden; the ☰ Filters button toggles it.
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
    filterPanel.element.hidden = true;
    stickyTop.appendChild(filterPanel.element);
    updateFilterBtn(); // reflect initial state (Unverified checked → shows "(1)")

    filterToggleBtn.addEventListener('click', () => {
      filterPanel.element.hidden = !filterPanel.element.hidden;
    });

    // Inject Tags + Labels dropdowns into the toolbar as the first group.
    const filterControlsGroup = makeToolbarGroup(...filterPanel.toolbarControls);
    toolbar.insertBefore(filterControlsGroup, toolbar.firstChild);
    toolbar.insertBefore(makeToolbarSep(), toolbar.children[1]);

    // ── Filter presets + dataset export (hidden in modal/preview contexts) ──
    if (datasetActions) {
    const _PRESETS_KEY = 'review-filter-presets';
    const _loadPresets = () => {
      try { return JSON.parse(localStorage.getItem(_PRESETS_KEY) ?? '[]'); } catch { return []; }
    };
    const _savePresets = (list) => localStorage.setItem(_PRESETS_KEY, JSON.stringify(list));

    const presetsSelect = document.createElement('select');
    presetsSelect.className = 'filter-presets-select';
    presetsSelect.title = 'Apply a saved filter preset';

    function _rebuildPresetsDropdown() {
      presetsSelect.innerHTML = '';
      const ph = document.createElement('option');
      ph.value = '';
      ph.textContent = '— Presets —';
      presetsSelect.appendChild(ph);
      for (const p of _loadPresets()) {
        const opt = document.createElement('option');
        opt.value = p.name;
        opt.textContent = p.name;
        presetsSelect.appendChild(opt);
      }
      presetsSelect.value = '';
    }
    _rebuildPresetsDropdown();

    presetsSelect.addEventListener('change', () => {
      const name = presetsSelect.value;
      if (!name) return;
      const preset = _loadPresets().find(p => p.name === name);
      if (preset) filterPanel.setState(preset.state);
      presetsSelect.value = '';
    });

    const savePresetBtn = document.createElement('button');
    savePresetBtn.className = 'review-toolbar__btn';
    savePresetBtn.textContent = '⊙ Save Filter Preset';
    savePresetBtn.title = 'Save current filters as a named preset';
    savePresetBtn.addEventListener('click', () => {
      const name = prompt('Preset name:');
      if (!name?.trim()) return;
      const trimmed = name.trim();
      const presets = _loadPresets();
      const idx = presets.findIndex(p => p.name === trimmed);
      const state = filterPanel.getState();
      if (idx !== -1) presets[idx].state = state;
      else presets.push({ name: trimmed, state });
      _savePresets(presets);
      _rebuildPresetsDropdown();
      toast(`Preset "${trimmed}" saved`, 3000);
    });

    // ── Export to Datasets ────────────────────────────────────────────
    const exportToDatasetBtn = document.createElement('button');
    exportToDatasetBtn.className = 'review-toolbar__btn';
    exportToDatasetBtn.textContent = '⊞ Open in Datasets Tab';
    exportToDatasetBtn.title = 'Export current filters to a new dataset in the Datasets editor';
    exportToDatasetBtn.addEventListener('click', () => {
      const { filterJson, labelUuids } = reviewStateToDatasetForm(filterPanel.getState());
      events.publish(EventTypes.EXPORT_FILTERS_TO_DATASET, { filterJson, labelUuids });
      toast('Opening Datasets editor…', 2000);
    });

    toolbar.append(makeToolbarSep(), makeToolbarGroup(presetsSelect, savePresetBtn, exportToDatasetBtn));

    // ── "Label most unique first" — diversity-ranked unverified queue ────────
    const uniqueFirstBtn = document.createElement('button');
    uniqueFirstBtn.className = 'review-toolbar__btn';
    uniqueFirstBtn.textContent = '★ Most Unique First';
    uniqueFirstBtn.title = 'Open the most visually distinct unverified images in a review panel (farthest-point sampling)';
    uniqueFirstBtn.addEventListener('click', () => openDissimilarGlobalModal());
    toolbar.append(makeToolbarSep(), makeToolbarGroup(uniqueFirstBtn));
    } // end if (datasetActions)
  }

  // ── Hover preview ─────────────────────────────────────────────────
  const _imageCache = new Map();
  const hoverPreview = createHoverPreview({ imageCache: _imageCache });
  const _thumbLoader = new ThumbnailLoader();

  // ── State ─────────────────────────────────────────────────────────
  // _allEntries holds ALL entries for the current page (both image and bbox).
  // _viewFilter controls which subset is rendered to DOM.
  let _allEntries = [];       // Array<{ kind: 'bbox', bbox, img } | { kind: 'image', img }>
  let _viewFilter = initialViewFilter;  // 'all' | 'images' | 'bboxes'
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
  let _renderGeneration = 0;
  let _hasMore = false;
  let _totalCount = 0;

  // _visibleEntries is the filtered subset rendered to DOM (updated by renderFromEntries)
  let _visibleEntries = [];
  let _cardUnsubs = [];  // per-card EventBus unsubs; drained before each page render
  const _drainCardUnsubs = () => _cardUnsubs.splice(0).forEach(fn => fn());

  const entryId = e => e.kind === 'image' ? e.img.uuid : e.bbox.uuid;

  // ── All UUIDs treated as verification indicators for the Clear action ─────
  const ALL_VERIFICATION_UUIDS = new Set([
    REJECTION_TAG_UUID,
    ...VERIFICATION_TAG_UUIDS,
  ]);

  // ── Review bbox adapter factory ────────────────────────────────────────────
  function makeReviewBboxAdapter(entry, card) {
    const { bbox, img } = entry;

    function currentTagUuids() {
      return (bbox.tags ?? []).map(t => t.uuid);
    }

    async function _applyUpdate(labelUuid, tagUuids) {
      await updateBbox(bbox.uuid, { labelUuid, tagUuids });
      await invalidateBboxThumbnails([bbox.uuid]);
      // Update in-memory bbox tags, preserving icon values for known tags
      const existing = new Map((bbox.tags ?? []).map(t => [t.uuid, t]));
      bbox.tags = tagUuids.map(u => existing.get(u) ?? { uuid: u, icon: null });
    }

    function _publishBboxChanged({ deleted = false } = {}) {
      events.publish(EventTypes.IMAGE_BBOXES_CHANGED, {
        imageUuid: img.uuid,
        imageFilename: img.filename,
        updatedBboxes: deleted ? new Map() : new Map([[bbox.uuid, bbox]]),
        deletedBboxUuids: deleted ? [bbox.uuid] : [],
      });
    }

    return {
      getBboxUuid() { return bbox.uuid ?? null; },
      getImageUuid() { return img.uuid ?? null; },
      getLabelUuid() { return bbox.labelUuid ?? bbox.label?.uuid ?? null; },
      getTagUuids() { return currentTagUuids(); },

      async applyLabel(uuid) {
        const existingTagUuids = currentTagUuids();
        // Assigning a label strips verification tags and marks VERIFIED
        const newTagUuids = uuid
          ? [...existingTagUuids.filter(u => !VERIFICATION_TAG_UUIDS.has(u)), VERIFIED_TAG_UUID]
          : existingTagUuids;
        await _applyUpdate(uuid, newTagUuids);
        // Update in-memory label
        if (uuid) {
          bbox.labelUuid = uuid;
          const label = labels.get(uuid);
          if (label) bbox.label = label;
        } else {
          bbox.labelUuid = null;
          bbox.label = null;
        }
        // Refresh card
        const thumb = card.querySelector('.thumb');
        if (thumb) thumb.src = getBboxThumbnailUrl(img.filename, bbox.uuid, 300) + `&t=${Date.now()}`;
        if (uuid) {
          const label = labels.get(uuid);
          if (label) {
            const chip = card.querySelector('.label-chip');
            if (chip) {
              chip.textContent = label.name;
              chip.style.cssText = `background:${label.color}33;border:1px solid ${label.color};`;
            }
            const bboxInImg = img.bboxes?.find(b => b.uuid === bbox.uuid);
            if (bboxInImg) { bboxInImg.labelUuid = uuid; bboxInImg.label = label; }
          }
          markCardStale(card, 'relabeled');
        }
        _imageCardRefreshers.get(img.uuid)?.();
        _publishBboxChanged();
      },

      async toggleTag(uuid) {
        const tags = currentTagUuids();
        const has = tags.includes(uuid);
        let newTagUuids;
        if (has) {
          newTagUuids = tags.filter(u => u !== uuid);
        } else if (VERIFICATION_TAG_UUIDS.has(uuid)) {
          // Swap within the verification exclusive group
          newTagUuids = swapVerificationTag(tags, uuid);
        } else {
          newTagUuids = [...tags, uuid];
        }
        const labelUuid = bbox.labelUuid ?? bbox.label?.uuid;
        await _applyUpdate(labelUuid, newTagUuids);
        markCardStale(card, 'tagged');
        _imageCardRefreshers.get(img.uuid)?.();
        _publishBboxChanged();
      },

      async setVerification(uuid) {
        const tags = currentTagUuids();
        let newTagUuids;
        if (uuid === null) {
          // Clear all verification indicators
          newTagUuids = tags.filter(u => !ALL_VERIFICATION_UUIDS.has(u));
        } else if (uuid === REJECTION_TAG_UUID) {
          // REJECTION has no exclusive group — just add
          newTagUuids = tags.includes(uuid) ? tags : [...tags, uuid];
        } else {
          // Swap within verification exclusive group (UNVERIFIED ↔ FALSE_POS)
          newTagUuids = swapVerificationTag(tags, uuid);
        }
        const labelUuid = bbox.labelUuid ?? bbox.label?.uuid;
        await _applyUpdate(labelUuid, newTagUuids);
        // Refresh thumb
        const thumb = card.querySelector('.thumb');
        if (thumb) thumb.src = getBboxThumbnailUrl(img.filename, bbox.uuid, 300) + `&t=${Date.now()}`;
        markCardStale(card, 'verified');
        _imageCardRefreshers.get(img.uuid)?.();
        _publishBboxChanged();
      },

      async remove() {
        await deleteBbox(bbox.uuid);
        await invalidateBboxThumbnails([bbox.uuid]);
        markCardStale(card, 'deleted');
        const idx = img.bboxes?.findIndex(b => b.uuid === bbox.uuid) ?? -1;
        if (idx !== -1) img.bboxes.splice(idx, 1);
        _imageCardRefreshers.get(img.uuid)?.();
        _publishBboxChanged({ deleted: true });
      },
    };
  }

  // ── Review image adapter factory ───────────────────────────────────────────
  function makeReviewImageAdapter(entry) {
    const { img } = entry;
    return {
      getPath() { return img.filename; },
      getUuid() { return img.uuid ?? null; },

      openOnCanvas() {
        events.publish(EventTypes.LOAD_IMAGE_ONTO_CANVAS, {
          path: img.filename,
          showCanvas: true,
          runtimeId: getPrimaryRuntimeId(),
        });
      },

      async setLabelPresence(uuid, status) {
        await setImageLabel(img.filename, uuid, status);
        events.publish(EventTypes.IMAGE_LABELS_CHANGED, { imageUuid: img.uuid, imagePath: img.filename });
      },

      getTagUuids() { return (img.tags ?? []).map(t => t.uuid); },

      async toggleTag(uuid) {
        const tags = img.tags ?? [];
        const has = tags.some(t => t.uuid === uuid);
        await setImageTag(img.filename, uuid, has ? 'remove' : 'add');
        img.tags = has ? tags.filter(t => t.uuid !== uuid) : [...tags, { uuid, icon: null }];
      },

      async setTagStatus(uuid, status) {
        await setImageTagStatus(img.filename, uuid, status);
        // img.tags only ever lists PRESENT tags — absent/uncertain both mean "not in the list".
        const tags = img.tags ?? [];
        img.tags = status === 'present'
          ? (tags.some(t => t.uuid === uuid) ? tags : [...tags, { uuid, icon: null }])
          : tags.filter(t => t.uuid !== uuid);
      },
    };
  }

  // ── Context-object builder for cards ──────────────────────────────────────
  function _buildContextObject(entry, card, kind) {
    if (kind === 'bbox') {
      return { adapters: { bbox: makeReviewBboxAdapter(entry, card) } };
    }
    return { adapters: { image: makeReviewImageAdapter(entry) } };
  }

  function _resolveContextObjects(entry, card, kind) {
    const id = entryId(entry);
    if (_selected.has(id) && _selected.size > 1) {
      // Multi-select: return objects for all selected entries of the same kind
      return _visibleEntries
        .filter(e => e.kind === kind && _selected.has(entryId(e)))
        .map(e => {
          const eid = entryId(e);
          const selector = kind === 'image'
            ? `[data-image-uuid="${eid}"][data-kind="image"]`
            : `[data-bbox-uuid="${eid}"]`;
          const c = grid.querySelector(selector) ?? card;
          return _buildContextObject(e, c, kind);
        });
    }
    return [_buildContextObject(entry, card, kind)];
  }

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
    if (!fetchImages) {
      const countStr = _totalCount > 0 ? _totalCount.toLocaleString() : '?';
      bulkAllBtn.textContent = `Apply to All (${countStr})`;
      bulkAllBtn.disabled = _totalCount === 0;
    } else {
      // No server-side total in custom-fetchImages mode — the full candidate
      // count is only known once showApplyToAllCandidatesModal fetches it.
      bulkAllBtn.textContent = 'Apply to All';
      bulkAllBtn.disabled = _visibleEntries.length === 0;
    }
  }

  function updateStatusMsg() {
    statusMsg.classList.remove('review-toolbar__status--loading');
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
    // Refresh the verify chip to reflect the saved tag state
    const verifyChipEl = card.querySelector('.verify-chip');
    if (verifyChipEl) {
      const newChip = buildVerifyChip((uiBbox.tags ?? []).map(t => t.uuid));
      verifyChipEl.replaceWith(newChip);
    }
    const nowVerified = (uiBbox.tags ?? []).some(t => t.uuid === VERIFIED_TAG_UUID);
    if (nowVerified) markCardStale(card, 'verified');
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
      _drainCardUnsubs();
      entryView.setEntries([]);
      updateStatusMsg();
      return;
    }

    entryView.setEmptyText('');
    _drainCardUnsubs();
    entryView.setEntries(_visibleEntries);

    updateStatusMsg();
  }

  // ── Render (fetch + display) ──────────────────────────────────────
  async function render() {
    const _gen = ++_renderGeneration;
    _rendering = true;

    const MAX_AUTO_SKIP = 10;
    let autoSkipped = 0;
    let _slowTimer = null;

    fetchLoop: while (true) {
      // Bail out if a newer render has started (user clicked Refresh mid-load).
      if (_renderGeneration !== _gen) { clearTimeout(_slowTimer); return; }
      refreshBtn.disabled = false;
      prevBtn.disabled = true;
      nextBtn.disabled = true;
      statusMsg.textContent = 'Loading…';
      statusMsg.classList.add('review-toolbar__status--loading');
      clearTimeout(_slowTimer);
      _slowTimer = setTimeout(() => {
        if (_renderGeneration === _gen) {
          statusMsg.textContent = 'Still loading… click Refresh to cancel';
          statusMsg.classList.add('review-toolbar__status--loading');
        }
      }, 5000);
      _thumbLoader.reset();
      // Use setEntries([]) so entryView._entries and the DOM stay in sync — a
      // direct grid.innerHTML = '' would leave _entries stale, causing a
      // view-toggle click during the async fetch to re-render the old page.
      _drainCardUnsubs();
      entryView.setEntries([]);
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
            limit: _pageSize + 1,
          });
          // Sentinel pattern: request one extra row to distinguish "exactly limit" from
          // "more exist" without relying on an unreliable length === limit heuristic.
          hasMore = images.length > _pageSize;
          images = images.slice(0, _pageSize);
        } else {
          ({ images, hasMore, totalCount } = await fetchReviewQueue({
            tagUuids: fs.showAllBboxes ? [] : fs.tagUuids,
            tagExcludeUuids: fs.tagExcludeUuids ?? [],
            showAllBboxes: fs.showAllBboxes,
            tagMatchMode: fs.tagMatchMode,
            labelUuids: fs.labelUuids,
            labelExcludeUuids: fs.labelExcludeUuids ?? [],
            labelMatchMode: fs.labelMatchMode,
            labelIncludeChildren: fs.labelIncludeChildren,
            imageLabelConditions: fs.imageLabelConditions ?? [],
            requireBboxes: _viewFilter === 'bboxes',
            sortBy,
            sortOrder,
            filenameSearch: fs.filenameSearch || undefined,
            dateFrom: fs.dateFrom || undefined,
            dateTo: fs.dateTo || undefined,
            afterUuid: cursor?.uuid ?? null,
            afterSortValue: cursor?.sortValue ?? null,
            verificationCheck: fs.verificationCheck,
            limit: _pageSize,
          }));
        }
      } catch (err) {
        clearTimeout(_slowTimer);
        if (_renderGeneration !== _gen) return;
        statusMsg.classList.remove('review-toolbar__status--loading');
        statusMsg.textContent = `Error: ${err.message}`;
        refreshBtn.disabled = false;
        prevBtn.disabled = _pageCursors.length <= 1;
        nextBtn.disabled = true;
        _rendering = false;
        return;
      }

      if (_renderGeneration !== _gen) { clearTimeout(_slowTimer); return; }

      _hasMore = hasMore;
      // totalCount is -1 on non-first pages (server skips the expensive count
      // query when a keyset cursor is present). Preserve the cached value.
      if (totalCount >= 0) _totalCount = totalCount;

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
      const showAll = (fetchImages && !bboxFilterFn) || fs?.showAllBboxes;
      const tagSet = (fs && !showAll) ? new Set(fs.tagUuids) : null;
      const activeLabelUuids = (fs?.labelUuids?.length > 0)
        ? (fs.labelIncludeChildren
            ? expandWithDescendants(fs.labelUuids, labels)
            : new Set(fs.labelUuids))
        : null;

      for (const img of images) {
        const matching = (img.bboxes || []).filter(b => {
          if (bboxFilterFn) return bboxFilterFn(b);
          const tagMatch = showAll || (b.tags || []).some(t => tagSet.has(t.uuid));
          const labelMatch = !activeLabelUuids || activeLabelUuids.has(b.labelUuid);
          return tagMatch && labelMatch;
        });
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

      // Auto-skip empty pages (when client-side tag filter leaves nothing).
      // Skip only in full-filter mode — in custom fetchImages mode (preview) we
      // want to show image entries even when they have no bbox annotations.
      const bboxEntries = _allEntries.filter(e => e.kind === 'bbox');
      if (!fetchImages && bboxEntries.length === 0 && _hasMore && autoSkipped < MAX_AUTO_SKIP) {
        autoSkipped++;
        _pageCursors[_pageCursors.length - 1] = _lastCursor;
        continue fetchLoop;
      }

      break fetchLoop;
    }

    const totalPages = _totalCount > 0 ? Math.ceil(_totalCount / _pageSize) : null;
    const pageLabel = totalPages ? `Page ${_pageCursors.length} of ${totalPages}` : `Page ${_pageCursors.length}`;
    const cardCount = _allEntries.length;
    pageInfo.textContent = `${pageLabel} · ${cardCount} card${cardCount === 1 ? '' : 's'}`;
    prevBtn.disabled = _pageCursors.length <= 1;
    nextBtn.disabled = !_hasMore;

    clearTimeout(_slowTimer);
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
    const bboxTagUuids = (bbox.tags ?? []).map(t => t.uuid);
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

    if (referenceBboxUuids.has(bbox.uuid)) {
      card.classList.add('review-card--reference');
      const refBadge = document.createElement('div');
      refBadge.className = 'card-state-badge state-reference';
      refBadge.textContent = '◉ Reference';
      info.appendChild(refBadge);
    }

    const chip = document.createElement('span');
    chip.className = 'label-chip';
    chip.textContent = labelName;
    chip.style.cssText = `background:${labelColor}33;border:1px solid ${labelColor};`;

    const verifyChip = buildVerifyChip(bboxTagUuids);

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

    attachContext(card, ev => _resolveContextObjects({ kind: 'bbox', bbox, img }, card, 'bbox'));

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

    if (referenceImageUuids.has(img.uuid)) {
      card.classList.add('review-card--reference');
      const refBadge = document.createElement('div');
      refBadge.className = 'card-state-badge state-reference';
      refBadge.textContent = '◉ Reference';
      info.appendChild(refBadge);
    }

    // Deduplicated label chips (bbox-level)
    const bboxes = img.bboxes ?? [];
    const chipsRow = document.createElement('div');
    chipsRow.style.cssText = 'display:flex;flex-wrap:wrap;gap:2px;margin-bottom:0.2em;';
    appendLabelChips(chipsRow, bboxes, labels);

    // Whole-image label chips — fetched async, distinct from bbox chips
    const imageLabelChipsRow = document.createElement('div');
    imageLabelChipsRow.style.cssText = 'display:flex;flex-wrap:wrap;gap:2px;margin-bottom:0.2em;';

    async function _renderImageLabelChips() {
      imageLabelChipsRow.innerHTML = '';
      let labelMap;
      try { labelMap = await fetchImageLabels(img.filename); } catch { return; }
      for (const [luuid, status] of labelMap) {
        if (status !== 'present') continue;
        const lbl = labels.get(luuid);
        if (!lbl) continue;
        const chip = document.createElement('span');
        chip.className = 'image-label-chip';
        chip.textContent = lbl.name;
        chip.style.cssText = `background:${lbl.color}22;border:1px dashed ${lbl.color};color:${lbl.color};`;
        imageLabelChipsRow.appendChild(chip);
      }
    }

    _renderImageLabelChips();

    // Keep chips up to date when labels change from context menu or canvas save.
    // Subscriptions are drained by _drainCardUnsubs() before each page render and in destroy().
    async function _onRelevantChange({ imageUuid: u } = {}) {
      if (u !== img.uuid) return;
      await _renderImageLabelChips();
    }
    _cardUnsubs.push(
      events.subscribe(EventTypes.IMAGE_LABELS_CHANGED, _onRelevantChange),
      events.subscribe(EventTypes.IMAGE_ANNOTATIONS_SAVED, _onRelevantChange),
    );

    // Verification status line (verified / unverified counts).
    const statusRow = document.createElement('div');
    statusRow.style.cssText = 'color:#aaa;font-size:0.75em;margin-bottom:0.15em;';
    if (bboxes.length > 0) {
      let verified = 0;
      let unverified = 0;
      for (const bbox of bboxes) {
        const bboxTags = bbox.tags ?? [];
        if (bboxTags.some(t => t.uuid === VERIFIED_TAG_UUID)) verified++;
        else if (bboxTags.some(t => t.uuid === UNVERIFIED_TAG_UUID)) unverified++;
      }
      const parts = [];
      if (verified > 0) parts.push(`✓ ${verified}`);
      if (unverified > 0) parts.push(`? ${unverified}`);
      statusRow.textContent = parts.join('  ');
    }

    const meta = document.createElement('div');
    meta.style.cssText = 'color:#aaa;font-size:0.75em;';
    meta.textContent = img.filename.split('/').slice(-2).join('/');

    info.appendChild(chipsRow);
    info.appendChild(imageLabelChipsRow);
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

    attachContext(card, ev => _resolveContextObjects({ kind: 'image', img }, card, 'image'));

    return card;
  }

  // ── Bbox list-row builder (compact card for list view) ────────────
  function buildBboxRow(bbox, img, labels, index) {
    const label = bbox.label ?? (bbox.labelUuid ? labels.get(bbox.labelUuid) : null);
    const labelName = label?.name ?? '(unknown)';
    const labelColor = label?.color ?? '#888';
    const rowTagUuids = (bbox.tags ?? []).map(t => t.uuid);
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

    if (referenceBboxUuids.has(bbox.uuid)) {
      row.classList.add('review-card--reference');
      const refBadge = document.createElement('span');
      refBadge.className = 'card-state-badge state-reference';
      refBadge.textContent = '◉ Reference';
      info.appendChild(refBadge);
    }

    const chip = document.createElement('span');
    chip.className = 'label-chip';
    chip.textContent = labelName;
    chip.style.cssText = `background:${labelColor}33;border:1px solid ${labelColor};`;

    const verifyChip = buildVerifyChip(rowTagUuids);

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

    attachContext(row, ev => _resolveContextObjects({ kind: 'bbox', bbox, img }, row, 'bbox'));

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

    if (referenceImageUuids.has(img.uuid)) {
      row.classList.add('review-card--reference');
      const refBadge = document.createElement('span');
      refBadge.className = 'card-state-badge state-reference';
      refBadge.textContent = '◉ Reference';
      info.appendChild(refBadge);
    }

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
      await showImageBulkActionModal(selectedEntries, labels, async (action, labelUuid, tagUuids) => {
        if (action === 'delete') {
          const n = selectedEntries.length;
          const noun = n === 1 ? 'image' : 'images';
          const confirmed = await showConfirmModal({
            title: `Delete ${n} ${noun}`,
            message: `Permanently delete ${n} selected ${noun}? This removes the files and all their annotations from disk. This cannot be undone.`,
            confirmText: 'Delete',
            danger: true,
          });
          if (!confirmed) return;

          let failCount = 0;
          for (const { img } of selectedEntries) {
            try {
              await deleteImage(img.uuid);
              const card = grid.querySelector(`[data-image-uuid="${img.uuid}"][data-kind="image"]`);
              if (card) markCardStale(card, 'deleted');
              events.publish(EventTypes.IMAGE_DELETED, { imageUuid: img.uuid, imagePath: img.filename });
            } catch (err) {
              Logger.error(`Delete image failed for ${img.uuid}: ${err.message}`);
              failCount++;
            }
          }
          clearSelection();
          updateStatusMsg();
          if (failCount > 0) {
            statusMsg.textContent += ` (${failCount} deletion${failCount === 1 ? '' : 's'} failed — check console)`;
          }
          return;
        }

        const { failCount, succeededEntries } = await executeBulkImageAction(action, selectedEntries, labelUuid, tagUuids);
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
      await showBulkActionModal(selectedEntries, labels, async (action, newLabelUuid, selectedTagUuids, tagsMap) => {
        const { failCount, succeededEntries } = await executeBulkAction(action, selectedEntries, newLabelUuid, selectedTagUuids, tagsMap);

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
              // Assigning a real label strips verification tags and adds VERIFIED — mirror in memory
              // and refresh the verify chip so the card reflects the new state immediately.
              bbox.tags = [
                ...(bbox.tags ?? []).filter(t => !VERIFICATION_TAG_UUIDS.has(t.uuid)),
                { uuid: VERIFIED_TAG_UUID, icon: null },
              ];
              const verifyChipEl = card.querySelector('.verify-chip');
              if (verifyChipEl) {
                verifyChipEl.replaceWith(buildVerifyChip(bbox.tags.map(t => t.uuid)));
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

  // ── Bulk Apply All ────────────────────────────────────────────────
  // In custom-fetchImages mode (similarity/diversity search popups) there is
  // no server-side filter to re-derive the candidate set from, so "Apply to
  // All" instead fetches the already-known full candidate list in one call
  // and applies the action via a single bulk-by-uuid-list backend request —
  // not a loop of one HTTP call per item.
  async function applyToAllCandidates() {
    statusMsg.classList.add('review-toolbar__status--loading');
    statusMsg.textContent = 'Loading candidates…';
    let images;
    try {
      images = await fetchImages({ afterUuid: null, limit: Number.MAX_SAFE_INTEGER });
    } catch (err) {
      statusMsg.classList.remove('review-toolbar__status--loading');
      statusMsg.textContent = `Error: ${err.message}`;
      return;
    }
    statusMsg.classList.remove('review-toolbar__status--loading');

    const kind = _selectionKind ?? (_viewFilter === 'images' ? 'image' : 'bbox');
    const entries = kind === 'image'
      ? images.map(img => ({ kind: 'image', img }))
      : images.flatMap(img => (img.bboxes || []).map(bbox => ({ kind: 'bbox', bbox, img })));

    if (entries.length === 0) {
      statusMsg.textContent = 'No candidates to apply to.';
      return;
    }

    if (kind === 'image') {
      await showImageBulkActionModal(entries, labels, async (action, labelUuid, tagUuids) => {
        if (action === 'delete') {
          const n = entries.length;
          const noun = n === 1 ? 'image' : 'images';
          const confirmed = await showConfirmModal({
            title: `Delete ${n} ${noun}`,
            message: `Permanently delete ${n} ${noun} from the full candidate list? This removes the files and all their annotations from disk. This cannot be undone.`,
            confirmText: 'Delete',
            danger: true,
          });
          if (!confirmed) return;
        }
        statusMsg.classList.add('review-toolbar__status--loading');
        statusMsg.textContent = `Applying to ${entries.length.toLocaleString()} images…`;
        try {
          const { modifiedCount, failCount } = await bulkApplyImagesByUuids({
            imageUuids: entries.map(e => e.img.uuid),
            action,
            labelUuid,
            tagUuids,
          });
          if (action === 'delete') {
            for (const { img } of entries) {
              events.publish(EventTypes.IMAGE_DELETED, { imageUuid: img.uuid, imagePath: img.filename });
            }
          }
          statusMsg.classList.remove('review-toolbar__status--loading');
          statusMsg.textContent = `Applied to ${modifiedCount.toLocaleString()} images` +
            (failCount > 0 ? ` (${failCount} failed — check console)` : '');
        } catch (err) {
          statusMsg.classList.remove('review-toolbar__status--loading');
          statusMsg.textContent = `Bulk apply failed: ${err.message}`;
          Logger.error('applyToAllCandidates (images) failed:', err);
        }
        await render();
      }, `Apply to all ${entries.length.toLocaleString()} matched image${entries.length === 1 ? '' : 's'}`);
    } else {
      await showBulkActionModal(entries, labels, async (action, newLabelUuid, selectedTagUuids) => {
        statusMsg.classList.add('review-toolbar__status--loading');
        statusMsg.textContent = `Applying to ${entries.length.toLocaleString()} bboxes…`;
        try {
          const { modifiedCount, failCount } = await bulkApplyBboxesByUuids({
            bboxUuids: entries.map(e => e.bbox.uuid),
            action,
            labelUuid: newLabelUuid ?? null,
            tagUuids: selectedTagUuids,
          });
          statusMsg.classList.remove('review-toolbar__status--loading');
          statusMsg.textContent = `Applied to ${modifiedCount.toLocaleString()} bboxes` +
            (failCount > 0 ? ` (${failCount} failed — check console)` : '');
        } catch (err) {
          statusMsg.classList.remove('review-toolbar__status--loading');
          statusMsg.textContent = `Bulk apply failed: ${err.message}`;
          Logger.error('applyToAllCandidates (bboxes) failed:', err);
        }
        await render();
      }, `Apply to all ${entries.length.toLocaleString()} matched bbox${entries.length === 1 ? '' : 'es'}`, { showPreview: true });
    }
  }

  bulkAllBtn.addEventListener('click', () => {
    if (fetchImages) applyToAllCandidates();
    else showBulkAllPagesModal();
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

  // Subscribe to IMAGE_ANNOTATIONS_SAVED so saves from any nested canvas modal
  // (or any other panel) propagate back to this panel's cards automatically.
  const _unsubImageSaved = events.subscribe(EventTypes.IMAGE_ANNOTATIONS_SAVED, ({ imageUuid, imageFilename, bboxes } = {}) => {
    if (!bboxes?.size || !imageUuid) return;
    for (const [bboxUuid, uiBbox] of bboxes) {
      const card = grid.querySelector(`[data-bbox-uuid="${bboxUuid}"]`);
      if (card) _refreshBboxCard(card, uiBbox, imageFilename, bboxUuid);
    }
    _imageCardRefreshers.get(imageUuid)?.();
  });

  // Mark image cards stale-deleted when an image is deleted from any panel.
  const _unsubImageDeleted = events.subscribe(EventTypes.IMAGE_DELETED, ({ imageUuid } = {}) => {
    if (!imageUuid) return;
    const card = grid.querySelector(`[data-image-uuid="${imageUuid}"][data-kind="image"]`);
    if (card) markCardStale(card, 'deleted');
  });

  // Subscribe to IMAGE_BBOXES_CHANGED so incremental bbox edits (context-menu
  // label/tag actions, bulk ops) done in sub-modals or sibling panels are
  // reflected in this panel's cards without a full page reload.
  const _unsubBboxChanged = events.subscribe(EventTypes.IMAGE_BBOXES_CHANGED, ({ imageUuid, imageFilename, updatedBboxes, deletedBboxUuids } = {}) => {
    if (!imageUuid) return;
    // Update cards for modified bboxes
    for (const [bboxUuid, uiBbox] of (updatedBboxes ?? [])) {
      // Keep the in-memory entry in sync so image-card overlays and hover
      // previews reflect the new label/tag state without a re-fetch.
      for (const entry of _allEntries) {
        if (entry.kind === 'bbox' && entry.bbox.uuid === bboxUuid) {
          if (uiBbox.label !== undefined) {
            entry.bbox.label = uiBbox.label;
            entry.bbox.labelUuid = uiBbox.labelUuid ?? uiBbox.label?.uuid ?? entry.bbox.labelUuid;
          }
          if (uiBbox.tags !== undefined) entry.bbox.tags = uiBbox.tags;
          break;
        }
      }
      const card = grid.querySelector(`[data-bbox-uuid="${bboxUuid}"]`);
      if (!card) continue;
      // Refresh thumbnail with cache-buster
      const thumb = card.querySelector('.thumb');
      if (thumb) thumb.src = getBboxThumbnailUrl(imageFilename, bboxUuid, 300) + `&t=${Date.now()}`;
      // Refresh label chip — handle both UIBbox (.text) and API bbox (.name) label shapes
      const labelChip = card.querySelector('.label-chip');
      if (labelChip && uiBbox.label) {
        labelChip.textContent = uiBbox.label.text ?? uiBbox.label.name;
        labelChip.style.cssText = `background:${uiBbox.label.color}33;border:1px solid ${uiBbox.label.color};`;
      }
      // Refresh verify chip
      const verifyChipEl = card.querySelector('.verify-chip');
      if (verifyChipEl && uiBbox.tags !== undefined) {
        verifyChipEl.replaceWith(buildVerifyChip((uiBbox.tags ?? []).map(t => t.uuid)));
      }
    }
    // Handle deleted bboxes
    for (const bboxUuid of (deletedBboxUuids ?? [])) {
      const card = grid.querySelector(`[data-bbox-uuid="${bboxUuid}"]`);
      if (card) markCardStale(card, 'deleted');
      // Remove from in-memory image bboxes so overlay redraws correctly
      for (const entry of _allEntries) {
        if (entry.img?.uuid === imageUuid && Array.isArray(entry.img.bboxes)) {
          const idx = entry.img.bboxes.findIndex(b => b.uuid === bboxUuid);
          if (idx !== -1) { entry.img.bboxes.splice(idx, 1); break; }
        }
      }
    }
    // Redraw the image-card bbox overlay for affected images
    _imageCardRefreshers.get(imageUuid)?.();
  });

  render();

  // ── Bulk Apply All Pages modal ────────────────────────────────────────────
  // Shows the action picker plus a "first 3 / last 3" preview of the bbox
  // range that will be affected, so the user can sanity-check the scope.
  async function showBulkAllPagesModal() {
    const fs = filterPanel.getState();
    const sortBy = sortBySelect.value;
    const sortOrder = sortOrderSelect.value;
    const totalBboxCount = _totalCount;

    // ── Build bbox-level filter (mirrors the render() subset logic) ──────
    const showAll = fs.showAllBboxes;
    const bboxTagFilter = showAll ? [] : (fs.tagUuids ?? []);
    const bboxLabelFilter = fs.labelUuids?.length > 0
      ? (fs.labelIncludeChildren
          ? Array.from(expandWithDescendants(fs.labelUuids, labels))
          : fs.labelUuids)
      : [];
    const bboxTagSet = showAll ? null : new Set(bboxTagFilter);
    const bboxLabelSet = bboxLabelFilter.length > 0 ? new Set(bboxLabelFilter) : null;

    // ── Fetch tags + labels for pickers ─────────────────────────────────
    const [tags, allLabels] = await Promise.all([
      fetchTags().catch(() => new Map()),
      fetchLabels().catch(() => new Map()),
    ]);

    // ── Build modal shell ────────────────────────────────────────────────
    const overlay = document.createElement('div');
    overlay.style.cssText = `position:fixed;inset:0;background:#0008;display:flex;
      align-items:center;justify-content:center;z-index:${nextModalZIndex()};`;

    const dialog = document.createElement('div');
    dialog.style.cssText = `background:#222;border:1px solid #555;border-radius:6px;
      padding:1.5em;min-width:420px;max-width:600px;display:flex;flex-direction:column;gap:0.9em;`;

    const title = document.createElement('strong');
    title.textContent = `Apply to All ${totalBboxCount.toLocaleString()} Matching Bboxes`;

    // ── Preview section ──────────────────────────────────────────────────
    const previewSection = document.createElement('div');
    previewSection.style.cssText = 'display:flex;flex-direction:column;gap:0.4em;';

    const previewHdr = document.createElement('div');
    previewHdr.style.cssText = 'font-size:0.82em;color:#aaa;text-transform:uppercase;letter-spacing:0.04em;';
    previewHdr.textContent = 'Preview range';

    const previewRow = document.createElement('div');
    previewRow.style.cssText = 'display:flex;align-items:center;gap:0.5em;min-height:88px;';

    const previewStatus = document.createElement('span');
    previewStatus.style.cssText = 'color:#aaa;font-size:0.85em;';
    previewStatus.textContent = 'Loading preview…';
    previewRow.appendChild(previewStatus);

    previewSection.append(previewHdr, previewRow);

    // ── Action picker (same as showBulkActionModal) ─────────────────────
    const actions = [
      { value: 'relabel', label: 'Change Label' },
      { value: 'markVerified', label: '✓ Mark Verified' },
      { value: 'markUnverified', label: '? Mark Unverified' },
      { value: 'markFalsePos', label: '✗ Mark False Positive' },
      { value: 'addTags', label: 'Add Tags' },
      { value: 'removeTags', label: 'Remove Tags' },
      { value: 'delete', label: 'Delete Bboxes' },
    ];

    const radioGroup = document.createElement('div');
    radioGroup.style.cssText = 'display:flex;flex-direction:column;gap:0.4em;';
    let selectedAction = 'relabel';

    let _selectedLabelUuid = undefined;
    const dynPicker = createDynamicLabelPicker(allLabels, {
      onSelect: (uuid) => { _selectedLabelUuid = uuid; },
    });
    const labelPickerRow = document.createElement('div');
    labelPickerRow.style.cssText = 'display:flex;flex-direction:column;gap:0.3em;';
    const labelPickerHdr = document.createElement('div');
    labelPickerHdr.textContent = 'New label:';
    labelPickerHdr.style.cssText = 'font-size:0.85em;color:#aaa;';
    labelPickerRow.append(labelPickerHdr, dynPicker.element);

    const tagOptions = filterVisibleTags(tags.values())
      .map(tag => ({ key: tag.uuid, text: (tag.icon ? tag.icon + ' ' : '') + tag.name, color: tag.color || '#cccccc' }));
    const tagCheckboxField = new CheckboxField({ value: [], options: tagOptions });
    const tagPickerRow = document.createElement('div');
    tagPickerRow.style.cssText = 'max-height:180px;overflow-y:auto;display:none;flex-direction:column;gap:0.3em;padding:0.3em 0;';
    tagPickerRow.appendChild(await tagCheckboxField.renderEdit());

    for (const action of actions) {
      const lbl = document.createElement('label');
      lbl.style.cssText = 'display:flex;align-items:center;gap:0.5em;cursor:pointer;';
      const radio = document.createElement('input');
      radio.type = 'radio';
      radio.name = 'bulk-all-action';
      radio.value = action.value;
      if (action.value === 'relabel') radio.checked = true;
      radio.addEventListener('change', () => {
        selectedAction = action.value;
        labelPickerRow.style.display = selectedAction === 'relabel' ? '' : 'none';
        tagPickerRow.style.display = (selectedAction === 'addTags' || selectedAction === 'removeTags') ? 'flex' : 'none';
      });
      lbl.append(radio, document.createTextNode(action.label));
      radioGroup.appendChild(lbl);
    }

    // ── Warning ──────────────────────────────────────────────────────────
    const warning = document.createElement('div');
    warning.style.cssText = 'font-size:0.85em;color:#e8a020;background:#e8a02015;border:1px solid #e8a02044;border-radius:4px;padding:0.5em 0.7em;';
    warning.textContent = `⚠ This will modify all ${totalBboxCount.toLocaleString()} bboxes matching your current filters, across every page. This cannot be undone.`;

    // ── Buttons ──────────────────────────────────────────────────────────
    const btnRow = document.createElement('div');
    btnRow.style.cssText = 'display:flex;gap:0.5em;justify-content:flex-end;';

    function close() { unregisterEscHandler(close); dynPicker.destroy(); overlay.remove(); }

    const cancelBtn = document.createElement('button');
    cancelBtn.textContent = 'Cancel';
    cancelBtn.addEventListener('click', close);

    const okBtn = document.createElement('button');
    okBtn.textContent = `Apply to All ${totalBboxCount.toLocaleString()}`;
    okBtn.style.cssText = 'background:#ff003322;border:1px solid #ff0033;border-radius:3px;padding:0.2em 0.8em;cursor:pointer;';
    okBtn.addEventListener('click', async () => {
      close();
      const newLabelUuid = selectedAction === 'relabel' ? (_selectedLabelUuid ?? undefined) : undefined;
      // Verification shortcuts are sent as addTags with the appropriate system tag UUID.
      const resolvedAction = _VERIF_TAG_MAP[selectedAction] ? 'addTags' : selectedAction;
      const selectedTagUuids = _VERIF_TAG_MAP[selectedAction]
        ? [_VERIF_TAG_MAP[selectedAction]]
        : (selectedAction === 'addTags' || selectedAction === 'removeTags') ? tagCheckboxField.getValue() : [];
      statusMsg.textContent = `Applying to all ${totalBboxCount.toLocaleString()} bboxes…`;
      statusMsg.classList.add('review-toolbar__status--loading');
      try {
        const { modifiedCount, failCount } = await bulkApplyAll({
          tagUuids: fs.showAllBboxes ? [] : (fs.tagUuids ?? []),
          tagExcludeUuids: fs.tagExcludeUuids ?? [],
          showAllBboxes: fs.showAllBboxes,
          tagMatchMode: fs.tagMatchMode,
          labelUuids: fs.labelUuids ?? [],
          labelExcludeUuids: fs.labelExcludeUuids ?? [],
          labelMatchMode: fs.labelMatchMode,
          labelIncludeChildren: fs.labelIncludeChildren,
          requireBboxes: _viewFilter === 'bboxes',
          sortBy,
          sortOrder,
          filenameSearch: fs.filenameSearch || undefined,
          dateFrom: fs.dateFrom || undefined,
          dateTo: fs.dateTo || undefined,
          verificationCheck: fs.verificationCheck,
          bboxTagFilter,
          bboxLabelFilter,
          action: resolvedAction,
          labelUuid: newLabelUuid ?? null,
          selectedTagUuids,
        });
        statusMsg.classList.remove('review-toolbar__status--loading');
        statusMsg.textContent = `Applied to ${modifiedCount.toLocaleString()} bboxes` +
          (failCount > 0 ? ` (${failCount} failed — check console)` : '');
        // Reload current page so stale cards are cleared.
        render();
      } catch (err) {
        statusMsg.classList.remove('review-toolbar__status--loading');
        statusMsg.textContent = `Bulk apply failed: ${err.message}`;
        Logger.error('Bulk apply all failed:', err);
      }
    });

    btnRow.append(cancelBtn, okBtn);
    dialog.append(title, previewSection, radioGroup, labelPickerRow, tagPickerRow, warning, btnRow);
    overlay.appendChild(dialog);
    document.body.appendChild(overlay);
    applyModalCascade(dialog);
    registerEscHandler(close);
    overlay.addEventListener('click', e => { if (e.target === overlay) close(); });

    // ── Populate preview asynchronously ─────────────────────────────────
    // First 3: already in _visibleEntries (free)
    const first3 = _visibleEntries.filter(e => e.kind === 'bbox').slice(0, 3);

    // Last 3: reverse-sort query (3 images should be enough to get 3 bboxes)
    const reversedOrder = sortOrder === 'desc' ? 'asc' : 'desc';
    let last3 = [];
    try {
      const { images: lastImgs } = await fetchReviewQueue({
        tagUuids: fs.showAllBboxes ? [] : (fs.tagUuids ?? []),
        tagExcludeUuids: fs.tagExcludeUuids ?? [],
        showAllBboxes: fs.showAllBboxes,
        tagMatchMode: fs.tagMatchMode,
        labelUuids: fs.labelUuids ?? [],
        labelExcludeUuids: fs.labelExcludeUuids ?? [],
        labelMatchMode: fs.labelMatchMode,
        labelIncludeChildren: fs.labelIncludeChildren,
        requireBboxes: true,
        sortBy,
        sortOrder: reversedOrder,
        filenameSearch: fs.filenameSearch || undefined,
        dateFrom: fs.dateFrom || undefined,
        dateTo: fs.dateTo || undefined,
        verificationCheck: fs.verificationCheck,
        limit: 5,
      });
      const rawEntries = [];
      for (const img of lastImgs) {
        for (const bbox of img.bboxes ?? []) {
          if (bboxTagSet && !bbox.tags?.some(t => bboxTagSet.has(t.uuid))) continue;
          if (bboxLabelSet && !bboxLabelSet.has(bbox.labelUuid)) continue;
          rawEntries.push({ kind: 'bbox', bbox, img });
        }
      }
      // Reverse to restore natural sort order, take up to 3.
      last3 = rawEntries.reverse().slice(0, 3);
    } catch (err) {
      Logger.error('showBulkAllPagesModal: failed to fetch last-3 preview:', err);
    }

    // ── Render preview thumbnails ────────────────────────────────────────
    previewRow.innerHTML = '';

    _populateBboxPreviewRow(previewRow, first3, last3);
  }

  return {
    destroy() {
      _drainCardUnsubs();
      _unsubImageSaved();
      _unsubImageDeleted();
      _unsubBboxChanged();
      _thumbLoader.reset();
      hoverPreview.destroy();
      entryView.destroy();
      container.innerHTML = '';
    },
    filterPanel,
    refresh: resetAndRender,
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
 *   fetchImages?: (params: object) => Promise<object[]>,
 *   editableDataset?: {
 *     filterJson: object,
 *     labelUuids: string[],
 *     onUpdateFilters: (filterJson: object, labelUuids: string[]) => void
 *   }
 * }} options
 *   title           — shown in the modal header bar.
 *   fetchImages     — passed straight through to createReviewPanel; see its docs.
 *   editableDataset — when provided, shows a live filter panel pre-populated from
 *                     the dataset's conditions. "Update Dataset" button pushes changes
 *                     back to the caller via onUpdateFilters (caller must still Save).
 */
export async function showReviewModal({ title = 'Browse Images', fetchImages, editableDataset, toolbarControls, referenceBboxUuids = new Set(), referenceImageUuids = new Set(), initialViewFilter = 'images', bboxFilterFn = null } = {}) {
  const overlay = document.createElement('div');
  const _modalZ = nextModalZIndex();
  overlay.style.cssText = `position:fixed;inset:0;background:#0009;display:flex;align-items:center;justify-content:center;z-index:${_modalZ};`;

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

  header.append(titleEl);

  // (caller-supplied toolbar controls are passed to createReviewPanel instead)

  // "Update Dataset" button — shown when modal is in editable-dataset mode
  if (editableDataset?.onUpdateFilters) {
    const updateBtn = document.createElement('button');
    updateBtn.textContent = 'Update Dataset';
    updateBtn.title = 'Apply current filters back to the dataset form (you must still Save to persist)';
    updateBtn.style.cssText = 'cursor:pointer;padding:0.2em 0.8em;';
    // Will be wired up after the review panel is created and filterPanel is accessible
    header.appendChild(updateBtn);
    // Store for wiring after panel init
    header._updateBtn = updateBtn;
  }

  header.appendChild(closeBtn);

  // Scrollable content area for the panel
  const content = document.createElement('div');
  content.style.cssText = 'flex:1;overflow:auto;display:flex;flex-direction:column;min-height:0;';

  dialog.append(header, content);
  overlay.appendChild(dialog);
  document.body.appendChild(overlay);
  applyModalCascade(dialog);

  // Use a ref so close() can call panel.destroy() even though panel is assigned below.
  const panelRef = { current: null };
  function close() { unregisterEscHandler(close); panelRef.current?.destroy(); overlay.remove(); }
  registerEscHandler(close);
  closeBtn.addEventListener('click', close);
  overlay.addEventListener('click', (e) => { if (e.target === overlay) close(); });

  // editableDataset mode: show live-filtered panel pre-populated from dataset conditions.
  // fetchImages is wired to previewDatasetImages with the current filter state.
  if (editableDataset) {
    // Use a ref so the fetchImages closure can always use the latest filter state.
    let _currentFilterState = datasetFormToReviewState(editableDataset.filterJson, editableDataset.labelUuids);

    const panel = await createReviewPanel(content, {
      // null fetchImages → full filter panel is shown
      // We'll pre-populate it below via filterPanel.setState()
      datasetActions: false,
      toolbarControls,
    });
    panelRef.current = panel;
    // Guard: if close() fired during the await, destroy the now-orphaned panel.
    if (!document.body.contains(overlay)) { panel.destroy(); return panel.refresh; }

    // Wire "Update Dataset" button now that panel is ready
    if (header._updateBtn && panel.filterPanel) {
      header._updateBtn.addEventListener('click', () => {
        const { filterJson, labelUuids } = reviewStateToDatasetForm(panel.filterPanel.getState());
        editableDataset.onUpdateFilters(filterJson, labelUuids);
        toast('Filters applied to dataset form — click Save to persist', 3000);
      });
    }

    // Pre-populate the filter panel from the dataset's saved conditions
    if (panel.filterPanel) {
      panel.filterPanel.setState(_currentFilterState);
    }
    return panel.refresh;
  } else {
    const panel = await createReviewPanel(content, { fetchImages, referenceBboxUuids, referenceImageUuids, initialViewFilter, toolbarControls, bboxFilterFn });
    panelRef.current = panel;
    // Guard: if close() fired during the await, destroy the now-orphaned panel.
    if (!document.body.contains(overlay)) { panel.destroy(); return panel.refresh; }
    return panel.refresh;
  }
}

// ── Bulk action execution ─────────────────────────────────────────────

// Verification shortcut actions → tag UUID they resolve to when sent as 'addTags'.
const _VERIF_TAG_MAP = {
  markVerified: VERIFIED_TAG_UUID,
  markUnverified: UNVERIFIED_TAG_UUID,
  markFalsePos: FALSE_POS_TAG_UUID,
};

async function executeBulkAction(action, entries, newLabelUuid, selectedTagUuids = [], tagsMap = new Map()) {
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
        // For tags in an exclusive group: remove any existing tags from the same group
        // before adding the new one (e.g. adding FALSE_POS must remove UNVERIFIED).
        let base = [...currentTagUuids];
        for (const tagUuid of selectedTagUuids) {
          const tag = tagsMap.get(tagUuid);
          if (tag?.exclusiveGroup) {
            base = base.filter(u => tagsMap.get(u)?.exclusiveGroup !== tag.exclusiveGroup);
          }
        }
        const tagUuids = [...new Set([...base, ...selectedTagUuids])];
        const labelUuid = bbox.labelUuid ?? bbox.label?.uuid;
        await updateBbox(bbox.uuid, { labelUuid, tagUuids });
      } else if (action === 'removeTags') {
        const currentTagUuids = (bbox.tags || []).map(t => t.uuid);
        const removeSet = new Set(selectedTagUuids);
        const tagUuids = currentTagUuids.filter(u => !removeSet.has(u));
        const labelUuid = bbox.labelUuid ?? bbox.label?.uuid;
        await updateBbox(bbox.uuid, { labelUuid, tagUuids });
      } else {
        // relabel: change the label. When a real label is assigned, strip verification
        // tags and add VERIFIED so the bbox leaves the review queue.
        // If newLabelUuid is undefined (picker not used), keep existing label unchanged.
        const existingLabelUuid = bbox.labelUuid ?? bbox.label?.uuid;
        const assigningLabel = newLabelUuid !== undefined && !!newLabelUuid;
        const labelUuid = assigningLabel ? newLabelUuid : existingLabelUuid;
        let currentTagUuids = (bbox.tags || []).map(t => t.uuid);
        if (assigningLabel) {
          currentTagUuids = [...currentTagUuids.filter(u => !VERIFICATION_TAG_UUIDS.has(u)), VERIFIED_TAG_UUID];
        }
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
 * Compute which bbox UUIDs differ between the undo-baseline snapshot (taken at
 * image load) and the current canvas state.  Used for conflict detection when
 * an external IMAGE_BBOXES_CHANGED event arrives while the canvas is dirty.
 * @param {Map<string,object>|null} baseline  _undoStack[0] from the runtime
 * @param {Map<string,object>} current        runtime.state.image.bboxes
 * @returns {Set<string>}
 */
function _computeUserChangedBboxUuids(baseline, current) {
  const changed = new Set();
  if (!baseline) return changed;
  for (const [uuid] of current) {
    if (!baseline.has(uuid)) { changed.add(uuid); continue; }
    const b = baseline.get(uuid);
    const c = current.get(uuid);
    const bTags = (b.tags ?? []).map(t => t.uuid).sort().join(',');
    const cTags = (c.tags ?? []).map(t => t.uuid).sort().join(',');
    if (b.labelUuid !== c.labelUuid || bTags !== cTags ||
        b.x !== c.x || b.y !== c.y || b.width !== c.width || b.height !== c.height) {
      changed.add(uuid);
    }
  }
  for (const [uuid] of baseline) {
    if (!current.has(uuid)) changed.add(uuid); // deleted by user
  }
  return changed;
}

/**
 * Apply external bbox changes to the canvas runtime, skipping any bboxes the
 * user has already modified.  Re-renders the canvas layer afterwards.
 * @param {import('@image_tagging_types').ImageTaggingRuntime} runtime
 * @param {Map<string,object>|null} updatedBboxes
 * @param {string[]} deletedBboxUuids
 * @param {Set<string>} userChangedUuids
 */
function _mergeExternalBboxChanges(runtime, updatedBboxes, deletedBboxUuids, userChangedUuids) {
  const bboxes = runtime.state.image.bboxes;
  let anyChange = false;
  for (const [uuid, uiBbox] of (updatedBboxes ?? [])) {
    if (userChangedUuids.has(uuid)) continue;
    const existing = bboxes.get(uuid);
    if (!existing) continue; // bbox not present on this canvas — skip
    if (uiBbox.label !== undefined) existing.label = uiBbox.label;
    if (uiBbox.labelUuid !== undefined) existing.labelUuid = uiBbox.labelUuid;
    if (uiBbox.tags !== undefined) existing.tags = uiBbox.tags;
    anyChange = true;
  }
  for (const uuid of (deletedBboxUuids ?? [])) {
    if (userChangedUuids.has(uuid)) continue;
    if (bboxes.delete(uuid)) anyChange = true;
  }
  if (anyChange) rerenderBboxViews(runtime.canvas, bboxes);
}

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

  // Content area: canvas (left) + sidebar with tabs (Labels / Inference / Tags)
  const contentArea = document.createElement('div');
  contentArea.className = 'canvas-modal-content';

  const canvasWrap = document.createElement('div');
  canvasWrap.className = 'canvas-modal-canvas';

  // ── Right sidebar: tab header + two panes ────────────────────────
  const MODAL_TABS = { labels: 'modal-tab-labels', tags: 'modal-tab-tags' };

  const sidebarEl = document.createElement('div');
  sidebarEl.className = 'canvas-modal-sidebar';

  const sidebarTabHeader = document.createElement('div');
  sidebarTabHeader.className = 'sidebar-tab-header';

  const _mkTabBtn = (tab, label) => {
    const btn = document.createElement('span');
    btn.className = 'sidebar-tab-button';
    btn.dataset.tab = tab;
    btn.textContent = label;
    return btn;
  };
  sidebarTabHeader.append(
    _mkTabBtn(MODAL_TABS.labels, 'Labels'),
    _mkTabBtn(MODAL_TABS.tags,   'Tags'),
  );

  const _mkTabPane = (id) => {
    const pane = document.createElement('div');
    pane.className = 'sidebar-tab-pane';
    pane.id = id;
    return pane;
  };
  const paneLabels = _mkTabPane(MODAL_TABS.labels);
  const paneTags   = _mkTabPane(MODAL_TABS.tags);

  const labelListEl = document.createElement('div');
  paneLabels.appendChild(labelListEl);

  sidebarEl.append(sidebarTabHeader, paneLabels, paneTags);

  const modalResizeHandleEl = document.createElement('div');
  modalResizeHandleEl.className = 'split-resize-handle';
  contentArea.append(canvasWrap, modalResizeHandleEl, sidebarEl);

  // Footer — Save / Cancel / Save & Close; Save buttons start dimmed (no unsaved changes yet)
  const footer = document.createElement('div');
  footer.className = 'canvas-modal-footer';

  const cancelBtn = document.createElement('button');
  cancelBtn.textContent = 'Cancel';
  cancelBtn.style.cssText = 'border:1px solid #555;border-radius:3px;padding:0.3em 1em;cursor:pointer;color:#aaa;margin-left:auto;';

  const saveOnlyBtn = document.createElement('button');
  saveOnlyBtn.textContent = 'Save';
  saveOnlyBtn.style.cssText = 'background:#33333322;border:1px solid #555;border-radius:3px;padding:0.3em 1em;cursor:pointer;color:#888;';

  const saveBtn = document.createElement('button');
  saveBtn.textContent = 'Save & Close';
  saveBtn.style.cssText = 'background:#33333322;border:1px solid #555;border-radius:3px;padding:0.3em 1em;cursor:pointer;color:#888;';
  footer.append(cancelBtn, saveOnlyBtn, saveBtn);

  dialog.append(header, toolbarEl, contentArea, footer);
  overlay.appendChild(dialog);
  overlay.style.zIndex = nextModalZIndex();
  document.body.appendChild(overlay);
  applyModalCascade(dialog);

  // ── Create canvas instance ─────────────────────────────────────────
  let instance;
  let toolbarHandle    = { destroy: () => {} };
  let labelHandle      = { destroy: () => {} };
  let tagPanelHandle   = { destroy: () => {} };
  let modalResizeHandle = { destroy: () => {} };
  let unsubToolChanged = () => {};
  let unsubDblClick    = () => {};
  let unsubBboxChanged = () => {};

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

  // ── Setup sidebar tabs ─────────────────────────────────────────────
  const { activateTab: activateSidebarTab } = setupSidebarTabs(sidebarEl);

  // ── Sidebar drag-resize ────────────────────────────────────────────
  modalResizeHandle = initSplitResize(modalResizeHandleEl, sidebarEl, {
    storageKey: 'wtch_modal_sidebar_width',
    defaultWidth: 220,
    min: 120,
    max: 500,
  });

  // ── Wire label list ────────────────────────────────────────────────
  labelHandle = await createLabelList(labelListEl, { mode: 'select', runtime: instance.runtime });

  // ── Wire tag panel (class + tag toggles, responds to bbox clicks) ──
  tagPanelHandle = createTagPanel(paneTags, instance.runtime);

  // ── Sidebar tab-switching rules ────────────────────────────────────
  // BBox tool → Labels tab; tag tool → Tags tab (mirrors main canvas sidebar).
  unsubToolChanged = events.subscribeToCanvas(instance.runtime.id, EventTypes.CANVAS_TOOL_CHANGED, ({ tool }) => {
    const { tool: kind } = parseToolUuid(tool);
    if (kind === 'bbox') activateSidebarTab(MODAL_TABS.labels);
    else if (kind === 'tag') activateSidebarTab(MODAL_TABS.tags);
  });

  // Double-clicking a bbox → switch to Tags tab.
  unsubDblClick = events.subscribeToCanvas(instance.runtime.id, EventTypes.CANVAS_BBOX_DOUBLE_CLICKED, () => {
    activateSidebarTab(MODAL_TABS.tags);
  });

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
  const _SAVE_BTN_CLEAN = 'background:#33333322;border:1px solid #555;border-radius:3px;padding:0.3em 1em;cursor:pointer;color:#888;';
  const _SAVE_BTN_DIRTY = 'background:#ff003322;border:1px solid #ff0033;border-radius:3px;padding:0.3em 1em;cursor:pointer;color:#ff6666;';
  function _updateSaveBtn() {
    if (saveBtn.disabled) return; // don't override while saving
    const dirty = instance.runtime.isDirty;
    saveBtn.style.cssText = dirty ? _SAVE_BTN_DIRTY : _SAVE_BTN_CLEAN;
    saveOnlyBtn.style.cssText = dirty ? _SAVE_BTN_DIRTY : _SAVE_BTN_CLEAN;
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

  // ── External-change handler ────────────────────────────────────────
  // Listens for bbox modifications made in sub-modals (e.g. similar-search
  // panels) while this canvas is open.
  //   • Clean canvas  → auto-reload the image so the user sees the latest state.
  //   • Dirty, no overlap → merge the external changes for non-user-touched bboxes.
  //   • Dirty, overlap    → prompt: reload (losing edits) or keep canvas changes.
  let _handlingExternalChange = false;
  unsubBboxChanged = events.subscribe(EventTypes.IMAGE_BBOXES_CHANGED, async ({ imageUuid, imageFilename, updatedBboxes, deletedBboxUuids } = {}) => {
    if (!imageUuid || imageUuid !== _currentImg?.uuid) return;
    if (_handlingExternalChange) return;
    _handlingExternalChange = true;
    try {
      if (!instance.runtime.isDirty) {
        // Canvas has no unsaved edits — reload transparently.
        try {
          await instance.loadImage(_currentImg.filename);
          toast('Canvas refreshed with external changes', 2000);
        } catch (err) {
          Logger.error('Canvas modal: auto-reload after external change failed', err);
        }
        return;
      }

      const baseline = instance.runtime.undoBaseline;
      const current  = instance.runtime.state.image.bboxes;
      const userChangedUuids = _computeUserChangedBboxUuids(baseline, current);

      const externalUuids = new Set([
        ...(updatedBboxes?.keys() ?? []),
        ...(deletedBboxUuids ?? []),
      ]);
      const hasOverlap = [...externalUuids].some(u => userChangedUuids.has(u));

      if (!hasOverlap) {
        // No conflict — merge external changes into non-user-touched bboxes.
        _mergeExternalBboxChanges(instance.runtime, updatedBboxes, deletedBboxUuids, userChangedUuids);
        toast('External changes merged into canvas', 2000);
        return;
      }

      // Overlap — ask the user what to do.
      const choice = await showExternalChangesModal({ conflictCount: [...externalUuids].filter(u => userChangedUuids.has(u)).length });
      if (choice === 'reload') {
        try {
          await instance.loadImage(_currentImg.filename);
        } catch (err) {
          Logger.error('Canvas modal: conflict reload failed', err);
          toast('Reload failed', 3000, 'error');
        }
      }
      // 'keep': leave the canvas as-is
    } finally {
      _handlingExternalChange = false;
    }
  });

  // ── Teardown ───────────────────────────────────────────────────────
  let _canvasEscClose = null;
  function closeModal() {
    if (_canvasEscClose) unregisterEscHandler(_canvasEscClose);
    unsubBboxChanged();
    unsubDirty();
    if (unsubNavLoad) unsubNavLoad();
    unsubToolChanged();
    unsubDblClick();
    tagPanelHandle.destroy();
    toolbarHandle.destroy();
    labelHandle.destroy();
    modalResizeHandle.destroy();
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
    // Broadcast to all review panels (including nested / parent modals) so
    // every card showing a bbox from this image gets refreshed.
    events.publish(EventTypes.IMAGE_ANNOTATIONS_SAVED, {
      imageUuid: currentImg?.uuid,
      imageFilename: currentImg?.filename ?? currentImg?.image_path,
      bboxes: currentBboxesMap,
    });
    if (currentImg?.uuid !== img.uuid) {
      // Navigated to a different image before saving — let navContext refresh its cards.
      navContext?.onSaved?.(currentImg, currentBboxesMap);
    }
    navContext?.onClose?.(currentImg);
    await onSaved(updatedUIBbox, currentImg);
  }

  // ── Close guard: prompt when there are unsaved changes ────────────
  function _setAllButtonsDisabled(disabled) {
    saveBtn.disabled = disabled;
    saveOnlyBtn.disabled = disabled;
    cancelBtn.disabled = disabled;
    closeBtn.disabled = disabled;
  }

  async function tryClose() {
    if (instance.runtime.isDirty) {
      const choice = await showUnsavedChangesModal();
      if (choice === 'cancel') return;
      if (choice === 'save') {
        _setAllButtonsDisabled(true);
        const saved = await saveAnnotations(instance.runtime);
        if (!saved) {
          _setAllButtonsDisabled(false);
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

  // ESC key — behaves like the Cancel button (shows unsaved-changes dialog if dirty)
  _canvasEscClose = () => { if (!closeBtn.disabled) tryClose(); };
  registerEscHandler(_canvasEscClose);

  // ── Save (without closing) button ─────────────────────────────────
  saveOnlyBtn.addEventListener('click', async () => {
    if (!instance.runtime.isDirty) return; // nothing to save — button is dimmed
    _setAllButtonsDisabled(true);
    saveOnlyBtn.textContent = 'Saving…';

    let saved = false;
    try {
      saved = await saveAnnotations(instance.runtime);
    } catch (err) {
      Logger.error('Canvas modal: save failed', err);
    }

    _setAllButtonsDisabled(false);
    saveOnlyBtn.textContent = 'Save';
    _updateSaveBtn();

    if (saved) {
      // Broadcast to other panels without closing
      const currentImg = _currentImg;
      const currentBboxesMap = new Map(instance.runtime.state.image.bboxes);
      events.publish(EventTypes.IMAGE_ANNOTATIONS_SAVED, {
        imageUuid: currentImg?.uuid,
        imageFilename: currentImg?.filename ?? currentImg?.image_path,
        bboxes: currentBboxesMap,
      });
    }
  });

  // ── Save & Close button ────────────────────────────────────────────
  saveBtn.addEventListener('click', async () => {
    if (!instance.runtime.isDirty) return; // nothing to save — button is dimmed
    _setAllButtonsDisabled(true);
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
      _setAllButtonsDisabled(false);
      _updateSaveBtn();
      saveBtn.textContent = 'Save & Close';
    }
  });
}

// ── Shared preview helpers ────────────────────────────────────────────

function _makeBboxThumbGroup(entries, label) {
  const group = document.createElement('div');
  group.style.cssText = 'display:flex;flex-direction:column;align-items:center;gap:0.3em;';
  const thumbs = document.createElement('div');
  thumbs.style.cssText = 'display:flex;gap:4px;';
  for (const { bbox, img } of entries) {
    const wrapper = document.createElement('div');
    wrapper.style.cssText = 'position:relative;';
    const thumb = document.createElement('img');
    thumb.style.cssText = 'width:80px;height:80px;object-fit:cover;border-radius:3px;border:1px solid #555;display:block;';
    thumb.src = getBboxThumbnailUrl(img.filename, bbox.uuid, 80);
    wrapper.appendChild(thumb);
    thumbs.appendChild(wrapper);
  }
  const lbl = document.createElement('div');
  lbl.style.cssText = 'font-size:0.75em;color:#888;';
  lbl.textContent = label;
  group.append(thumbs, lbl);
  return group;
}

/** Populate a preview row with first-N / last-N bbox thumbnails. */
function _populateBboxPreviewRow(row, first3, last3) {
  if (first3.length === 0 && last3.length === 0) {
    const msg = document.createElement('span');
    msg.style.cssText = 'color:#888;font-size:0.85em;';
    msg.textContent = 'No preview available.';
    row.appendChild(msg);
    return;
  }
  const first3Uuids = new Set(first3.map(e => e.bbox.uuid));
  const uniqueLast3 = last3.filter(e => !first3Uuids.has(e.bbox.uuid));
  if (first3.length > 0) row.appendChild(_makeBboxThumbGroup(first3, 'First'));
  if (uniqueLast3.length > 0) {
    const sep = document.createElement('div');
    sep.style.cssText = 'color:#555;font-size:1.2em;letter-spacing:0.1em;align-self:center;padding:0 0.2em;';
    sep.textContent = '···';
    row.appendChild(sep);
    row.appendChild(_makeBboxThumbGroup(uniqueLast3, 'Last'));
  }
}

// ── Bulk action modal ─────────────────────────────────────────────────

async function showBulkActionModal(entries, labels, onConfirm, titleText = null, opts = {}) {
  // Fetch fresh tags and ALL labels (standard + dynamic) for the pickers.
  const [tags, allLabels] = await Promise.all([
    fetchTags().catch(() => new Map()),
    fetchLabels().catch(() => new Map()),
  ]);

  const overlay = document.createElement('div');
  overlay.style.cssText = `position:fixed;inset:0;background:#0008;display:flex;
    align-items:center;justify-content:center;z-index:${nextModalZIndex()};`;

  const dialog = document.createElement('div');
  dialog.style.cssText = `background:#222;border:1px solid #555;border-radius:6px;
    padding:1.5em;min-width:320px;display:flex;flex-direction:column;gap:0.8em;`;

  const title = document.createElement('strong');
  title.textContent = titleText ?? `Apply to ${entries.length} selected bbox${entries.length === 1 ? '' : 'es'}`;

  // ── Preview section (opt-in) ─────────────────────────────────────────
  if (opts.showPreview) {
    const bboxEntries = entries.filter(e => e.kind === 'bbox');
    const previewSection = document.createElement('div');
    previewSection.style.cssText = 'display:flex;flex-direction:column;gap:0.4em;';

    const previewHdr = document.createElement('div');
    previewHdr.style.cssText = 'font-size:0.82em;color:#aaa;text-transform:uppercase;letter-spacing:0.04em;';
    previewHdr.textContent = 'Preview range';

    const previewRow = document.createElement('div');
    previewRow.style.cssText = 'display:flex;align-items:center;gap:0.5em;min-height:88px;';

    _populateBboxPreviewRow(previewRow, bboxEntries.slice(0, 3), bboxEntries.slice(-3));

    previewSection.append(previewHdr, previewRow);
    dialog.appendChild(title);
    dialog.appendChild(previewSection);
  } else {
    dialog.appendChild(title);
  }

  // Action radio buttons
  const actions = [
    { value: 'relabel', label: 'Change Label' },
    { value: 'markVerified', label: '✓ Mark Verified' },
    { value: 'markUnverified', label: '? Mark Unverified' },
    { value: 'markFalsePos', label: '✗ Mark False Positive' },
    { value: 'addTags', label: 'Add Tags' },
    { value: 'removeTags', label: 'Remove Tags' },
    { value: 'delete', label: 'Delete Bboxes' },
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

  // Label picker — dynamic search/create picker showing ALL labels (standard + dynamic)
  let _selectedLabelUuid = undefined; // undefined = no change; null = clear label
  const dynPicker = createDynamicLabelPicker(allLabels, {
    onSelect: (uuid) => { _selectedLabelUuid = uuid; },
  });
  const labelPickerRow = document.createElement('div');
  labelPickerRow.style.cssText = 'display:flex;flex-direction:column;gap:0.3em;';
  const labelPickerHdr = document.createElement('div');
  labelPickerHdr.textContent = 'New label:';
  labelPickerHdr.style.cssText = 'font-size:0.85em;color:#aaa;';
  labelPickerRow.appendChild(labelPickerHdr);
  labelPickerRow.appendChild(dynPicker.element);

  // Tag picker (shown only for addTags / removeTags)
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

  function close() { unregisterEscHandler(close); dynPicker.destroy(); overlay.remove(); }

  const cancelBtn = document.createElement('button');
  cancelBtn.textContent = 'Cancel';
  cancelBtn.addEventListener('click', close);

  const okBtn = document.createElement('button');
  okBtn.textContent = 'Apply';
  okBtn.style.cssText = 'background:#ff003322;border:1px solid #ff0033;border-radius:3px;padding:0.2em 0.8em;cursor:pointer;';
  okBtn.addEventListener('click', async () => {
    close();
    // _selectedLabelUuid: undefined = picker not used (no change), null = clear label, string = set label
    const newLabelUuid = selectedAction === 'relabel' ? (_selectedLabelUuid ?? undefined) : undefined;
    // Verification shortcuts are sent as addTags with the appropriate system tag UUID.
    const resolvedAction = _VERIF_TAG_MAP[selectedAction] ? 'addTags' : selectedAction;
    const selectedTagUuids = _VERIF_TAG_MAP[selectedAction]
      ? [_VERIF_TAG_MAP[selectedAction]]
      : (selectedAction === 'addTags' || selectedAction === 'removeTags') ? tagCheckboxField.getValue() : [];
    await onConfirm(resolvedAction, newLabelUuid, selectedTagUuids, tags);
  });

  btnRow.appendChild(cancelBtn);
  btnRow.appendChild(okBtn);
  dialog.append(radioGroup, labelPickerRow, tagPickerRow, btnRow);
  overlay.appendChild(dialog);
  document.body.appendChild(overlay);
  applyModalCascade(dialog);
  registerEscHandler(close);

  overlay.addEventListener('click', e => { if (e.target === overlay) close(); });
}

const TAG_ACTIONS = ['addTags', 'removeTags', 'setTagAbsent', 'setTagUncertain'];

async function showImageBulkActionModal(entries, labels, onConfirm, titleText = null) {
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

async function executeBulkImageAction(action, entries, labelUuid, tagUuids = []) {
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
