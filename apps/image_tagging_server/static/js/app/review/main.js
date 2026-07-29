import { fetchLabels } from '/app-static/js/shared/api/labels.js';
import { createDynamicLabelPicker } from '/app-static/js/ui/utils/dynamicLabelPicker.js';
import { fetchTags } from '/app-static/js/shared/api/tags.js';
import { filterVisibleTags } from '/app-static/js/shared/activeModel.js';
import { createFilterPanel } from '/app-static/js/app/review/filters.js';
import {
  buildVerifyChip,
  createCardGrid,
  buildBboxCard,
  buildImageCard,
  buildBboxRow,
  buildImageRow,
} from '/app-static/js/app/review/cardGrid.js';
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
import { fetchImageLabels, fetchImageLabelsBatch, setImageLabel } from '/app-static/js/shared/api/imageLabels.js';
import { activeTrainingConfig } from '/app-static/js/shared/activeTrainingConfig.js';
import { setImageTag, setImageTagStatus } from '/app-static/js/shared/api/imageTags.js';
import { attachContext } from '/app-static/js/ui/contextMenu/index.js';
import { getPrimaryRuntimeId } from '/app-static/js/app/image-tagging/runtime.js';
import { Logger, toast } from '/app-static/js/ui/utils/logging.js';
import { showUnsavedChangesModal, showExternalChangesModal, showConfirmModal } from '/app-static/js/ui/utils/confirmModal.js';
import { deleteImage } from '/app-static/js/shared/api/images.js';
import { rerenderBboxViews, resetCanvasZoom } from '/app-static/js/app/image-tagging/canvas/manager.js';
import { nextModalZIndex, applyModalCascade, registerEscHandler, unregisterEscHandler } from '/app-static/js/ui/utils/modalZIndex.js';
import { CheckboxField } from '/app-static/js/ui/utils/fields/CheckboxField.js';
import { createCanvasInstance } from '/app-static/js/app/image-tagging/canvas/instance.js';
import { saveAnnotations, addInferenceResults, seedRuntimeBboxes } from '/app-static/js/app/image-tagging/canvas/image.js';
import { createInferencePanel } from '/app-static/js/app/image-tagging/inferencePanel.js';
import { runDetection } from '/app-static/js/shared/api/inference.js';
import { initSplitResize } from '/app-static/js/ui/utils/splitResize.js';
import { events, EventTypes } from '/app-static/js/shared/events/index.js';
import { init as initToolbar } from '/app-static/js/app/image-tagging/toolbar.js';
import { createTagPanel } from '/app-static/js/ui/inspector.js';
import { setupSidebarTabs } from '/app-static/js/app/utils/tabs.js';
import { parseToolUuid } from '/app-static/js/app/image-tagging/canvas/tools.js';
import { createHoverPreview } from '/app-static/js/ui/thumbnailHoverPreview.js';
import { drawLetterboxedBboxes } from '/app-static/js/ui/utils/bboxThumbOverlay.js';
import { createEntryView } from '/app-static/js/ui/entryView.js';
import { reviewStateToDatasetForm, datasetFormToReviewState } from '/app-static/js/shared/filterUtils.js';
import { openDissimilarGlobalModal } from '/app-static/js/ui/dissimilarSearch.js';
import { runModelOnResultsFlow, makeBboxFilterPredicate } from '/app-static/js/app/review/runModelOnResults.js';
import { runFilterChainOnResultsFlow } from '/app-static/js/app/review/runFilterChainOnResults.js';
import { showImageBulkActionModal, executeBulkImageAction } from '/app-static/js/app/review/imageBulkActionModal.js';
import { createBboxVerificationControl } from '/app-static/js/app/review/bboxVerificationControl.js';
import { defaultCuratedFilterState, bboxMatchesFilter, imageMatchesFilter } from '/app-static/js/app/review/filterPredicate.js';
import { createMultiSelectDropdown } from '/app-static/js/ui/utils/index.js';
import { flattenHierarchy } from '/app-static/js/ui/labelList.js';

const DEFAULT_PAGE_SIZE = 250;
const MAX_ENTRIES = 1000; // hard cap on total cards per page
const SCROLL_LOAD_MARGIN_PX = 400; // infinite-scroll (M2) preload distance from the bottom of the scroll container

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
export async function createReviewPanel(container, { fetchImages = null, datasetActions = true, referenceBboxUuids = new Set(), referenceImageUuids = new Set(), initialViewFilter = 'images', toolbarControls = null, bboxFilterFn = null } = {}) {
  container.innerHTML = '';

  const labelsPromise = fetchLabels().catch(() => new Map());
  // Tags are needed both in filter mode (createFilterPanel) and in fetchImages
  // mode (curated Tags control, M1) — fetch unconditionally.
  const tagsPromise = fetchTags().catch(() => new Map());
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
      cardGrid.selected.clear();
      cardGrid.selectionKind = null;
      // Config Ready (V2) is whole-image-only — disable it outside images/all view.
      filterPanel?.setConfigReadyAvailable(_viewFilter !== 'bboxes');
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

  // Run Model on Results — proposes relabels for the current selection/filter
  // scope by running a detector and matching detections to existing bboxes
  // by IoU (see app/review/runModelOnResults.js). Enabled in both full-filter
  // and custom-fetchImages modes; scope resolution happens on click.
  const runModelBtn = document.createElement('button');
  runModelBtn.className = 'review-toolbar__btn';
  runModelBtn.textContent = 'Run Model on Results…';
  runModelBtn.addEventListener('click', () => runModelOnResultsFlow({
    fetchImages,
    filterPanel,
    viewFilter: _viewFilter,
    selectionKind: cardGrid.selectionKind,
    selected: cardGrid.selected,
    visibleEntries: _visibleEntries,
    bboxFilterFn,
    labels,
    sortBy: sortBySelect.value,
    sortOrder: sortOrderSelect.value,
    refresh: () => render(),
  }));

  // Run Filter Chain on Results — runs a stored FilterChain (relabel/tag/add-box
  // steps) over the images behind the current selection/filter. Same scope
  // machinery as Run Model on Results, but image-scoped (see runFilterChainOnResults.js).
  const runChainBtn = document.createElement('button');
  runChainBtn.className = 'review-toolbar__btn';
  runChainBtn.textContent = 'Run Filter Chain on Results…';
  runChainBtn.addEventListener('click', () => runFilterChainOnResultsFlow({
    fetchImages,
    filterPanel,
    viewFilter: _viewFilter,
    selectionKind: cardGrid.selectionKind,
    selected: cardGrid.selected,
    visibleEntries: _visibleEntries,
    bboxFilterFn,
    labels,
    sortBy: sortBySelect.value,
    sortOrder: sortOrderSelect.value,
    refresh: () => render(),
  }));

  // Refresh
  const refreshBtn = document.createElement('button');
  refreshBtn.className = 'review-toolbar__btn';
  refreshBtn.textContent = 'Refresh';

  // Pagination — filter mode only. Custom/fetchImages mode uses infinite
  // scroll instead (M2), so Prev/Next never apply there.
  const pageInfo = document.createElement('span');
  pageInfo.className = 'review-toolbar__page-info';
  const prevBtn = document.createElement('button');
  prevBtn.className = 'review-toolbar__btn';
  prevBtn.textContent = '‹ Prev';
  const nextBtn = document.createElement('button');
  nextBtn.className = 'review-toolbar__btn';
  nextBtn.textContent = 'Next ›';
  if (fetchImages) { prevBtn.hidden = true; nextBtn.hidden = true; }

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
      ? buildImageCard(cardGrid, entry, _cardOpts(labels, i))
      : buildBboxCard(cardGrid, entry, _cardOpts(labels, i)),
    renderRow: (entry, i) => entry.kind === 'image'
      ? buildImageRow(cardGrid, entry, _cardOpts(labels, i))
      : buildBboxRow(cardGrid, entry, _cardOpts(labels, i)),
    // Cards subscribe to EventBus on build; drain before every rebuild or a
    // grid/list toggle doubles the subscriptions each click.
    onBeforeRender: () => _drainCardUnsubs(),
  });
  // `grid` aliases the live container holding cards/rows — selection, stale-marking,
  // group-highlight, overlay refresh, bulk apply and nav-context queries all go through it.
  const grid = entryView.layout;

  // Card-grid controller — OWNS the selection state (selected / selectionKind /
  // lastClickedIndex / deletedUuids / staleUuids) and the click/shift-range/
  // stale/type-fading logic that used to live inline here. Everything host-
  // specific it needs (visible entries, id extraction, bulk-button refresh,
  // toast, select-all reset) is injected via `ctx`; the arrow wrappers below
  // defer resolution so the controller can be built before those closures are
  // declared further down.
  const cardGrid = createCardGrid(grid, {
    getVisibleEntries: () => _visibleEntries,
    entryId: e => entryId(e),
    onSelectionChange: () => updateBulkBtn(),
    hooks: {
      toast,
      resetSelectAll: () => { selectAllCb.checked = false; selectAllCb.indeterminate = false; },
    },
  });

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
    makeToolbarGroup(selectAllLabel, bulkBtn, bulkAllBtn, runModelBtn, runChainBtn),
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
    // Config Ready (V2) is whole-image-only — start disabled if we're opening
    // straight into bboxes view. Kept in sync by the view-toggle handler below.
    filterPanel.setConfigReadyAvailable(initialViewFilter !== 'bboxes');

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

  // ── Curated result-filter surface (M1, fetchImages/custom mode only) ──────
  // Filter mode gets the full createFilterPanel; custom/similarity mode never
  // built any filter UI at all before this. M1 adds the *curated* subset
  // (locked decision 3: bbox Verification + Label + Tag only — no date/
  // filename/image-label conditions) as an in-memory predicate
  // (filterPredicate.js) re-applied to the already-fetched candidate objects.
  // `_curated` starts at the no-op default so an unfiltered modal behaves
  // exactly as before this unit landed.
  let _curated = defaultCuratedFilterState();
  if (fetchImages) {
    const tags = await tagsPromise;

    function _onCuratedChange() {
      // Mirror the existing filter-panel refresh path: reset paging state and
      // re-render so the new predicate takes effect immediately.
      _pageCursors = [null];
      _lastCursor = null;
      render();
    }

    // Bbox verification tri-state (V1) — Any/Verified/Unverified/False-Pos.
    const verificationControl = createBboxVerificationControl({
      value: _curated.bboxVerification,
      onChange: (value) => {
        _curated = { ..._curated, bboxVerification: value };
        _onCuratedChange();
      },
    });

    // Labels multi-select — object-scoped labels only, with "include child
    // labels" hierarchy toggle, mirroring filters.js's buildLabelsControls.
    const labelsContainer = document.createElement('div');
    let _labelsDropdown = null;
    function _buildCuratedLabelsDropdown() {
      const regularMap = new Map([...labels].filter(([, l]) => l.labelType !== 'dynamic' && l.scope !== 'image'));
      // Dynamic labels are always filterable here — see filters.js's
      // buildLabelsControls for why this doesn't gate on the "Show dynamic
      // labels" picker-clutter preference.
      const dynamicMap = new Map([...labels].filter(([, l]) => l.labelType === 'dynamic' && l.scope !== 'image'));
      const flatRegular = flattenHierarchy(regularMap);
      const flatDynamic = flattenHierarchy(dynamicMap);
      const toItem = l => ({ uuid: l.uuid, name: l.name, color: l.color || null, depth: l.depth, parentUuid: l.parentUuid ?? null });
      const labelItems = [
        ...flatRegular.map(toItem),
        ...(flatDynamic.length > 0 ? [{ isSeparator: true, label: 'Dynamic' }, ...flatDynamic.map(toItem)] : []),
      ];
      return createMultiSelectDropdown({
        label: 'Labels',
        items: labelItems,
        defaultWidth: '280px',
        defaultHeight: '400px',
        defaultSelected: _curated.labelUuids,
        defaultMode: _curated.labelMatchMode,
        filterKey: 'curated-labels',
        hierarchyToggle: {
          label: 'Include child labels',
          default: _curated.labelIncludeChildren,
          onChange: (val) => {
            _curated = { ..._curated, labelIncludeChildren: val };
            _onCuratedChange();
          },
        },
        onChange: (selected, excluded, mode) => {
          _curated = { ..._curated, labelUuids: selected, labelExcludeUuids: excluded, labelMatchMode: mode };
          _onCuratedChange();
        },
      });
    }
    _labelsDropdown = _buildCuratedLabelsDropdown();
    labelsContainer.appendChild(_labelsDropdown.element);

    // Tags multi-select (include/exclude/mode) — visible tags minus the
    // verification tags, which the tri-state control above owns exclusively
    // (same "only one way to filter verification" rule V1 established in
    // filters.js).
    const tagsContainer = document.createElement('div');
    let _tagsDropdown = null;
    function _buildCuratedTagsDropdown() {
      const visible = filterVisibleTags(tags.values()).filter(t => !VERIFICATION_TAG_UUIDS.has(t.uuid));
      const items = visible.map(t => ({ uuid: t.uuid, name: t.name, color: t.color || null }));
      return createMultiSelectDropdown({
        label: 'Tags',
        items,
        defaultSelected: _curated.tagUuids,
        defaultMode: _curated.tagMatchMode,
        filterKey: 'curated-tags',
        defaultWidth: '220px',
        defaultHeight: '300px',
        onChange: (selected, excluded, mode) => {
          _curated = { ..._curated, tagUuids: selected, tagExcludeUuids: excluded, tagMatchMode: mode };
          _onCuratedChange();
        },
      });
    }
    _tagsDropdown = _buildCuratedTagsDropdown();
    tagsContainer.appendChild(_tagsDropdown.element);

    // Append after the caller's own toolbarControls (threshold/count/switch) —
    // toolbarControls is a positional contract elsewhere (createFilterPanel's
    // toolbarControls array); here it's simpler (a single element/fragment),
    // but the same "append, don't insert" rule applies so caller-built
    // controls never shift position (NOTES.md).
    const curatedGroup = makeToolbarGroup(verificationControl.element, labelsContainer, tagsContainer);
    curatedGroup.classList.add('modal-toolbar-row');
    toolbar.append(makeToolbarSep(), curatedGroup);
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
  // Selection state (_selected / _selectionKind / _lastClickedIndex /
  // _deletedUuids / _staleUuids) now lives in the `cardGrid` controller
  // (createCardGrid, above) — read/write it through that controller.
  let _pageSize = DEFAULT_PAGE_SIZE;
  let _imageCardRefreshers = new Map(); // img.uuid → () => void — redraws bbox overlay
  // Whole-image label statuses for the current page, fetched in ONE batch call
  // per render — cards read from here instead of issuing a GET per card.
  // img.uuid → (labelUuid → 'present'|'absent')
  let _pageImageLabels = new Map();

  let _pageCursors = [null];
  let _lastCursor = null;
  let _rendering = false;
  let _renderGeneration = 0;
  let _hasMore = false;
  let _totalCount = 0;

  // ── Infinite-scroll state (M2, fetchImages mode only) ────────────────────
  // `_curatedCursor` is the afterUuid to resume from; `_curatedExhausted`
  // means fetchImages returned fewer than a full chunk (no more candidates).
  // Filter mode ignores these — it keeps fixed-page keyset pagination.
  let _curatedCursor = null;
  let _curatedExhausted = false;
  let _loadingMore = false;
  let _loadMorePending = false;
  let _scrollHandler = null;

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

  // V2: "Config Ready" (verificationCheck) is repointed from the Labels-dropdown
  // selection to the active training config's target labels, and is whole-image
  // (not meaningful in bboxes view). Returns [] when there's no verificationCheck
  // to run, no active config, or we're in bboxes view — callers only attach
  // verificationLabelUuids to the request when this is non-empty.
  function _verificationLabelUuidsFor(fs) {
    if (!fs.verificationCheck) return [];
    if (_viewFilter === 'bboxes') return [];
    return [...activeTrainingConfig.getLabelUuids()];
  }

  // Gate verificationCheck itself out of the request in bboxes view — the
  // check is whole-image-only (V2), so sending it there would silently filter
  // bbox-view results by a check that's meaningless in that context.
  function _verificationCheckFor(fs) {
    return _viewFilter === 'bboxes' ? null : fs.verificationCheck;
  }

  // Full set of image UUIDs matching this review panel's current scope — the
  // active filter panel state (top-level tab) or, inside a nested search-results
  // modal, the modal's own candidate list (fetchImages). Used by image/bbox
  // context-menu adapters to scope "search with current filters" similar/dissimilar
  // queries.
  async function _getFilteredImageUuids() {
    if (fetchImages) {
      const images = await fetchImages({ afterUuid: null, limit: Number.MAX_SAFE_INTEGER });
      // Apply the curated predicate (M1) so "search with current filters" from
      // inside a modal reflects the curated Verification/Label/Tag selection,
      // parallel to how filter-mode already re-runs the full server query below.
      return images.filter(img => imageMatchesFilter(img, _curated, labels)).map(i => i.uuid);
    }
    const fs = filterPanel.getState();
    const { images } = await fetchReviewQueue({
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
      sortBy: sortBySelect.value,
      sortOrder: sortOrderSelect.value,
      filenameSearch: fs.filenameSearch || undefined,
      dateFrom: fs.dateFrom || undefined,
      dateTo: fs.dateTo || undefined,
      verificationCheck: _verificationCheckFor(fs),
      verificationLabelUuids: _verificationLabelUuidsFor(fs),
      bboxVerification: fs.bboxVerification,
      limit: Number.MAX_SAFE_INTEGER,
    });
    return images.map(i => i.uuid);
  }

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
      getFilteredImageUuids: _getFilteredImageUuids,

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

      getFilteredImageUuids: _getFilteredImageUuids,
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
    const _selected = cardGrid.selected;
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
    const _selected = cardGrid.selected;
    const _selectionKind = cardGrid.selectionKind;
    const _deletedUuids = cardGrid.deletedUuids;
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
    const stale = cardGrid.staleUuids.size;

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

  // markCardStale moved to the cardGrid controller (createCardGrid). Local alias
  // keeps the many call sites below unchanged.
  const markCardStale = (card, state) => cardGrid.markCardStale(card, state);

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

  // updateTypeFading / clearSelection moved to the cardGrid controller. Local
  // aliases keep call sites below unchanged.
  const updateTypeFading = () => cardGrid.updateTypeFading();
  const clearSelection = () => cardGrid.clearSelection();

  // ── Card-builder wiring ───────────────────────────────────────────
  // Opens the canvas editor for an entry. Reproduces the old builders'
  // double-click handlers exactly. Image-initiated opens (from an image
  // card/row double-click) use the lightweight onSaved that only refreshes
  // bulk/status; bbox-initiated opens refresh/mark-stale the originating card.
  function onOpenEditor(entry) {
    const { bbox, img, imageInitiated } = entry;
    const navCtx = _buildNavContext(img.uuid);
    if (imageInitiated) {
      showCanvasModal(bbox, img, labels, async (_updatedUIBbox, _currentImg) => {
        updateBulkBtn();
        updateStatusMsg();
      }, navCtx);
      return;
    }
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
  }

  // Subscribe an image card's chip-refresher to label-change events, fetching
  // fresh labels on change. Pushes unsubs to _cardUnsubs (drained per render).
  function subscribeImageLabelChanges(imageUuid, imageFilename, renderImageLabelChips) {
    async function _onRelevantChange({ imageUuid: u } = {}) {
      if (u !== imageUuid) return;
      let labelMap;
      try { labelMap = await fetchImageLabels(imageFilename); } catch { return; }
      _pageImageLabels.set(imageUuid, labelMap);
      renderImageLabelChips(labelMap);
    }
    _cardUnsubs.push(
      events.subscribe(EventTypes.IMAGE_LABELS_CHANGED, _onRelevantChange),
      events.subscribe(EventTypes.IMAGE_ANNOTATIONS_SAVED, _onRelevantChange),
    );
  }

  // Assemble the opts object the cardGrid.js builders destructure. Sourced
  // from the closures above; `index` is the render index.
  function _cardOpts(labels, index) {
    return {
      gridEl: grid,
      labels,
      index,
      referenceBboxUuids,
      referenceImageUuids,
      hoverPreview,
      thumbLoader: _thumbLoader,
      makeAdapter: makeReviewBboxAdapter,
      resolveContext: _resolveContextObjects,
      onOpenEditor,
      registerImageRefresher: (uuid, fn) => _imageCardRefreshers.set(uuid, fn),
      getPageImageLabels: uuid => _pageImageLabels.get(uuid),
      subscribeImageLabelChanges,
    };
  }

  // ── Render filtered entries to grid ──────────────────────────────
  function renderFromEntries() {
    cardGrid.resetForRender();

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

  // ── Page/scroll status line (M2: text + Prev/Next differ by mode) ─────────
  function updatePageInfo() {
    const cardCount = _allEntries.length;
    if (fetchImages) {
      pageInfo.textContent = `${cardCount} card${cardCount === 1 ? '' : 's'}` +
        (_hasMore ? ' · scroll for more' : '');
    } else {
      const totalPages = _totalCount > 0 ? Math.ceil(_totalCount / _pageSize) : null;
      const pageLabel = totalPages ? `Page ${_pageCursors.length} of ${totalPages}` : `Page ${_pageCursors.length}`;
      pageInfo.textContent = `${pageLabel} · ${cardCount} card${cardCount === 1 ? '' : 's'}`;
    }
    prevBtn.disabled = fetchImages ? true : _pageCursors.length <= 1;
    nextBtn.disabled = fetchImages ? true : !_hasMore;
  }

  // Build image+bbox entries for one chunk of curated (fetchImages-mode)
  // candidates — shared by render()'s initial bootstrap and _loadMoreCurated()
  // so the two never drift on what counts as a "match".
  function _buildCuratedEntries(images) {
    const bboxMatches = makeBboxFilterPredicate({ fetchImagesMode: true, bboxFilterFn, fs: null, labels });
    const entries = [];
    for (const img of images) {
      const matching = (img.bboxes || []).filter(b => bboxMatches(b) && bboxMatchesFilter(b, _curated, labels));
      if (imageMatchesFilter(img, _curated, labels)) {
        entries.push({ kind: 'image', img });
        for (const bbox of matching) entries.push({ kind: 'bbox', bbox, img });
      }
    }
    return entries;
  }

  // ── Infinite scroll (M2, fetchImages mode only) ─────────────────────────────
  // `container` is the modal's own scroll parent (showReviewModal's `content`
  // div, overflow:auto). `_nearBottom` reads its live geometry directly rather
  // than through an IntersectionObserver + sentinel element — a plain 'scroll'
  // listener (wired below, near render()'s initial call) already fires
  // reliably on every scrollTop change, so there's no need for the observer's
  // separate (and edge-triggered, easy-to-race) notification cycle.
  function _nearBottom() {
    if (!fetchImages || !container.isConnected) return false;
    return container.scrollTop + container.clientHeight >= container.scrollHeight - SCROLL_LOAD_MARGIN_PX;
  }

  // A call that arrives while another is already running must not just be
  // dropped — scroll events fire in bursts, and by the time one lands the
  // page may already have moved further. `_loadMorePending` records "someone
  // asked again while I was busy" so the running call loops once more before
  // it actually exits, instead of assuming a later scroll event will retry it.
  async function _loadMoreCurated() {
    if (!fetchImages || _rendering) return;
    if (_loadingMore) { _loadMorePending = true; return; }
    _loadingMore = true;
    const _gen = _renderGeneration;
    try {
      do {
        _loadMorePending = false;
        while (!_curatedExhausted && _allEntries.length < MAX_ENTRIES && _nearBottom()) {
          const raw = await fetchImages({ afterUuid: _curatedCursor, afterSortValue: null, limit: _pageSize + 1 });
          if (_renderGeneration !== _gen) return; // superseded by a full render() mid-flight

          const exhausted = raw.length <= _pageSize;
          const pageImages = raw.slice(0, _pageSize);
          if (pageImages.length > 0) _curatedCursor = pageImages[pageImages.length - 1].uuid;
          _curatedExhausted = exhausted;
          _hasMore = !exhausted;
          if (pageImages.length === 0) break;

          let newEntries = _buildCuratedEntries(pageImages);
          if (_allEntries.length + newEntries.length > MAX_ENTRIES) {
            newEntries = newEntries.slice(0, MAX_ENTRIES - _allEntries.length);
            _hasMore = false;
            _curatedExhausted = true;
          }
          if (!newEntries.length) continue;

          _allEntries.push(...newEntries);

          const newImageUuids = [...new Set(newEntries.map(e => e.img.uuid))];
          try {
            const more = await fetchImageLabelsBatch(newImageUuids);
            for (const [uuid, statuses] of more) _pageImageLabels.set(uuid, statuses);
          } catch (err) {
            Logger.error('review: batch image-labels fetch (load-more) failed', err);
          }
          if (_renderGeneration !== _gen) return;

          const newVisible = newEntries.filter(e =>
            _viewFilter === 'all' || (_viewFilter === 'images' ? e.kind === 'image' : e.kind === 'bbox'));
          if (newVisible.length) {
            entryView.setEmptyText('');
            _visibleEntries.push(...newVisible);
            entryView.appendEntries(newVisible);
          }
          updateBulkBtn();
          updatePageInfo();
          updateStatusMsg();
        }
      } while (_loadMorePending && !_curatedExhausted && _allEntries.length < MAX_ENTRIES);
    } catch (err) {
      Logger.error('review: load-more (infinite scroll) failed', err);
    } finally {
      _loadingMore = false;
    }
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
      cardGrid.resetForRender();
      _allEntries = [];
      _visibleEntries = [];
      _lastCursor = null;
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
          // M2: infinite-scroll bootstrap. Fetch chunks (chunk size = _pageSize)
          // until the first screenful is filled or the candidate list is
          // exhausted; `_loadMoreCurated()` (wired to a scroll sentinel) takes
          // over from `_curatedCursor` as the user scrolls further. Entries are
          // built directly into `_allEntries` here via `_buildCuratedEntries`,
          // so `images` is left empty — the generic per-image entry-building
          // loop below is then a no-op for this branch.
          _curatedCursor = null;
          _curatedExhausted = false;
          while (_allEntries.length < _pageSize && !_curatedExhausted) {
            if (_renderGeneration !== _gen) { clearTimeout(_slowTimer); return; }
            const raw = await fetchImages({ afterUuid: _curatedCursor, afterSortValue: null, limit: _pageSize + 1 });
            // Re-check after the await before touching shared cursor/entry state:
            // a filter change or Refresh mid-fetch starts a newer render() that
            // has already reset _allEntries/_curatedCursor, and writing this
            // superseded chunk would corrupt it (mirrors _loadMoreCurated).
            if (_renderGeneration !== _gen) { clearTimeout(_slowTimer); return; }
            const exhausted = raw.length <= _pageSize;
            const pageImages = raw.slice(0, _pageSize);
            if (pageImages.length > 0) _curatedCursor = pageImages[pageImages.length - 1].uuid;
            _curatedExhausted = exhausted;
            if (pageImages.length === 0) break;
            _allEntries.push(..._buildCuratedEntries(pageImages));
          }
          hasMore = !_curatedExhausted;
          images = [];
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
            verificationCheck: _verificationCheckFor(fs),
            verificationLabelUuids: _verificationLabelUuidsFor(fs),
            bboxVerification: fs.bboxVerification,
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
      // Shared with runModelOnResults.js's candidate-scope resolution so both
      // agree on which bboxes "match" the current filter/bboxFilterFn.
      const bboxMatches = makeBboxFilterPredicate({ fetchImagesMode: !!fetchImages, bboxFilterFn, fs, labels });

      for (const img of images) {
        // In fetchImages/custom mode (M1), further narrow by the curated
        // result-filter predicate (Verification/Label/Tag) on top of whatever
        // the caller's bboxFilterFn already narrowed (e.g. a similarity modal
        // pre-narrows img.bboxes to the matched-similar set) — compose, don't
        // replace, bboxFilterFn.
        const matching = (img.bboxes || []).filter(b =>
          bboxMatches(b) && (!fetchImages || bboxMatchesFilter(b, _curated, labels)));
        // Filter mode: always add an image-level entry (unchanged behavior).
        // fetchImages mode: only add the image entry when it has at least one
        // curated-matching bbox — otherwise a non-matching image would leak
        // into images-view (previously every fetched image was shown
        // unconditionally, since there was no curated filter to gate on).
        if (!fetchImages || imageMatchesFilter(img, _curated, labels)) {
          _allEntries.push({ kind: 'image', img });
          for (const bbox of matching) {
            _allEntries.push({ kind: 'bbox', bbox, img });
          }
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

    // One batch request resolves whole-image label chips for every card on the
    // page (was: one GET per card — up to page-size parallel requests).
    try {
      const imageUuids = [...new Set(_allEntries.map(e => e.img.uuid))];
      _pageImageLabels = await fetchImageLabelsBatch(imageUuids);
    } catch (err) {
      Logger.error('review: batch image-labels fetch failed', err);
      _pageImageLabels = new Map();
    }
    if (_renderGeneration !== _gen) return;

    updatePageInfo();

    clearTimeout(_slowTimer);
    renderFromEntries();

    refreshBtn.disabled = false;
    _rendering = false;

    // M2: top up past the bootstrap chunk if the viewport isn't full yet.
    // Don't rely solely on the IntersectionObserver's own initial callback —
    // it can race `_rendering` above and, being edge-triggered, never retry
    // if that one callback is dropped (see _loadMoreCurated's doc comment).
    if (fetchImages) _loadMoreCurated();
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

  // ── Select all ────────────────────────────────────────────────────
  selectAllCb.addEventListener('change', () => {
    if (!selectAllCb.checked) {
      // Unchecked → clear everything
      clearSelection();
      cardGrid.lastClickedIndex = -1;
      return;
    }

    const _selected = cardGrid.selected;
    const _deletedUuids = cardGrid.deletedUuids;

    // Determine kind: use current selection's kind, or the kind of the first
    // visible non-deleted entry (so selecting nothing then hitting select-all
    // anchors to whatever type comes first on the page).
    const firstEntry = _visibleEntries.find(e => !_deletedUuids.has(entryId(e)));
    const targetKind = cardGrid.selectionKind ?? firstEntry?.kind ?? null;
    if (!targetKind) {
      selectAllCb.checked = false;
      return;
    }

    cardGrid.selectionKind = targetKind;
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

    cardGrid.lastClickedIndex = -1;
    updateBulkBtn();
    updateTypeFading();
  });

  // ── Bulk Apply ────────────────────────────────────────────────────
  bulkBtn.addEventListener('click', async () => {
    const _selected = cardGrid.selected;
    if (_selected.size === 0) return;
    const selectedEntries = _visibleEntries.filter(e => _selected.has(entryId(e)));

    if (cardGrid.selectionKind === 'image') {
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

    // M3: apply the curated Verification/Label/Tag predicate (same one
    // render() uses) so the count + first-3/last-3 preview below reflect the
    // *filtered* candidate set, not the raw unfiltered fetch.
    const curatedImages = images.filter(img => imageMatchesFilter(img, _curated, labels));
    const kind = cardGrid.selectionKind ?? (_viewFilter === 'images' ? 'image' : 'bbox');
    const entries = kind === 'image'
      ? curatedImages.map(img => ({ kind: 'image', img }))
      : curatedImages.flatMap(img =>
          (img.bboxes || [])
            .filter(b => (!bboxFilterFn || bboxFilterFn(b)) && bboxMatchesFilter(b, _curated, labels))
            .map(bbox => ({ kind: 'bbox', bbox, img })));

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

  // ── Infinite-scroll listener (M2, fetchImages mode only) ───────────────────
  // Only needs to *start* _loadMoreCurated on each scroll — the initial
  // viewport fill is handled explicitly by the `_loadMoreCurated()` call at
  // the end of render() above.
  if (fetchImages) {
    _scrollHandler = () => _loadMoreCurated();
    container.addEventListener('scroll', _scrollHandler, { passive: true });
  }

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
      { value: 'revertRelabel', label: '↩ Revert Filter-Chain Relabel' },
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
        : (selectedAction === 'addTags' || selectedAction === 'removeTags') ? tagCheckboxField.getValue()
        // revertRelabel: strip only the task-generated review tag(s) among the
        // current filter — never a system tag like UNVERIFIED that may also be
        // part of the active filter (e.g. filtering by review-tag + UNVERIFIED
        // together must not strip UNVERIFIED from every reverted bbox).
        : (selectedAction === 'revertRelabel')
          ? (fs.tagUuids ?? []).filter(uuid => tags.get(uuid)?.kind === 'task_generated')
        : [];
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
          verificationCheck: _verificationCheckFor(fs),
          verificationLabelUuids: _verificationLabelUuidsFor(fs),
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
        verificationCheck: _verificationCheckFor(fs),
        verificationLabelUuids: _verificationLabelUuidsFor(fs),
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
      filterPanel?.destroy();
      _thumbLoader.reset();
      hoverPreview.destroy();
      entryView.destroy();
      if (_scrollHandler) container.removeEventListener('scroll', _scrollHandler);
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
export function _mergeExternalBboxChanges(runtime, updatedBboxes, deletedBboxUuids, userChangedUuids) {
  const bboxes = runtime.state.image.bboxes;
  let anyChange = false;
  for (const [uuid, uiBbox] of (updatedBboxes ?? [])) {
    if (userChangedUuids.has(uuid)) continue;
    const existing = bboxes.get(uuid);
    if (!existing) continue; // bbox not present on this canvas — skip
    if (uiBbox.label !== undefined) {
      // Incoming bboxes are review-API shaped (label.name); canvas renderers
      // read label.text — normalize so merged chips don't render blank.
      existing.label = uiBbox.label === null
        ? null
        : { ...uiBbox.label, text: uiBbox.label.text ?? uiBbox.label.name };
    }
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
export async function showCanvasModal(bbox, img, labels, onSaved, navContext = null, editOpts = {}) {
  if (_modalOpen) return;
  _modalOpen = true;

  // ── Editor options (all defaulted to today's review-grid behavior) ──
  // • persist(runtime)   → DB save (saveAnnotations); preview injects an
  //                        in-memory persist that skips the DB.
  // • seedBboxes         → not-yet-persisted candidate boxes to inject after
  //                        loadImage (the bboxUuid:'new' case). Default: none.
  // • broadcast          → gate the IMAGE_ANNOTATIONS_SAVED / IMAGE_BBOXES_CHANGED
  //                        publishes so preview edits don't ripple into the live
  //                        grid. Default: true (publish exactly as today).
  // • allowNavigation    → wire the [ ] prev/next nav handlers. Default: true.
  const _persist = editOpts.persist ?? saveAnnotations;
  const _seedBboxes = editOpts.seedBboxes ?? null;
  const _broadcast = editOpts.broadcast ?? true;
  const _allowNavigation = editOpts.allowNavigation ?? true;

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

  // ── Right sidebar: tab header + panes ────────────────────────
  const MODAL_TABS = { labels: 'modal-tab-labels', inference: 'modal-tab-inference', tags: 'modal-tab-tags' };

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
    _mkTabBtn(MODAL_TABS.inference, 'Inference'),
    _mkTabBtn(MODAL_TABS.tags,   'Tags'),
  );

  const _mkTabPane = (id) => {
    const pane = document.createElement('div');
    pane.className = 'sidebar-tab-pane';
    pane.id = id;
    return pane;
  };
  const paneLabels    = _mkTabPane(MODAL_TABS.labels);
  const paneInference = _mkTabPane(MODAL_TABS.inference);
  const paneTags      = _mkTabPane(MODAL_TABS.tags);

  const labelListEl = document.createElement('div');
  paneLabels.appendChild(labelListEl);

  sidebarEl.append(sidebarTabHeader, paneLabels, paneInference, paneTags);

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
  let inferencePanelHandle = { destroy: () => {} };
  let modalResizeHandle = { destroy: () => {} };
  let unsubToolChanged = () => {};
  let unsubDblClick    = () => {};
  let unsubBboxChanged = () => {};
  let unsubInference   = () => {};

  try {
    instance = await createCanvasInstance(canvasWrap, { tool: 'select' });
    await instance.loadImage(img.filename);
    // Inject not-yet-persisted candidate boxes (preview bboxUuid:'new' case).
    // No-op when absent, so the default review-grid path is untouched.
    if (_seedBboxes?.length) seedRuntimeBboxes(instance.runtime, _seedBboxes);
  } catch (err) {
    Logger.error('Canvas modal: failed to load image', err);
    instance?.destroy();
    overlay.remove();
    _modalOpen = false;
    return;
  }

  // ── Wire shared toolbar (select / bbox / delete) ───────────────────
  toolbarHandle = initToolbar(instance.runtime, toolbarEl, {
    buttons: ['undo', 'redo', 'select', 'bbox', 'clear', 'delete'],
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

  // The first loadImage() above measured canvasWrap before the toolbar
  // (populated by initToolbar, just before this) and the sidebar (resized
  // to its persisted width by initSplitResize, just above) reached their
  // final layout size, so its fit-to-view zoom can be wrong. Re-fit now
  // that the modal chrome has settled.
  resetCanvasZoom(instance.runtime.canvas);

  // ── Wire label list ────────────────────────────────────────────────
  labelHandle = await createLabelList(labelListEl, { mode: 'select', runtime: instance.runtime });

  // ── Wire tag panel (class + tag toggles, responds to bbox clicks) ──
  tagPanelHandle = createTagPanel(paneTags, instance.runtime);

  // ── Wire inference panel (model list, Run/Refresh/Clear-Grey, confidence) ──
  // Use a child wrapper, not paneInference itself: createInferencePanel adds the
  // `.inference-panel` class (display:flex, unconditional) to its container, which
  // would collide with `.sidebar-tab-pane`'s display:none-unless-.active toggling.
  const inferencePanelContentEl = document.createElement('div');
  paneInference.appendChild(inferencePanelContentEl);
  inferencePanelHandle = await createInferencePanel(instance.runtime, inferencePanelContentEl);
  unsubInference = events.subscribeToCanvas(
    instance.runtime.id, EventTypes.RUN_INFERENCE_ON_CANVAS,
    async ({ modelName, confThresh, replaceExisting }) => {
      try {
        const image = instance.runtime.state?.image?.img;
        if (!image) throw new Error('No image loaded');
        const result = await runDetection(modelName, image, confThresh);
        await addInferenceResults(instance.runtime, result, { replaceExisting });
      } catch (err) {
        Logger.error('Canvas modal: failed to run inference', err);
      }
    },
  );

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
  if (_allowNavigation && navContext?.items?.length) {
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
    unsubInference();
    tagPanelHandle.destroy();
    inferencePanelHandle.destroy();
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
    if (_broadcast) {
      events.publish(EventTypes.IMAGE_ANNOTATIONS_SAVED, {
        imageUuid: currentImg?.uuid,
        imageFilename: currentImg?.filename ?? currentImg?.image_path,
        bboxes: currentBboxesMap,
      });
    }
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
        const saved = await _persist(instance.runtime);
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
      saved = await _persist(instance.runtime);
    } catch (err) {
      Logger.error('Canvas modal: save failed', err);
    }

    _setAllButtonsDisabled(false);
    saveOnlyBtn.textContent = 'Save';
    _updateSaveBtn();

    if (saved && _broadcast) {
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
      saved = await _persist(instance.runtime);
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

// TAG_ACTIONS / showImageBulkActionModal / executeBulkImageAction moved to
// imageBulkActionModal.js so they can be reused outside the review grid's own
// toolbar (e.g. runClassifierOnResults.js's classifier-match confirmation step).
