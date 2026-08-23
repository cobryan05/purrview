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
} from '/app-static/js/app/review/cardGrid.js';
import { expandWithDescendants } from '/app-static/js/ui/labelList.js';
import {
  fetchReviewQueue,
  bulkApplyAll,
  bulkApplyBboxesByUuids,
  bulkApplyImagesByUuids,
  bulkApplyImages,
  countMatchingBboxes,
  updateImageBboxes,
  updateBbox,
  deleteBbox,
  getBboxThumbnailUrl,
  getImageThumbnailUrl,
  invalidateBboxThumbnails,
  UNVERIFIED_TAG_UUID,
  VERIFIED_TAG_UUID,
  REJECTION_TAG_UUID,
  VERIFICATION_TAG_UUIDS,
  swapVerificationTag,
} from '/app-static/js/shared/api/review.js';
import { fetchImageLabels, fetchImageLabelsBatch, setImageLabel } from '/app-static/js/shared/api/imageLabels.js';
import { activeTrainingConfig } from '/app-static/js/shared/activeTrainingConfig.js';
import { fetchTrainingConfigImageGaps } from '/app-static/js/shared/api/trainingConfigs.js';
import { setImageTag, setImageTagStatus } from '/app-static/js/shared/api/imageTags.js';
import { attachContext } from '/app-static/js/ui/contextMenu/index.js';
import { getPrimaryRuntimeId } from '/app-static/js/app/image-tagging/runtime.js';
import { Logger, toast } from '/app-static/js/ui/utils/logging.js';
import { showUnsavedChangesModal, showExternalChangesModal, showConfirmModal } from '/app-static/js/ui/utils/confirmModal.js';
import { deleteImage, prefetchImage } from '/app-static/js/shared/api/images.js';
import { rerenderBboxViews, resetCanvasZoom } from '/app-static/js/app/image-tagging/canvas/manager.js';
import { acquireModalLayer, applyModalCascade, registerEscHandler, unregisterEscHandler } from '/app-static/js/ui/utils/modalZIndex.js';
import { bindOutsideDismiss } from '/app-static/js/ui/utils/anchoredPopup.js';
import { trapModalFocus } from '/app-static/js/ui/utils/modalShell.js';
import { attachScrollChrome } from '/app-static/js/app/utils/scrollChrome.js';
import { MultiSelectField } from '/app-static/js/ui/utils/fields/MultiSelectField.js';
import { createCanvasInstance } from '/app-static/js/app/image-tagging/canvas/instance.js';
import { selectBboxView } from '/app-static/js/app/image-tagging/canvas/interaction.js';
import { next_image, prev_image } from '/app-static/js/app/image-tagging/files.js';
import { saveAnnotations, seedRuntimeBboxes } from '/app-static/js/app/image-tagging/canvas/image.js';
import { registerCanvasAction } from '/app-static/js/app/image-tagging/canvasActions.js';
import { createInferencePanel } from '/app-static/js/app/image-tagging/inferencePanel.js';
import {
  wireInferenceRunner,
  createSidebarPanels,
  wireSidebarTabRules,
} from '/app-static/js/app/image-tagging/canvasWiring.js';
import { initSplitResize } from '/app-static/js/ui/utils/splitResize.js';
import { events, EventTypes } from '/app-static/js/shared/events/index.js';
import { init as initToolbar } from '/app-static/js/app/image-tagging/toolbar.js';
import { setupSidebarTabs } from '/app-static/js/app/utils/tabs.js';
import { createHoverPreview } from '/app-static/js/ui/thumbnailHoverPreview.js';
import { drawLetterboxedBboxes } from '/app-static/js/ui/utils/bboxThumbOverlay.js';
import { createEntryView } from '/app-static/js/ui/entryView.js';
import { reviewStateToDatasetForm, datasetFormToReviewState, reviewStateToFilterOpts } from '/app-static/js/shared/filterUtils.js';
import { fetchViews, createDataset, updateDataset } from '/app-static/js/shared/api/datasets.js';
import { openDissimilarGlobalModal } from '/app-static/js/ui/dissimilarSearch.js';
import { runModelOnResultsFlow, makeBboxFilterPredicate } from '/app-static/js/app/review/runModelOnResults.js';
import { runFilterChainOnResultsFlow } from '/app-static/js/app/review/runFilterChainOnResults.js';
import { showImageBulkActionModal, executeBulkImageAction, isRemovalAction, confirmRemovalAction, countImagesWithVerifiedWork, HIGH_RISK_REMOVAL_COUNT, removeSingleImage } from '/app-static/js/app/review/imageBulkActionModal.js';
import { createBboxVerificationControl } from '/app-static/js/app/review/bboxVerificationControl.js';
import { createConfidenceRangeControl } from '/app-static/js/app/review/confidenceRangeControl.js';
import { defaultCuratedFilterState, bboxMatchesFilter, imageMatchesFilter } from '/app-static/js/app/review/filterPredicate.js';
import { createMultiSelectDropdown } from '/app-static/js/ui/utils/index.js';
import { flattenHierarchy } from '/app-static/js/ui/labelList.js';

const DEFAULT_PAGE_SIZE = 250;
const MAX_ENTRIES = 1000; // hard cap on total cards per page
const SCROLL_LOAD_MARGIN_PX = 400; // infinite-scroll (M2) preload distance from the bottom of the scroll container

// How far around the canvas modal's current position to prefetch images, as
// offsets from _navIndex. Two each way rather than one so a second keypress
// arriving before the first load finishes still lands warm; kept small because
// each entry pins a decoded full-resolution image (see IMAGE_CACHE_MAX, which
// must stay >= this window plus the current image or the window evicts itself).
const NAV_PREFETCH_OFFSETS = [1, -1, 2, -2];

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
export async function createReviewPanel(container, { fetchImages = null, datasetActions = true, referenceBboxUuids = new Set(), referenceImageUuids = new Set(), initialViewFilter = 'images', toolbarControls = null, bboxFilterFn = null, gapsConfig = null } = {}) {
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
  toolbar.dataset.testid = 'review-toolbar';

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
  // Slice 3, PR 3.2: no longer a "☰ Filters (n)" toggle for a hidden inline
  // panel — the chip bar below is always visible and shows what's active, so
  // this only opens the popover editor.
  //
  // That popover is now the ONE place every filter lives. The Tags and Labels
  // dropdowns used to be spliced into the toolbar below (filterPanel's old
  // `toolbarControls`), which meant label/tag filtering was split across two
  // surfaces with nothing to say which held what — and this button's tooltip
  // listed only some of what was actually behind it. Keep the tooltip in step
  // with filterPanel.element if a dimension is added.
  const filterToggleBtn = document.createElement('button');
  filterToggleBtn.className = 'review-toolbar__btn review-filter-chip-bar__add';
  filterToggleBtn.dataset.testid = 'review-filter-toggle';
  filterToggleBtn.textContent = '+ Add filter';
  filterToggleBtn.title = 'Tags, labels, source, date, filename, confidence and verification filters';
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

  // Apply to Selected button (checked items on current page)
  const bulkBtn = document.createElement('button');
  bulkBtn.className = 'review-toolbar__btn';
  bulkBtn.dataset.testid = 'review-bulk-apply';
  bulkBtn.textContent = 'Apply to Selected (0)';
  bulkBtn.disabled = true;

  // Apply to Parent Images button — visible only when the current selection is
  // bbox-kind. Apply to Selected (above) can only offer bbox-level actions (relabel/
  // delete/tags) for a bbox selection; this resolves the DISTINCT parent images
  // behind the selected bboxes and routes them into the image-level bulk-action
  // flow instead (e.g. filter by a bbox label, then mark the image-level label
  // for a *different* object confirmed-absent across those images).
  const bulkParentImagesBtn = document.createElement('button');
  bulkParentImagesBtn.className = 'review-toolbar__btn';
  bulkParentImagesBtn.dataset.testid = 'review-bulk-apply-parent';
  bulkParentImagesBtn.textContent = 'Apply to Parent Images';
  bulkParentImagesBtn.disabled = true;
  bulkParentImagesBtn.hidden = true;

  // Bulk Apply All button — all matching results across all pages (filter mode),
  // or the full candidate list (custom fetchImages mode, e.g. similarity search).
  const bulkAllBtn = document.createElement('button');
  bulkAllBtn.className = 'review-toolbar__btn';
  bulkAllBtn.textContent = 'Apply to All';
  bulkAllBtn.dataset.testid = 'review-bulk-apply-all';
  bulkAllBtn.disabled = true;

  // Apply to All Matching Images — filter-mode only sibling of bulkAllBtn.
  // bulkAllBtn (→ showBulkAllPagesModal) only ever offers bbox-level actions,
  // no matter which view is active. This uses the new filter-based
  // bulkApplyImages endpoint to apply an image-level action (setPresent/
  // setAbsent/tags/delete) to every image matching the CURRENT filter, across
  // all pages, without downloading the candidate set client-side. Not offered
  // in custom/fetchImages (similarity) mode — there is no ImageSearchParams
  // filter to reuse there; applyToAllCandidates's uuid-list path covers that
  // mode instead.
  const bulkAllImagesBtn = document.createElement('button');
  bulkAllImagesBtn.className = 'review-toolbar__btn';
  bulkAllImagesBtn.textContent = 'Apply to Filtered Images';
  bulkAllImagesBtn.dataset.testid = 'review-bulk-apply-filtered-images';
  bulkAllImagesBtn.hidden = !!fetchImages;

  // Run Model on Results — proposes relabels for the current selection/filter
  // scope by running a detector and matching detections to existing bboxes
  // by IoU (see app/review/runModelOnResults.js). Enabled in both full-filter
  // and custom-fetchImages modes; scope resolution happens on click.
  const runModelBtn = document.createElement('button');
  runModelBtn.className = 'review-toolbar__btn';
  runModelBtn.dataset.testid = 'review-run-model-on-results';
  runModelBtn.textContent = 'Run Model on Results…';
  runModelBtn.addEventListener('click', async () => {
    // Scope resolution (counting the filter/selection before the dialog can
    // even open) can take a moment on a large review set — otherwise the
    // button just sits there with no feedback until the dialog appears.
    const label = runModelBtn.textContent;
    runModelBtn.disabled = true;
    runModelBtn.textContent = 'Run Model on Results… (loading)';
    try {
      await runModelOnResultsFlow({
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
      });
    } finally {
      runModelBtn.disabled = false;
      runModelBtn.textContent = label;
    }
  });

  // Saved Previews — every persisted "Run Model on Results" run, so a large
  // triage can be picked up later instead of having to finish in one sitting.
  //
  // Review tab only (`!fetchImages`): saved runs are a property of the whole
  // project, not of the transient candidate set a similarity/search modal is
  // showing, and the run-model flow inside a modal opens its own run's panel
  // directly anyway. Building it in modal mode would put a second
  // "Saved **Prev**iews" button in the toolbar the pagination controls live in.
  const savedPreviewsBtn = document.createElement('button');
  savedPreviewsBtn.className = 'review-toolbar__btn';
  savedPreviewsBtn.dataset.testid = 'review-saved-previews';
  savedPreviewsBtn.textContent = 'Saved Previews';
  savedPreviewsBtn.addEventListener('click', async () => {
    const { openSavedPreviewsModal } = await import('/app-static/js/app/review/previewPanel.js');
    await openSavedPreviewsModal({
      labels,
      onApplied: () => render(),
      onClose: () => _refreshSavedPreviewCount(),
    });
  });

  async function _refreshSavedPreviewCount() {
    try {
      const { fetchPreviewRuns } = await import('/app-static/js/shared/api/previewRuns.js');
      const runs = await fetchPreviewRuns();
      savedPreviewsBtn.textContent = runs.length > 0 ? `Saved Previews (${runs.length})` : 'Saved Previews';
    } catch {
      // A count nobody asked for is not worth a toast; the button still opens
      // the list, which reports the failure properly.
      savedPreviewsBtn.textContent = 'Saved Previews';
    }
  }
  if (!fetchImages) _refreshSavedPreviewCount();

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

  // Refresh — in the action bar, immediately right of the status line ("N images
  // on this page"). That line is what tells you the page is out of date, so the
  // button that reloads it belongs against it rather than tucked behind Next.
  // It carries the accent variant: every other button in these bars is a muted
  // outline, and Refresh is the one you reach for repeatedly.
  const refreshBtn = document.createElement('button');
  refreshBtn.className = 'review-toolbar__btn review-toolbar__btn--accent';
  refreshBtn.dataset.testid = 'review-refresh';
  refreshBtn.textContent = '⟳ Refresh';
  refreshBtn.title = 'Reload the current page';

  // ── Persistent utility group (slice 3, PR 3.1) ─────────────────────
  // "Most Unique First", "Open in Datasets Tab", and any
  // caller-supplied controls (e.g. a similarity-search threshold slider)
  // read as one "toolbar" cluster instead of being scattered across
  // separately-appended groups. Created once and appended to `toolbar` in
  // its final position below; the buttons built inside later conditional
  // blocks (uniqueFirstBtn, exportToDatasetBtn) are appended into this same
  // node rather than creating new sibling groups, so this group's DOM
  // position never has to be revisited.
  const persistentUtilityGroup = makeToolbarGroup();
  persistentUtilityGroup.classList.add('review-toolbar__group--persistent');
  if (toolbarControls) persistentUtilityGroup.appendChild(toolbarControls);

  // Pagination — filter mode only. Custom/fetchImages mode uses infinite
  // scroll instead (M2), so Prev/Next never apply there.
  const pageInfo = document.createElement('span');
  pageInfo.className = 'review-toolbar__page-info';
  const prevBtn = document.createElement('button');
  prevBtn.className = 'review-toolbar__btn';
  prevBtn.dataset.testid = 'review-page-prev';
  prevBtn.textContent = '‹ Prev';
  const nextBtn = document.createElement('button');
  nextBtn.className = 'review-toolbar__btn';
  nextBtn.dataset.testid = 'review-page-next';
  nextBtn.textContent = 'Next ›';
  if (fetchImages) { prevBtn.hidden = true; nextBtn.hidden = true; }

  // Status message
  const statusMsg = document.createElement('span');
  statusMsg.className = 'review-toolbar__status';
  statusMsg.dataset.testid = 'review-status';

  /**
   * Flash the status message for a bulk apply that the server does in one
   * request (bulk-apply / bulk-apply-by-uuids and their bbox counterparts).
   *
   * The tab is only waiting for the reply — the handler runs to completion
   * whether or not anyone is watching, so nobody has to sit in front of a
   * pulsing "Applying…" for a run over tens of thousands of images. Say so.
   * Only use this for paths that are a single request; the per-item loops
   * (image removal, bbox selection apply) really do need the tab open, since
   * the remaining calls are only ever issued from here.
   */
  function setServerSideApplyStatus(text) {
    statusMsg.classList.add('review-toolbar__status--loading');
    statusMsg.textContent = `${text} (running on the server — you can leave this page)`;
  }

  /**
   * The counterpart of setServerSideApplyStatus for a run this tab drives
   * itself, one request per item (bbox selection apply, image removal). Nobody
   * else can finish it, so the message asks the user to wait and counts up
   * instead of pulsing an opaque "Applying…".
   */
  function setClientSideApplyStatus(noun, done, total) {
    statusMsg.classList.add('review-toolbar__status--loading');
    statusMsg.textContent =
      `Applying to ${done.toLocaleString()} of ${total.toLocaleString()} ${noun}… ` +
      'please wait — this one runs from this browser tab';
  }

  /**
   * Ask the browser to confirm a close/reload while a per-item run is in
   * flight. The unsent items exist nowhere but in this tab, so leaving abandons
   * the run part-applied — unlike the server-side bulk endpoints, which finish
   * on their own and therefore get no guard.
   *
   * @returns {() => void} removes the guard; always call it from a `finally`.
   */
  function guardUnloadDuringRun() {
    const onBeforeUnload = (e) => { e.preventDefault(); e.returnValue = ''; };
    window.addEventListener('beforeunload', onBeforeUnload);
    return () => window.removeEventListener('beforeunload', onBeforeUnload);
  }

  /**
   * Aborted when the panel is torn down (tab closed, similarity modal closed,
   * grid re-created). Per-item loops check it between items so a run whose UI
   * no longer exists stops at the next boundary rather than writing on into a
   * dead panel — the work already committed stands, the rest is simply not sent.
   */
  const _panelLifetime = new AbortController();

  /** Note a run that stopped early, appended to whatever status lands after it. */
  function appendStoppedEarlyNote(done, total, noun) {
    statusMsg.textContent += ` (stopped after ${done.toLocaleString()} of ${total.toLocaleString()} ${noun}` +
      ' — the rest were not applied)';
  }

  // ── Entry view (grid only) ────────────────────────────────────────
  // Review results are thumbnails first — the list rendering carried no
  // information the cards don't, so the toggle and the row renderers are gone.
  const entryView = createEntryView({
    storageKey: 'review-view',
    defaultView: 'grid',
    showToggle: false,
    gridClass: 'review-grid',
    renderCell: (entry, i) => entry.kind === 'image'
      ? buildImageCard(cardGrid, entry, _cardOpts(labels, i))
      : buildBboxCard(cardGrid, entry, _cardOpts(labels, i)),
    // Cards subscribe to EventBus on build; drain before every rebuild or a
    // re-render doubles the subscriptions.
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

  // ── Query bar: filter · sort · view mode · page size ──────────────
  // Controls that change *what* is shown. Stays compact so it never wraps.
  // Refresh used to be the one thing this group always held. Now that it has
  // moved to the action bar the group can end up empty (modal/preview mode,
  // where neither Most Unique First nor Open in Datasets Tab is built), which
  // would leave a separator with nothing after it — so keep a reference and
  // drop both once the conditional buttons below have had their say.
  const persistentUtilitySep = makeToolbarSep();
  toolbar.append(
    makeToolbarGroup(sortLabel, sortBySelect, sortOrderSelect),
    makeToolbarSep(),
    makeToolbarGroup(viewToggle, pageSizeSelect),
    persistentUtilitySep,
    persistentUtilityGroup,
  );

  // ── Action bar: select-all, prev/next, status (always visible) ────
  const actionBar = document.createElement('div');
  actionBar.className = 'review-action-bar';
  actionBar.dataset.testid = 'review-action-bar';
  actionBar.append(
    makeToolbarGroup(selectAllLabel),
    makeToolbarGroup(prevBtn, pageInfo, nextBtn),
    makeToolbarGroup(statusMsg, refreshBtn),
  );

  // ── Filter-scoped action bar (slice 3, PR 3.3) ─────────────────────
  // Apply to Filtered Images / Apply to All / Run Model on Results… / Run
  // Filter Chain on Results… all operate on the current filter or candidate
  // scope, NOT a checked selection — none of them are disabled or hidden by
  // selection state today (confirmed: bulkAllBtn gates on _totalCount,
  // runModelBtn/runChainBtn are never gated at all; a curated-mode Playwright
  // test clicks "Apply to All" with nothing selected). They must stay
  // reachable at zero selection, so — despite UI_REDESIGN §32.6's original
  // table putting them in the selection-gated contextual bar's ⋯ overflow —
  // they live here instead, always visible. Apply to Selected / Apply to Parent
  // Images are the only genuinely selection-dependent actions; see the
  // contextual action bar below.
  const filterScopedActionsBar = document.createElement('div');
  filterScopedActionsBar.className = 'review-filter-scoped-actions';
  filterScopedActionsBar.dataset.testid = 'review-filter-scoped-actions';
  filterScopedActionsBar.append(bulkAllImagesBtn, bulkAllBtn, runModelBtn, runChainBtn);
  if (!fetchImages) filterScopedActionsBar.insertBefore(savedPreviewsBtn, runChainBtn);

  // ── Contextual action bar (slice 3, PR 3.3) ────────────────────────
  // Appears only with an active selection — hidden the rest of the time.
  // Apply to Selected always shows; Apply to Parent Images additionally requires a
  // bbox-kind selection (unchanged condition from updateBulkBtn below).
  const clearSelectionBtn = document.createElement('button');
  clearSelectionBtn.className = 'review-toolbar__btn';
  clearSelectionBtn.dataset.testid = 'review-clear-selection';
  clearSelectionBtn.textContent = 'Clear selection';
  clearSelectionBtn.addEventListener('click', () => {
    clearSelection();
    updateBulkBtn();
  });

  const contextualActionBar = document.createElement('div');
  contextualActionBar.className = 'review-contextual-action-bar';
  contextualActionBar.dataset.testid = 'review-contextual-action-bar';
  contextualActionBar.hidden = true;
  const contextualRightGroup = makeToolbarGroup(clearSelectionBtn);
  contextualRightGroup.classList.add('review-contextual-action-bar__right');
  contextualActionBar.append(bulkBtn, bulkParentImagesBtn, contextualRightGroup);

  // ── Filter chip bar (slice 3, PR 3.2) ──────────────────────────────
  // Always-visible summary of active filters, replacing the old "☰ Filters
  // (n)" toggle-to-reveal-a-hidden-panel pattern. filterPanel.element (Date /
  // Filename / Config Ready / Bbox Verification / Image Label Filters) is
  // repositioned as a popover anchored under this bar instead of an inline
  // block that pushed the grid down — see .review-filter-popover in app.css.
  // Full-filter mode only; fetchImages/curated mode has no filter panel.
  const filterChipBarWrap = document.createElement('div');
  filterChipBarWrap.className = 'review-filter-chip-bar-wrap';
  const filterChipBar = document.createElement('div');
  filterChipBar.className = 'review-filter-chip-bar';
  filterChipBar.dataset.testid = 'review-filter-chip-bar';
  filterChipBarWrap.appendChild(filterChipBar);

  // ── Sticky top wrapper ────────────────────────────────────────────
  // Pins toolbar + chip bar + action bars to the top of the scroll container.
  const stickyTop = document.createElement('div');
  stickyTop.className = 'review-sticky-top';
  stickyTop.append(toolbar);
  if (!fetchImages) stickyTop.append(filterChipBarWrap);
  stickyTop.append(filterScopedActionsBar, actionBar, contextualActionBar);
  container.appendChild(stickyTop);
  container.appendChild(entryView.layout);

  // In the tab context the page scrolls and app/main.js's page-level
  // scrollChrome already hides these bars on scroll-down. In a modal the
  // container *is* the scroller, so it needs its own listener — same behaviour,
  // scoped to this container. Detected from computed overflow rather than a
  // flag so any future scrolling host gets it for free.
  const _containerScrolls = ['auto', 'scroll'].includes(getComputedStyle(container).overflowY);
  const _scrollChrome = _containerScrolls ? attachScrollChrome(container) : null;

  // ── Filter panel (full mode only) ─────────────────────────────────
  // Lives inside filterChipBarWrap as a popover; opened by "+ Add filter".
  let filterPanel = null;
  if (!fetchImages) {
    const tags = await tagsPromise;

    function renderChipBar() {
      filterToggleBtn.classList.toggle('filter-active', !filterPanel.isDefault());
      filterChipBar.querySelectorAll('.review-filter-chip').forEach(el => el.remove());
      for (const { key, label, onRemove } of filterPanel.getChipDescriptors()) {
        const chip = document.createElement('span');
        chip.className = 'review-filter-chip';
        chip.dataset.chipKey = key;
        const text = document.createElement('span');
        text.className = 'review-filter-chip__label';
        text.textContent = label;
        const remove = document.createElement('button');
        remove.type = 'button';
        remove.className = 'review-filter-chip__remove';
        remove.textContent = '✕';
        remove.title = 'Remove filter';
        remove.addEventListener('click', onRemove);
        chip.append(text, remove);
        filterChipBar.insertBefore(chip, filterToggleBtn);
      }
    }

    filterPanel = createFilterPanel(labels, tags, () => {
      renderChipBar();
      _pageCursors = [null];
      _lastCursor = null;
      render();
    }, {
      // Names only — redraw the chips without re-running the image query.
      onNamesChanged: () => renderChipBar(),
    });
    filterPanel.element.hidden = true;
    filterPanel.element.classList.add('review-filter-popover');
    filterChipBarWrap.appendChild(filterPanel.element);
    filterChipBar.appendChild(filterToggleBtn);
    renderChipBar(); // reflect initial state (Unverified checked → shows a chip)
    // Config Ready (V2) is whole-image-only — start disabled if we're opening
    // straight into bboxes view. Kept in sync by the view-toggle handler below.
    filterPanel.setConfigReadyAvailable(initialViewFilter !== 'bboxes');

    // ── Popover open/close ──────────────────────────────────────────────────
    // Clicking "+ Add filter" a second time was the ONLY way to dismiss this,
    // which nothing on screen suggested. It now also closes on ESC, on an
    // outside click, and via its own ✕ button. Dismiss wiring is shared with
    // anchoredPopup.js (bindOutsideDismiss) rather than hand-rolled here — this
    // panel keeps its own CSS-anchored positioning, so it can't use
    // openAnchoredPopup itself, but it must not drift from the shared contract.
    let _unbindFilterDismiss = null;

    function setFilterPopoverOpen(open) {
      if (open === !filterPanel.element.hidden) return;  // already in that state
      filterPanel.element.hidden = !open;
      filterToggleBtn.setAttribute('aria-expanded', String(open));
      if (open) {
        _unbindFilterDismiss = bindOutsideDismiss(filterPanel.element, filterToggleBtn, () =>
          setFilterPopoverOpen(false),
        );
      } else {
        _unbindFilterDismiss?.();
        _unbindFilterDismiss = null;
        // ESC/outside-click can fire while focus sits on a control inside the
        // now-hidden panel, which would strand focus on an unreachable element.
        filterToggleBtn.focus();
      }
    }

    filterToggleBtn.setAttribute('aria-haspopup', 'dialog');
    filterToggleBtn.setAttribute('aria-expanded', 'false');
    filterPanel.element.setAttribute('role', 'dialog');
    filterPanel.element.setAttribute('aria-label', 'Add filter');

    // Explicit close affordance — the popover is a large panel, and the only
    // previous exit was the button that opened it, well away from where the
    // pointer ends up after editing a filter.
    const filterPopoverClose = document.createElement('button');
    filterPopoverClose.type = 'button';
    filterPopoverClose.className = 'review-filter-popover__close';
    filterPopoverClose.dataset.testid = 'review-filter-popover-close';
    filterPopoverClose.textContent = '✕';
    filterPopoverClose.title = 'Close (Esc)';
    filterPopoverClose.setAttribute('aria-label', 'Close filter panel');
    filterPopoverClose.addEventListener('click', () => setFilterPopoverOpen(false));
    filterPanel.element.insertBefore(filterPopoverClose, filterPanel.element.firstChild);

    filterToggleBtn.addEventListener('click', () => {
      setFilterPopoverOpen(filterPanel.element.hidden);
    });

    // The Tags + Labels dropdowns used to be spliced in here as the toolbar's
    // first group. They now sit in the filter popover with every other filter
    // (see createFilterPanel's Labels & Tags row) — the toolbar is for view,
    // sort and actions, not for a subset of the filters.

    // ── Views (server-backed saved searches) + dataset export (hidden in
    // modal/preview contexts) — UI_REDESIGN_PLAN.md slice 5, PR 5.2. Replaces
    // the old per-browser `localStorage['review-filter-presets']` store: a
    // "preset" was already exactly a View (name + full filter state), just
    // stored client-side. Views are unpromoted rows in the `datasets` table
    // (decision register #10) — createDataset/updateDataset default to
    // `promoted: true`, so every write here passes `promoted: false` explicitly.
    if (datasetActions) {
    let _viewsByName = new Map();

    const viewsSelect = document.createElement('select');
    viewsSelect.className = 'filter-views-select';
    viewsSelect.title = 'Apply a saved View';
    viewsSelect.dataset.testid = 'review-views-select';

    // Fetching Views must never block the rest of toolbar setup below (Unique
    // First, Open in Datasets Tab) — an unhandled rejection here previously
    // aborted the whole synchronous setup function partway through, silently
    // dropping every control appended after it.
    async function _rebuildViewsDropdown() {
      let viewsMap;
      try {
        viewsMap = await fetchViews();
      } catch (err) {
        toast(err.message || 'Failed to load Views', 5000, 'error');
        return;
      }
      _viewsByName = new Map([...viewsMap.values()].map(v => [v.name, v]));
      viewsSelect.innerHTML = '';
      const ph = document.createElement('option');
      ph.value = '';
      ph.textContent = '— Views —';
      viewsSelect.appendChild(ph);
      for (const name of _viewsByName.keys()) {
        const opt = document.createElement('option');
        opt.value = name;
        opt.textContent = name;
        viewsSelect.appendChild(opt);
      }
      viewsSelect.value = '';
    }
    _rebuildViewsDropdown(); // fire-and-forget; toolbar setup continues synchronously below

    viewsSelect.addEventListener('change', () => {
      const name = viewsSelect.value;
      if (!name) return;
      const view = _viewsByName.get(name);
      if (view) filterPanel.setState(datasetFormToReviewState(view.filterJson, view.labelUuids));
      viewsSelect.value = '';
    });

    const saveAsViewBtn = document.createElement('button');
    saveAsViewBtn.className = 'review-toolbar__btn';
    saveAsViewBtn.textContent = '⊙ Save as View';
    saveAsViewBtn.title = 'Save current filters as a named View';
    saveAsViewBtn.dataset.testid = 'review-save-as-view';
    saveAsViewBtn.addEventListener('click', async () => {
      const name = prompt('View name:');
      if (!name?.trim()) return;
      const trimmed = name.trim();
      const { filterJson, labelUuids } = reviewStateToDatasetForm(filterPanel.getState());
      try {
        const existing = _viewsByName.get(trimmed);
        if (existing) {
          await updateDataset(existing.uuid, { filterJson, labelUuids });
        } else {
          await createDataset({ name: trimmed, filterJson, labelUuids, promoted: false });
        }
        await _rebuildViewsDropdown();
        toast(`View "${trimmed}" saved`, 3000);
      } catch (err) {
        toast(err.message || 'Failed to save View', 5000, 'error');
      }
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

    toolbar.append(makeToolbarSep(), makeToolbarGroup(viewsSelect, saveAsViewBtn));
    // Joins the persistent utility group (Refresh, ★ Most Unique First) rather
    // than its own toolbar row — see the group's definition above.
    persistentUtilityGroup.appendChild(exportToDatasetBtn);

    // ── "Label most unique first" — diversity-ranked unverified queue ────────
    const uniqueFirstBtn = document.createElement('button');
    uniqueFirstBtn.className = 'review-toolbar__btn';
    uniqueFirstBtn.textContent = '★ Most Unique First';
    uniqueFirstBtn.title = 'Open the most visually distinct images in a review panel (farthest-point sampling). '
      + 'Scoped to the current filters, or to unverified images when no filter is active.';
    // With no filter active the server keeps its own unverified-only scope; once
    // the user has filtered, the walk must cover exactly what the grid shows, so
    // the filtered UUID set is pushed down as the candidate pool.
    uniqueFirstBtn.addEventListener('click', async () => {
      let candidateUuids = null;
      if (!filterPanel.isDefault()) {
        // Resolving the filtered set is a full unpaged review query, so the modal
        // can be a second or more away on a large filter — disable the button and
        // say so rather than leaving the click looking like it did nothing.
        const label = uniqueFirstBtn.textContent;
        uniqueFirstBtn.disabled = true;
        uniqueFirstBtn.textContent = '★ Loading…';
        try {
          candidateUuids = await _getFilteredImageUuids();
        } catch (err) {
          toast(err.message || 'Failed to read current filters', 5000, 'error');
          return;
        } finally {
          uniqueFirstBtn.disabled = false;
          uniqueFirstBtn.textContent = label;
        }
      }
      openDissimilarGlobalModal({ candidateUuids });
    });
    // Insert before exportToDatasetBtn so the group reads ★ Most Unique First →
    // ⊞ Open in Datasets Tab, matching creation-order-independent display order
    // (UI_REDESIGN_PLAN.md, slice 3, PR 3.1).
    persistentUtilityGroup.insertBefore(uniqueFirstBtn, exportToDatasetBtn);
    } // end if (datasetActions)
  }

  // Every conditional member has now been appended, so an empty group here is
  // final, not just early. Drop it and its leading separator rather than
  // rendering a divider that separates nothing.
  if (persistentUtilityGroup.childElementCount === 0) {
    persistentUtilitySep.remove();
    persistentUtilityGroup.remove();
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

    // Bbox verification filter — Any/Verified/Unverified.
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

    // Model-confidence band — bbox-scoped like the verification/label/tag
    // controls above, applied by the same in-memory predicate. Percent in,
    // 0–1 out; see createConfidenceRangeControl.
    const confidenceControl = createConfidenceRangeControl({
      min: _curated.confidenceMin,
      max: _curated.confidenceMax,
      includeMissing: _curated.confidenceIncludeMissing,
      onChange: ({ min, max, includeMissing }) => {
        _curated = {
          ..._curated,
          confidenceMin: min,
          confidenceMax: max,
          confidenceIncludeMissing: includeMissing,
        };
        _onCuratedChange();
      },
    });

    // Filename substring — image-scoped, so unlike the three controls above it
    // is not part of the bbox predicate; `imageMatchesFilter` applies it as its
    // own gate. Same 400ms debounce + `.trim()` as filters.js's filename input
    // so typing in a modal doesn't re-render (and, in modal mode, re-page) on
    // every keystroke. Matches the FILES_ROOT-*relative* path — see
    // `imageMatchesFilename` for why that differs from the main page's
    // server-side LIKE.
    const filenameContainer = document.createElement('div');
    filenameContainer.className = 'curated-filename';
    const filenameLabel = document.createElement('span');
    filenameLabel.className = 'filter-label';
    filenameLabel.textContent = 'File:';
    const filenameInput = document.createElement('input');
    filenameInput.type = 'text';
    filenameInput.placeholder = 'substring…';
    filenameInput.title = 'Filter to images whose path contains this text (case-insensitive)';
    filenameInput.setAttribute('data-testid', 'curated-filename-input');
    let _filenameDebounce = null;
    filenameInput.addEventListener('input', () => {
      clearTimeout(_filenameDebounce);
      _filenameDebounce = setTimeout(() => {
        const next = filenameInput.value.trim();
        if (next === _curated.filenameSearch) return;
        _curated = { ..._curated, filenameSearch: next };
        _onCuratedChange();
      }, 400);
    });
    filenameContainer.append(filenameLabel, filenameInput);

    // Created-date range — also image-scoped, also client-side. Mirrors
    // filters.js's pair of datetime-local inputs, including the conversion to
    // an ISO string on change; `imageMatchesDate` handles the fact that the
    // API's createdAt comes back without a UTC offset.
    const dateContainer = document.createElement('div');
    dateContainer.className = 'curated-date';
    const dateLabel = document.createElement('span');
    dateLabel.className = 'filter-label';
    dateLabel.textContent = 'Date:';
    const dateFromInput = document.createElement('input');
    dateFromInput.type = 'datetime-local';
    dateFromInput.title = 'From date (image created)';
    dateFromInput.setAttribute('data-testid', 'curated-date-from');
    dateFromInput.addEventListener('change', () => {
      _curated = { ..._curated, dateFrom: dateFromInput.value ? new Date(dateFromInput.value).toISOString() : null };
      _onCuratedChange();
    });
    const dateToInput = document.createElement('input');
    dateToInput.type = 'datetime-local';
    dateToInput.title = 'To date (image created)';
    dateToInput.setAttribute('data-testid', 'curated-date-to');
    dateToInput.addEventListener('change', () => {
      _curated = { ..._curated, dateTo: dateToInput.value ? new Date(dateToInput.value).toISOString() : null };
      _onCuratedChange();
    });
    dateContainer.append(dateLabel, dateFromInput, document.createTextNode('–'), dateToInput);

    // Append after the caller's own toolbarControls (threshold/count/switch):
    // the "append, don't insert" rule keeps caller-built controls from shifting
    // position (NOTES.md). This is the curated/modal toolbar — the main page's
    // filters all live in the filter popover instead.
    const curatedGroup = makeToolbarGroup(verificationControl.element, confidenceControl.element, labelsContainer, tagsContainer, filenameContainer, dateContainer);
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
  // Unresolved training-run labels for the current page, same batching deal as
  // _pageImageLabels. img.uuid → [{ labelUuid, labelName, labelScope }]; images with
  // nothing outstanding are absent from the map (not mapped to []).
  let _pageImageGaps = new Map();
  // Which run the gap annotations are judged against: the run the modal was opened
  // from, else the session's active run. Resolved once per panel — a mid-session
  // change of active run doesn't retro-relabel an already-open modal, which is the
  // honest behaviour (the cards' gap rows would otherwise silently change meaning).
  const _gapsConfig = gapsConfig ?? activeTrainingConfig.getActive() ?? null;

  /** Batch-fetch gaps for the given image uuids and merge into _pageImageGaps. */
  async function _loadImageGaps(imageUuids, { replace = false } = {}) {
    if (replace) _pageImageGaps = new Map();
    if (!_gapsConfig || !imageUuids.length) return;
    try {
      const gaps = await fetchTrainingConfigImageGaps(_gapsConfig.uuid, imageUuids);
      // Images resolved to zero gaps are omitted from the response, so a stale entry
      // from a previous fetch has to be cleared explicitly rather than overwritten.
      for (const uuid of imageUuids) _pageImageGaps.delete(uuid);
      for (const [uuid, entries] of gaps) _pageImageGaps.set(uuid, entries);
    } catch (err) {
      Logger.error('review: batch image-gaps fetch failed', err);
    }
  }

  let _pageCursors = [null];
  let _lastCursor = null;
  let _rendering = false;
  let _renderGeneration = 0;
  let _hasMore = false;
  // Matching IMAGES (what the review query paginates by). Do NOT reuse this as
  // a bbox count — one image carries many bboxes.
  let _totalCount = 0;
  // Matching BBOXES, fetched separately (null = not resolved yet).
  let _totalBboxCount = null;

  // ── Infinite-scroll state (M2, fetchImages mode only) ────────────────────
  // `_curatedCursor` is the afterUuid to resume from; `_curatedExhausted`
  // means fetchImages returned fewer than a full chunk (no more candidates).
  // Filter mode ignores these — it keeps fixed-page keyset pagination.
  let _curatedCursor = null;
  // ⚠️ The uuid alone is NOT a usable cursor. The server-side custom fetchers
  // (the training-config/coverage modals) keyset-paginate on `(created_at,
  // uuid)`; given a uuid with no sort value they cannot build that predicate,
  // and the page stops advancing. Since these loops page until exhaustion, a
  // curated filter that rejects a whole chunk then loops forever — that is the
  // "Needs Review + filename filter never finishes" hang.
  //
  // Always `createdAt`, never `sortBySelect.value`: modal mode hides the sort
  // controls and never passes `sortBy` to `fetchImages`, so every custom fetcher
  // is using its own `created_at` default. Deriving from the select could send
  // an `updatedAt` against a created_at-ordered query and skip rows instead.
  let _curatedCursorSortValue = null;
  const _curatedSortValueOf = (img) => img?.createdAt ?? null;
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

  // Filter fields shared by every review-query call site here — everything
  // except pagination (afterUuid/afterSortValue/limit), requireBboxes and
  // sortBy/sortOrder, which differ per call site and stay inline at the call
  // site. The mapping itself lives in shared/filterUtils.js so the "Run Model
  // on Results" scope resolvers (a separate module) resolve the same set these
  // counts are built from — the inline copies had drifted, dropping the
  // negate flags and inverting the filter (see reviewStateToFilterOpts).
  function _baseReviewFilterOpts(fs) {
    return reviewStateToFilterOpts(fs, {
      viewFilter: _viewFilter,
      trainingLabelUuids: activeTrainingConfig.getLabelUuids(),
    });
  }

  // Bbox-level subset of the current filter — the same narrowing render()
  // applies client-side when building bbox entries. Sent to the bulk-apply and
  // bbox-count endpoints so both agree with what the grid shows.
  function _bboxSubsetFilters(fs) {
    const bboxTagFilter = fs.showAllBboxes ? [] : (fs.tagUuids ?? []);
    const bboxLabelFilter = fs.labelUuids?.length > 0
      ? (fs.labelIncludeChildren
          ? Array.from(expandWithDescendants(fs.labelUuids, labels))
          : fs.labelUuids)
      : [];
    return { bboxTagFilter, bboxLabelFilter };
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
      ..._baseReviewFilterOpts(fs),
      requireBboxes: _viewFilter === 'bboxes',
      sortBy: sortBySelect.value,
      sortOrder: sortOrderSelect.value,
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
      // `bbox` is normally the same object as its entry in `img.bboxes`, but a
      // canvas save replaces that array wholesale (IMAGE_ANNOTATIONS_SAVED) and
      // leaves this adapter closed over the old object. The image card renders
      // its ✓/? chips and status row from `img.bboxes`, so it would keep showing
      // the pre-toggle state unless the new object is updated too.
      const bboxInImg = img.bboxes?.find(b => b.uuid === bbox.uuid);
      if (bboxInImg && bboxInImg !== bbox) bboxInImg.tags = bbox.tags;
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

      // `refreshImageCard: false` is for callers flipping a run of boxes on one
      // image: every refresh redraws the whole card, so N boxes cost N full
      // re-renders. They refresh once when the run finishes instead.
      async setVerification(uuid, { refreshImageCard = true } = {}) {
        const tags = currentTagUuids();
        let newTagUuids;
        if (uuid === null) {
          // Clear all verification indicators
          newTagUuids = tags.filter(u => !ALL_VERIFICATION_UUIDS.has(u));
        } else if (uuid === REJECTION_TAG_UUID) {
          // REJECTION has no exclusive group — just add
          newTagUuids = tags.includes(uuid) ? tags : [...tags, uuid];
        } else {
          // Swap within verification exclusive group (UNVERIFIED ↔ VERIFIED)
          newTagUuids = swapVerificationTag(tags, uuid);
        }
        const labelUuid = bbox.labelUuid ?? bbox.label?.uuid;
        await _applyUpdate(labelUuid, newTagUuids);
        // Refresh thumb
        const thumb = card.querySelector('.thumb');
        if (thumb) thumb.src = getBboxThumbnailUrl(img.filename, bbox.uuid, 300) + `&t=${Date.now()}`;
        markCardStale(card, 'verified');
        if (refreshImageCard) _imageCardRefreshers.get(img.uuid)?.();
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
        const res = await setImageLabel(img.filename, uuid, status);
        // Published even when `res.skipped` — the write was refused, so the UI
        // may be showing a state the server never accepted and needs to re-sync
        // to the stored truth. The caller (contextMenu/providers/imageProvider.js)
        // surfaces the refusal itself; returning `res` is what makes that possible.
        events.publish(EventTypes.IMAGE_LABELS_CHANGED, { imageUuid: img.uuid, imagePath: img.filename });
        return res;
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
    bulkBtn.textContent = `Apply to Selected (${_selected.size})`;
    bulkBtn.disabled = _selected.size === 0;
    selectAllCb.checked = _selected.size > 0 && _selected.size === activeCount;
    selectAllCb.indeterminate = _selected.size > 0 && _selected.size < activeCount;
    // Contextual action bar (slice 3, PR 3.3) — only visible with a selection.
    contextualActionBar.hidden = _selected.size === 0;

    // "Apply to Parent Images" only makes sense for a bbox-kind selection —
    // it resolves the distinct images behind the selected bboxes.
    const isBboxSelection = _selectionKind === 'bbox' && _selected.size > 0;
    bulkParentImagesBtn.hidden = !isBboxSelection;
    bulkParentImagesBtn.disabled = !isBboxSelection;
    if (isBboxSelection) {
      const selectedBboxEntries = _visibleEntries.filter(e => _selected.has(entryId(e)));
      const distinctCount = _distinctParentImageEntries(selectedBboxEntries).length;
      bulkParentImagesBtn.textContent = `Apply to Parent Images (${distinctCount})`;
    }
    if (!fetchImages) {
      const imgCountStr = _totalCount > 0 ? _totalCount.toLocaleString() : '?';
      const bboxCountStr = _totalBboxCount === null ? '…' : _totalBboxCount.toLocaleString();
      bulkAllBtn.textContent = `Apply to All Filtered Bboxes (${bboxCountStr})`;
      // Gated on the image total, not the bbox count: the bbox count is
      // advisory (and may be pending or have failed), and "no matching images"
      // already implies no matching bboxes.
      bulkAllBtn.disabled = _totalCount === 0;
      // In bboxes view the same button acts on the images *containing* the
      // matched bboxes (the filter already carries requireBboxes), so it is
      // labelled for that scope rather than implying it acts on the bboxes.
      bulkAllImagesBtn.textContent = _viewFilter === 'bboxes'
        ? `Apply to Images Containing Filtered Bboxes (${imgCountStr})`
        : `Apply to Filtered Images (${imgCountStr})`;
      bulkAllImagesBtn.disabled = _totalCount === 0;
      // bulkAllBtn is only shown in the view whose entries it acts on; the
      // image-level button is offered everywhere, since acting on the images
      // behind the visible bboxes is a legitimate move in bboxes view too.
      bulkAllBtn.hidden = _viewFilter === 'images';
      bulkAllImagesBtn.hidden = false;
    } else {
      // No server-side total in custom-fetchImages mode — the full candidate
      // count is only known once showApplyToAllCandidatesModal fetches it.
      // That path resolves image-vs-bbox scope from the current view itself.
      bulkAllBtn.textContent = 'Apply to All';
      bulkAllBtn.disabled = _visibleEntries.length === 0;
      bulkAllBtn.hidden = false;
      // The filter-based path has no filter to reuse here, but the candidate
      // list itself can still be collapsed to its distinct parent images, so
      // bboxes view gets the image-level option in this mode too.
      bulkAllImagesBtn.textContent = 'Apply to Images Containing Bboxes';
      bulkAllImagesBtn.disabled = _visibleEntries.length === 0;
      bulkAllImagesBtn.hidden = _viewFilter !== 'bboxes';
    }
  }

  /**
   * Refresh the matching-bbox total behind "Apply to All Filtered Bboxes".
   * Generation-guarded like render(): a filter change mid-flight starts a newer
   * render whose count must win.
   */
  async function _refreshBboxCount(gen) {
    if (fetchImages) return;
    const fs = filterPanel.getState();
    try {
      const count = await countMatchingBboxes({
        ..._baseReviewFilterOpts(fs),
        requireBboxes: _viewFilter === 'bboxes',
        ..._bboxSubsetFilters(fs),
      });
      if (_renderGeneration !== gen) return;
      _totalBboxCount = count;
    } catch (err) {
      if (_renderGeneration !== gen) return;
      Logger.error('review: bbox count fetch failed', err);
      _totalBboxCount = null;
    }
    updateBulkBtn();
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
        chip.classList.remove('label-chip--fallback');
        chip.style.cssText = `background:${uiBbox.label.color}33;border:1px solid ${uiBbox.label.color};`;
      } else {
        chip.textContent = '(unknown)';
        chip.style.cssText = '';
        chip.classList.add('label-chip--fallback');
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
      gapsConfig: _gapsConfig,
      getPageImageGaps: uuid => _pageImageGaps.get(uuid),
      makeImageAdapter: makeReviewImageAdapter,
      // Re-reads gaps for ONE image after a chip resolves it. Scoped to that image
      // on purpose — the rest of the page is untouched by the write, and refetching
      // the whole page on every chip click would be a request per click.
      refreshImageGaps: async (imageUuid) => {
        await _loadImageGaps([imageUuid]);
        return _pageImageGaps.get(imageUuid) ?? [];
      },
    };
  }

  // Does this entry survive the current view toggle? Shared with render()'s
  // auto-skip check so "the page looks empty" can never mean one thing to the
  // grid and another to the thing that decides to throw the page away.
  function _matchesViewFilter(e) {
    if (_viewFilter === 'images') return e.kind === 'image';
    if (_viewFilter === 'bboxes') return e.kind === 'bbox';
    return true; // 'all'
  }

  // ── Render filtered entries to grid ──────────────────────────────
  function renderFromEntries() {
    cardGrid.resetForRender();

    // Filter entries by view mode
    _visibleEntries = _allEntries.filter(_matchesViewFilter);

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
          const raw = await fetchImages({
            afterUuid: _curatedCursor,
            afterSortValue: _curatedCursorSortValue,
            limit: _pageSize + 1,
          });
          if (_renderGeneration !== _gen) return; // superseded by a full render() mid-flight

          const exhausted = raw.length <= _pageSize;
          const pageImages = raw.slice(0, _pageSize);
          if (pageImages.length > 0) {
            const last = pageImages[pageImages.length - 1];
            _curatedCursor = last.uuid;
            _curatedCursorSortValue = _curatedSortValueOf(last);
          }
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
          await _loadImageGaps(newImageUuids);
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
          _curatedCursorSortValue = null;
          _curatedExhausted = false;
          while (_allEntries.length < _pageSize && !_curatedExhausted) {
            if (_renderGeneration !== _gen) { clearTimeout(_slowTimer); return; }
            const raw = await fetchImages({
              afterUuid: _curatedCursor,
              afterSortValue: _curatedCursorSortValue,
              limit: _pageSize + 1,
            });
            // Re-check after the await before touching shared cursor/entry state:
            // a filter change or Refresh mid-fetch starts a newer render() that
            // has already reset _allEntries/_curatedCursor, and writing this
            // superseded chunk would corrupt it (mirrors _loadMoreCurated).
            if (_renderGeneration !== _gen) { clearTimeout(_slowTimer); return; }
            const exhausted = raw.length <= _pageSize;
            const pageImages = raw.slice(0, _pageSize);
            if (pageImages.length > 0) {
              const last = pageImages[pageImages.length - 1];
              _curatedCursor = last.uuid;
              _curatedCursorSortValue = _curatedSortValueOf(last);
            }
            _curatedExhausted = exhausted;
            if (pageImages.length === 0) break;
            _allEntries.push(..._buildCuratedEntries(pageImages));
          }
          hasMore = !_curatedExhausted;
          images = [];
        } else {
          ({ images, hasMore, totalCount } = await fetchReviewQueue({
            ..._baseReviewFilterOpts(fs),
            requireBboxes: _viewFilter === 'bboxes',
            sortBy,
            sortOrder,
            afterUuid: cursor?.uuid ?? null,
            afterSortValue: cursor?.sortValue ?? null,
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

      // Auto-skip empty pages (when the client-side bbox filter leaves nothing
      // to render). Skip only in full-filter mode — in custom fetchImages mode
      // (preview) we want to show image entries even when they have no bbox
      // annotations.
      //
      // "Empty" has to mean *what this view would render*, not "has no bboxes".
      // Only the bboxes view is blank on a page with no bbox entries; in the
      // images and all views every returned image is a card in its own right,
      // so a bbox-less page is a full page. Testing bbox count regardless of
      // view made any filter selecting mostly-unannotated images (a camera
      // source, say — 1535 images, 1 of them annotated) burn all ten skips in
      // a row and walk to the end of the result set. And because a skip
      // *overwrites* the current page cursor rather than pushing one, the pager
      // still read "Page 1 of 7" with Prev disabled while the grid showed the
      // last page — no click involved, and nothing on screen to explain it.
      const visibleEntryCount = _allEntries.filter(_matchesViewFilter).length;
      if (!fetchImages && visibleEntryCount === 0 && _hasMore && autoSkipped < MAX_AUTO_SKIP) {
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
    await _loadImageGaps([...new Set(_allEntries.map(e => e.img.uuid))], { replace: true });
    if (_renderGeneration !== _gen) return;

    updatePageInfo();

    clearTimeout(_slowTimer);
    renderFromEntries();

    // Not awaited — the grid must not wait on a count query. The previous
    // value stays on the button until it resolves (no '…' flicker on paging).
    _refreshBboxCount(_gen);

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
      // Same rule as the image-card double-click handler in cardGrid.js: an
      // image-kind entry (no specific bbox of its own) must navigate to
      // nothing pre-selected, not silently to its first bbox.
      navItems.push({ img: entry.img, bbox: entry.bbox ?? null });
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

  // Resolve the DISTINCT parent images behind a list of bbox-kind entries,
  // deduped by image uuid, as image-kind entries suitable for
  // showImageBulkActionModal / executeBulkImageAction.
  function _distinctParentImageEntries(bboxEntries) {
    const seen = new Map();
    for (const { img } of bboxEntries) {
      if (!seen.has(img.uuid)) seen.set(img.uuid, { kind: 'image', img });
    }
    return [...seen.values()];
  }

  // Shared by the "Apply to Selected" image-selection branch and "Apply to Parent
  // Images" (bbox selection → distinct parent images). titleText overrides
  // showImageBulkActionModal's default "Apply to N selected images" title —
  // callers acting on parent images MUST pass an explicit title so the modal
  // never implies it's acting on bboxes.
  async function _runImageBulkAction(imageEntries, titleText = null) {
    await showImageBulkActionModal(imageEntries, labels, async (action, labelUuids, tagUuids) => {
      if (isRemovalAction(action)) {
        // 'delete' takes the file with it; 'removeFromDb' leaves it on disk.
        // Everything downstream (card staleness, IMAGE_DELETED) is identical —
        // as far as the rest of the app is concerned the image is gone.
        const confirmed = await confirmRemovalAction(action, imageEntries.length, '', {
          verifiedCount: countImagesWithVerifiedWork(imageEntries),
        });
        if (!confirmed) return;

        // One request per image, issued from here — see setClientSideApplyStatus.
        let failCount = 0;
        let processed = 0;
        const releaseUnloadGuard = guardUnloadDuringRun();
        try {
          setClientSideApplyStatus('images', 0, imageEntries.length);
          for (const { img } of imageEntries) {
            if (_panelLifetime.signal.aborted) break;
            try {
              await removeSingleImage(action, img.uuid);
              const card = grid.querySelector(`[data-image-uuid="${img.uuid}"][data-kind="image"]`);
              if (card) markCardStale(card, 'deleted');
              events.publish(EventTypes.IMAGE_DELETED, { imageUuid: img.uuid, imagePath: img.filename });
            } catch (err) {
              Logger.error(`${action} image failed for ${img.uuid}: ${err.message}`);
              failCount++;
            }
            setClientSideApplyStatus('images', ++processed, imageEntries.length);
          }
        } finally {
          releaseUnloadGuard();
          statusMsg.classList.remove('review-toolbar__status--loading');
        }
        if (_panelLifetime.signal.aborted) return;
        clearSelection();
        updateStatusMsg();
        if (failCount > 0) {
          statusMsg.textContent += ` (${failCount} removal${failCount === 1 ? '' : 's'} failed — check console)`;
        }
        if (processed < imageEntries.length) appendStoppedEarlyNote(processed, imageEntries.length, 'images');
        return;
      }

      const { failCount, skippedCount, succeededEntries } = await executeBulkImageAction(action, imageEntries, labelUuids, tagUuids);
      for (const { img } of succeededEntries) {
        const card = grid.querySelector(`[data-image-uuid="${img.uuid}"][data-kind="image"]`);
        if (card) markCardStale(card, 'tagged');
      }
      clearSelection();
      updateStatusMsg();
      if (skippedCount > 0) {
        statusMsg.textContent += ` (${skippedCount} skipped — verified bbox present)`;
      }
      if (failCount > 0) {
        statusMsg.textContent += ` (${failCount} action${failCount === 1 ? '' : 's'} failed — check console)`;
      }
    }, titleText);
  }

  // ── Apply to Parent Images (bbox selection → distinct parent images) ──
  bulkParentImagesBtn.addEventListener('click', async () => {
    const _selected = cardGrid.selected;
    if (_selected.size === 0 || cardGrid.selectionKind !== 'bbox') return;
    const selectedBboxEntries = _visibleEntries.filter(e => _selected.has(entryId(e)));
    const imageEntries = _distinctParentImageEntries(selectedBboxEntries);
    if (imageEntries.length === 0) return;
    const bboxCount = selectedBboxEntries.length;
    const imgCount = imageEntries.length;
    // Title is explicit about acting on N distinct IMAGES, not the M selected
    // bboxes — the two counts often differ (multiple bboxes per image) and
    // must never be conflated.
    const titleText = `Apply to ${imgCount} distinct image${imgCount === 1 ? '' : 's'} `
      + `(behind ${bboxCount} selected bbox${bboxCount === 1 ? '' : 'es'})`;
    await _runImageBulkAction(imageEntries, titleText);
  });

  // ── Apply to Selected ────────────────────────────────────────────────────
  bulkBtn.addEventListener('click', async () => {
    const _selected = cardGrid.selected;
    if (_selected.size === 0) return;
    const selectedEntries = _visibleEntries.filter(e => _selected.has(entryId(e)));

    if (cardGrid.selectionKind === 'image') {
      await _runImageBulkAction(selectedEntries);
    } else {
      await showBulkActionModal(selectedEntries, labels, async (action, newLabelUuid, selectedTagUuids, tagsMap) => {
        const releaseUnloadGuard = guardUnloadDuringRun();
        let failCount, succeededEntries, processed, stoppedEarly;
        try {
          setClientSideApplyStatus('bboxes', 0, selectedEntries.length);
          ({ failCount, succeededEntries, processed, stoppedEarly } = await executeBulkAction(
            action, selectedEntries, newLabelUuid, selectedTagUuids, tagsMap,
            {
              signal: _panelLifetime.signal,
              onProgress: (done, total) => setClientSideApplyStatus('bboxes', done, total),
            },
          ));
        } finally {
          releaseUnloadGuard();
          statusMsg.classList.remove('review-toolbar__status--loading');
        }
        // The panel this was updating is gone; its DOM work would land nowhere.
        if (_panelLifetime.signal.aborted) return;

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
        if (stoppedEarly) appendStoppedEarlyNote(processed, selectedEntries.length, 'bboxes');
      });
    }
  });

  // ── Bulk Apply All ────────────────────────────────────────────────
  // In custom-fetchImages mode (similarity/diversity search popups) there is
  // no server-side filter to re-derive the candidate set from, so "Apply to
  // All" instead fetches the already-known full candidate list in one call
  // and applies the action via a single bulk-by-uuid-list backend request —
  // not a loop of one HTTP call per item.
  //
  // With `parentImagesOfBboxes`, the candidate bboxes are resolved to their
  // DISTINCT parent images and an image-level action is applied to those
  // instead — the "Apply to Images Containing Bboxes" button in bboxes view.
  async function applyToAllCandidates({ parentImagesOfBboxes = false } = {}) {
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
    const kind = parentImagesOfBboxes
      ? 'image'
      : (cardGrid.selectionKind ?? (_viewFilter === 'images' ? 'image' : 'bbox'));
    const bboxEntries = (!parentImagesOfBboxes && kind === 'image') ? [] : curatedImages.flatMap(img =>
      (img.bboxes || [])
        .filter(b => (!bboxFilterFn || bboxFilterFn(b)) && bboxMatchesFilter(b, _curated, labels))
        .map(bbox => ({ kind: 'bbox', bbox, img })));
    const entries = parentImagesOfBboxes
      ? _distinctParentImageEntries(bboxEntries)
      : kind === 'image'
        ? curatedImages.map(img => ({ kind: 'image', img }))
        : bboxEntries;

    if (entries.length === 0) {
      statusMsg.textContent = 'No candidates to apply to.';
      return;
    }

    if (kind === 'image') {
      await showImageBulkActionModal(entries, labels, async (action, labelUuids, tagUuids) => {
        if (isRemovalAction(action)) {
          const confirmed = await confirmRemovalAction(action, entries.length, 'from the full candidate list', {
            verifiedCount: countImagesWithVerifiedWork(entries),
          });
          if (!confirmed) return;
        }
        setServerSideApplyStatus(`Applying to ${entries.length.toLocaleString()} images…`);
        try {
          const { modifiedCount, failCount, skippedCount } = await bulkApplyImagesByUuids({
            imageUuids: entries.map(e => e.img.uuid),
            action,
            labelUuids,
            tagUuids,
          });
          if (isRemovalAction(action)) {
            for (const { img } of entries) {
              events.publish(EventTypes.IMAGE_DELETED, { imageUuid: img.uuid, imagePath: img.filename });
            }
          }
          statusMsg.classList.remove('review-toolbar__status--loading');
          statusMsg.textContent = `Applied to ${modifiedCount.toLocaleString()} images` +
            (skippedCount > 0 ? ` (${skippedCount} skipped — verified bbox present)` : '') +
            (failCount > 0 ? ` (${failCount} failed — check console)` : '');
        } catch (err) {
          statusMsg.classList.remove('review-toolbar__status--loading');
          statusMsg.textContent = `Bulk apply failed: ${err.message}`;
          Logger.error('applyToAllCandidates (images) failed:', err);
        }
        await render();
      }, parentImagesOfBboxes
        // Must never read as "N bboxes" — the two counts differ whenever an
        // image holds more than one matched bbox.
        ? `Apply to ${entries.length.toLocaleString()} distinct image${entries.length === 1 ? '' : 's'} `
          + `(behind ${bboxEntries.length.toLocaleString()} matched bbox${bboxEntries.length === 1 ? '' : 'es'})`
        : `Apply to all ${entries.length.toLocaleString()} matched image${entries.length === 1 ? '' : 's'}`);
    } else {
      await showBulkActionModal(entries, labels, async (action, newLabelUuid, selectedTagUuids) => {
        setServerSideApplyStatus(`Applying to ${entries.length.toLocaleString()} bboxes…`);
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

  // ── Apply to All Images (filtered) ─────────────────────────────────
  // Filter-mode-only sibling of "Apply to All": applies an image-level action
  // (setPresent/setAbsent/setUncertain/tags/delete) to every image matching
  // the CURRENT filter — paginated server-side via the new bulkApplyImages
  // endpoint, never downloading the full candidate list client-side. This is
  // what makes "filter by label X, then mark label Y absent on all of them"
  // tractable at scale without selecting every bbox by hand.
  async function applyToAllImagesByFilter() {
    const fs = filterPanel.getState();
    const sortBy = sortBySelect.value;
    const sortOrder = sortOrderSelect.value;
    const totalImageCount = _totalCount;
    // In bboxes view the scope is still images — the ones containing the
    // matched bboxes (requireBboxes is already part of the filter below) — so
    // the title says so rather than implying a bbox-level action.
    const plural = totalImageCount === 1 ? '' : 's';
    const titleText = _viewFilter === 'bboxes'
      ? `Apply to All ${totalImageCount.toLocaleString()} Image${plural} Containing Filtered Bboxes`
      : `Apply to All ${totalImageCount.toLocaleString()} Filtered Image${plural}`;

    // ── Preview section: first 3 / last 3 images of the filtered range ────
    // Fetched from the server rather than read off _visibleEntries so it shows
    // the true ends of the filtered set, not just whichever page is open.
    const previewSection = document.createElement('div');
    previewSection.className = 'bulk-modal-preview-section';
    const previewHdr = document.createElement('div');
    previewHdr.className = 'bulk-modal-preview-hdr';
    previewHdr.textContent = 'Preview range';
    const previewRow = document.createElement('div');
    previewRow.className = 'bulk-modal-preview-row';
    const previewStatus = document.createElement('span');
    previewStatus.className = 'bulk-modal-preview-status';
    previewStatus.textContent = 'Loading preview…';
    previewRow.appendChild(previewStatus);
    previewSection.append(previewHdr, previewRow);

    const endOpts = {
      ..._baseReviewFilterOpts(fs),
      requireBboxes: _viewFilter === 'bboxes',
      sortBy,
      limit: 3,
    };
    // Start both end queries now; render them once the modal is on screen.
    const previewPromise = Promise.all([
      fetchReviewQueue({ ...endOpts, sortOrder }),
      fetchReviewQueue({ ...endOpts, sortOrder: sortOrder === 'desc' ? 'asc' : 'desc' }),
    ]).then(([firstPage, lastPage]) => ({
      first: firstPage.images ?? [],
      // Reverse so the "Last" group still reads in natural sort order.
      last: [...(lastPage.images ?? [])].reverse(),
    })).catch(err => {
      Logger.error('applyToAllImagesByFilter: failed to fetch preview range:', err);
      return { first: [], last: [] };
    });

    await showImageBulkActionModal([], labels, async (action, labelUuids, tagUuids) => {
      if (isRemovalAction(action)) {
        // The images behind this button were never on screen — the count came
        // from the server — so how much human work is in them has to come from
        // the server too. One extra count query, and only once the removal is
        // already big enough to warrant the second step; below that the guard
        // never asks, so neither do we.
        let verifiedCount = null;
        if (totalImageCount > HIGH_RISK_REMOVAL_COUNT) {
          try {
            const { totalCount } = await fetchReviewQueue({
              ..._baseReviewFilterOpts(fs),
              requireBboxes: _viewFilter === 'bboxes',
              requireVerifiedBbox: true,
              limit: 1,
            });
            // -1 is the endpoint's "count query timed out" sentinel, not a
            // count. Treat it as unknown rather than as a number, or the
            // dialog headlines "-1 human-verified images".
            verifiedCount = totalCount >= 0 ? totalCount : null;
          } catch (err) {
            // Leave it null: "could not check" is not "nothing to lose", and
            // confirmRemovalAction reads null as the loud case on purpose.
            Logger.warn('applyToAllImagesByFilter: verified-count probe failed', err);
          }
        }
        const confirmed = await confirmRemovalAction(
          action, totalImageCount, 'matching the current filter, across all pages', { verifiedCount },
        );
        if (!confirmed) return;
      }
      setServerSideApplyStatus('Applying to all matching images…');
      try {
        const { modifiedCount, failCount, skippedCount } = await bulkApplyImages({
          ..._baseReviewFilterOpts(fs),
          requireBboxes: _viewFilter === 'bboxes',
          sortBy: sortBySelect.value,
          sortOrder: sortOrderSelect.value,
          action,
          selectedLabelUuids: labelUuids,
          selectedTagUuids: tagUuids,
        });
        statusMsg.classList.remove('review-toolbar__status--loading');
        statusMsg.textContent = `Applied to ${modifiedCount.toLocaleString()} images` +
          (skippedCount > 0 ? ` (${skippedCount} skipped — verified bbox present)` : '') +
          (failCount > 0 ? ` (${failCount} failed — check console)` : '');
        render();
      } catch (err) {
        statusMsg.classList.remove('review-toolbar__status--loading');
        statusMsg.textContent = `Bulk apply failed: ${err.message}`;
        Logger.error('applyToAllImagesByFilter failed:', err);
      }
    }, titleText, { previewNode: previewSection });

    const { first, last } = await previewPromise;
    _populateImagePreviewRow(previewRow, first, last);
  }

  bulkAllImagesBtn.addEventListener('click', () => {
    // Same scope resolution as bulkAllBtn: filter mode re-derives the set
    // server-side, candidate mode collapses the known bbox list to its images.
    if (fetchImages) applyToAllCandidates({ parentImagesOfBboxes: true });
    else applyToAllImagesByFilter();
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

  /**
   * Convert a canvas UIBbox into the API-shaped bbox the review entries hold.
   * UIBbox labels are `{uuid, text, color}`; entry bboxes carry `{uuid, name,
   * color}` plus a flat `labelUuid`, which is what the cards and adapters read.
   */
  function _uiBboxToEntryBbox(uiBbox) {
    const label = uiBbox.label
      ? { uuid: uiBbox.label.uuid, name: uiBbox.label.text ?? uiBbox.label.name, color: uiBbox.label.color }
      : null;
    return {
      uuid: uiBbox.uuid,
      x: uiBbox.x,
      y: uiBbox.y,
      width: uiBbox.width,
      height: uiBbox.height,
      label,
      labelUuid: label?.uuid ?? null,
      tags: uiBbox.tags ?? [],
      metadataJson: uiBbox.metadataJson ?? null,
    };
  }

  // Subscribe to IMAGE_ANNOTATIONS_SAVED so saves from any nested canvas modal
  // (or any other panel) propagate back to this panel's cards automatically.
  //
  // The payload's `bboxes` is the AUTHORITATIVE post-save set for the image, so
  // the in-memory entry is resynced from it wholesale — that is what makes
  // added and deleted boxes show up. Without this the image card re-rendered
  // from its pre-save `img.bboxes` and looked unchanged.
  const _unsubImageSaved = events.subscribe(EventTypes.IMAGE_ANNOTATIONS_SAVED, ({ imageUuid, imageFilename, bboxes } = {}) => {
    // NB: no `!bboxes.size` guard — deleting every box on an image is a real
    // save that the card must reflect.
    if (!imageUuid || !bboxes) return;

    for (const [bboxUuid, uiBbox] of bboxes) {
      const card = grid.querySelector(`[data-bbox-uuid="${bboxUuid}"]`);
      if (card) _refreshBboxCard(card, uiBbox, imageFilename, bboxUuid);
    }

    const saved = [...bboxes.values()].map(_uiBboxToEntryBbox);
    for (const entry of _allEntries) {
      if (entry.img?.uuid !== imageUuid) continue;
      // Mutate in place: the image card's hover-preview meta and the hover
      // thumbnail cache both hold this array by reference.
      if (Array.isArray(entry.img.bboxes)) {
        entry.img.bboxes.length = 0;
        entry.img.bboxes.push(...saved);
      } else {
        entry.img.bboxes = saved;
      }
      // A bbox entry whose box no longer exists was deleted in the canvas.
      if (entry.kind === 'bbox' && entry.bbox && !bboxes.has(entry.bbox.uuid)) {
        const bboxCard = grid.querySelector(`[data-bbox-uuid="${entry.bbox.uuid}"]`);
        if (bboxCard) markCardStale(bboxCard, 'deleted');
      }
    }

    _imageCardRefreshers.get(imageUuid)?.();
    updateStatusMsg();
  });

  // Mark every card for a deleted image stale — both its image-kind card (if
  // shown) and any bbox-kind cards for it in the "bboxes"/"all" views, since
  // deleting the whole image takes its bboxes with it too.
  const _unsubImageDeleted = events.subscribe(EventTypes.IMAGE_DELETED, ({ imageUuid } = {}) => {
    if (!imageUuid) return;
    grid.querySelectorAll(`[data-image-uuid="${imageUuid}"]`).forEach(card => markCardStale(card, 'deleted'));
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
    // Server-side count of the bboxes this will actually modify (the image
    // total is a different, much smaller number — see _totalCount).
    const totalBboxCount = _totalBboxCount;
    const totalBboxStr = totalBboxCount === null ? '' : ` ${totalBboxCount.toLocaleString()}`;

    // ── Build bbox-level filter (mirrors the render() subset logic) ──────
    const showAll = fs.showAllBboxes;
    const { bboxTagFilter, bboxLabelFilter } = _bboxSubsetFilters(fs);
    const bboxTagSet = showAll ? null : new Set(bboxTagFilter);
    const bboxLabelSet = bboxLabelFilter.length > 0 ? new Set(bboxLabelFilter) : null;

    // ── Fetch tags + labels for pickers ─────────────────────────────────
    const [tags, allLabels] = await Promise.all([
      fetchTags().catch(() => new Map()),
      fetchLabels().catch(() => new Map()),
    ]);

    // ── Build modal shell ────────────────────────────────────────────────
    const overlay = document.createElement('div');
    overlay.className = 'bulk-modal-overlay';
    const _layer = acquireModalLayer({ dialog: true, overlay });
    overlay.style.zIndex = _layer.zIndex;

    const dialog = document.createElement('div');
    dialog.className = 'bulk-modal-dialog bulk-modal-dialog--wide';

    const title = document.createElement('strong');
    title.textContent = `Apply to All${totalBboxStr} Filtered Bboxes`;

    // ── Preview section ──────────────────────────────────────────────────
    const previewSection = document.createElement('div');
    previewSection.className = 'bulk-modal-preview-section';

    const previewHdr = document.createElement('div');
    previewHdr.className = 'bulk-modal-preview-hdr';
    previewHdr.textContent = 'Preview range';

    const previewRow = document.createElement('div');
    previewRow.className = 'bulk-modal-preview-row';

    const previewStatus = document.createElement('span');
    previewStatus.className = 'bulk-modal-preview-status';
    previewStatus.textContent = 'Loading preview…';
    previewRow.appendChild(previewStatus);

    previewSection.append(previewHdr, previewRow);

    // ── Action picker (same as showBulkActionModal) ─────────────────────
    const actions = [
      { value: 'relabel', label: 'Change Label' },
      { value: 'markVerified', label: '✓ Mark Verified' },
      { value: 'markUnverified', label: '? Mark Unverified' },
      { value: 'addTags', label: 'Add Tags' },
      { value: 'removeTags', label: 'Remove Tags' },
      { value: 'delete', label: 'Delete Bboxes' },
      { value: 'revertRelabel', label: '↩ Revert Filter-Chain Relabel' },
    ];

    const radioGroup = document.createElement('div');
    radioGroup.className = 'bulk-modal-radio-group';
    let selectedAction = 'relabel';

    let _selectedLabelUuid = undefined;
    const dynPicker = createDynamicLabelPicker(allLabels, {
      onSelect: (uuid) => { _selectedLabelUuid = uuid; },
    });
    const labelPickerRow = document.createElement('div');
    labelPickerRow.className = 'bulk-modal-label-picker-row';
    const labelPickerHdr = document.createElement('div');
    labelPickerHdr.textContent = 'New label:';
    labelPickerHdr.className = 'bulk-modal-label-picker-hdr';
    labelPickerRow.append(labelPickerHdr, dynPicker.element);

    const tagOptions = filterVisibleTags(tags.values())
      .map(tag => ({ key: tag.uuid, text: (tag.icon ? tag.icon + ' ' : '') + tag.name, color: tag.color || '#cccccc' }));
    const tagField = new MultiSelectField({ label: 'Tags', value: [], options: tagOptions, emptyText: 'No tags defined yet' });
    const tagPickerRow = document.createElement('div');
    tagPickerRow.className = 'bulk-modal-tag-picker-row';
    tagPickerRow.appendChild(await tagField.renderEdit());

    for (const action of actions) {
      const lbl = document.createElement('label');
      lbl.className = 'bulk-modal-action-label';
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
    warning.className = 'bulk-modal-warning';
    warning.textContent = `⚠ This will modify all${totalBboxStr} bboxes matching your current filters, across every page. This cannot be undone.`;

    // ── Buttons ──────────────────────────────────────────────────────────
    const btnRow = document.createElement('div');
    btnRow.className = 'bulk-modal-btn-row';

    let releaseFocusTrap = null;
    function close() { unregisterEscHandler(close); _layer.release(); releaseFocusTrap?.(); dynPicker.destroy(); overlay.remove(); }

    const cancelBtn = document.createElement('button');
    cancelBtn.textContent = 'Cancel';
    cancelBtn.addEventListener('click', close);

    const okBtn = document.createElement('button');
    okBtn.textContent = `Apply to All${totalBboxStr}`;
    okBtn.className = 'bulk-modal-ok-btn';
    okBtn.addEventListener('click', async () => {
      close();
      const newLabelUuid = selectedAction === 'relabel' ? (_selectedLabelUuid ?? undefined) : undefined;
      // Verification shortcuts are sent as addTags with the appropriate system tag UUID.
      const resolvedAction = _VERIF_TAG_MAP[selectedAction] ? 'addTags' : selectedAction;
      const selectedTagUuids = _VERIF_TAG_MAP[selectedAction]
        ? [_VERIF_TAG_MAP[selectedAction]]
        : (selectedAction === 'addTags' || selectedAction === 'removeTags') ? tagField.getValue()
        // revertRelabel: strip only the task-generated review tag(s) among the
        // current filter — never a system tag like UNVERIFIED that may also be
        // part of the active filter (e.g. filtering by review-tag + UNVERIFIED
        // together must not strip UNVERIFIED from every reverted bbox).
        : (selectedAction === 'revertRelabel')
          ? (fs.tagUuids ?? []).filter(uuid => tags.get(uuid)?.kind === 'task_generated')
        : [];
      setServerSideApplyStatus(`Applying to all${totalBboxStr} bboxes…`);
      try {
        const { modifiedCount, failCount } = await bulkApplyAll({
          // Spread the shared opts rather than re-listing them: this call site
          // used to spell out its own subset and had silently fallen behind
          // _baseReviewFilterOpts by two dimensions the grid IS filtered by
          // (bboxVerification, imageLabelConditions), so Apply to All modified
          // more than it showed.
          ..._baseReviewFilterOpts(fs),
          requireBboxes: _viewFilter === 'bboxes',
          sortBy,
          sortOrder,
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
    applyModalCascade(dialog, _layer.depth);
    registerEscHandler(close);
    overlay.addEventListener('click', e => { if (e.target === overlay) close(); });
    releaseFocusTrap = trapModalFocus(dialog);

    // ── Populate preview asynchronously ─────────────────────────────────
    // First 3: already in _visibleEntries (free)
    const first3 = _visibleEntries.filter(e => e.kind === 'bbox').slice(0, 3);

    // Last 3: reverse-sort query (3 images should be enough to get 3 bboxes)
    const reversedOrder = sortOrder === 'desc' ? 'asc' : 'desc';
    let last3 = [];
    try {
      const { images: lastImgs } = await fetchReviewQueue({
        ..._baseReviewFilterOpts(fs),
        requireBboxes: true,
        sortBy,
        sortOrder: reversedOrder,
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
      // Stops any per-item bulk loop still running against this panel.
      _panelLifetime.abort();
      _drainCardUnsubs();
      _unsubImageSaved();
      _unsubImageDeleted();
      _unsubBboxChanged();
      filterPanel?.destroy();
      _thumbLoader.reset();
      hoverPreview.destroy();
      entryView.destroy();
      if (_scrollHandler) container.removeEventListener('scroll', _scrollHandler);
      _scrollChrome?.destroy();
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
/**
 * @param {object} [opts]
 * @param {{uuid: string, name: string}} [opts.gapsConfig]
 *   Training run whose unresolved labels are annotated onto each image card. Pass the
 *   run the modal was opened from (a gaps/coverage cell) so the cards are judged by the
 *   same run the numbers came from; omit to fall back to the session's active run.
 */
export async function showReviewModal({ title = 'Browse Images', fetchImages, editableDataset, toolbarControls, referenceBboxUuids = new Set(), referenceImageUuids = new Set(), initialViewFilter = 'images', bboxFilterFn = null, gapsConfig = null } = {}) {
  const overlay = document.createElement('div');
  overlay.className = 'compare-modal-overlay';
  const _layer = acquireModalLayer({ dialog: true, overlay });
  overlay.style.zIndex = _layer.zIndex;

  const dialog = document.createElement('div');
  dialog.className = 'compare-modal-dialog';

  // Header
  const header = document.createElement('div');
  header.className = 'compare-modal-header';

  const titleEl = document.createElement('strong');
  titleEl.textContent = title;

  const closeBtn = document.createElement('button');
  closeBtn.textContent = '✕';
  closeBtn.title = 'Close';
  closeBtn.className = 'compare-modal-close-btn';

  header.append(titleEl);

  // (caller-supplied toolbar controls are passed to createReviewPanel instead)

  // "Update Dataset" button — shown when modal is in editable-dataset mode
  if (editableDataset?.onUpdateFilters) {
    const updateBtn = document.createElement('button');
    updateBtn.textContent = 'Update Dataset';
    updateBtn.title = 'Apply current filters back to the dataset form (you must still Save to persist)';
    updateBtn.className = 'compare-modal-update-btn';
    // Will be wired up after the review panel is created and filterPanel is accessible
    header.appendChild(updateBtn);
    // Store for wiring after panel init
    header._updateBtn = updateBtn;
  }

  header.appendChild(closeBtn);

  // Scrollable content area for the panel
  const content = document.createElement('div');
  content.className = 'compare-modal-content';

  dialog.append(header, content);
  overlay.appendChild(dialog);
  document.body.appendChild(overlay);
  applyModalCascade(dialog, _layer.depth);
  // Without this, focus stays on whatever triggered the modal (e.g. a search
  // input behind it), so arrow keys/Tab keep acting on the background panel
  // instead of the modal that's visually on top.
  const releaseFocusTrap = trapModalFocus(dialog);

  // Use a ref so close() can call panel.destroy() even though panel is assigned below.
  const panelRef = { current: null };
  function close() { unregisterEscHandler(close); _layer.release(); releaseFocusTrap(); panelRef.current?.destroy(); overlay.remove(); }
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
    const panel = await createReviewPanel(content, { fetchImages, referenceBboxUuids, referenceImageUuids, initialViewFilter, toolbarControls, bboxFilterFn, gapsConfig });
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
};

/**
 * @param {{ signal?: AbortSignal, onProgress?: (done: number, total: number) => void }} [opts]
 *   `signal` stops the loop at the next item boundary (panel torn down mid-run);
 *   already-committed items are reported as succeeded and `stoppedEarly` is set.
 */
async function executeBulkAction(action, entries, newLabelUuid, selectedTagUuids = [], tagsMap = new Map(), { signal = null, onProgress = null } = {}) {
  const toInvalidate = [];
  const succeededEntries = [];
  let failCount = 0;
  let processed = 0;

  for (const entry of entries) {
    if (signal?.aborted) break;
    const { bbox } = entry;
    try {
      if (action === 'delete') {
        await deleteBbox(bbox.uuid);
      } else if (action === 'addTags') {
        const currentTagUuids = (bbox.tags || []).map(t => t.uuid);
        // For tags in an exclusive group: remove any existing tags from the same group
        // before adding the new one (e.g. adding VERIFIED must remove UNVERIFIED).
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
    onProgress?.(++processed, entries.length);
  }

  if (toInvalidate.length) await invalidateBboxThumbnails(toInvalidate);
  return { failCount, succeededEntries, processed, stoppedEarly: processed < entries.length };
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
  overlay.dataset.testid = 'canvas-edit-modal';

  const dialog = document.createElement('div');
  dialog.className = 'canvas-modal-dialog';
  dialog.dataset.testid = 'canvas-edit-modal-dialog';

  // Header
  const header = document.createElement('div');
  header.className = 'canvas-modal-header';
  const titleEl = document.createElement('span');
  titleEl.className = 'canvas-modal-title';
  titleEl.dataset.testid = 'canvas-edit-modal-title';
  titleEl.textContent = `Edit: ${img.filename.split('/').slice(-2).join('/')}`;

  // Visible fallback for the [ ] nav hotkeys — same navObj, same next_image/
  // prev_image calls, so it stays in lockstep whether the user clicks or
  // presses a key. Hidden entirely when there's nothing to navigate through.
  const prevBtn = document.createElement('button');
  prevBtn.className = 'canvas-modal-nav-btn canvas-modal-nav-btn--prev';
  prevBtn.dataset.testid = 'canvas-edit-modal-prev';
  prevBtn.textContent = '‹ Prev';
  prevBtn.title = 'Previous image';
  prevBtn.style.display = 'none';

  const nextBtn = document.createElement('button');
  nextBtn.className = 'canvas-modal-nav-btn canvas-modal-nav-btn--next';
  nextBtn.dataset.testid = 'canvas-edit-modal-next';
  nextBtn.textContent = 'Next ›';
  nextBtn.title = 'Next image';
  nextBtn.style.display = 'none';

  const deleteImageBtn = document.createElement('button');
  deleteImageBtn.className = 'canvas-modal-delete-btn';
  deleteImageBtn.dataset.testid = 'canvas-edit-modal-delete-image';
  deleteImageBtn.textContent = '🗑';
  deleteImageBtn.title = 'Delete this image';

  const closeBtn = document.createElement('button');
  closeBtn.className = 'canvas-modal-close';
  closeBtn.dataset.testid = 'canvas-edit-modal-close';
  closeBtn.textContent = '✕';
  closeBtn.title = 'Close';

  const headerActions = document.createElement('div');
  headerActions.className = 'canvas-modal-header-actions';
  headerActions.append(prevBtn, nextBtn, deleteImageBtn, closeBtn);
  header.append(titleEl, headerActions);

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
  cancelBtn.className = 'canvas-modal-cancel-btn';

  const saveOnlyBtn = document.createElement('button');
  saveOnlyBtn.textContent = 'Save';
  saveOnlyBtn.className = 'save-btn--clean';

  const saveBtn = document.createElement('button');
  saveBtn.textContent = 'Save & Close';
  saveBtn.className = 'save-btn--clean';
  // Its label switches to 'Close' whenever the canvas is clean, so anything
  // that needs to find this button must go through the testid, not the text.
  saveBtn.dataset.testid = 'canvas-edit-modal-save-close';
  footer.append(cancelBtn, saveOnlyBtn, saveBtn);

  // The overlay goes into the DOM immediately below, but these buttons' click
  // handlers aren't attached until the very end of this function — after the
  // canvas load and the awaited sidebar/inference panel fetches. Start them
  // disabled so a click landing in that window can't hit a dead button; the
  // final wiring step re-enables them.
  closeBtn.disabled = true;
  deleteImageBtn.disabled = true;
  cancelBtn.disabled = true;
  saveOnlyBtn.disabled = true;
  saveBtn.disabled = true;
  prevBtn.disabled = true;
  nextBtn.disabled = true;

  dialog.append(header, toolbarEl, contentArea, footer);
  overlay.appendChild(dialog);
  // Held from here until closeModal (or the load-failure path below). The ESC
  // handler is not registered until well after the awaits that follow, so the
  // layer — not the ESC stack — is what marks this modal as on screen.
  const _layer = acquireModalLayer({ dialog: true, overlay });
  overlay.style.zIndex = _layer.zIndex;
  document.body.appendChild(overlay);
  applyModalCascade(dialog, _layer.depth);
  // Every button above is still disabled, so this parks focus on the dialog
  // itself for now — the important part is moving it off whatever was
  // focused behind the modal before the awaits below start.
  const releaseFocusTrap = trapModalFocus(dialog);

  // ── Create canvas instance ─────────────────────────────────────────
  let instance;
  let toolbarHandle    = { destroy: () => {} };
  let sidebarPanelsHandle = { destroy: () => {} };
  let inferencePanelHandle = { destroy: () => {} };
  let modalResizeHandle = { destroy: () => {} };
  let unsubTabRules    = () => {};
  let unsubBboxChanged = () => {};
  let unsubInference   = () => {};
  let unregisterSaveAction = () => {};

  try {
    instance = await createCanvasInstance(canvasWrap, { tool: 'select' });
    await instance.loadImage(img.filename);
    // Inject not-yet-persisted candidate boxes (preview bboxUuid:'new' case).
    // No-op when absent, so the default review-grid path is untouched.
    if (_seedBboxes?.length) seedRuntimeBboxes(instance.runtime, _seedBboxes);
  } catch (err) {
    Logger.error('Canvas modal: failed to load image', err);
    instance?.destroy();
    _layer.release();
    releaseFocusTrap();
    overlay.remove();
    _modalOpen = false;
    return;
  }

  // ── Wire shared toolbar (select / bbox / delete) ───────────────────
  toolbarHandle = initToolbar(instance.runtime, toolbarEl, {
    buttons: ['undo', 'redo', 'select', 'bbox', 'clear', 'delete', 'copy', 'paste', 'enhance', 'search'],
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

  // Highlight the initially-opened bbox.
  //
  // This must happen before the awaited panel setup below, not after it: the
  // toolbar is live from here on, and its Delete button acts on the current
  // selection via deleteSelected(), which is a silent no-op when nothing is
  // selected. Highlighting only after createSidebarPanels/createInferencePanel
  // (the latter awaits three cold model fetches) left a window in which Delete
  // was clickable but did nothing.
  function _highlightBbox(bboxUuid) {
    const layer = instance.runtime.canvas.layer;
    const view = layer.findOne('#' + bboxUuid);
    if (view) {
      // Goes through the shared selection path rather than attaching the
      // transformer directly: the highlight puts live resize anchors on screen
      // before the user has clicked anything, and those anchors have to mark
      // the canvas dirty like any other selection's would.
      selectBboxView(instance.runtime, view);
      layer.draw();
    }
  }
  // No bbox when the canvas was opened from an image with none yet — there is
  // simply nothing to highlight, and the editor opens ready to draw one.
  if (bbox) _highlightBbox(bbox.uuid);

  // ── Wire label list + tag panel ────────────────────────────────────
  sidebarPanelsHandle = await createSidebarPanels(instance.runtime, {
    labelsListEl: labelListEl,
    tagsPaneEl: paneTags,
  });

  // ── Wire inference panel (model list, Run/Refresh/Clear-Grey, confidence) ──
  // Use a child wrapper, not paneInference itself: createInferencePanel adds the
  // `.inference-panel` class (display:flex, unconditional) to its container, which
  // would collide with `.sidebar-tab-pane`'s display:none-unless-.active toggling.
  const inferencePanelContentEl = document.createElement('div');
  paneInference.appendChild(inferencePanelContentEl);
  inferencePanelHandle = await createInferencePanel(instance.runtime, inferencePanelContentEl);
  unsubInference = wireInferenceRunner(instance.runtime, { logPrefix: 'Canvas modal: ' });

  // ── Sidebar tab-switching rules ────────────────────────────────────
  unsubTabRules = wireSidebarTabRules(instance.runtime, {
    activateTab: activateSidebarTab,
    labelsTabId: MODAL_TABS.labels,
    tagsTabId: MODAL_TABS.tags,
  });

  // ── Save button state: highlight only when there are unsaved changes ──
  // With nothing to save, "Save & Close" becomes plain "Close" and does just
  // that. It used to keep the save wording and return early on click, so the
  // most obvious button in the footer was a dead control whenever the canvas
  // was clean — which is most of the time when browsing.
  function _updateSaveBtn() {
    if (saveBtn.disabled) return; // don't override while saving
    const dirty = instance.runtime.isDirty;
    saveBtn.className = dirty ? 'save-btn--dirty' : 'save-btn--clean';
    saveBtn.textContent = dirty ? 'Save & Close' : 'Close';
    saveOnlyBtn.className = dirty ? 'save-btn--dirty' : 'save-btn--clean';
  }
  const unsubDirty = events.subscribeToCanvas(
    instance.runtime.id, EventTypes.CANVAS_BBOXES_CHANGED, _updateSaveBtn,
  );

  // ── Navigation: wire [ ] hotkeys to navigate within the results page ──
  let unsubNavLoad = null;
  let _updateNavButtons = () => {};
  let _prefetchNavNeighbours = () => {};
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

    // Visible fallback for the [ ] hotkeys — same navObj, so click and keypress
    // stay in lockstep. Only shown when there's actually something to navigate.
    prevBtn.style.display = '';
    nextBtn.style.display = '';
    _updateNavButtons = () => {
      prevBtn.disabled = _navIndex <= 0;
      nextBtn.disabled = _navIndex >= navContext.items.length - 1;
    };
    _updateNavButtons();
    prevBtn.addEventListener('click', () => prev_image(instance.runtime));
    nextBtn.addEventListener('click', () => next_image(instance.runtime));

    // Warm the images on either side of where we are, so a `]` press on a slow
    // DB reads an already-decoded copy instead of paying the full
    // read-file + base64 + metadata-query round trip again.
    //
    // Both directions, not just forward: back-stepping through a run of boxes
    // is as common as forward, and the cache is shared, so a step that reverses
    // direction still lands on something warm.
    _prefetchNavNeighbours = () => {
      for (const offset of NAV_PREFETCH_OFFSETS) {
        prefetchImage(navContext.items[_navIndex + offset]?.img.filename);
      }
    };
    _prefetchNavNeighbours();

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
        // Nav buttons already reflect the new _navIndex's boundary state via
        // _updateNavButtons(); force both disabled during the load itself so a
        // slow fetch (large DB, cold thumbnail cache) can't be double-clicked.
        prevBtn.disabled = true;
        nextBtn.disabled = true;
        prevBtn.classList.add('review-toolbar__status--loading');
        nextBtn.classList.add('review-toolbar__status--loading');
        try {
          // The only load allowed to come from cache: stepping through a list
          // resolved when the modal opened, where a prefetched copy is still
          // what the server would return. Save/external-change reloads below
          // stay uncached on purpose.
          await instance.loadImage(item.img.filename, { allowCached: true });
        } catch (err) {
          Logger.error('Canvas modal: failed to load image during navigation', err);
          toast('Failed to load image', 3000, 'error');
          titleEl.textContent = `Edit: ${_currentImg.filename.split('/').slice(-2).join('/')}`;
          return; // Don't update state — canvas and _currentImg stay in sync
        } finally {
          prevBtn.classList.remove('review-toolbar__status--loading');
          nextBtn.classList.remove('review-toolbar__status--loading');
          _updateNavButtons();
        }
        _currentImg = item.img;
        _currentBbox = targetBbox;
        if (_currentBbox) _highlightBbox(_currentBbox.uuid);
        // Slide the prefetch window along AFTER the visible load has settled,
        // so speculative fetches never compete with the one the user is waiting
        // on for a connection.
        _prefetchNavNeighbours();
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
    _layer.release();
    releaseFocusTrap();
    unsubBboxChanged();
    unsubDirty();
    if (unsubNavLoad) unsubNavLoad();
    unsubTabRules();
    unsubInference();
    unregisterSaveAction();
    sidebarPanelsHandle.destroy();
    inferencePanelHandle.destroy();
    toolbarHandle.destroy();
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
      // Always include the originally-opened bbox — absent when the canvas was
      // opened on an image that had none.
      ...(bbox ? [bbox.uuid] : []),
    ])];
    await invalidateBboxThumbnails(allUuids);
    const currentBboxesMap = new Map(instance.runtime.state.image.bboxes);
    const openedUuid = currentBbox?.uuid ?? bbox?.uuid;
    const updatedUIBbox = openedUuid ? currentBboxesMap.get(openedUuid) : undefined;
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
    deleteImageBtn.disabled = disabled;
    if (disabled) {
      prevBtn.disabled = true;
      nextBtn.disabled = true;
    } else {
      _updateNavButtons(); // restore prev/next's own boundary-based disabled state
    }
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

  // ── Delete-image button: permanently removes the whole image, not just the
  // annotations being edited. Skips the unsaved-changes prompt entirely — once
  // the image is gone, any pending bbox edits are moot.
  deleteImageBtn.addEventListener('click', async () => {
    const targetImg = _currentImg;
    const filename = targetImg.filename?.split('/').pop() ?? targetImg.filename;
    const confirmed = await showConfirmModal({
      title: 'Delete image',
      message: `Permanently delete "${filename}"? This removes the file and all its annotations from disk. This cannot be undone.`,
      confirmText: 'Delete',
      danger: true,
    });
    if (!confirmed) return;

    _setAllButtonsDisabled(true);
    try {
      await deleteImage(targetImg.uuid);
    } catch (err) {
      Logger.error('Canvas modal: delete image failed', err);
      toast(`Failed to delete image: ${err.message}`, 4000, 'error');
      _setAllButtonsDisabled(false);
      return;
    }

    events.publish(EventTypes.IMAGE_DELETED, { imageUuid: targetImg.uuid, imagePath: targetImg.filename });
    closeModal();
    navContext?.onClose?.(targetImg);
  });

  // ESC key — behaves like the Cancel button (shows unsaved-changes dialog if dirty)
  _canvasEscClose = () => { if (!closeBtn.disabled) tryClose(); };
  registerEscHandler(_canvasEscClose);

  // ── Save (without closing) ────────────────────────────────────────
  // Named rather than inline: the Save hotkey fires this same path (registered
  // as this runtime's 'save' action below), so keypress and click cannot drift
  // apart — both go through _persist, both restore the button states, and both
  // broadcast under the same conditions.
  async function _doSaveOnly() {
    if (saveOnlyBtn.disabled) return;      // a save (or a close) is already in flight
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
  }

  saveOnlyBtn.addEventListener('click', _doSaveOnly);
  unregisterSaveAction = registerCanvasAction(instance.runtime.id, 'save', _doSaveOnly);

  // ── Save & Close button ────────────────────────────────────────────
  saveBtn.addEventListener('click', async () => {
    // Reads "Close" in this state. Routed through tryClose rather than
    // closeModal so it behaves exactly like Cancel — including re-prompting if
    // the canvas turned dirty between the label update and the click.
    if (!instance.runtime.isDirty) {
      await tryClose();
      return;
    }
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
      // _updateSaveBtn owns the label — a failed save leaves the canvas dirty,
      // so it restores 'Save & Close' itself.
      _setAllButtonsDisabled(false);
      _updateSaveBtn();
    }
  });

  // Every handler is attached — release the buttons disabled at build time.
  _setAllButtonsDisabled(false);
  _updateSaveBtn();
}

// ── Shared preview helpers ────────────────────────────────────────────

function _makeBboxThumbGroup(entries, label) {
  const group = document.createElement('div');
  group.className = 'image-compare-group';
  const thumbs = document.createElement('div');
  thumbs.className = 'image-compare-thumbs';
  for (const { bbox, img } of entries) {
    const wrapper = document.createElement('div');
    wrapper.className = 'image-compare-thumb-wrap';
    const thumb = document.createElement('img');
    thumb.className = 'image-compare-thumb';
    thumb.src = getBboxThumbnailUrl(img.filename, bbox.uuid, 80);
    wrapper.appendChild(thumb);
    thumbs.appendChild(wrapper);
  }
  const lbl = document.createElement('div');
  lbl.className = 'image-compare-group-label';
  lbl.textContent = label;
  group.append(thumbs, lbl);
  return group;
}

function _makeImageThumbGroup(images, label) {
  const group = document.createElement('div');
  group.className = 'image-compare-group';
  const thumbs = document.createElement('div');
  thumbs.className = 'image-compare-thumbs';
  for (const img of images) {
    const wrapper = document.createElement('div');
    wrapper.className = 'image-compare-thumb-wrap';
    const thumb = document.createElement('img');
    thumb.className = 'image-compare-thumb';
    thumb.src = getImageThumbnailUrl(img.filename, 80);
    thumb.title = img.filename ?? '';
    wrapper.appendChild(thumb);
    thumbs.appendChild(wrapper);
  }
  const lbl = document.createElement('div');
  lbl.className = 'image-compare-group-label';
  lbl.textContent = label;
  group.append(thumbs, lbl);
  return group;
}

/**
 * Populate a preview row with first-N / last-N image thumbnails — the
 * image-level twin of _populateBboxPreviewRow, so "Apply to Filtered Images"
 * shows the same both-ends-of-the-range sanity check as "Apply to All".
 */
function _populateImagePreviewRow(row, firstImgs, lastImgs) {
  row.innerHTML = '';
  if (firstImgs.length === 0 && lastImgs.length === 0) {
    const msg = document.createElement('span');
    msg.className = 'image-compare-empty-msg';
    msg.textContent = 'No preview available.';
    row.appendChild(msg);
    return;
  }
  const firstUuids = new Set(firstImgs.map(i => i.uuid));
  const uniqueLast = lastImgs.filter(i => !firstUuids.has(i.uuid));
  if (firstImgs.length > 0) row.appendChild(_makeImageThumbGroup(firstImgs, 'First'));
  if (uniqueLast.length > 0) {
    const sep = document.createElement('div');
    sep.className = 'image-compare-sep';
    sep.textContent = '···';
    row.appendChild(sep);
    row.appendChild(_makeImageThumbGroup(uniqueLast, 'Last'));
  }
}

/** Populate a preview row with first-N / last-N bbox thumbnails. */
function _populateBboxPreviewRow(row, first3, last3) {
  if (first3.length === 0 && last3.length === 0) {
    const msg = document.createElement('span');
    msg.className = 'image-compare-empty-msg';
    msg.textContent = 'No preview available.';
    row.appendChild(msg);
    return;
  }
  const first3Uuids = new Set(first3.map(e => e.bbox.uuid));
  const uniqueLast3 = last3.filter(e => !first3Uuids.has(e.bbox.uuid));
  if (first3.length > 0) row.appendChild(_makeBboxThumbGroup(first3, 'First'));
  if (uniqueLast3.length > 0) {
    const sep = document.createElement('div');
    sep.className = 'image-compare-sep';
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
  overlay.className = 'bulk-modal-overlay';
  const _layer = acquireModalLayer({ dialog: true, overlay });
  overlay.style.zIndex = _layer.zIndex;

  const dialog = document.createElement('div');
  dialog.className = 'bulk-modal-dialog';

  const title = document.createElement('strong');
  title.textContent = titleText ?? `Apply to ${entries.length} selected bbox${entries.length === 1 ? '' : 'es'}`;

  // ── Preview section (opt-in) ─────────────────────────────────────────
  if (opts.showPreview) {
    const bboxEntries = entries.filter(e => e.kind === 'bbox');
    const previewSection = document.createElement('div');
    previewSection.className = 'bulk-modal-preview-section';

    const previewHdr = document.createElement('div');
    previewHdr.className = 'bulk-modal-preview-hdr';
    previewHdr.textContent = 'Preview range';

    const previewRow = document.createElement('div');
    previewRow.className = 'bulk-modal-preview-row';

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
    { value: 'addTags', label: 'Add Tags' },
    { value: 'removeTags', label: 'Remove Tags' },
    { value: 'delete', label: 'Delete Bboxes' },
  ];

  const radioGroup = document.createElement('div');
  radioGroup.className = 'bulk-modal-radio-group';
  let selectedAction = 'relabel';

  for (const action of actions) {
    const lbl = document.createElement('label');
    lbl.className = 'bulk-modal-action-label';
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
  labelPickerRow.className = 'bulk-modal-label-picker-row';
  const labelPickerHdr = document.createElement('div');
  labelPickerHdr.textContent = 'New label:';
  labelPickerHdr.className = 'bulk-modal-label-picker-hdr';
  labelPickerRow.appendChild(labelPickerHdr);
  labelPickerRow.appendChild(dynPicker.element);

  // Tag picker (shown only for addTags / removeTags)
  const tagOptions = filterVisibleTags(tags.values())
    .map(tag => ({
      key: tag.uuid,
      text: (tag.icon ? tag.icon + ' ' : '') + tag.name,
      color: tag.color || '#cccccc',
    }));
  const tagField = new MultiSelectField({ label: 'Tags', value: [], options: tagOptions, emptyText: 'No tags defined yet' });
  const tagPickerRow = document.createElement('div');
  tagPickerRow.className = 'bulk-modal-tag-picker-row';
  tagPickerRow.appendChild(await tagField.renderEdit());

  const btnRow = document.createElement('div');
  btnRow.className = 'bulk-modal-btn-row';

  function close() { unregisterEscHandler(close); _layer.release(); dynPicker.destroy(); overlay.remove(); }

  const cancelBtn = document.createElement('button');
  cancelBtn.textContent = 'Cancel';
  cancelBtn.addEventListener('click', close);

  const okBtn = document.createElement('button');
  okBtn.textContent = 'Apply';
  okBtn.className = 'bulk-modal-ok-btn';
  okBtn.addEventListener('click', async () => {
    close();
    // _selectedLabelUuid: undefined = picker not used (no change), null = clear label, string = set label
    const newLabelUuid = selectedAction === 'relabel' ? (_selectedLabelUuid ?? undefined) : undefined;
    // Verification shortcuts are sent as addTags with the appropriate system tag UUID.
    const resolvedAction = _VERIF_TAG_MAP[selectedAction] ? 'addTags' : selectedAction;
    const selectedTagUuids = _VERIF_TAG_MAP[selectedAction]
      ? [_VERIF_TAG_MAP[selectedAction]]
      : (selectedAction === 'addTags' || selectedAction === 'removeTags') ? tagField.getValue() : [];
    await onConfirm(resolvedAction, newLabelUuid, selectedTagUuids, tags);
  });

  btnRow.appendChild(cancelBtn);
  btnRow.appendChild(okBtn);
  dialog.append(radioGroup, labelPickerRow, tagPickerRow, btnRow);
  overlay.appendChild(dialog);
  document.body.appendChild(overlay);
  applyModalCascade(dialog, _layer.depth);
  registerEscHandler(close);

  overlay.addEventListener('click', e => { if (e.target === overlay) close(); });
}

// TAG_ACTIONS / showImageBulkActionModal / executeBulkImageAction moved to
// imageBulkActionModal.js so they can be reused outside the review grid's own
// toolbar (e.g. runClassifierOnResults.js's classifier-match confirmation step).
