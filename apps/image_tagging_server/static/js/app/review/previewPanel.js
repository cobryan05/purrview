/**
 * Triage panel for a **persisted preview run** — the "Run Model on Results"
 * confirmation screen, rebuilt on server-side state.
 *
 * What this replaces and why (PREVIEW_REMODEL.md §1, §7.2):
 *
 *  - The old confirmation modal held every candidate in a JS array inside its
 *    own closure. Closing it — backdrop click, ESC, a stray click-away — threw
 *    away the whole run, and re-getting it meant re-running inference.
 *  - There was no per-candidate state, so a 3,000-candidate run had to be
 *    triaged in one sitting or not at all. Decisions here are persisted the
 *    moment they are made (debounced PATCH), so a run is resumable.
 *  - Every card was built up front, with no pagination, no filtering and no
 *    sort control. This pages against the server and filters on candidate
 *    fields — proposed label, confidence band, decision, new-vs-existing —
 *    which are the ones that matter for triaging model output.
 *
 * ⚠️ This is deliberately **not** `createReviewPanel` (§3). That panel builds
 * its entries from live `ImageRecordRead`s, and a candidate is not one: it
 * carries a proposed label / confidence / proposed geometry that no bbox row
 * has, and an `isNew` candidate has no bbox row at all. It does share
 * `cardGrid.js`, the modal primitives and the context-menu system.
 *
 * `createPreviewPanel(container, opts)` returns `{ refresh, destroy }` — the
 * same shape as `createReviewPanel`, so it can mount in a modal or a tab. Only
 * the modal is wired today (§12 Q3); the shape costs nothing and keeps the tab
 * option open.
 */

import {
  applyPreviewRun,
  deletePreviewRun,
  fetchPreviewRun,
  fetchPreviewRuns,
  queryPreviewCandidates,
  updatePreviewCandidates,
  updatePreviewRun,
  isReportRun,
  PreviewDecision,
  PreviewDecisionAction,
} from '/app-static/js/shared/api/previewRuns.js';
import { fetchLabels } from '/app-static/js/shared/api/labels.js';
import { deleteImage as apiDeleteImage } from '/app-static/js/shared/api/images.js';
import { events, EventTypes } from '/app-static/js/shared/events/index.js';
import { flattenHierarchy } from '/app-static/js/ui/labelList.js';
import { expandLabelFilterSet } from '/app-static/js/app/review/filterPredicate.js';
import { buildCandidateCard, createCardGrid } from '/app-static/js/app/review/cardGrid.js';
import { labelChip } from '/app-static/js/app/review/labelChip.js';
import { createHoverPreview } from '/app-static/js/ui/thumbnailHoverPreview.js';
import { createModalShell } from '/app-static/js/ui/utils/modalShell.js';
import { createMultiSelectDropdown } from '/app-static/js/ui/utils/createMultiSelectDropdown.js';
import { prefs } from '/app-static/js/shared/prefs.js';
import { showConfirmModal } from '/app-static/js/ui/utils/confirmModal.js';
import { toast, Logger } from '/app-static/js/ui/utils/logging.js';
// Shared with the event modal, which names the same findings from the other end of the
// pipeline. Two spellings of "detector dropout" read as two different problems.
import { blipKindText } from '/app-static/js/shared/domain/blips.js';

/** Candidates per page, until the user says otherwise. Matches the review grid's feel
 * on a full-width modal. */
const DEFAULT_PAGE_SIZE = 60;

/** The page sizes on offer.
 *
 * A run over a whole filter routinely returns five figures, and at a fixed 60 that is
 * "page 1 / 500" — a number that makes the pager useless as navigation and the grid
 * useless as a survey. The big sizes are there to be *chosen*, not defaulted to: 500
 * cards is a real scroll and a real number of thumbnails, which is exactly why the
 * default stays where it was.
 */
const PAGE_SIZES = [30, 60, 120, 250, 500];

/** Remembered across runs and sessions: a page size is a working preference, not a
 * property of the run being triaged. */
const PAGE_SIZE_KEY = 'ww_preview_page_size';

/** How long a decision sits in the pending-write buffer before it is flushed. */
const PATCH_DEBOUNCE_MS = 400;

const DECISION_FILTER_ITEMS = [
  { uuid: PreviewDecision.PENDING, name: 'Unreviewed' },
  { uuid: PreviewDecision.ACCEPTED, name: 'Accepted' },
  { uuid: PreviewDecision.REJECTED, name: 'Rejected' },
  { uuid: PreviewDecision.APPLIED, name: 'Applied' },
  { uuid: PreviewDecision.STALE, name: 'Stale' },
];

/**
 * Ordering modes. The two `*_confidence` modes are *groupings*: the server
 * keeps every candidate of one image / one proposed label contiguous, ranked by
 * that group's best candidate, so a group survives page boundaries. Label
 * grouping additionally gets a header per class in the grid — triaging "all the
 * cats the model found" is the case the flat confidence sort could not serve.
 */
const SORT_MODES = [
  { value: 'label_confidence', text: 'Group: label', group: 'label' },
  { value: 'image_confidence', text: 'Group: image', group: 'image' },
  // Tracking-blip runs only. `HarvestBlipFramesTask` writes one incident's frames
  // consecutively and in time order, so the server's existing found-order sort already
  // returns them contiguous and chronological — the grouping is the only new part, and
  // no new server ordering is needed. `serverSort` is what actually goes on the wire.
  { value: 'blip', text: 'Group: tracking incident', group: 'blip', serverSort: 'created_at' },
  { value: 'confidence', text: 'Sort: confidence', group: null },
  { value: 'created_at', text: 'Sort: found order', group: null },
];


/**
 * Mount the triage panel for one run.
 *
 * @param {HTMLElement} container
 * @param {Object} opts
 * @param {string} opts.runUuid
 * @param {Map<string, object>} [opts.labels] - label lookup; fetched if omitted.
 * @param {() => void} [opts.onApplied] - reload the host review page after an apply.
 * @param {(counts: object) => void} [opts.onCountsChanged]
 * @returns {Promise<{refresh: () => Promise<void>, destroy: () => void}>}
 */
export async function createPreviewPanel(container, { runUuid, labels = null, onApplied = null, onCountsChanged = null } = {}) {
  const _labels = labels ?? await fetchLabels().catch(() => new Map());
  const hoverPreview = createHoverPreview();

  let run = await fetchPreviewRun(runUuid);
  let page = 0;
  // Read once here rather than on every render: a stored value that is no longer on
  // offer (an older build's size) falls back rather than putting the select into a
  // state its own options cannot express.
  const _storedSize = prefs.read(PAGE_SIZE_KEY, DEFAULT_PAGE_SIZE);
  let pageSize = PAGE_SIZES.includes(_storedSize) ? _storedSize : DEFAULT_PAGE_SIZE;
  let totalCount = 0;
  /** @type {object[]} candidates on the current page, in render order. */
  let candidates = [];
  let destroyed = false;

  // ── Pending decision writes ────────────────────────────────────────────
  // Decisions are persisted eagerly so closing the panel loses nothing — the
  // actual fix for the click-away complaint. They are also *debounced and
  // coalesced by target state*: rubber-banding a selection of 200 cards through
  // accept → reject → accept must be one PATCH, not 600.
  /** @type {Map<string, {decision?: string, decisionAction?: string, proposedLabelUuid?: string, geometry?: object}>} */
  const _pending = new Map();
  let _flushTimer = null;
  let _inFlight = Promise.resolve();

  function _queuePatch(candidateUuids, changes) {
    for (const uuid of candidateUuids) {
      _pending.set(uuid, { ..._pending.get(uuid), ...changes });
    }
    clearTimeout(_flushTimer);
    _flushTimer = setTimeout(() => { _flushPending(); }, PATCH_DEBOUNCE_MS);
  }

  async function _flushPending() {
    clearTimeout(_flushTimer);
    if (_pending.size === 0) return _inFlight;
    // Group by identical change-set so one PATCH covers every candidate moving
    // to the same state — the common case for a bulk accept.
    const byChange = new Map();
    for (const [uuid, changes] of _pending) {
      const key = JSON.stringify(changes);
      if (!byChange.has(key)) byChange.set(key, { changes, uuids: [] });
      byChange.get(key).uuids.push(uuid);
    }
    _pending.clear();

    _inFlight = _inFlight.then(async () => {
      for (const { changes, uuids } of byChange.values()) {
        try {
          await updatePreviewCandidates(runUuid, uuids, changes);
        } catch (err) {
          Logger.error('Failed to persist preview decisions:', err);
          toast(`Could not save ${uuids.length} decision(s): ${err.message}`, 5000, 'error');
        }
      }
    });
    return _inFlight;
  }

  // ── Layout ─────────────────────────────────────────────────────────────
  // Layout/spacing lives in app.css (.preview-panel*): the chrome around the
  // grid is the scarce resource here — every row of toolbar is a row of
  // candidates the user cannot see — so it is styled in one place that can be
  // tightened, not spread across inline cssText.
  const root = document.createElement('div');
  root.className = 'preview-panel';
  root.dataset.testid = 'preview-panel';
  container.appendChild(root);

  const toolbar = document.createElement('div');
  toolbar.className = 'preview-panel__toolbar';
  root.appendChild(toolbar);

  const grid = document.createElement('div');
  grid.className = 'preview-panel__grid relabel-modal-grid review-grid';
  const body = document.createElement('div');
  body.className = 'preview-panel__body';
  body.appendChild(grid);
  root.appendChild(body);

  const footer = document.createElement('div');
  footer.className = 'preview-panel__footer';
  root.appendChild(footer);

  // ── Filter bar (candidate filters — see §3 for why not createFilterPanel) ──
  const filterState = {
    decisions: [],
    proposedLabelUuids: [],
    // The "≠" halves. Every other multi-select in the app offers exclusion, and its
    // absence here was felt hardest in exactly the place it is most useful: a run of
    // twelve thousand candidates is triaged by taking classes *away*, and "everything
    // except the ones I have already accepted" had no expression at all.
    excludeDecisions: [],
    excludeProposedLabelUuids: [],
    confidenceMin: null,
    confidenceMax: null,
    isNew: null,
    // Label grouping is the default view: triage is per class far more often
    // than per frame, and the grouped grid reads as "all the cats, then all the
    // dogs" rather than a confidence-ordered mix.
    sortBy: 'label_confidence',
    sortOrder: 'desc',
  };

  /** The grouping the current sort implies, or null for a flat ordering. */
  function _activeGrouping() {
    return SORT_MODES.find(m => m.value === filterState.sortBy)?.group ?? null;
  }

  /**
   * The filter state as the server understands it.
   *
   * A grouping mode may be a client-side reading of an ordering the server already
   * supports, in which case only the grouping is local and `serverSort` is what goes on
   * the wire. Sending the local name instead would reach an endpoint that has never
   * heard of it — which does not fail, it silently falls back to a default order, and
   * the grouping then looks broken rather than the sort.
   */
  function _queryState() {
    const mode = SORT_MODES.find(m => m.value === filterState.sortBy);
    return { ...filterState, sortBy: mode?.serverSort ?? filterState.sortBy };
  }

  const decisionDd = createMultiSelectDropdown({
    label: 'Decision',
    items: DECISION_FILTER_ITEMS,
    showModeToggle: false,
    filterKey: 'preview-decision',
    onChange: (selected, excluded) => {
      filterState.decisions = selected;
      filterState.excludeDecisions = excluded;
      _reload(0);
    },
  });
  decisionDd.element.dataset.testid = 'preview-filter-decision';
  toolbar.appendChild(decisionDd.element);

  // Hierarchical, with "include child labels" — the same contract the review
  // filter panel's Labels dropdown has. Picking a parent and getting only its
  // own boxes is the surprising answer when the vocabulary is a tree.
  //
  // The expansion is done here rather than server-side: the candidates endpoint
  // filters on `proposedLabelUuids` membership, so the panel sends the already
  // expanded set. `expandLabelFilterSet` is the same helper the review grid's
  // predicate uses, so the two can't drift on what "include children" means.
  let _labelSelection = [];
  let _labelExclusion = [];
  let _labelIncludeChildren = false;

  function _applyLabelFilter() {
    // Both halves expand the same way. "Not this parent" has to mean "not its children
    // either" when the toggle is on, or excluding "Cat" would leave every named cat
    // behind and read as having done nothing.
    filterState.proposedLabelUuids = _labelSelection.length > 0
      ? [...expandLabelFilterSet(_labelSelection, _labels, _labelIncludeChildren)]
      : [];
    filterState.excludeProposedLabelUuids = _labelExclusion.length > 0
      ? [...expandLabelFilterSet(_labelExclusion, _labels, _labelIncludeChildren)]
      : [];
    _reload(0);
  }

  const labelDd = createMultiSelectDropdown({
    label: 'Label',
    items: flattenHierarchy(_labels).map(l => ({
      uuid: l.uuid,
      name: l.name,
      color: l.color || null,
      depth: l.depth,
      parentUuid: l.parentUuid ?? null,
    })),
    showModeToggle: false,
    searchable: true,
    filterKey: 'preview-proposed-label',
    hierarchyToggle: {
      label: 'Include child labels',
      default: false,
      onChange: (val) => { _labelIncludeChildren = val; _applyLabelFilter(); },
    },
    onChange: (selected, excluded) => {
      _labelSelection = selected;
      _labelExclusion = excluded;
      _applyLabelFilter();
    },
  });
  labelDd.element.dataset.testid = 'preview-filter-label';
  toolbar.appendChild(labelDd.element);

  // Confidence band. A number pair rather than a slider: triage is usually
  // "everything above 90%" or "the uncertain 40–60% band", both of which are
  // typed faster than dragged.
  const confWrap = document.createElement('label');
  confWrap.className = 'preview-panel__conf';
  confWrap.title = 'Confidence band (%)';
  confWrap.appendChild(document.createTextNode('Conf'));
  const confMin = document.createElement('input');
  const confMax = document.createElement('input');
  for (const [el, testid, placeholder] of [[confMin, 'preview-conf-min', '0'], [confMax, 'preview-conf-max', '100']]) {
    el.type = 'number';
    el.min = '0';
    el.max = '100';
    el.step = '5';
    el.placeholder = placeholder;
    el.dataset.testid = testid;
  }
  const _readConfidence = () => {
    const lo = confMin.value === '' ? null : Number(confMin.value) / 100;
    const hi = confMax.value === '' ? null : Number(confMax.value) / 100;
    filterState.confidenceMin = Number.isFinite(lo) ? lo : null;
    filterState.confidenceMax = Number.isFinite(hi) ? hi : null;
    _reload(0);
  };
  confMin.addEventListener('change', _readConfidence);
  confMax.addEventListener('change', _readConfidence);
  confWrap.append(confMin, document.createTextNode('–'), confMax);
  toolbar.appendChild(confWrap);

  const kindSelect = document.createElement('select');
  kindSelect.dataset.testid = 'preview-filter-kind';
  kindSelect.title = 'Show existing boxes, proposed new boxes, or both';
  for (const [value, text] of [['', 'All boxes'], ['existing', 'Existing'], ['new', 'New']]) {
    const opt = document.createElement('option');
    opt.value = value;
    opt.textContent = text;
    kindSelect.appendChild(opt);
  }
  kindSelect.addEventListener('change', () => {
    filterState.isNew = kindSelect.value === '' ? null : kindSelect.value === 'new';
    _reload(0);
  });
  toolbar.appendChild(kindSelect);

  const sortSelect = document.createElement('select');
  sortSelect.dataset.testid = 'preview-sort';
  sortSelect.title = 'Grouping / ordering. Grouped modes keep a whole image or a '
    + 'whole class together, across pages.';
  for (const { value, text } of SORT_MODES) {
    const opt = document.createElement('option');
    opt.value = value;
    opt.textContent = text;
    sortSelect.appendChild(opt);
  }
  sortSelect.value = filterState.sortBy;
  sortSelect.addEventListener('change', () => { filterState.sortBy = sortSelect.value; _reload(0); });
  toolbar.appendChild(sortSelect);

  const countLabel = document.createElement('span');
  countLabel.dataset.testid = 'preview-count';
  countLabel.className = 'preview-panel__count';
  toolbar.appendChild(countLabel);

  // ── Selection controller (click / shift-range, shared with the review grid) ──
  const _visibleEntries = [];
  const cardGrid = createCardGrid(grid, {
    getVisibleEntries: () => _visibleEntries,
    entryId: (entry) => entry.id,
    onSelectionChange: () => _updateDecisionButtons(),
    hooks: { toast: (m) => toast(m, 3000, 'warning') },
  });

  /** Candidates the decision buttons act on: the selection, or the whole page. */
  function _targets() {
    if (cardGrid.selected.size === 0) return [];
    return _visibleEntries.filter(e => cardGrid.selected.has(e.id)).map(e => e.candidate);
  }

  // ── Decision actions (replacing the old global Relabel/Delete radio) ────
  // The radio could only be one or the other for a whole run. Decisions are per
  // candidate now, which is strictly more expressive: a run can relabel most of
  // its boxes and delete a handful.
  function _setDecision(targets, decision, decisionAction = undefined) {
    if (targets.length === 0) return;
    const changes = { decision };
    if (decisionAction !== undefined) changes.decisionAction = decisionAction;
    const counts = run?.decisionCounts;
    for (const c of targets) {
      // Keep the run's decision tally in step locally. The PATCH is debounced
      // and the run is only re-fetched on reload, so without this the Apply
      // button would still read "Apply 0 accepted" (and stay disabled)
      // immediately after the user accepted a screenful of candidates.
      if (counts && c.decision !== decision) {
        counts[c.decision] = Math.max(0, (counts[c.decision] ?? 0) - 1);
        counts[decision] = (counts[decision] ?? 0) + 1;
      }
      c.decision = decision;
      if (decisionAction !== undefined) c.decisionAction = decisionAction;
      const card = grid.querySelector(`[data-candidate-uuid="${c.uuid}"]`);
      if (card) card.dataset.decision = decision;
    }
    _queuePatch(targets.map(c => c.uuid), changes);
    _renderPage();
  }

  // The selection count lives here once, rather than being appended to all four
  // button labels — four widening buttons is what wrapped the footer onto a
  // second row mid-triage, and a row of footer is a row of candidates.
  const selCount = document.createElement('span');
  selCount.className = 'preview-panel__selcount';
  selCount.dataset.testid = 'preview-selection-count';
  footer.appendChild(selCount);

  /**
   * Select (or deselect) a set of entries, keeping the controller's Set, the
   * card class and the card checkbox in step — the three things `applySelect`
   * normally updates together. Shared by the select-all/none buttons and the
   * per-label group header's "Select group".
   */
  function _setSelection(entries, selected) {
    for (const en of entries) {
      if (selected) cardGrid.selected.add(en.id);
      else cardGrid.selected.delete(en.id);
      en.card?.classList.toggle('selected', selected);
      const cb = en.card?.querySelector('input[type=checkbox]');
      if (cb && !cb.disabled) cb.checked = selected;
    }
    if (selected) cardGrid.selectionKind = 'bbox';
    _updateDecisionButtons();
  }

  // Select-all / deselect-all over the current page. Page-scoped on purpose:
  // decisions apply to what you can see, and the whole-run equivalent is the
  // filter plus "Accept all", which is right next to these.
  const selectAllBtn = document.createElement('button');
  selectAllBtn.className = 'secondary';
  selectAllBtn.dataset.testid = 'preview-select-all';
  selectAllBtn.textContent = 'All';
  selectAllBtn.title = 'Select every candidate on this page';
  selectAllBtn.addEventListener('click', () => _setSelection(_visibleEntries, true));
  footer.appendChild(selectAllBtn);

  const selectNoneBtn = document.createElement('button');
  selectNoneBtn.className = 'secondary';
  selectNoneBtn.dataset.testid = 'preview-select-none';
  selectNoneBtn.textContent = 'None';
  selectNoneBtn.title = 'Clear the selection';
  selectNoneBtn.addEventListener('click', () => {
    // clearSelection also resets the controller's select-all checkbox state and
    // type fading (and fires onSelectionChange, which refreshes the buttons) —
    // a bare Set.clear() would leave all of that stale.
    cardGrid.clearSelection();
  });
  footer.appendChild(selectNoneBtn);

  function _decisionButton(text, testid, title, onClick) {
    const btn = document.createElement('button');
    btn.className = 'secondary';
    btn.dataset.testid = testid;
    btn.textContent = text;
    btn.title = title;
    btn.addEventListener('click', onClick);
    footer.appendChild(btn);
    return btn;
  }

  const acceptBtn = _decisionButton('Accept', 'preview-accept',
    'Accept the selected candidates as relabels', () =>
      _setDecision(_targets(), PreviewDecision.ACCEPTED, PreviewDecisionAction.RELABEL));
  const acceptDeleteBtn = _decisionButton('Accept ⌫', 'preview-accept-delete',
    'Accept as delete: applying will delete these boxes instead of relabeling them', () =>
      _setDecision(_targets(), PreviewDecision.ACCEPTED, PreviewDecisionAction.DELETE));
  const rejectBtn = _decisionButton('Reject', 'preview-reject',
    'Reject the selected candidates (reversible)', () =>
      _setDecision(_targets(), PreviewDecision.REJECTED));
  const resetBtn = _decisionButton('Unreview', 'preview-reset',
    'Put the selected candidates back to unreviewed', () =>
      _setDecision(_targets(), PreviewDecision.PENDING));
  const deleteImageBtn = _decisionButton('Delete image', 'preview-delete-image',
    'Permanently delete the image(s) behind the selected candidates — the whole file and every '
    + 'annotation on it, not just this candidate', () => _deleteSelectedImages());
  deleteImageBtn.classList.add('preview-panel__danger-btn');

  /**
   * Deletes the underlying image(s) for the current selection, not just the
   * candidate — a candidate can only be rejected, but a harvested frame that
   * turned out worthless (blur, empty, duplicate scenery) needs to be gone
   * from the dataset entirely, and there was previously no way to do that
   * without leaving the preview run for the Review tab and re-finding it there.
   */
  async function _deleteSelectedImages() {
    const targets = _targets();
    if (!targets.length) return;
    const imageUuids = [...new Set(targets.map(c => c.imageUuid).filter(Boolean))];
    if (!imageUuids.length) return;
    const confirmed = await showConfirmModal({
      title: imageUuids.length > 1 ? 'Delete images' : 'Delete image',
      message: `Permanently delete ${imageUuids.length > 1 ? `${imageUuids.length} images` : 'this image'} `
        + 'and every annotation on it? This cannot be undone.',
      confirmText: 'Delete',
      danger: true,
    });
    if (!confirmed) return;
    try {
      await Promise.all(imageUuids.map(uuid => apiDeleteImage(uuid)));
      toast(`Deleted ${imageUuids.length.toLocaleString()} image(s)`, 4000);
      cardGrid.clearSelection();
      await _reload(page);
    } catch (err) {
      Logger.error(err);
      toast(`Failed to delete: ${err.message}`, 5000, 'error');
    }
  }

  // Bulk over the *filter*, not the page: "accept everything above 90%" should
  // not require paging through 40 pages checking boxes.
  const bulkBtn = document.createElement('button');
  bulkBtn.className = 'secondary';
  bulkBtn.dataset.testid = 'preview-accept-all-filtered';
  bulkBtn.title = 'Accept every candidate matching the current filter, not just this page';
  bulkBtn.addEventListener('click', async () => {
    const confirmed = await showConfirmModal({
      title: 'Accept all filtered',
      message: `Mark all ${totalCount.toLocaleString()} candidate(s) matching the current filter as accepted?`,
      confirmText: 'Accept all',
    });
    if (!confirmed) return;
    await _setDecisionOverFilter(PreviewDecision.ACCEPTED, PreviewDecisionAction.RELABEL);
  });
  footer.appendChild(bulkBtn);

  /**
   * Apply a decision to every candidate matching the current filter, not just
   * the page. Fetches the matching uuids in one unpaginated pass — bounded by
   * the run, and only paid when the user asks for it.
   */
  async function _setDecisionOverFilter(decision, decisionAction) {
    try {
      const { candidates: all } = await queryPreviewCandidates(runUuid, {
        ..._queryState(),
        limit: Number.MAX_SAFE_INTEGER,
        offset: 0,
      });
      if (all.length === 0) return;
      await updatePreviewCandidates(runUuid, all.map(c => c.uuid), { decision, decisionAction });
      toast(`${all.length.toLocaleString()} candidate(s) marked ${decision}`, 4000, 'success');
      await _reload(page);
    } catch (err) {
      Logger.error('Bulk decision failed:', err);
      toast(`Bulk decision failed: ${err.message}`, 5000, 'error');
    }
  }

  // Off by default: a model's suggestion isn't a human verification.
  const verifyLabel = document.createElement('label');
  verifyLabel.className = 'preview-panel__verify';
  verifyLabel.title = 'Off: applied boxes keep their current verification state (still reviewable). '
    + 'On: treat this apply as a human decision and mark the boxes verified.';
  const verifyCheckbox = document.createElement('input');
  verifyCheckbox.type = 'checkbox';
  verifyCheckbox.className = 'relabel-modal-mark-verified';
  verifyLabel.append(verifyCheckbox, document.createTextNode('Verified'));
  footer.appendChild(verifyLabel);

  const applyBtn = document.createElement('button');
  applyBtn.dataset.testid = 'preview-apply';
  applyBtn.addEventListener('click', async () => {
    // Nothing in the buffer may be lost to the apply — flush before asking the
    // server which candidates are accepted.
    await _flushPending();
    const accepted = run?.decisionCounts?.[PreviewDecision.ACCEPTED] ?? 0;
    const confirmed = await showConfirmModal({
      title: `Apply ${accepted.toLocaleString()} accepted candidate(s)`,
      message: 'Accepted candidates will be written to the database. Rejected and unreviewed '
        + 'candidates are left alone and stay in this run.',
      confirmText: 'Apply',
    });
    if (!confirmed) return;

    applyBtn.disabled = true;
    try {
      const result = await applyPreviewRun(runUuid, { markVerified: verifyCheckbox.checked });
      const parts = [];
      if (result.modified) parts.push(`${result.modified} relabeled`);
      if (result.added) parts.push(`${result.added} added`);
      if (result.deleted) parts.push(`${result.deleted} deleted`);
      if (result.failed) parts.push(`${result.failed} failed`);
      // Never silently skipped: a run can sit for days, and the user needs to
      // know which candidates the world moved out from under.
      if (result.stale) parts.push(`${result.stale} skipped (changed since the run)`);
      toast(parts.length ? `Applied: ${parts.join(', ')}` : 'Nothing to apply', 6000,
        result.failed || result.stale ? 'warning' : 'success');
      onApplied?.();
      await _reload(page);
    } catch (err) {
      Logger.error('Apply preview run failed:', err);
      toast(`Apply failed: ${err.message}`, 5000, 'error');
    } finally {
      applyBtn.disabled = false;
    }
  });
  footer.appendChild(applyBtn);

  // ── Pagination ─────────────────────────────────────────────────────────
  const pager = document.createElement('div');
  pager.className = 'preview-panel__pager';
  const prevBtn = document.createElement('button');
  prevBtn.className = 'secondary';
  prevBtn.dataset.testid = 'preview-prev-page';
  prevBtn.textContent = '‹ Prev';
  prevBtn.addEventListener('click', () => _reload(page - 1));
  const nextBtn = document.createElement('button');
  nextBtn.className = 'secondary';
  nextBtn.dataset.testid = 'preview-next-page';
  nextBtn.textContent = 'Next ›';
  nextBtn.addEventListener('click', () => _reload(page + 1));
  const pageLabel = document.createElement('span');
  pageLabel.dataset.testid = 'preview-page-label';

  // How many cards a page holds. A run over a whole filter can return five figures,
  // and at a fixed 60 the pager reads "page 1 / 500" — at which point paging is not
  // navigation, it is scrolling with extra steps.
  //
  // The page is reset to the first on every change rather than trying to keep the
  // user's position: card 4,000 is on page 67 at 60 and page 8 at 500, and a "clever"
  // remap would land somewhere neither number predicts.
  const sizeSelect = document.createElement('select');
  sizeSelect.className = 'preview-panel__page-size';
  sizeSelect.dataset.testid = 'preview-page-size';
  sizeSelect.title = 'How many candidates to show per page';
  sizeSelect.setAttribute('aria-label', 'Candidates per page');
  for (const size of PAGE_SIZES) {
    const opt = document.createElement('option');
    opt.value = String(size);
    opt.textContent = `${size} / page`;
    opt.selected = size === pageSize;
    sizeSelect.appendChild(opt);
  }
  sizeSelect.addEventListener('change', () => {
    pageSize = Number(sizeSelect.value) || DEFAULT_PAGE_SIZE;
    prefs.write(PAGE_SIZE_KEY, pageSize);
    void _reload(0);
  });

  pager.append(prevBtn, pageLabel, nextBtn, sizeSelect);
  toolbar.appendChild(pager);

  // ── Candidate editing (context menu + canvas editor) ───────────────────

  /**
   * Adapter the `candidateBbox` context provider drives off. Unlike the old
   * in-memory version, `applyLabel` and `remove` must also persist — a relabel
   * or a removal the user makes here has to survive closing the panel just as a
   * decision does.
   */
  function _makeAdapter(entry) {
    const c = entry.candidate;
    return {
      getBboxUuid() { return c.isNew ? null : (c.bboxUuid ?? null); },
      getImageUuid() { return c.imageUuid ?? null; },
      getLabelUuid() { return c.proposedLabelUuid ?? null; },
      getLabelName() { return _labels.get(c.proposedLabelUuid)?.name ?? null; },
      getLabelColor() { return _labels.get(c.proposedLabelUuid)?.color ?? null; },
      getConfidence() { return c.confidence ?? null; },

      async applyLabel(labelUuid) {
        c.proposedLabelUuid = labelUuid;
        _queuePatch([c.uuid], { proposedLabelUuid: labelUuid });
        _renderPage();
      },
      /** "Remove from preview" is a rejection now — a persisted, reversible
       *  decision rather than dropping the row on the floor. */
      async remove() {
        _setDecision([c], PreviewDecision.REJECTED);
      },
      openEditor() { _openCandidateEditor(entry); },
    };
  }

  function _resolveContext(entry) {
    const selected = cardGrid.selected;
    const targets = (selected.has(entry.id) && selected.size > 1)
      ? _visibleEntries.filter(en => selected.has(en.id))
      : [entry];
    return targets.map(en => ({ adapters: { candidateBbox: _makeAdapter(en) } }));
  }

  /**
   * Build the canvas-modal nav item for one candidate: the image/bbox pair the
   * modal loads, plus `candidate` (so persist can find the right record after
   * navigating) and `seedBboxes` (so an `isNew` candidate's box — which has no
   * DB row — survives navigating away and back; see the seedBboxes doc on
   * `showCanvasModal` in main.js).
   */
  function _buildNavItem(entry) {
    const c = entry.candidate;
    const img = { uuid: c.imageUuid, filename: c.imageFilename, bboxes: [] };
    const bbox = {
      uuid: c.isNew ? 'new' : c.bboxUuid,
      labelUuid: c.proposedLabelUuid,
      label: _labels.get(c.proposedLabelUuid) ?? null,
      ...(c.geometry ?? {}),
      tags: [],
    };
    const seedBboxes = (c.isNew && c.geometry)
      ? [{
          uuid: 'new',
          label: _labels.get(c.proposedLabelUuid)
            ? { uuid: c.proposedLabelUuid, text: _labels.get(c.proposedLabelUuid).name, color: _labels.get(c.proposedLabelUuid).color }
            : undefined,
          ...c.geometry,
          dirty: false,
        }]
      : null;
    return { img, bbox, candidate: c, seedBboxes };
  }

  /**
   * Open the shared canvas editor on a candidate's box, navigable across the
   * rest of the current page via the modal's own [ ]/Prev/Next. Geometry
   * edited here lands on whichever candidate is on screen when saved and is
   * PATCHed — it must not touch the real bbox, which only the apply path may
   * write.
   */
  async function _openCandidateEditor(entry) {
    const { showCanvasModal } = await import('/app-static/js/app/review/main.js');
    const navItems = _visibleEntries.map(_buildNavItem);
    const startIndex = _visibleEntries.indexOf(entry);
    const start = navItems[startIndex] ?? _buildNavItem(entry);
    showCanvasModal(start.bbox, start.img, _labels, async () => {}, {
      items: navItems,
      startIndex,
    }, {
      allowNavigation: true,
      broadcast: false,
      seedBboxes: start.seedBboxes,
      persist: (runtime, navItem) => {
        const c = navItem?.candidate ?? entry.candidate;
        const uiBboxes = [...runtime.state.image.bboxes.values()];
        const box = c.isNew
          ? (uiBboxes.find(b => b.uuid === 'new') ?? uiBboxes[0])
          : uiBboxes.find(b => b.uuid === c.bboxUuid);
        if (box) {
          c.geometry = { x: box.x, y: box.y, width: box.width, height: box.height };
          _queuePatch([c.uuid], { geometry: c.geometry });
        }
        return true;
      },
    });
  }

  // ── Rendering ──────────────────────────────────────────────────────────

  function _updateDecisionButtons() {
    const n = cardGrid.selected.size;
    selCount.textContent = n > 0 ? `${n.toLocaleString()} selected` : 'none selected';
    selCount.classList.toggle('preview-panel__selcount--active', n > 0);
    for (const btn of [acceptBtn, acceptDeleteBtn, rejectBtn, resetBtn, deleteImageBtn]) btn.disabled = n === 0;
    bulkBtn.textContent = `Accept all (${totalCount.toLocaleString()})`;
    bulkBtn.disabled = totalCount === 0;
    const accepted = run?.decisionCounts?.[PreviewDecision.ACCEPTED] ?? 0;
    applyBtn.textContent = `Apply ${accepted.toLocaleString()}`;
    applyBtn.title = `Write the ${accepted.toLocaleString()} accepted candidate(s) to the database`;
    applyBtn.disabled = accepted === 0;
  }

  /**
   * Header for one label group in the grid, spanning the full row. Only drawn
   * in label-grouping mode: image groups are usually one or two candidates, so
   * a header each would cost more grid rows than the grouping saves.
   *
   * `Select group` acts on this page's members of the group — the whole-run
   * equivalent is the label filter plus "Accept all", which is already there.
   */
  function _buildGroupHeader(labelUuid, entries) {
    const header = document.createElement('div');
    header.className = 'preview-panel__group-header';
    header.dataset.testid = 'preview-group-header';
    header.dataset.labelUuid = labelUuid ?? '';

    const chip = labelChip(_labels.get(labelUuid)?.name, labelUuid, _labels);
    // `entries` is filled as the group's cards are built, so the count is
    // written once the page is complete (see the end of _renderPage).
    const count = document.createElement('span');
    count.className = 'preview-panel__group-count';
    header._countEl = count;
    header._entries = entries;

    const selectBtn = document.createElement('button');
    selectBtn.className = 'secondary preview-panel__group-select';
    selectBtn.dataset.testid = 'preview-group-select';
    selectBtn.textContent = 'Select group';
    selectBtn.addEventListener('click', () => _setSelection(entries, true));

    header.append(chip, count, selectBtn);
    return header;
  }

  /**
   * Header for one tracking incident: the claim the analyser made about it.
   *
   * This is what makes a harvested frame reviewable at all. Judging a cropped still on
   * its own is the ambiguity that produced the mistake in the first place — but
   * "9 of 121 frames disagree" is a claim you can sanity-check at a glance, and when it
   * looks wrong the clip is one click away.
   */
  function _buildBlipGroupHeader(candidate, entries) {
    const geometry = candidate.geometry ?? {};
    const header = document.createElement('div');
    header.className = 'preview-panel__group-header';
    header.dataset.testid = 'preview-blip-header';
    header.dataset.blipUuid = geometry.sourceBlipUuid ?? '';

    const track = document.createElement('span');
    track.className = 'preview-panel__group-count';
    track.textContent = geometry.sourceTrackId != null ? `Track #${geometry.sourceTrackId}` : 'Track';

    const claim = document.createElement('span');
    claim.className = 'preview-panel__blip-claim';
    claim.dataset.testid = 'preview-blip-claim';
    claim.textContent = _claimText(geometry, candidate.proposedLabelUuid);

    header.append(track, claim);

    // The escape hatch. Only offered when the frame actually records where it came
    // from — an older run, or one harvested before provenance existed, would otherwise
    // show a button that cannot answer.
    if (geometry.sourceEventUuid) {
      const clipBtn = document.createElement('button');
      clipBtn.className = 'secondary preview-panel__group-select';
      clipBtn.dataset.testid = 'preview-open-clip';
      const at = geometry.sourceClipTimeS;
      clipBtn.textContent = at != null ? `▶ Clip at ${Number(at).toFixed(1)}s` : '▶ Open clip';
      clipBtn.addEventListener('click', () => {
        events.publish(EventTypes.OPEN_EVENT_CLIP, {
          eventUuid: geometry.sourceEventUuid,
          atS: at ?? null,
        });
      });
      header.appendChild(clipBtn);
    }

    const selectBtn = document.createElement('button');
    selectBtn.className = 'secondary preview-panel__group-select';
    selectBtn.dataset.testid = 'preview-group-select';
    selectBtn.textContent = 'Select group';
    selectBtn.addEventListener('click', () => _setSelection(entries, true));
    header.appendChild(selectBtn);

    return header;
  }

  /** "identity flip · 9 of 121 frames disagree · proposing cat_a" */
  function _claimText(geometry, proposedLabelUuid) {
    const parts = [blipKindText(geometry.sourceBlipKind)];
    const disagreed = geometry.sourceBlipSamples;
    const total = geometry.sourceTrackSamples;
    // Only stated when both numbers are known. "9 frames disagree" without the total is
    // the half of the claim that carries no weight — nine out of twelve is a different
    // story from nine out of a hundred and twenty-one.
    if (disagreed != null && total != null) {
      parts.push(`${disagreed} of ${total} frames disagree`);
    } else if (disagreed != null) {
      parts.push(`${disagreed} frame(s) affected`);
    }
    const proposed = _labels.get(proposedLabelUuid)?.name;
    if (proposed) parts.push(`proposing ${proposed}`);
    return parts.join(' · ');
  }

  function _renderPage() {
    grid.innerHTML = '';
    _visibleEntries.length = 0;
    const grouping = _activeGrouping();
    // Candidates arrive already contiguous per group (the server orders them
    // that way), so a header is due whenever the key changes.
    let groupKey;
    let groupEntries = null;
    let index = 0;
    for (const candidate of candidates) {
      const entry = { id: candidate.uuid, kind: 'bbox', candidate };
      _visibleEntries.push(entry);
      if (grouping === 'label' && candidate.proposedLabelUuid !== groupKey) {
        groupKey = candidate.proposedLabelUuid;
        groupEntries = [];
        grid.appendChild(_buildGroupHeader(groupKey, groupEntries));
      } else if (grouping === 'blip' && candidate.geometry?.sourceBlipUuid !== groupKey) {
        groupKey = candidate.geometry?.sourceBlipUuid;
        groupEntries = [];
        grid.appendChild(_buildBlipGroupHeader(candidate, groupEntries));
      }
      groupEntries?.push(entry);
      const card = buildCandidateCard(cardGrid, entry, {
        gridEl: grid,
        labels: _labels,
        index,
        hoverPreview,
        labelChip,
        resolveContext: _resolveContext,
        onOpenEditor: _openCandidateEditor,
      });
      entry.card = card;
      // A decision edit re-renders the page without clearing the selection, so
      // the rebuilt card has to be put back into the state the controller still
      // believes it is in — including its checkbox, which applySelect reads.
      if (cardGrid.selected.has(entry.id)) {
        card.classList.add('selected');
        const cb = card.querySelector('input[type=checkbox]');
        if (cb) cb.checked = true;
      }
      grid.appendChild(card);
      index += 1;
    }

    for (const header of grid.querySelectorAll('.preview-panel__group-header')) {
      // Not every header carries a running count. A tracking-incident header states a
      // claim instead, and its own text must not be overwritten with a tally.
      if (!header._countEl) continue;
      const n = header._entries?.length ?? 0;
      header._countEl.textContent = `${n} on this page`;
    }

    const first = totalCount === 0 ? 0 : page * pageSize + 1;
    const last = Math.min(totalCount, (page + 1) * pageSize);
    countLabel.textContent = totalCount === 0 ? 'No candidates' : `${first}–${last} of ${totalCount.toLocaleString()}`;
    pageLabel.textContent = `Page ${page + 1} / ${Math.max(1, Math.ceil(totalCount / pageSize))}`;
    prevBtn.disabled = page === 0;
    nextBtn.disabled = last >= totalCount;
    _updateDecisionButtons();
    onCountsChanged?.(run?.decisionCounts ?? {});
  }

  // Re-entrancy guard: a fast filter change while a fetch is in flight would
  // otherwise interleave two renders (see the pattern in configuration/landing.js).
  let _loading = false;
  async function _reload(nextPage = page) {
    if (destroyed || _loading) return;
    _loading = true;
    try {
      page = Math.max(0, nextPage);
      const [{ candidates: rows, totalCount: total }, freshRun] = await Promise.all([
        queryPreviewCandidates(runUuid, { ..._queryState(), limit: pageSize, offset: page * pageSize }),
        fetchPreviewRun(runUuid).catch(() => run),
      ]);
      if (destroyed) return;
      candidates = rows;
      totalCount = total;
      run = freshRun ?? run;
      // A page that emptied out (everything on it got rejected and filtered
      // away) should not strand the user on a blank grid past the end.
      if (candidates.length === 0 && page > 0 && totalCount > 0) {
        _loading = false;
        return _reload(Math.min(page - 1, Math.ceil(totalCount / pageSize) - 1));
      }
      cardGrid.selected.clear();
      _renderPage();
    } catch (err) {
      Logger.error('Failed to load preview candidates:', err);
      toast(`Failed to load candidates: ${err.message}`, 5000, 'error');
    } finally {
      _loading = false;
    }
  }

  await _reload(0);

  return {
    get run() { return run; },
    refresh: () => _reload(page),
    destroy() {
      destroyed = true;
      clearTimeout(_flushTimer);
      // Fire-and-forget: the panel is going away, but the decisions in the
      // buffer are the user's and must still land.
      _flushPending().catch(err => Logger.error('Final preview decision flush failed:', err));
      hoverPreview.destroy();
      root.remove();
    },
  };
}

/**
 * Open one run's triage panel in a full-surface modal.
 *
 * @param {Object} opts
 * @param {string} opts.runUuid
 * @param {Map<string, object>} [opts.labels]
 * @param {() => void} [opts.onApplied]
 */
export async function openPreviewRunModal({ runUuid, labels = null, onApplied = null } = {}) {
  let panel = null;
  const { dialog, open, close } = createModalShell({ onClose: () => panel?.destroy() });
  dialog.classList.add('app-modal-dialog--full', 'preview-run-modal', 'relabel-confirm-modal');

  // One header row instead of a tall title row plus a footer row that held a
  // single Close button: both were pure chrome around the grid, which is the
  // only thing in this modal worth vertical space.
  const header = document.createElement('div');
  header.className = 'preview-run-modal__header';

  const title = document.createElement('h3');
  title.className = 'app-modal-title preview-run-modal__title';
  title.dataset.testid = 'preview-run-title';

  const subtitle = document.createElement('span');
  subtitle.className = 'preview-run-modal__subtitle';
  subtitle.dataset.testid = 'preview-run-subtitle';

  const closeBtn = document.createElement('button');
  closeBtn.className = 'secondary preview-run-modal__close';
  closeBtn.dataset.testid = 'preview-run-close';
  closeBtn.textContent = '✕';
  closeBtn.title = 'Close (decisions are already saved)';
  closeBtn.setAttribute('aria-label', 'Close');
  closeBtn.addEventListener('click', close);

  header.append(title, subtitle, closeBtn);
  dialog.appendChild(header);

  open();

  try {
    panel = await createPreviewPanel(dialog, { runUuid, labels, onApplied });
    title.textContent = panel.run?.name ?? 'Model run';
    subtitle.textContent = panel.run?.modelName ?? '';
  } catch (err) {
    Logger.error('Failed to open model run:', err);
    title.textContent = 'Failed to load this run';
    toast(`Failed to open this run: ${err.message}`, 5000, 'error');
  }

  return { close };
}

/**
 * The saved-runs list — every persisted preview, with its triage progress.
 *
 * Built on `.panel-box` rather than a hand-rolled list screen (ARCH.md §
 * Panels, cards and list editors: four near-identical list editors already
 * exist; this is not becoming the fifth).
 */
export async function openPreviousModelResultsModal({ labels = null, onApplied = null, onClose = null } = {}) {
  // onClose (not the awaited return of this function) is when the caller's run
  // count is actually stale: this resolves as soon as the list has rendered,
  // while the modal is still open and still being edited.
  const { dialog, open, close } = createModalShell({ onClose: () => onClose?.() });
  dialog.classList.add('app-modal-dialog--md', 'previous-model-results-modal');

  const panelBox = document.createElement('div');
  panelBox.className = 'panel-box';
  const bar = document.createElement('div');
  bar.className = 'panel-box__bar';
  const barTitle = document.createElement('div');
  barTitle.className = 'panel-box__title';
  barTitle.textContent = 'Previous Model Results';
  bar.appendChild(barTitle);
  const listBody = document.createElement('div');
  listBody.className = 'panel-box__body';
  listBody.dataset.testid = 'previous-model-results-list';
  listBody.style.cssText = 'max-height:60vh;overflow-y:auto;';
  panelBox.append(bar, listBody);
  dialog.appendChild(panelBox);

  const footer = document.createElement('div');
  footer.className = 'classify-modal-footer';
  footer.style.cssText = 'display:flex;justify-content:flex-end;';
  const closeBtn = document.createElement('button');
  closeBtn.className = 'secondary';
  closeBtn.textContent = 'Close';
  closeBtn.addEventListener('click', close);
  footer.appendChild(closeBtn);
  dialog.appendChild(footer);

  async function _render() {
    listBody.innerHTML = '';
    let runs = [];
    try {
      runs = await fetchPreviewRuns();
    } catch (err) {
      Logger.error('Failed to list preview runs:', err);
      listBody.textContent = `Failed to load previous model results: ${err.message}`;
      return;
    }
    // Scoring a report's stats lives with the verification dialog that produced
    // them; this list only needs the one line, and only when a report is in it.
    let reportRowSummary = () => '';
    if (runs.some(isReportRun)) {
      ({ reportRowSummary } = await import('/app-static/js/app/review/compareModels.js'));
    }
    if (runs.length === 0) {
      const empty = document.createElement('div');
      empty.className = 'relabel-modal-empty';
      empty.textContent = 'No previous model results yet. A run\u2019s results are kept for 7 days unless you pin them.';
      listBody.appendChild(empty);
      return;
    }
    for (const run of runs) listBody.appendChild(_buildRunRow(run, reportRowSummary));
  }

  function _buildRunRow(run, reportRowSummary) {
    const row = document.createElement('div');
    row.className = 'preview-run-row';
    row.dataset.testid = 'preview-run-row';
    row.dataset.runUuid = run.uuid;
    row.style.cssText = 'display:flex;align-items:center;gap:0.6em;padding:0.4em 0;border-bottom:1px solid var(--border);';

    const main = document.createElement('div');
    main.style.cssText = 'flex:1;min-width:0;';
    const name = document.createElement('div');
    name.textContent = run.name;
    name.style.cssText = 'font-weight:600;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;';
    const isReport = isReportRun(run);
    const counts = run.decisionCounts ?? {};
    const meta = document.createElement('div');
    meta.className = 'preview-run-row__meta';
    meta.style.cssText = 'font-size:0.8em;color:#aaa;';
    // A report has no candidates to count — its result is its stats.
    meta.textContent = isReport ? [run.modelName, reportRowSummary(run)].filter(Boolean).join(' · ') : [
      run.modelName,
      `${counts[PreviewDecision.PENDING] ?? 0} unreviewed`,
      `${counts[PreviewDecision.ACCEPTED] ?? 0} accepted`,
      `${counts[PreviewDecision.REJECTED] ?? 0} rejected`,
      `${counts[PreviewDecision.APPLIED] ?? 0} applied`,
    ].filter(Boolean).join(' · ');
    main.append(name, meta);

    const openBtn = document.createElement('button');
    openBtn.dataset.testid = 'preview-run-open';
    openBtn.textContent = 'Open';
    openBtn.addEventListener('click', async () => {
      if (isReport) {
        const { openSavedVerifyReport } = await import('/app-static/js/app/review/compareModels.js');
        await openSavedVerifyReport({ run, labels });
      } else {
        await openPreviewRunModal({ runUuid: run.uuid, labels, onApplied });
      }
      await _render();
    });

    // Pinned runs are exempt from the 7-day sweep — the one control that
    // decides whether a run is still there tomorrow.
    const pinBtn = document.createElement('button');
    pinBtn.className = 'secondary';
    pinBtn.dataset.testid = 'preview-run-pin';
    pinBtn.textContent = run.pinned ? '📌 Pinned' : 'Pin';
    pinBtn.title = run.pinned ? 'Unpin (run expires again)' : 'Pin to keep this run past its expiry';
    pinBtn.addEventListener('click', async () => {
      try {
        await updatePreviewRun(run.uuid, { pinned: !run.pinned });
        await _render();
      } catch (err) {
        toast(`Failed to pin: ${err.message}`, 4000, 'error');
      }
    });

    const delBtn = document.createElement('button');
    delBtn.className = 'secondary';
    delBtn.dataset.testid = 'preview-run-delete';
    delBtn.textContent = '🗑';
    delBtn.title = isReport ? 'Delete this report' : 'Delete this run';
    delBtn.addEventListener('click', async () => {
      const confirmed = await showConfirmModal({
        title: isReport ? 'Delete report' : 'Delete model run',
        message: isReport
          ? `Delete "${run.name}"? Its accuracy report is not recoverable — the run would have to be re-run.`
          : `Delete "${run.name}" and all of its candidates? Applied changes are not undone.`,
        confirmText: 'Delete',
        danger: true,
      });
      if (!confirmed) return;
      try {
        await deletePreviewRun(run.uuid);
        await _render();
      } catch (err) {
        toast(`Failed to delete: ${err.message}`, 4000, 'error');
      }
    });

    row.append(main, openBtn, pinBtn, delBtn);
    return row;
  }

  open();
  await _render();
  return { close, refresh: _render };
}
