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
  PreviewDecision,
  PreviewDecisionAction,
} from '/app-static/js/shared/api/previewRuns.js';
import { fetchLabels } from '/app-static/js/shared/api/labels.js';
import { buildCandidateCard, createCardGrid } from '/app-static/js/app/review/cardGrid.js';
import { labelChip } from '/app-static/js/app/review/labelChip.js';
import { createHoverPreview } from '/app-static/js/ui/thumbnailHoverPreview.js';
import { createModalShell } from '/app-static/js/ui/utils/modalShell.js';
import { createMultiSelectDropdown } from '/app-static/js/ui/utils/createMultiSelectDropdown.js';
import { showConfirmModal } from '/app-static/js/ui/utils/confirmModal.js';
import { toast, Logger } from '/app-static/js/ui/utils/logging.js';

/** Candidates per page. Matches the review grid's feel on a full-width modal. */
const PAGE_SIZE = 60;

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
  const root = document.createElement('div');
  root.className = 'preview-panel';
  root.dataset.testid = 'preview-panel';
  root.style.cssText = 'display:flex;flex-direction:column;gap:0.5em;flex:1;min-height:0;';
  container.appendChild(root);

  const toolbar = document.createElement('div');
  toolbar.className = 'preview-panel__toolbar';
  toolbar.style.cssText = 'display:flex;align-items:center;gap:0.6em;flex-wrap:wrap;flex-shrink:0;';
  root.appendChild(toolbar);

  const grid = document.createElement('div');
  grid.className = 'preview-panel__grid relabel-modal-grid review-grid';
  const body = document.createElement('div');
  body.className = 'preview-panel__body';
  body.style.cssText = 'flex:1;overflow-y:auto;min-height:0;';
  body.appendChild(grid);
  root.appendChild(body);

  const footer = document.createElement('div');
  footer.className = 'preview-panel__footer';
  footer.style.cssText = 'display:flex;align-items:center;gap:0.5em;flex-shrink:0;flex-wrap:wrap;';
  root.appendChild(footer);

  // ── Filter bar (candidate filters — see §3 for why not createFilterPanel) ──
  const filterState = {
    decisions: [],
    proposedLabelUuids: [],
    confidenceMin: null,
    confidenceMax: null,
    isNew: null,
    sortBy: 'image_confidence',
    sortOrder: 'desc',
  };

  const decisionDd = createMultiSelectDropdown({
    label: 'Decision',
    items: DECISION_FILTER_ITEMS,
    showModeToggle: false,
    allowExclude: false,
    filterKey: 'preview-decision',
    onChange: (selected) => { filterState.decisions = selected; _reload(0); },
  });
  decisionDd.element.dataset.testid = 'preview-filter-decision';
  toolbar.appendChild(decisionDd.element);

  const labelDd = createMultiSelectDropdown({
    label: 'Proposed label',
    items: [..._labels.values()].map(l => ({ uuid: l.uuid, name: l.name, color: l.color })),
    showModeToggle: false,
    allowExclude: false,
    searchable: true,
    filterKey: 'preview-proposed-label',
    onChange: (selected) => { filterState.proposedLabelUuids = selected; _reload(0); },
  });
  labelDd.element.dataset.testid = 'preview-filter-label';
  toolbar.appendChild(labelDd.element);

  // Confidence band. A number pair rather than a slider: triage is usually
  // "everything above 90%" or "the uncertain 40–60% band", both of which are
  // typed faster than dragged.
  const confWrap = document.createElement('label');
  confWrap.style.cssText = 'display:flex;align-items:center;gap:0.3em;font-size:0.85em;';
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
    el.style.cssText = 'width:4.5em;padding:0.1em 0.3em;';
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
  confWrap.append(confMin, document.createTextNode('–'), confMax, document.createTextNode('%'));
  toolbar.appendChild(confWrap);

  const kindSelect = document.createElement('select');
  kindSelect.dataset.testid = 'preview-filter-kind';
  kindSelect.style.cssText = 'padding:0.1em 0.3em;font-size:0.85em;';
  for (const [value, text] of [['', 'All boxes'], ['existing', 'Existing only'], ['new', 'New only']]) {
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
  sortSelect.style.cssText = 'padding:0.1em 0.3em;font-size:0.85em;';
  for (const [value, text] of [
    ['image_confidence', 'Group by image'],
    ['confidence', 'Confidence'],
    ['created_at', 'Found order'],
  ]) {
    const opt = document.createElement('option');
    opt.value = value;
    opt.textContent = text;
    sortSelect.appendChild(opt);
  }
  sortSelect.addEventListener('change', () => { filterState.sortBy = sortSelect.value; _reload(0); });
  toolbar.appendChild(sortSelect);

  const countLabel = document.createElement('span');
  countLabel.dataset.testid = 'preview-count';
  countLabel.style.cssText = 'margin-left:auto;font-size:0.85em;color:#aaa;';
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

  function _decisionButton(text, testid, onClick) {
    const btn = document.createElement('button');
    btn.className = 'secondary';
    btn.dataset.testid = testid;
    btn.textContent = text;
    btn.addEventListener('click', onClick);
    footer.appendChild(btn);
    return btn;
  }

  const acceptBtn = _decisionButton('Accept', 'preview-accept', () =>
    _setDecision(_targets(), PreviewDecision.ACCEPTED, PreviewDecisionAction.RELABEL));
  const acceptDeleteBtn = _decisionButton('Accept as delete', 'preview-accept-delete', () =>
    _setDecision(_targets(), PreviewDecision.ACCEPTED, PreviewDecisionAction.DELETE));
  const rejectBtn = _decisionButton('Reject', 'preview-reject', () =>
    _setDecision(_targets(), PreviewDecision.REJECTED));
  const resetBtn = _decisionButton('Unreview', 'preview-reset', () =>
    _setDecision(_targets(), PreviewDecision.PENDING));

  // Bulk over the *filter*, not the page: "accept everything above 90%" should
  // not require paging through 40 pages checking boxes.
  const bulkBtn = document.createElement('button');
  bulkBtn.className = 'secondary';
  bulkBtn.dataset.testid = 'preview-accept-all-filtered';
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
        ...filterState,
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
  verifyLabel.style.cssText = 'display:flex;align-items:center;gap:0.3em;font-size:0.85em;margin-left:auto;';
  verifyLabel.title = 'Off: applied boxes keep their current verification state (still reviewable). '
    + 'On: treat this apply as a human decision and mark the boxes verified.';
  const verifyCheckbox = document.createElement('input');
  verifyCheckbox.type = 'checkbox';
  verifyCheckbox.className = 'relabel-modal-mark-verified';
  verifyLabel.append(verifyCheckbox, document.createTextNode('Mark as verified'));
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
  pager.style.cssText = 'display:flex;align-items:center;gap:0.4em;font-size:0.85em;';
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
  pager.append(prevBtn, pageLabel, nextBtn);
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
   * Open the shared canvas editor on a candidate's box. Geometry edited here
   * lands on the candidate and is PATCHed — it must not touch the real bbox,
   * which only the apply path may write.
   */
  async function _openCandidateEditor(entry) {
    const c = entry.candidate;
    const { showCanvasModal } = await import('/app-static/js/app/review/main.js');
    const img = { uuid: c.imageUuid, filename: c.imageFilename, bboxes: [] };
    const candidateBbox = {
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
    showCanvasModal(candidateBbox, img, _labels, async () => {}, null, {
      allowNavigation: false,
      broadcast: false,
      seedBboxes,
      persist: (runtime) => {
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
    const suffix = n > 0 ? ` (${n})` : '';
    acceptBtn.textContent = `Accept${suffix}`;
    acceptDeleteBtn.textContent = `Accept as delete${suffix}`;
    rejectBtn.textContent = `Reject${suffix}`;
    resetBtn.textContent = `Unreview${suffix}`;
    for (const btn of [acceptBtn, acceptDeleteBtn, rejectBtn, resetBtn]) btn.disabled = n === 0;
    bulkBtn.textContent = `Accept all ${totalCount.toLocaleString()} filtered`;
    bulkBtn.disabled = totalCount === 0;
    const accepted = run?.decisionCounts?.[PreviewDecision.ACCEPTED] ?? 0;
    applyBtn.textContent = `Apply ${accepted.toLocaleString()} accepted`;
    applyBtn.disabled = accepted === 0;
  }

  function _renderPage() {
    grid.innerHTML = '';
    _visibleEntries.length = 0;
    let index = 0;
    for (const candidate of candidates) {
      const entry = { id: candidate.uuid, kind: 'bbox', candidate };
      _visibleEntries.push(entry);
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

    const first = totalCount === 0 ? 0 : page * PAGE_SIZE + 1;
    const last = Math.min(totalCount, (page + 1) * PAGE_SIZE);
    countLabel.textContent = totalCount === 0 ? 'No candidates' : `${first}–${last} of ${totalCount.toLocaleString()}`;
    pageLabel.textContent = `Page ${page + 1} / ${Math.max(1, Math.ceil(totalCount / PAGE_SIZE))}`;
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
        queryPreviewCandidates(runUuid, { ...filterState, limit: PAGE_SIZE, offset: page * PAGE_SIZE }),
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
        return _reload(Math.min(page - 1, Math.ceil(totalCount / PAGE_SIZE) - 1));
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

  const title = document.createElement('h3');
  title.className = 'app-modal-title';
  title.style.flexShrink = '0';
  title.dataset.testid = 'preview-run-title';
  dialog.appendChild(title);

  open();

  try {
    panel = await createPreviewPanel(dialog, { runUuid, labels, onApplied });
    title.textContent = panel.run?.name ?? 'Preview run';
  } catch (err) {
    Logger.error('Failed to open preview run:', err);
    title.textContent = 'Failed to load preview run';
    toast(`Failed to open preview run: ${err.message}`, 5000, 'error');
  }

  const footer = document.createElement('div');
  footer.className = 'classify-modal-footer';
  footer.style.cssText = 'display:flex;justify-content:flex-end;flex-shrink:0;';
  const closeBtn = document.createElement('button');
  closeBtn.className = 'secondary';
  closeBtn.dataset.testid = 'preview-run-close';
  closeBtn.textContent = 'Close';
  closeBtn.addEventListener('click', close);
  footer.appendChild(closeBtn);
  dialog.appendChild(footer);

  return { close };
}

/**
 * The saved-runs list — every persisted preview, with its triage progress.
 *
 * Built on `.panel-box` rather than a hand-rolled list screen (ARCH.md §
 * Panels, cards and list editors: four near-identical list editors already
 * exist; this is not becoming the fifth).
 */
export async function openSavedPreviewsModal({ labels = null, onApplied = null, onClose = null } = {}) {
  // onClose (not the awaited return of this function) is when the caller's run
  // count is actually stale: this resolves as soon as the list has rendered,
  // while the modal is still open and still being edited.
  const { dialog, open, close } = createModalShell({ onClose: () => onClose?.() });
  dialog.classList.add('app-modal-dialog--md', 'saved-previews-modal');

  const panelBox = document.createElement('div');
  panelBox.className = 'panel-box';
  const bar = document.createElement('div');
  bar.className = 'panel-box__bar';
  const barTitle = document.createElement('div');
  barTitle.className = 'panel-box__title';
  barTitle.textContent = 'Saved Previews';
  bar.appendChild(barTitle);
  const listBody = document.createElement('div');
  listBody.className = 'panel-box__body';
  listBody.dataset.testid = 'saved-previews-list';
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
      listBody.textContent = `Failed to load saved previews: ${err.message}`;
      return;
    }
    if (runs.length === 0) {
      const empty = document.createElement('div');
      empty.className = 'relabel-modal-empty';
      empty.textContent = 'No saved previews. Runs are kept for 7 days unless you pin them.';
      listBody.appendChild(empty);
      return;
    }
    for (const run of runs) listBody.appendChild(_buildRunRow(run));
  }

  function _buildRunRow(run) {
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
    const counts = run.decisionCounts ?? {};
    const meta = document.createElement('div');
    meta.className = 'preview-run-row__meta';
    meta.style.cssText = 'font-size:0.8em;color:#aaa;';
    meta.textContent = [
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
      await openPreviewRunModal({ runUuid: run.uuid, labels, onApplied });
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
    delBtn.title = 'Delete this preview run';
    delBtn.addEventListener('click', async () => {
      const confirmed = await showConfirmModal({
        title: 'Delete preview run',
        message: `Delete "${run.name}" and all of its candidates? Applied changes are not undone.`,
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
