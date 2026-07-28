/**
 * "Run Model on Results" — Chunk B.
 *
 * Wires the review-browser toolbar button to:
 *  1. resolve the candidate bbox/image scope from whatever the user currently
 *     has selected/filtered (mirrors the existing Bulk Apply / Apply to All split),
 *  2. warn on large candidate counts (above WARN_THRESHOLD) and let the user cap
 *     how many to process — but with no hard ceiling; "process all" is allowed,
 *  3. let the user pick a detector model + conf/IoU thresholds in a run dialog,
 *  4. launch `PreviewModelRelabelTask` (a temporary, one-shot task config),
 *     poll it to completion with a cancel button, and
 *  5. hand the resulting candidate relabels off to the (Chunk C) confirmation
 *     modal via `openRelabelConfirmationModal`.
 *
 * This module owns all of that logic so review/main.js only needs a button
 * and a one-line call into `runModelOnResultsFlow`.
 */

import {
  fetchReviewQueue,
  getBboxThumbnailUrl,
  getImageThumbnailUrl,
  bulkApplyBboxesByUuids,
  invalidateBboxThumbnails,
  addBbox,
  UNVERIFIED_TAG_UUID,
} from '/app-static/js/shared/api/review.js';
import { activeTrainingConfig } from '/app-static/js/shared/activeTrainingConfig.js';
import { fetchModelsList, fetchModelsKinds, clearModelsCache } from '/app-static/js/shared/api/models.js';
import {
  createTaskConfig,
  startTask,
  fetchTasksStatus,
  fetchTasksResult,
  cancelTasks,
  deleteTasks,
  TaskStatus,
} from '/app-static/js/shared/api/tasks.js';
import { createModalShell } from '/app-static/js/ui/utils/modalShell.js';
import { createHoverPreview } from '/app-static/js/ui/thumbnailHoverPreview.js';
import { drawLetterboxedBboxes } from '/app-static/js/ui/utils/bboxThumbOverlay.js';
import { toast, Logger } from '/app-static/js/ui/utils/logging.js';
import { expandLabelFilterSet, bboxHasAnyTag, VERIFICATION_TAG_BY_STATE } from '/app-static/js/app/review/filterPredicate.js';
import { openClassifierTagConfirmationModal } from '/app-static/js/app/review/runClassifierOnResults.js';
import { labelChip as _labelChip } from '/app-static/js/app/review/labelChip.js';
import { createCardGrid } from '/app-static/js/app/review/cardGrid.js';
import { attachContext } from '/app-static/js/ui/contextMenu/index.js';

const _CLASSIFIER_KINDS = new Set(['image_classifier', 'multi_label_classifier']);

// Above this many candidate bboxes, warn the user (large runs take a while) and
// let them cap how many to process before launching. There is no hard ceiling —
// "process all" is always allowed (the server no longer enforces one either).
export const WARN_THRESHOLD = 1000;

const POLL_INTERVAL_MS = 700;

/**
 * Rebuilds the per-image bbox predicate used by review/main.js's `render()` to
 * decide which of an image's bboxes count as "matching" the current filter —
 * shared here so scope resolution (bbox candidates for a whole filtered set,
 * or for selected images) applies exactly the same rule the grid is showing.
 *
 * @param {Object} opts
 * @param {boolean} opts.fetchImagesMode - true when the panel is in custom-fetchImages
 *   (nested search/dataset-preview) mode.
 * @param {(bbox: object) => boolean} [opts.bboxFilterFn] - caller-supplied override filter.
 * @param {object|null} opts.fs - filterPanel.getState() result, or null in fetchImages mode.
 * @param {Map<string, object>} opts.labels
 * @returns {(bbox: object) => boolean}
 */
export function makeBboxFilterPredicate({ fetchImagesMode, bboxFilterFn, fs, labels }) {
  const showAll = (fetchImagesMode && !bboxFilterFn) || fs?.showAllBboxes;
  const tagSet = (fs && !showAll) ? new Set(fs.tagUuids) : null;
  // Share the label-set expansion helper with filterPredicate.js so the
  // include/child-expansion policy can't drift between the two filter paths.
  const activeLabelUuids = (fs?.labelUuids?.length > 0)
    ? expandLabelFilterSet(fs.labelUuids, labels, fs.labelIncludeChildren)
    : null;
  // Bbox verification tri-state (V1) — reuse filterPredicate.js's tag-membership
  // helper and verification-tag mapping so this render-time predicate can't
  // drift from bboxMatchesFilter's semantics.
  const wantVerificationTagUuid = (fs?.bboxVerification && fs.bboxVerification !== 'any')
    ? VERIFICATION_TAG_BY_STATE[fs.bboxVerification]
    : null;

  return (b) => {
    const baseMatch = bboxFilterFn
      ? bboxFilterFn(b)
      : showAll || (b.tags || []).some(t => tagSet.has(t.uuid));
    const labelMatch = !activeLabelUuids || activeLabelUuids.has(b.labelUuid);
    const verificationMatch = !wantVerificationTagUuid || bboxHasAnyTag(b, new Set([wantVerificationTagUuid]));
    return baseMatch && labelMatch && verificationMatch;
  };
}

function _collectFromImages(images, predicate) {
  const bboxUuids = [];
  const imageUuids = [];
  // Parallel to bboxUuids: the owning image UUID for each candidate bbox, so a
  // later `truncateScope` can drop the images left with no candidates.
  const bboxImageUuids = [];
  for (const img of images) {
    const matches = (img.bboxes || []).filter(predicate);
    if (matches.length === 0) continue;
    imageUuids.push(img.uuid);
    for (const b of matches) {
      bboxUuids.push(b.uuid);
      bboxImageUuids.push(img.uuid);
    }
  }
  return { bboxUuids, imageUuids, bboxImageUuids };
}

/**
 * Caps a resolved scope to at most `limit` candidate bboxes, keeping only the
 * images those retained bboxes belong to (so the task doesn't waste round-trips
 * fetching images with no candidates left). A non-finite or non-positive limit
 * means "process all" and returns the scope's full bbox/image lists unchanged.
 *
 * @param {{bboxUuids: string[], imageUuids: string[], bboxImageUuids: string[]}} scope
 * @param {number} limit
 * @returns {{bboxUuids: string[], imageUuids: string[]}}
 */
export function truncateScope(scope, limit) {
  if (!Number.isFinite(limit) || limit <= 0 || scope.bboxUuids.length <= limit) {
    return { bboxUuids: scope.bboxUuids, imageUuids: scope.imageUuids };
  }
  const bboxUuids = scope.bboxUuids.slice(0, limit);
  const imageUuids = [...new Set(scope.bboxImageUuids.slice(0, limit))];
  return { bboxUuids, imageUuids };
}

/**
 * Resolves the candidate bbox/image scope for "Run Model on Results", mirroring
 * the existing Bulk Apply / Apply to All selection split:
 *   - bboxes selected  → those bboxes, in memory.
 *   - images selected  → those images' bboxes, filtered by the shared predicate.
 *   - nothing selected, fetchImages mode → enumerate via fetchImages({limit: MAX}).
 *   - nothing selected, full-filter mode → re-run fetchReviewQueue at a high limit.
 *
 * @param {Object} ctx - see `runModelOnResultsFlow` for field docs.
 * @returns {Promise<{bboxUuids: string[], imageUuids: string[], bboxImageUuids: string[]}>}
 *   `bboxImageUuids` is parallel to `bboxUuids` (owning image per candidate), for `truncateScope`.
 */
export async function resolveCandidateScope(ctx) {
  const {
    fetchImages, filterPanel, viewFilter, selectionKind, selected,
    visibleEntries, bboxFilterFn, labels, sortBy, sortOrder,
  } = ctx;

  if (selectionKind === 'bbox' && selected.size > 0) {
    const entries = visibleEntries.filter(e => e.kind === 'bbox' && selected.has(e.bbox.uuid));
    const bboxUuids = entries.map(e => e.bbox.uuid);
    const bboxImageUuids = entries.map(e => e.img.uuid);
    const imageUuids = [...new Set(bboxImageUuids)];
    return { bboxUuids, imageUuids, bboxImageUuids };
  }

  if (selectionKind === 'image' && selected.size > 0) {
    const selectedImages = visibleEntries
      .filter(e => e.kind === 'image' && selected.has(e.img.uuid))
      .map(e => e.img);
    const fs = fetchImages ? null : filterPanel.getState();
    const predicate = makeBboxFilterPredicate({ fetchImagesMode: !!fetchImages, bboxFilterFn, fs, labels });
    return _collectFromImages(selectedImages, predicate);
  }

  if (fetchImages) {
    const images = await fetchImages({ afterUuid: null, limit: Number.MAX_SAFE_INTEGER });
    const predicate = makeBboxFilterPredicate({ fetchImagesMode: true, bboxFilterFn, fs: null, labels });
    return _collectFromImages(images, predicate);
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
    requireBboxes: viewFilter === 'bboxes',
    sortBy,
    sortOrder,
    filenameSearch: fs.filenameSearch || undefined,
    dateFrom: fs.dateFrom || undefined,
    dateTo: fs.dateTo || undefined,
    // V2: "Config Ready" is whole-image-only — gate both the check itself and
    // its label-uuids override out of the request in bboxes view (mirrors
    // main.js's _verificationCheckFor / _verificationLabelUuidsFor).
    verificationCheck: viewFilter === 'bboxes' ? null : fs.verificationCheck,
    verificationLabelUuids: (fs.verificationCheck && viewFilter !== 'bboxes')
      ? [...activeTrainingConfig.getLabelUuids()]
      : [],
    bboxVerification: fs.bboxVerification,
    limit: Number.MAX_SAFE_INTEGER,
  });
  const predicate = makeBboxFilterPredicate({ fetchImagesMode: false, bboxFilterFn, fs, labels });
  return _collectFromImages(images, predicate);
}

/**
 * Per-card adapter for a preview candidate — the object the shared cardGrid
 * builders and the `candidateBbox` context-menu provider drive off. It mutates
 * the in-memory candidate / preview state ONLY; nothing hits the DB until the
 * user hits Apply (the existing batch contract). Scope decision 3: no
 * `setVerification` / `toggleTag` (the `candidateBbox` provider never calls
 * them, so their absence is safe).
 *
 * Geometry edited in the canvas editor lands back on `candidate.geometry`
 * (the `{x,y,width,height}` field the Apply path reads: existing boxes pass it
 * as `bboxGeometry[bboxUuid]`, new boxes as `addBbox`'s x/y/width/height).
 *
 * @param {object} candidate - the transient preview candidate object.
 * @param {HTMLElement} cardEl - the candidate's card element.
 * @param {{
 *   labels: Map<string, object>,
 *   removeCandidate: (candidate: object) => void,
 *   rerenderProposedChip: (candidate: object) => void,
 *   openEditor: (candidate: object) => void,
 * }} previewController
 */
export function makeCandidateBboxAdapter(candidate, cardEl, previewController) {
  const { labels, removeCandidate, rerenderProposedChip, openEditor } = previewController;
  return {
    // ── Display getters (used by the context menu / any card builder) ──────
    // New-box candidates have no real uuid — return null so the similar-search
    // menu item (which `?.`-guards getBboxUuid) self-disables/filters them out.
    getBboxUuid() { return candidate.isNew ? null : (candidate.bboxUuid ?? null); },
    getImageUuid() { return candidate.imageUuid ?? null; },
    getLabelUuid() { return candidate.proposedLabelUuid ?? null; },
    getLabelName() { return candidate.proposedLabelName ?? null; },
    getLabelColor() { return labels.get(candidate.proposedLabelUuid)?.color ?? null; },
    getConfidence() { return candidate.confidence ?? null; },
    getThumbnailUrl() {
      return candidate.isNew
        ? getImageThumbnailUrl(candidate.imagePath, 300)
        : getBboxThumbnailUrl(candidate.imagePath, candidate.bboxUuid, 300);
    },

    // ── Relabel (no DB) ────────────────────────────────────────────────────
    // Sets the proposed label on the candidate and re-renders its chip in place.
    // The name comes from the labels Map (fetchLabels() returns a Map).
    async applyLabel(labelUuid) {
      candidate.proposedLabelUuid = labelUuid;
      candidate.proposedLabelName = labelUuid ? (labels.get(labelUuid)?.name ?? null) : null;
      rerenderProposedChip(candidate);
    },

    // ── Remove from preview (no DB) ────────────────────────────────────────
    // Drops the candidate so Apply won't include it (unchecks + removes the row).
    async remove() {
      removeCandidate(candidate);
    },

    // ── Edit geometry (opens the shared canvas editor; no DB) ──────────────
    openEditor() { openEditor(candidate); },
  };
}

/**
 * Confirmation modal for "Run Model on Results" (Chunk C). Lets the user
 * review the server's proposed relabels, pick relabel-vs-delete, check/uncheck
 * individual candidates, and apply the result via `bulkApplyBboxesByUuids`
 * (grouped by proposed label for the relabel case, one call for delete).
 *
 * @param {Array<{bboxUuid: string, imageUuid: string, imagePath: string,
 *   currentLabelUuid: ?string, currentLabelName: ?string, proposedLabelUuid: string,
 *   proposedLabelName: ?string, confidence: number,
 *   geometry: ?{x: number, y: number, width: number, height: number}}>} candidates
 * @param {Object} ctx
 * @param {string} ctx.modelName - detector used to generate the candidates.
 * @param {() => void} ctx.refresh - reload the current review-panel page (e.g. `render()`).
 * @param {Map<string, object>} ctx.labels - label lookup, for display.
 * @param {{imagesFailed: number, noDetectionCount: number, noOpCount: number,
 *   sampleErrors: string[]}} [ctx.stats] - diagnostic breakdown from the task result,
 *   shown when there are zero candidates so "0 results" isn't a silent mystery.
 */
export async function openRelabelConfirmationModal(candidates, ctx) {
  // Same hover-zoom popover used for cards in review/main.js and the file
  // browser — fetches the image at a larger thumbnail size and draws the
  // relevant bbox on a canvas overlay, positioned near the hovered card.
  const hoverPreview = createHoverPreview();
  const { dialog, open, close } = createModalShell({ onClose: () => hoverPreview.destroy() });
  // Reuses showReviewModal's layout pattern (apps/image_tagging_server/static/js/app/review/main.js):
  // a max-height flex column with a fixed header/footer and a scrollable body,
  // so a large candidate grid can't push the footer buttons off-screen.
  dialog.style.cssText = 'max-width:900px;width:90vw;max-height:85vh;display:flex;flex-direction:column;';
  dialog.classList.add('relabel-confirm-modal');

  // Group candidates from the same image together (ranked by that image's best
  // candidate) so two boxes on one image are always adjacent in the grid —
  // with 500 images and 15 visible at a time, hover-highlighting alone isn't
  // enough to spot a same-image pair. Within a group, still highest-confidence first.
  const _imageMaxConfidence = new Map();
  for (const c of candidates) {
    const prev = _imageMaxConfidence.get(c.imageUuid) ?? -Infinity;
    _imageMaxConfidence.set(c.imageUuid, Math.max(prev, c.confidence ?? 0));
  }
  const sorted = [...candidates].sort((a, b) => {
    const groupDiff = _imageMaxConfidence.get(b.imageUuid) - _imageMaxConfidence.get(a.imageUuid);
    if (groupDiff !== 0) return groupDiff;
    if (a.imageUuid !== b.imageUuid) return a.imageUuid < b.imageUuid ? -1 : 1;
    return (b.confidence ?? 0) - (a.confidence ?? 0);
  });

  const title = document.createElement('h3');
  title.className = 'app-modal-title';
  title.style.flexShrink = '0';
  const _newCount = sorted.filter(c => c.isNew).length;
  const _suffix = _newCount > 0 ? ` (${_newCount} new)` : '';
  title.textContent = sorted.length === 0
    ? `No candidates from ${ctx.modelName}`
    : `${sorted.length} candidate${sorted.length === 1 ? '' : 's'} from ${ctx.modelName}${_suffix}`;
  dialog.appendChild(title);

  if (sorted.length === 0) {
    const empty = document.createElement('div');
    empty.className = 'relabel-modal-empty';
    const stats = ctx.stats;
    if (stats) {
      const parts = [];
      if (stats.noOpCount > 0) parts.push(`${stats.noOpCount} bbox(es) already matched the model's prediction`);
      if (stats.noDetectionCount > 0) parts.push(`${stats.noDetectionCount} bbox(es) had no matching detection`);
      if (stats.imagesFailed > 0) parts.push(`${stats.imagesFailed} image(s) failed to load/process`);
      empty.textContent = parts.length > 0
        ? `No candidates: ${parts.join('; ')}.`
        : 'No candidates above threshold.';
      if (stats.imagesFailed > 0 && stats.sampleErrors?.length) {
        const errDetail = document.createElement('div');
        errDetail.style.cssText = 'font-size:0.8em;color:#e88;margin-top:0.5em;';
        errDetail.textContent = `Sample error: ${stats.sampleErrors[0]}`;
        empty.appendChild(document.createElement('br'));
        empty.appendChild(errDetail);
      }
    } else {
      empty.textContent = 'No candidates above threshold.';
    }
    dialog.appendChild(empty);

    const footer = document.createElement('div');
    footer.className = 'classify-modal-footer';
    footer.style.flexShrink = '0';
    const closeBtn = document.createElement('button');
    closeBtn.className = 'secondary';
    closeBtn.textContent = 'Close';
    closeBtn.addEventListener('click', close);
    footer.appendChild(closeBtn);
    dialog.appendChild(footer);

    open();
    return;
  }

  // Scrollable body: everything between the fixed title and fixed footer.
  const body = document.createElement('div');
  body.style.cssText = 'flex:1;overflow-y:auto;min-height:0;';
  dialog.appendChild(body);

  // ── Action selector: Relabel to proposed vs Delete ────────────────────
  const actionRow = document.createElement('div');
  actionRow.className = 'relabel-modal-action-row';
  actionRow.style.cssText = 'display:flex;gap:1em;align-items:center;';

  const relabelLabel = document.createElement('label');
  relabelLabel.style.cssText = 'display:flex;align-items:center;gap:0.3em;cursor:pointer;';
  const relabelRadio = document.createElement('input');
  relabelRadio.type = 'radio';
  relabelRadio.name = 'relabel-modal-action';
  relabelRadio.value = 'relabel';
  relabelRadio.checked = true;
  relabelLabel.append(relabelRadio, document.createTextNode('Relabel to proposed'));

  const deleteLabel = document.createElement('label');
  deleteLabel.style.cssText = 'display:flex;align-items:center;gap:0.3em;cursor:pointer;';
  const deleteRadio = document.createElement('input');
  deleteRadio.type = 'radio';
  deleteRadio.name = 'relabel-modal-action';
  deleteRadio.value = 'delete';
  deleteLabel.append(deleteRadio, document.createTextNode('Delete'));

  actionRow.append(relabelLabel, deleteLabel);
  body.appendChild(actionRow);

  // ── Bulk select controls ───────────────────────────────────────────────
  const bulkRow = document.createElement('div');
  bulkRow.className = 'relabel-modal-bulk-row';
  bulkRow.style.cssText = 'display:flex;gap:0.5em;';
  const selectAllBtn = document.createElement('button');
  selectAllBtn.className = 'secondary';
  selectAllBtn.textContent = 'Select all';
  const selectNoneBtn = document.createElement('button');
  selectNoneBtn.className = 'secondary';
  selectNoneBtn.textContent = 'Select none';
  bulkRow.append(selectAllBtn, selectNoneBtn);
  body.appendChild(bulkRow);

  // ── Candidate grid ──────────────────────────────────────────────────────
  const grid = document.createElement('div');
  grid.className = 'relabel-modal-grid review-grid';
  body.appendChild(grid);

  /** @type {Array<{candidate: object, card: HTMLElement, checkbox: HTMLInputElement, proposedChip: HTMLElement}>} */
  const rows = [];

  // ── Shared selection controller (click / shift-range select) ────────────
  // The candidate cards get real review-card selection semantics via the shared
  // cardGrid controller — distinct from the per-card "will Apply" checkbox
  // (which stays the Apply bit). `_visibleEntries` mirrors `rows` in order so
  // shift-range select and (future) type-fading iterate the same list.
  const _visibleEntries = [];
  const cardGrid = createCardGrid(grid, {
    getVisibleEntries: () => _visibleEntries,
    entryId: (entry) => entry.id,
    onSelectionChange: () => {},
    hooks: { toast: (m) => toast(m, 3000, 'warning') },
  });

  // Re-render the proposed-label chip in place after an in-memory relabel.
  function _rerenderProposedChip(candidate) {
    const row = rows.find(r => r.candidate === candidate);
    if (!row) return;
    const fresh = _labelChip(candidate.proposedLabelName, candidate.proposedLabelUuid, ctx.labels);
    fresh.classList.add('relabel-modal-proposed-chip');
    // Preserve the delete-mode dimming currently applied to proposed chips.
    fresh.style.opacity = row.proposedChip.style.opacity;
    row.proposedChip.replaceWith(fresh);
    row.proposedChip = fresh;
  }

  // Drop a candidate from the preview so Apply won't include it. Uncheck (so the
  // "will Apply" bit is off even if the row lingered) and remove the row/card.
  function _removeCandidate(candidate) {
    const idx = rows.findIndex(r => r.candidate === candidate);
    if (idx === -1) return;
    const { card, checkbox } = rows[idx];
    checkbox.checked = false;
    cardGrid.selected.delete(rows[idx].id);
    rows.splice(idx, 1);
    const veIdx = _visibleEntries.findIndex(e => e.candidate === candidate);
    if (veIdx !== -1) _visibleEntries.splice(veIdx, 1);
    card.remove();
    _updateApplyBtn();
  }

  // Open the shared canvas editor on a candidate box. Editing geometry mutates
  // `candidate.geometry` in memory (the field Apply reads); Cancel discards.
  async function _openCandidateEditor(candidate) {
    const { showCanvasModal } = await import('/app-static/js/app/review/main.js');
    const img = {
      uuid: candidate.imageUuid,
      filename: candidate.imagePath,
      // The canvas loads the real image's bboxes itself; for a NEW box we seed
      // the proposed geometry as a not-yet-persisted box below.
      bboxes: [],
    };
    // A synthetic bbox object for the editor's initial-target argument.
    const candidateBbox = {
      uuid: candidate.isNew ? 'new' : candidate.bboxUuid,
      labelUuid: candidate.proposedLabelUuid,
      label: ctx.labels.get(candidate.proposedLabelUuid) ?? null,
      ...(candidate.geometry ?? {}),
      tags: [],
    };
    const seedBboxes = (candidate.isNew && candidate.geometry)
      ? [{
          uuid: 'new',
          label: ctx.labels.get(candidate.proposedLabelUuid)
            ? { uuid: candidate.proposedLabelUuid, text: candidate.proposedLabelName, color: ctx.labels.get(candidate.proposedLabelUuid).color }
            : undefined,
          x: candidate.geometry.x,
          y: candidate.geometry.y,
          width: candidate.geometry.width,
          height: candidate.geometry.height,
          dirty: false,
        }]
      : null;
    const previewEditOpts = {
      allowNavigation: false,
      broadcast: false,
      seedBboxes,
      // persist reads edited geometry back into the candidate(s) WITHOUT the DB.
      // Existing boxes are matched by uuid; a single new box takes the sole
      // seeded 'new' box's geometry.
      persist: (runtime) => {
        const uiBboxes = [...runtime.state.image.bboxes.values()];
        if (candidate.isNew) {
          const box = uiBboxes.find(b => b.uuid === 'new') ?? uiBboxes[0];
          if (box) candidate.geometry = { x: box.x, y: box.y, width: box.width, height: box.height };
        } else {
          const box = uiBboxes.find(b => b.uuid === candidate.bboxUuid);
          if (box) candidate.geometry = { x: box.x, y: box.y, width: box.width, height: box.height };
        }
        return true;
      },
    };
    showCanvasModal(candidateBbox, img, ctx.labels, async () => {}, null, previewEditOpts);
  }

  const _previewController = {
    labels: ctx.labels,
    removeCandidate: _removeCandidate,
    rerenderProposedChip: _rerenderProposedChip,
    openEditor: _openCandidateEditor,
  };

  // ── Same-image hover highlight ──────────────────────────────────────────
  // Candidates are already grouped adjacently by image (see `sorted` above);
  // this makes a same-image group visually obvious even when it's split
  // across two grid rows. Mirrors review/main.js's attachGroupHighlight.
  function _attachGroupHighlight(card, imageUuid) {
    card.addEventListener('mouseenter', () => {
      card.classList.add('group-highlight-self-bbox');
      grid.querySelectorAll(`[data-image-uuid="${imageUuid}"]`).forEach(c => {
        if (c === card) return;
        c.classList.add('group-highlight-bbox');
      });
    });
    card.addEventListener('mouseleave', () => {
      card.classList.remove('group-highlight-self-bbox');
      grid.querySelectorAll('.group-highlight-bbox').forEach(c => c.classList.remove('group-highlight-bbox'));
    });
  }

  let _cardIndex = 0;
  for (const candidate of sorted) {
    const card = document.createElement('div');
    card.className = 'review-card relabel-modal-card';
    // Stable id for the selection controller: real uuid for existing boxes, a
    // synthetic one for NEW candidates (no uuid yet). Cards carry data-kind and
    // data-bbox-uuid so the shared applySelect shift-range selectors resolve.
    const cardId = candidate.isNew
      ? `candidate-new-${_cardIndex}`
      : candidate.bboxUuid;
    card.dataset.imageUuid = candidate.imageUuid;
    card.dataset.bboxUuid = cardId;
    card.dataset.kind = 'bbox';
    _attachGroupHighlight(card, candidate.imageUuid);

    // Existing-bbox candidates: let the preview fetch the image's real bboxes
    // and highlight this one by uuid (no need to know its geometry ourselves).
    // New-box candidates: no uuid exists yet, so hand it the proposed geometry
    // directly as a single synthetic bbox.
    if (candidate.isNew) {
      hoverPreview.attach(card, {
        imagePath: candidate.imagePath,
        imageUuid: candidate.imageUuid,
        allBboxes: candidate.geometry ? [{ uuid: 'new', ...candidate.geometry,
          label: { color: ctx.labels.get(candidate.proposedLabelUuid)?.color } }] : [],
        hoveredBboxUuid: 'new',
      });
    } else {
      hoverPreview.attach(card, {
        imagePath: candidate.imagePath,
        imageUuid: candidate.imageUuid,
        hoveredBboxUuid: candidate.bboxUuid,
      });
    }

    const checkbox = document.createElement('input');
    checkbox.type = 'checkbox';
    checkbox.checked = true;
    checkbox.addEventListener('change', _updateApplyBtn);
    card.appendChild(checkbox);

    const thumb = document.createElement('img');
    thumb.className = 'thumb';
    thumb.loading = 'lazy';
    if (candidate.isNew) {
      // A new box has no bbox uuid to crop yet — show the whole image for
      // context, with the proposed geometry drawn as a canvas overlay (same
      // letterbox-aware approach as review/main.js's buildImageCard).
      thumb.src = getImageThumbnailUrl(candidate.imagePath, 300);
      const thumbWrap = document.createElement('div');
      thumbWrap.style.cssText = 'position:relative;';
      const overlay = document.createElement('canvas');
      overlay.className = 'thumb-overlay';
      overlay.style.cssText = 'position:absolute;inset:0;width:100%;height:100%;pointer-events:none;';

      const drawOverlay = () => {
        if (!candidate.geometry) return;
        drawLetterboxedBboxes(thumb, overlay, [{ ...candidate.geometry, color: '#5cc45c' }]);
      };

      if (thumb.complete && thumb.naturalWidth) {
        drawOverlay();
      } else {
        thumb.onload = drawOverlay;
      }

      thumbWrap.append(thumb, overlay);
      card.appendChild(thumbWrap);
    } else {
      thumb.src = getBboxThumbnailUrl(candidate.imagePath, candidate.bboxUuid, 300);
      card.appendChild(thumb);
    }

    const info = document.createElement('div');
    info.className = 'card-info';

    const labelsRow = document.createElement('div');
    labelsRow.className = 'relabel-modal-labels-row';
    labelsRow.style.cssText = 'display:flex;align-items:center;gap:0.4em;flex-wrap:wrap;';
    const proposedChip = _labelChip(candidate.proposedLabelName, candidate.proposedLabelUuid, ctx.labels);
    proposedChip.classList.add('relabel-modal-proposed-chip');
    if (candidate.isNew) {
      // New box: a "NEW" badge instead of a current→proposed transition.
      const newBadge = document.createElement('span');
      newBadge.className = 'label-chip relabel-modal-new-badge';
      newBadge.textContent = '＋ NEW';
      newBadge.style.cssText = 'background:#2e7d3233;border:1px solid #2e7d32;color:#5cc45c;';
      labelsRow.append(newBadge, proposedChip);
    } else {
      const currentChip = _labelChip(candidate.currentLabelName, candidate.currentLabelUuid, ctx.labels);
      const arrow = document.createElement('span');
      arrow.textContent = '→';
      labelsRow.append(currentChip, arrow, proposedChip);
    }
    info.appendChild(labelsRow);

    const pct = Math.round((candidate.confidence ?? 0) * 100);
    const barWrap = document.createElement('div');
    barWrap.className = 'classify-modal-bar-wrap';
    const bar = document.createElement('div');
    bar.className = 'classify-modal-bar';
    bar.style.width = `${pct}%`;
    bar.style.background = 'var(--pico-primary)';
    barWrap.appendChild(bar);
    const pctSpan = document.createElement('span');
    pctSpan.className = 'classify-modal-pct';
    pctSpan.textContent = `${pct}%`;
    const confRow = document.createElement('div');
    confRow.style.cssText = 'display:flex;align-items:center;gap:0.4em;';
    confRow.append(barWrap, pctSpan);
    info.appendChild(confRow);

    card.appendChild(info);
    grid.appendChild(card);

    const entry = { id: cardId, kind: 'bbox', candidate, card };
    _visibleEntries.push(entry);
    const index = _cardIndex;

    // ── Card-click selection (separate from the Apply checkbox) ────────────
    // The shared applySelect drives off a detached checkbox proxy so selection
    // (the `.selected` class + the controller's Set, used for multi-target
    // context-menu actions) stays independent of the visible "will Apply" box.
    const selProxy = document.createElement('input');
    selProxy.type = 'checkbox';
    let _selTimer = null;
    card.addEventListener('click', (e) => {
      // Ignore clicks on the Apply checkbox and interactive card chrome.
      if (e.target === checkbox) return;
      if (e.shiftKey) e.preventDefault();
      clearTimeout(_selTimer);
      const shiftKey = e.shiftKey;
      const newChecked = !cardGrid.selected.has(cardId);
      _selTimer = setTimeout(() => {
        if (!card.isConnected) return;
        selProxy.checked = !newChecked; // applySelect flips it to newChecked
        cardGrid.applySelect(selProxy, card, cardId, 'bbox', index, shiftKey, newChecked);
      }, 220);
    });

    // ── Double-click → open the shared canvas geometry editor ──────────────
    card.addEventListener('dblclick', (e) => {
      if (e.target === checkbox) return;
      clearTimeout(_selTimer);
      _openCandidateEditor(candidate);
    });

    // ── Right-click context menu (Relabel / Edit-geometry / Remove / Similar)
    attachContext(card, () => {
      const _selected = cardGrid.selected;
      // Multi-select: act on every selected candidate of this kind.
      const targets = (_selected.has(cardId) && _selected.size > 1)
        ? _visibleEntries.filter(en => _selected.has(en.id))
        : [entry];
      return targets.map(en => ({
        adapters: { candidateBbox: makeCandidateBboxAdapter(en.candidate, en.card, _previewController) },
      }));
    });

    rows.push({ id: cardId, candidate, card, checkbox, proposedChip });
    _cardIndex += 1;
  }

  function _setDeleteMode(isDelete) {
    for (const r of rows) {
      r.proposedChip.style.opacity = isDelete ? '0.35' : '1';
    }
  }
  relabelRadio.addEventListener('change', () => _setDeleteMode(false));
  deleteRadio.addEventListener('change', () => _setDeleteMode(true));

  selectAllBtn.addEventListener('click', () => {
    for (const r of rows) r.checkbox.checked = true;
    _updateApplyBtn();
  });
  selectNoneBtn.addEventListener('click', () => {
    for (const r of rows) r.checkbox.checked = false;
    _updateApplyBtn();
  });

  // ── Footer ────────────────────────────────────────────────────────────
  const footer = document.createElement('div');
  footer.className = 'classify-modal-footer';
  footer.style.cssText = 'display:flex;gap:0.5em;justify-content:flex-end;flex-shrink:0;';

  const applyBtn = document.createElement('button');
  const cancelBtn = document.createElement('button');
  cancelBtn.className = 'secondary';
  cancelBtn.textContent = 'Cancel';
  cancelBtn.addEventListener('click', close);
  footer.append(cancelBtn, applyBtn);
  dialog.appendChild(footer);

  function _updateApplyBtn() {
    const n = rows.filter(r => r.checkbox.checked).length;
    applyBtn.textContent = `Apply to ${n} checked`;
    applyBtn.disabled = n === 0;
  }
  _updateApplyBtn();

  applyBtn.addEventListener('click', async () => {
    const checkedRows = rows.filter(r => r.checkbox.checked);
    if (checkedRows.length === 0) return;

    applyBtn.disabled = true;
    cancelBtn.disabled = true;
    try {
      const isDelete = deleteRadio.checked;
      // New-box candidates have no bbox uuid yet — they're created (not
      // relabeled/deleted). The Relabel/Delete radio governs only existing boxes;
      // in Delete mode new boxes are simply not created.
      const newRows = checkedRows.filter(r => r.candidate.isNew);
      const existingRows = checkedRows.filter(r => !r.candidate.isNew);
      const existingUuids = existingRows.map(r => r.candidate.bboxUuid);
      let modifiedCount = 0;
      let addedCount = 0;
      let failCount = 0;

      if (isDelete) {
        if (existingUuids.length > 0) {
          const res = await bulkApplyBboxesByUuids({ bboxUuids: existingUuids, action: 'delete' });
          modifiedCount += res.modifiedCount;
          failCount += res.failCount;
        }
      } else {
        const groups = new Map();
        for (const r of existingRows) {
          const key = r.candidate.proposedLabelUuid;
          if (!groups.has(key)) groups.set(key, []);
          groups.get(key).push(r.candidate);
        }
        for (const [labelUuid, groupCandidates] of groups) {
          const bboxUuids = groupCandidates.map(c => c.bboxUuid);
          // The model that just ran is assumed more precise than whatever produced
          // the original box, so its suggested geometry (if any) replaces it.
          const bboxGeometry = {};
          for (const c of groupCandidates) {
            if (c.geometry) bboxGeometry[c.bboxUuid] = c.geometry;
          }
          const res = await bulkApplyBboxesByUuids({
            bboxUuids,
            action: 'relabel',
            labelUuid,
            bboxGeometry: Object.keys(bboxGeometry).length > 0 ? bboxGeometry : null,
          });
          modifiedCount += res.modifiedCount;
          failCount += res.failCount;
        }

        // Create the accepted new boxes (UNVERIFIED, so they land as reviewable).
        for (const r of newRows) {
          const c = r.candidate;
          try {
            await addBbox({
              imagePath: c.imagePath,
              labelUuid: c.proposedLabelUuid,
              x: c.geometry.x,
              y: c.geometry.y,
              width: c.geometry.width,
              height: c.geometry.height,
              tagUuids: [UNVERIFIED_TAG_UUID],
              metadataJson: { run_model_added: true, source_model: ctx.modelName },
            });
            addedCount += 1;
          } catch (err) {
            Logger.error('addBbox failed:', err);
            failCount += 1;
          }
        }
      }

      if (existingUuids.length > 0) {
        await invalidateBboxThumbnails(existingUuids).catch((err) => Logger.error('invalidateBboxThumbnails failed:', err));
      }

      const applied = [];
      if (modifiedCount > 0) applied.push(`${modifiedCount} bbox(es)`);
      if (addedCount > 0) applied.push(`${addedCount} new box(es)`);
      const head = applied.length > 0 ? `Applied to ${applied.join(' + ')}` : 'Nothing applied';
      const summary = failCount > 0 ? `${head}, ${failCount} failed` : head;
      toast(summary, 4000, failCount > 0 ? 'warning' : 'success');

      close();
      ctx.refresh?.();
    } catch (err) {
      Logger.error('Failed to apply relabel candidates:', err);
      toast(`Apply failed: ${err.message}`, 5000, 'error');
      applyBtn.disabled = false;
      cancelBtn.disabled = false;
    }
  });

  open();
}

function _fmtPct(x) {
  return `${Math.round((x ?? 0) * 100)}%`;
}

/**
 * Warns before a large "Run Model on Results" run (candidate count above
 * `WARN_THRESHOLD`) and lets the user cap how many candidates to process. The
 * count input defaults to blank, meaning "all of them".
 *
 * @param {number} total - total candidate bbox count.
 * @returns {Promise<number|null>} the chosen limit (a positive integer), or
 *   `Infinity` to process all, or `null` if the user cancels/dismisses.
 */
export function confirmLargeRun(total) {
  return new Promise((resolve) => {
    let resolved = false;
    const finish = (v) => { if (resolved) return; resolved = true; resolve(v); close(); };
    const { dialog, open, close } = createModalShell({ onClose: () => finish(null) });
    dialog.style.cssText = 'max-width:480px;display:flex;flex-direction:column;gap:0.8em;';
    dialog.classList.add('run-model-large-warning-dialog');

    const title = document.createElement('h3');
    title.className = 'app-modal-title';
    title.textContent = 'Large run';
    dialog.appendChild(title);

    const msg = document.createElement('div');
    msg.append(
      document.createTextNode("You're about to run the model on "),
      Object.assign(document.createElement('strong'), { textContent: total.toLocaleString() }),
      document.createTextNode(' candidate bbox(es). Large runs can take a while.'),
    );
    dialog.appendChild(msg);

    const inputRow = document.createElement('div');
    inputRow.style.cssText = 'display:flex;flex-direction:column;gap:0.3em;';
    const inputLabel = document.createElement('label');
    inputLabel.style.cssText = 'font-size:0.85em;color:#aaa;';
    inputLabel.textContent = 'How many to process? (leave blank for all)';
    const input = document.createElement('input');
    input.type = 'number';
    input.min = '1';
    input.max = String(total);
    input.placeholder = `all (${total.toLocaleString()})`;
    input.className = 'run-model-large-warning-limit';
    inputRow.append(inputLabel, input);
    dialog.appendChild(inputRow);

    const btnRow = document.createElement('div');
    btnRow.style.cssText = 'display:flex;gap:0.5em;justify-content:flex-end;';
    const cancelBtn = document.createElement('button');
    cancelBtn.className = 'secondary';
    cancelBtn.textContent = 'Cancel';
    cancelBtn.addEventListener('click', () => finish(null));
    const continueBtn = document.createElement('button');
    continueBtn.textContent = 'Continue';
    continueBtn.addEventListener('click', () => {
      const raw = input.value.trim();
      if (raw === '') { finish(Infinity); return; }
      const n = Math.floor(Number(raw));
      if (!Number.isFinite(n) || n <= 0) {
        toast('Enter a positive number, or leave blank to process all.', 3000, 'warning');
        return;
      }
      finish(Math.min(n, total));
    });
    btnRow.append(cancelBtn, continueBtn);
    dialog.appendChild(btnRow);

    open();
  });
}

/**
 * Builds and shows the "Run Model on Results" dialog: model picker (detectors
 * only), conf/IoU sliders, candidate-count display with cap warnings, and a
 * Run button that launches PreviewModelRelabelTask and switches the dialog to
 * a poll/progress/cancel view until the task completes.
 *
 * @param {{bboxUuids: string[], imageUuids: string[]}} scope
 * @param {Object} ctx - forwarded to `openRelabelConfirmationModal` (see its docs),
 *   plus the fields `resolveCandidateScope` needs.
 */
async function _showRunDialog(scope, ctx) {
  const { bboxUuids, imageUuids } = scope;

  const { dialog, open, close } = createModalShell();
  dialog.style.cssText = 'max-width:520px;max-height:85vh;display:flex;flex-direction:column;overflow-y:auto;';
  dialog.classList.add('run-model-on-results-dialog');

  const title = document.createElement('h3');
  title.className = 'app-modal-title';
  dialog.appendChild(title);
  title.textContent = 'Run Model on Results';

  const countLine = document.createElement('div');
  countLine.textContent = `${bboxUuids.length.toLocaleString()} candidate bbox(es) across ${imageUuids.length.toLocaleString()} image(s).`;
  dialog.appendChild(countLine);

  // The large-run warning (and optional cap) is handled in `runModelOnResultsFlow`
  // before this dialog is shown, so `bboxUuids` here is already the user-approved
  // set — no further gating needed.

  // ── Model + inference-input pickers, side by side to keep this compact ──
  const pickerRow = document.createElement('div');
  pickerRow.style.cssText = 'display:flex;gap:1em;';
  dialog.appendChild(pickerRow);

  const modelLabelRow = document.createElement('div');
  modelLabelRow.style.cssText = 'display:flex;align-items:center;justify-content:space-between;gap:0.5em;';
  const modelLabel = document.createElement('div');
  modelLabel.textContent = 'Model:';
  modelLabel.style.cssText = 'font-size:0.85em;color:#aaa;';
  const modelRefreshBtn = document.createElement('button');
  modelRefreshBtn.type = 'button';
  modelRefreshBtn.className = 'secondary run-model-on-results-refresh';
  modelRefreshBtn.textContent = 'Refresh';
  modelRefreshBtn.title = 'Reload the model list (picks up newly imported models)';
  modelRefreshBtn.style.cssText = 'padding:0.1em 0.6em;font-size:0.8em;';
  modelLabelRow.append(modelLabel, modelRefreshBtn);

  const modelRow = document.createElement('div');
  modelRow.style.cssText = 'display:flex;flex-direction:column;gap:0.3em;flex:1;';
  const modelSelect = document.createElement('select');
  modelSelect.className = 'run-model-on-results-model-select';
  modelRow.append(modelLabelRow, modelSelect);
  pickerRow.appendChild(modelRow);

  let models = [];
  /** @type {Map<string, string>} model name -> modelKind */
  let modelKinds = new Map();
  // Declared here (assigned below near the other dialog buttons) so
  // `_populateModelSelect` can toggle it whether it runs at initial load or
  // from a later refresh click, without a temporal-dead-zone reference error.
  let runBtn = null;

  async function _loadModels() {
    models = [];
    modelKinds = new Map();
    try {
      const allModels = await fetchModelsList();
      modelKinds = await fetchModelsKinds(allModels);
      models = allModels;
    } catch (err) {
      Logger.error('Failed to load models:', err);
    }
  }

  function _populateModelSelect() {
    const previousValue = modelSelect.value;
    modelSelect.innerHTML = '';
    if (models.length === 0) {
      const opt = document.createElement('option');
      opt.value = '';
      opt.textContent = '(no models found)';
      modelSelect.appendChild(opt);
    } else {
      const detectorNames = models.filter(m => !_CLASSIFIER_KINDS.has(modelKinds.get(m)));
      const classifierNames = models.filter(m => _CLASSIFIER_KINDS.has(modelKinds.get(m)));
      for (const [label, names] of [['Detectors', detectorNames], ['Classifiers', classifierNames]]) {
        if (names.length === 0) continue;
        const group = document.createElement('optgroup');
        group.label = label;
        for (const m of names) {
          const opt = document.createElement('option');
          opt.value = m;
          opt.textContent = m;
          group.appendChild(opt);
        }
        modelSelect.appendChild(group);
      }
      if (models.includes(previousValue)) modelSelect.value = previousValue;
    }
    if (runBtn) runBtn.disabled = models.length === 0;
  }

  modelRefreshBtn.addEventListener('click', async () => {
    modelRefreshBtn.disabled = true;
    try {
      clearModelsCache();
      await _loadModels();
      _populateModelSelect();
      _syncCropModeUI();
    } finally {
      modelRefreshBtn.disabled = false;
    }
  });

  await _loadModels();
  _populateModelSelect();

  // ── Crop mode: whole image vs. per-bbox crop ─────────────────────────
  let cropMode = 'image';
  const cropModeRow = document.createElement('div');
  cropModeRow.style.cssText = 'display:flex;flex-direction:column;gap:0.3em;flex:1;';
  const cropModeLabel = document.createElement('div');
  cropModeLabel.textContent = 'Inference input:';
  cropModeLabel.style.cssText = 'font-size:0.85em;color:#aaa;';
  const cropModeSelect = document.createElement('select');
  cropModeSelect.className = 'run-model-on-results-crop-mode-select';
  const cropModeImageOpt = document.createElement('option');
  cropModeImageOpt.value = 'image';
  cropModeImageOpt.textContent = 'Whole image (IoU match)';
  const cropModeBboxOpt = document.createElement('option');
  cropModeBboxOpt.value = 'bbox';
  cropModeBboxOpt.textContent = 'Bbox crop';
  cropModeSelect.append(cropModeImageOpt, cropModeBboxOpt);
  cropModeRow.append(cropModeLabel, cropModeSelect);
  pickerRow.appendChild(cropModeRow);

  // ── Confidence + IoU sliders ─────────────────────────────────────────
  let confThresh = 0.5;
  let iouThresh = 0.5;

  function makeSliderRow(labelText, initial, onChange) {
    const row = document.createElement('div');
    row.className = 'inference-conf-row';
    const label = document.createElement('label');
    label.className = 'inference-conf-label';
    label.textContent = labelText;
    const slider = document.createElement('input');
    slider.type = 'range';
    slider.min = '0';
    slider.max = '1';
    slider.step = '0.05';
    slider.value = String(initial);
    slider.className = 'inference-conf-slider';
    const value = document.createElement('span');
    value.className = 'inference-conf-value';
    value.textContent = _fmtPct(initial);
    slider.addEventListener('input', () => {
      const v = parseFloat(slider.value);
      value.textContent = _fmtPct(v);
      onChange(v);
    });
    row.append(label, slider, value);
    return row;
  }

  dialog.appendChild(makeSliderRow('Min confidence:', confThresh, (v) => { confThresh = v; }));
  const iouRow = makeSliderRow('Min IoU:', iouThresh, (v) => { iouThresh = v; });
  dialog.appendChild(iouRow);

  // ── Add-new-boxes toggle (whole-image mode only) ─────────────────────
  const addNewRow = document.createElement('label');
  addNewRow.style.cssText = 'display:flex;align-items:center;gap:0.4em;cursor:pointer;font-size:0.9em;';
  const addNewCheckbox = document.createElement('input');
  addNewCheckbox.type = 'checkbox';
  addNewCheckbox.className = 'run-model-on-results-add-new';
  addNewRow.append(
    addNewCheckbox,
    document.createTextNode('Also add new boxes (detections overlapping nothing)'),
  );
  dialog.appendChild(addNewRow);

  function _isClassifierSelected() {
    return _CLASSIFIER_KINDS.has(modelKinds.get(modelSelect.value));
  }

  function _syncCropModeUI() {
    cropMode = cropModeSelect.value;
    const isClassifier = _isClassifierSelected();
    // Crop mode, IoU matching, and "add new boxes" are all detector-only
    // concepts (whole-image classification has no geometry to match/refine).
    cropModeRow.style.display = isClassifier ? 'none' : '';
    iouRow.style.display = (isClassifier || cropMode === 'bbox') ? 'none' : '';
    // "Add new boxes" needs whole-image detections to reason about overlap, so
    // it's only meaningful in image crop_mode.
    addNewRow.style.display = (isClassifier || cropMode === 'bbox') ? 'none' : 'flex';
    if (isClassifier || cropMode === 'bbox') addNewCheckbox.checked = false;
  }
  cropModeSelect.addEventListener('change', _syncCropModeUI);
  modelSelect.addEventListener('change', _syncCropModeUI);
  _syncCropModeUI();

  // ── Progress area (hidden until Run) ─────────────────────────────────
  const progressWrap = document.createElement('div');
  progressWrap.style.cssText = 'display:none;flex-direction:column;gap:0.4em;';
  const progressLabel = document.createElement('div');
  progressLabel.style.cssText = 'font-size:0.85em;color:#aaa;';
  const progressBar = document.createElement('progress');
  progressBar.max = 100;
  progressBar.value = 0;
  progressBar.style.width = '100%';
  const progressCancelBtn = document.createElement('button');
  progressCancelBtn.className = 'secondary';
  progressCancelBtn.textContent = 'Cancel';
  progressWrap.append(progressLabel, progressBar, progressCancelBtn);
  dialog.appendChild(progressWrap);

  // ── Buttons ───────────────────────────────────────────────────────────
  const btnRow = document.createElement('div');
  btnRow.style.cssText = 'display:flex;gap:0.5em;justify-content:flex-end;';
  runBtn = document.createElement('button');
  runBtn.textContent = 'Run';
  runBtn.disabled = models.length === 0;
  const cancelBtn = document.createElement('button');
  cancelBtn.className = 'secondary';
  cancelBtn.textContent = 'Cancel';
  cancelBtn.addEventListener('click', close);
  btnRow.append(cancelBtn, runBtn);
  dialog.appendChild(btnRow);

  let taskUuid = null;
  let pollTimer = null;
  let cancelled = false;

  function _stopPolling() {
    if (pollTimer) { clearTimeout(pollTimer); pollTimer = null; }
  }

  progressCancelBtn.addEventListener('click', async () => {
    cancelled = true;
    progressCancelBtn.disabled = true;
    _stopPolling();
    if (taskUuid) {
      try {
        await cancelTasks({ taskUuids: [taskUuid] });
        await deleteTasks({ taskUuids: [taskUuid] });
      } catch (err) {
        toast(`Failed to cancel task: ${err.message}`, 5000, 'error');
      }
    }
    close();
  });

  runBtn.addEventListener('click', async () => {
    if (!modelSelect.value) { toast('Please select a model', 3000, 'warning'); return; }

    const modelName = modelSelect.value;
    const isClassifier = _isClassifierSelected();

    if (isClassifier) {
      if (imageUuids.length === 0) {
        toast('No images in the current selection/filter.', 4000, 'warning');
        return;
      }
    } else if (bboxUuids.length === 0) {
      toast('No candidate bboxes in the current selection/filter.', 4000, 'warning');
      return;
    }

    // Detector runs are already capped/warned on before this dialog opens
    // (see runModelOnResultsFlow); classifier runs are keyed off imageUuids
    // instead of bboxUuids, so they need their own large-run check here.
    let runImageUuids = imageUuids;
    if (isClassifier && imageUuids.length > WARN_THRESHOLD) {
      const limit = await confirmLargeRun(imageUuids.length);
      if (limit === null) return; // user cancelled/dismissed the warning
      runImageUuids = Number.isFinite(limit) ? imageUuids.slice(0, limit) : imageUuids;
    }

    runBtn.disabled = true;
    cancelBtn.disabled = true;
    modelSelect.disabled = true;
    cropModeSelect.disabled = true;
    progressWrap.style.display = 'flex';
    const totalImages = isClassifier ? runImageUuids.length : imageUuids.length;
    progressLabel.textContent = `Starting… (0/${totalImages} images)`;

    const params = isClassifier
      ? {
          model_name: modelName,
          conf_thresh: confThresh,
          image_uuids: runImageUuids,
        }
      : {
          model_name: modelName,
          crop_mode: cropMode,
          add_new: cropMode === 'image' && addNewCheckbox.checked,
          conf_thresh: confThresh,
          iou_thresh: iouThresh,
          image_uuids: imageUuids,
          bbox_uuids: bboxUuids,
        };

    try {
      const configUuid = await createTaskConfig({
        typename: isClassifier ? 'PreviewClassifierMatchTask' : 'PreviewModelRelabelTask',
        name: `Run Model on Results: ${modelName}`,
        params,
        persistent: false,
        isTemporary: true,
      });
      taskUuid = await startTask({ configUuid });

      const finalInfo = await _pollUntilDone(taskUuid, totalImages, (info) => {
        if (cancelled) return;
        const progress = info?.progress ?? 0;
        const processed = Math.round((progress / 100) * totalImages);
        progressBar.value = progress;
        progressLabel.textContent = `Processing… (${processed}/${totalImages} images)`;
      });

      if (cancelled) return;

      const status = finalInfo?.instance?.status;
      if (status !== TaskStatus.COMPLETED) {
        const msg = finalInfo?.instance?.errorMessage || `Task ended with status: ${status}`;
        toast(`Run failed: ${msg}`, 5000, 'error');
        await deleteTasks({ taskUuids: [taskUuid] }).catch(() => {});
        return;
      }

      progressLabel.textContent = 'Fetching results…';
      const resultsMap = await fetchTasksResult({ taskUuids: [taskUuid], cacheResults: false });
      const result = resultsMap.get(taskUuid);
      const data = result?.data ?? {};
      const candidates = data.candidates ?? [];
      // TaskResult.data is a raw passthrough dict (no camelCase alias conversion,
      // unlike most other API responses) — read the task's snake_case keys directly.

      await deleteTasks({ taskUuids: [taskUuid] }).catch((e) => Logger.error('Cleanup failed', e));

      close();
      if (isClassifier) {
        await openClassifierTagConfirmationModal(candidates, {
          modelName,
          refresh: ctx.refresh,
          labels: ctx.labels,
          stats: {
            imagesFailed: data.images_failed ?? 0,
            noMatchCount: data.no_match_count ?? 0,
            sampleErrors: data.sample_errors ?? [],
          },
        });
      } else {
        await openRelabelConfirmationModal(candidates, {
          modelName,
          refresh: ctx.refresh,
          labels: ctx.labels,
          stats: {
            imagesFailed: data.images_failed ?? 0,
            noDetectionCount: data.no_detection_count ?? 0,
            noOpCount: data.no_op_count ?? 0,
            sampleErrors: data.sample_errors ?? [],
          },
        });
      }
    } catch (err) {
      Logger.error('Run Model on Results failed:', err);
      toast(`Run failed: ${err.message}`, 5000, 'error');
      runBtn.disabled = false;
      cancelBtn.disabled = false;
      modelSelect.disabled = false;
      cropModeSelect.disabled = false;
      progressWrap.style.display = 'none';
    }
  });

  async function _pollUntilDone(uuid, totalImages, onUpdate) {
    return new Promise((resolve, reject) => {
      const _tick = async () => {
        if (cancelled) { resolve(null); return; }
        try {
          const { tasks } = await fetchTasksStatus({ taskUuids: [uuid] });
          const info = tasks?.[uuid];
          if (info) onUpdate(info);
          const status = info?.instance?.status;
          if (status === TaskStatus.COMPLETED || status === TaskStatus.ERROR || status === TaskStatus.CANCELLED) {
            resolve(info);
            return;
          }
        } catch (err) {
          reject(err);
          return;
        }
        if (!cancelled) pollTimer = setTimeout(_tick, POLL_INTERVAL_MS);
      };
      _tick();
    });
  }

  open();
}

/**
 * Entry point wired to the review-browser toolbar button. Resolves the
 * candidate scope from the current selection/filter state, warns (and lets the
 * user cap the count) on large runs, then opens the run dialog.
 *
 * @param {Object} ctx
 * @param {Function|null} ctx.fetchImages - the panel's custom data-fetcher (null in full-filter mode).
 * @param {Object|null} ctx.filterPanel - the panel's filter-panel instance (null in fetchImages mode).
 * @param {'all'|'images'|'bboxes'} ctx.viewFilter
 * @param {'bbox'|'image'|null} ctx.selectionKind
 * @param {Set<string>} ctx.selected
 * @param {Array<object>} ctx.visibleEntries
 * @param {(bbox: object) => boolean|null} ctx.bboxFilterFn
 * @param {Map<string, object>} ctx.labels
 * @param {string} ctx.sortBy
 * @param {string} ctx.sortOrder
 * @param {() => void} ctx.refresh - reload the current review-panel page after apply (Chunk C).
 */
export async function runModelOnResultsFlow(ctx) {
  let scope;
  try {
    scope = await resolveCandidateScope(ctx);
  } catch (err) {
    Logger.error('Failed to resolve candidate scope:', err);
    toast(`Failed to resolve candidates: ${err.message}`, 5000, 'error');
    return;
  }

  if (scope.bboxUuids.length === 0 && scope.imageUuids.length === 0) {
    toast('No candidates in the current selection/filter.', 4000, 'warning');
    return;
  }

  let runScope = scope;
  if (scope.bboxUuids.length > WARN_THRESHOLD) {
    const limit = await confirmLargeRun(scope.bboxUuids.length);
    if (limit === null) return; // user cancelled/dismissed the warning
    runScope = truncateScope(scope, limit);
  }

  await _showRunDialog(runScope, ctx);
}
