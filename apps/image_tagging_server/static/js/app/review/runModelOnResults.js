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
  getRegionThumbnailUrl,
  bulkApplyBboxesByUuids,
  invalidateBboxThumbnails,
  addBbox,
  UNVERIFIED_TAG_UUID,
  VERIFIED_TAG_UUID,
} from '/app-static/js/shared/api/review.js';
import { activeTrainingConfig } from '/app-static/js/shared/activeTrainingConfig.js';
import { reviewStateToFilterOpts } from '/app-static/js/shared/filterUtils.js';
import { fetchModelsList, fetchModelsKinds, clearModelsCache } from '/app-static/js/shared/api/models.js';
import { fetchModelGroups, clearModelGroupsCache } from '/app-static/js/shared/api/modelGroups.js';
import { expandModelGroupMembers } from '/app-static/js/shared/modelGroups/expand.js';
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
import { toast, Logger } from '/app-static/js/ui/utils/logging.js';
import { expandLabelFilterSet, bboxHasAnyTag, bboxMatchesConfidence, VERIFICATION_TAG_BY_STATE } from '/app-static/js/app/review/filterPredicate.js';
import { openClassifierTagConfirmationModal } from '/app-static/js/app/review/runClassifierOnResults.js';
import { labelChip as _labelChip } from '/app-static/js/app/review/labelChip.js';
import { createCardGrid } from '/app-static/js/app/review/cardGrid.js';
import { createPreviewRun, updatePreviewRun, deletePreviewRun } from '/app-static/js/shared/api/previewRuns.js';
import { attachContext } from '/app-static/js/ui/contextMenu/index.js';

const _CLASSIFIER_KINDS = new Set(['image_classifier', 'multi_label_classifier']);

const MODEL_ID_PREFIX = 'model:';
const GROUP_ID_PREFIX = 'group:';

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
    // The server's bboxConfidence term only decides which IMAGES come back; the
    // band still has to be re-applied per bbox here, or a matching image's
    // out-of-band boxes would render as cards of their own.
    const confidenceMatch = !fs || bboxMatchesConfidence(b, fs);
    return baseMatch && labelMatch && verificationMatch && confidenceMatch;
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
 * Like `_collectFromImages`, but for the "Filtered images" scope: every image
 * is included regardless of whether it has any bboxes at all (unlike the
 * bbox-filtered scope, which drops images with zero matching bboxes), and
 * every bbox on an included image is a candidate (no bbox-filter predicate —
 * the point of this scope is to widen past whatever bbox filter is active).
 */
function _collectAllFromImages(images) {
  const bboxUuids = [];
  const imageUuids = [];
  const bboxImageUuids = [];
  for (const img of images) {
    imageUuids.push(img.uuid);
    for (const b of (img.bboxes || [])) {
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
 * The current filter state as review-query options, shared by both scope
 * resolvers below. Delegates to the same mapping review/main.js's toolbar
 * counts use, so "Run Model on Results" resolves exactly the set the toolbar
 * counted. These call sites used to spell the mapping out inline and had
 * drifted — dropping `filenameNegate`/`dateNegate` turned a "does NOT contain"
 * filter into "contains", i.e. the complement of the user's filter.
 */
function _reviewFilterOpts(fs, viewFilter) {
  return reviewStateToFilterOpts(fs, {
    viewFilter,
    trainingLabelUuids: activeTrainingConfig.getLabelUuids(),
  });
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
    ..._reviewFilterOpts(fs, viewFilter),
    requireBboxes: viewFilter === 'bboxes',
    sortBy,
    sortOrder,
    limit: Number.MAX_SAFE_INTEGER,
  });
  const predicate = makeBboxFilterPredicate({ fetchImagesMode: false, bboxFilterFn, fs, labels });
  return _collectFromImages(images, predicate);
}

/**
 * Resolves the "Filtered images" scope — the image-scoped sibling of
 * `resolveCandidateScope` used for the "Run Model on Results" run dialog's
 * scope toggle. Unlike `resolveCandidateScope`, this includes every image
 * matching the current image-level filter/selection (even ones with zero
 * bboxes) and every bbox on each included image (not filtered by the bbox
 * tag/label filter) — the point of this scope is to widen past whatever bbox
 * filter produced the small "candidate bboxes" count.
 *
 * Mirrors the toolbar's "Apply to Filtered Images" button semantics
 * (`requireBboxes: viewFilter === 'bboxes'`), so its count matches what's
 * already shown there.
 *
 * @param {Object} ctx - see `runModelOnResultsFlow` for field docs.
 * @returns {Promise<{bboxUuids: string[], imageUuids: string[], bboxImageUuids: string[]}>}
 */
export async function resolveImageScope(ctx) {
  const { fetchImages, filterPanel, viewFilter, selectionKind, selected, visibleEntries, sortBy, sortOrder } = ctx;

  if (selectionKind === 'bbox' && selected.size > 0) {
    const entries = visibleEntries.filter(e => e.kind === 'bbox' && selected.has(e.bbox.uuid));
    const imagesByUuid = new Map();
    for (const e of entries) imagesByUuid.set(e.img.uuid, e.img);
    return _collectAllFromImages([...imagesByUuid.values()]);
  }

  if (selectionKind === 'image' && selected.size > 0) {
    const selectedImages = visibleEntries
      .filter(e => e.kind === 'image' && selected.has(e.img.uuid))
      .map(e => e.img);
    return _collectAllFromImages(selectedImages);
  }

  if (fetchImages) {
    const images = await fetchImages({ afterUuid: null, limit: Number.MAX_SAFE_INTEGER });
    return _collectAllFromImages(images);
  }

  const fs = filterPanel.getState();
  const { images } = await fetchReviewQueue({
    ..._reviewFilterOpts(fs, viewFilter),
    requireBboxes: viewFilter === 'bboxes',
    sortBy,
    sortOrder,
    limit: Number.MAX_SAFE_INTEGER,
  });
  return _collectAllFromImages(images);
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

  function thumbnailUrl() {
    if (!candidate.isNew) return getBboxThumbnailUrl(candidate.imagePath, candidate.bboxUuid, 300);
    return candidate.geometry
      ? getRegionThumbnailUrl(candidate.imagePath, candidate.geometry, {
          width: 300,
          color: labels.get(candidate.proposedLabelUuid)?.color ?? null,
        })
      : getImageThumbnailUrl(candidate.imagePath, 300);
  }

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
    getThumbnailUrl: thumbnailUrl,

    // ── Relabel (no DB) ────────────────────────────────────────────────────
    // Sets the proposed label on the candidate and re-renders its chip in place.
    // The name comes from the labels Map (fetchLabels() returns a Map).
    async applyLabel(labelUuid) {
      candidate.proposedLabelUuid = labelUuid;
      candidate.proposedLabelName = labelUuid ? (labels.get(labelUuid)?.name ?? null) : null;
      rerenderProposedChip(candidate);
      // A proposed NEW box is outlined in the proposed label's colour, and that
      // colour is baked into the crop URL (the box has no row for the server to
      // read a label off). Without this the chip said one label and the crop
      // stayed outlined in the previous one — most visible when relabelling a
      // multi-selection, where every card but the chip row looked untouched.
      const thumb = cardEl?.querySelector('.thumb');
      const url = thumbnailUrl();
      if (thumb && thumb.getAttribute('src') !== url) thumb.src = url;
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
  // a fixed-height flex column with a fixed header/footer and a scrollable body,
  // so a large candidate grid can't push the footer buttons off-screen. --full
  // matches .compare-modal-dialog (the search modal) rather than the 900px cap
  // this used to hard-code inline — triaging thousands of candidates wants the
  // same working surface browsing results gets.
  dialog.classList.add('app-modal-dialog--full', 'relabel-confirm-modal');

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

  // Accepting this preview applies a *model's* opinion, not a human's, so it must
  // NOT silently promote boxes to VERIFIED — the whole point of the review queue
  // is that model output stays reviewable. Opt-in only, default off.
  const verifyLabel = document.createElement('label');
  verifyLabel.style.cssText = 'display:flex;align-items:center;gap:0.3em;cursor:pointer;margin-left:auto;';
  verifyLabel.title = 'Off: relabeled boxes keep their current verification state (still reviewable). '
    + 'On: treat this apply as a human decision and mark the boxes verified.';
  const verifyCheckbox = document.createElement('input');
  verifyCheckbox.type = 'checkbox';
  verifyCheckbox.className = 'relabel-modal-mark-verified';
  verifyCheckbox.checked = false;
  verifyLabel.append(verifyCheckbox, document.createTextNode('Mark as verified'));

  actionRow.append(relabelLabel, deleteLabel, verifyLabel);
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
      // A new box has no bbox uuid to crop by, but its proposed geometry is
      // enough for a region crop — the whole frame this used to show left a
      // small proposal invisible. Hovering still gives the whole frame.
      thumb.src = candidate.geometry
        ? getRegionThumbnailUrl(candidate.imagePath, candidate.geometry, {
            width: 300,
            color: ctx.labels.get(candidate.proposedLabelUuid)?.color ?? '#5cc45c',
          })
        : getImageThumbnailUrl(candidate.imagePath, 300);
      card.appendChild(thumb);
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
    // Verification only applies to the relabel branch — deletes have nothing to verify.
    verifyCheckbox.disabled = isDelete;
    verifyLabel.style.opacity = isDelete ? '0.35' : '1';
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
      // Off by default: a model's suggestion isn't a human verification.
      const markVerified = verifyCheckbox.checked;
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
            markVerified,
          });
          modifiedCount += res.modifiedCount;
          failCount += res.failCount;
        }

        // Create the accepted new boxes — UNVERIFIED so they land as reviewable,
        // unless the user opted into treating this apply as a human verification.
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
              tagUuids: [markVerified ? VERIFIED_TAG_UUID : UNVERIFIED_TAG_UUID],
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
 * @param {number} total - total unit count for the scope being run.
 * @param {string} [noun] - what `total` counts, for the prompt text. The run
 *   dialog's scope toggle decides this: bbox scope counts bboxes, image scope
 *   (and any classifier run) counts images.
 * @returns {Promise<number|null>} the chosen limit (a positive integer), or
 *   `Infinity` to process all, or `null` if the user cancels/dismisses.
 */
export function confirmLargeRun(total, noun = 'candidate bbox(es)') {
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
      document.createTextNode(` ${noun}. Large runs can take a while.`),
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
 * Builds and shows the "Run Model on Results" dialog: scope toggle (filtered
 * bboxes vs. filtered images — see module docs), model picker (detectors
 * only), conf/IoU sliders, candidate-count display with cap warnings, and a
 * Run button that launches PreviewModelRelabelTask and switches the dialog to
 * a poll/progress/cancel view until the task completes.
 *
 * @param {{bboxUuids: string[], imageUuids: string[]}} bboxScope - the
 *   pre-resolved bbox-filtered scope (see `runModelOnResultsFlow`). Uncapped:
 *   the large-run prompt happens on Run, once the scope toggle has settled.
 *   The "Filtered images" scope is resolved via `resolveImageScope(ctx)` —
 *   lazily on first toggle, or up front when images view makes it the default.
 * @param {Object} ctx - forwarded to `openRelabelConfirmationModal` (see its docs),
 *   plus the fields `resolveCandidateScope`/`resolveImageScope` need.
 * @param {{bboxUuids: string[], imageUuids: string[], bboxImageUuids: string[]}|null} [preResolvedImageScope]
 *   the "Filtered images" scope, when the caller already had to resolve it (it
 *   does when the bbox scope came back empty). Saves a second enumeration and
 *   lets the dialog start on that scope.
 */
async function _showRunDialog(bboxScope, ctx, preResolvedImageScope = null) {
  // Active candidate lists the Run button reads from — reassigned by the scope
  // toggle below.
  let bboxUuids = bboxScope.bboxUuids;
  let imageUuids = bboxScope.imageUuids;
  // Owning image per entry in `bboxUuids`, kept in step with it so the Run
  // button's cap can drop whole images without orphaning their bboxes.
  let bboxImageUuids = bboxScope.bboxImageUuids ?? [];
  // Default to the view the user is already in: asking an image-view user
  // about bboxes (a unit they didn't choose) is the whole complaint the cap
  // prompt used to embody. Anything that isn't images view keeps bbox scope —
  // unless there are no candidate bboxes to relabel, in which case bbox scope
  // is a dead end and image scope is the only one that can produce anything.
  const noCandidateBboxes = bboxScope.bboxUuids.length === 0;
  let activeScopeKind = (ctx.viewFilter === 'images' || noCandidateBboxes) ? 'images' : 'bboxes';
  // Resolved on first switch to 'images' — lazily when bbox scope is the
  // default (avoids the extra fetch for users who never touch the toggle), or
  // immediately below when images view makes it the starting scope.
  let imageScope = preResolvedImageScope;

  const { dialog, open, close } = createModalShell();
  dialog.style.cssText = 'max-width:520px;max-height:85vh;display:flex;flex-direction:column;overflow-y:auto;';
  dialog.classList.add('run-model-on-results-dialog');

  const title = document.createElement('h3');
  title.className = 'app-modal-title';
  dialog.appendChild(title);
  title.textContent = 'Run Model on Results';

  // ── Scope toggle: filtered bboxes (today's default) vs. filtered images ──
  // (every image matching the filter, including ones with zero bboxes — see
  // `resolveImageScope`'s docs for why these can differ so much, e.g. "8
  // bboxes" vs. "219 images").
  const scopeRow = document.createElement('div');
  scopeRow.style.cssText = 'display:flex;flex-direction:column;gap:0.2em;font-size:0.9em;';
  const scopeLabel = document.createElement('div');
  scopeLabel.textContent = 'Scope:';
  scopeLabel.style.cssText = 'font-size:0.85em;color:#aaa;';
  scopeRow.appendChild(scopeLabel);

  function _makeScopeRadio(value, initialText) {
    const label = document.createElement('label');
    label.style.cssText = 'display:flex;align-items:center;gap:0.4em;cursor:pointer;';
    const radio = document.createElement('input');
    radio.type = 'radio';
    radio.name = 'run-model-on-results-scope';
    radio.value = value;
    radio.checked = value === activeScopeKind;
    const text = document.createElement('span');
    text.textContent = initialText;
    label.append(radio, text);
    scopeRow.appendChild(label);
    return { radio, text };
  }

  const bboxScopeRadio = _makeScopeRadio(
    'bboxes',
    `Filtered bboxes (${bboxScope.bboxUuids.length.toLocaleString()}) — relabel existing candidate bboxes`,
  );
  const imageScopeRadio = _makeScopeRadio(
    'images',
    'Filtered images (…) — run on every matching image, including ones with no bboxes',
  );
  // Nothing to relabel means bbox scope can only ever produce an empty run, so
  // it isn't offered — the dialog opens on image scope instead of refusing to
  // open at all, which is what used to happen.
  if (noCandidateBboxes) {
    bboxScopeRadio.radio.disabled = true;
    bboxScopeRadio.text.textContent = 'Filtered bboxes (0) — no existing bboxes to relabel';
  }
  dialog.appendChild(scopeRow);

  const countLine = document.createElement('div');
  dialog.appendChild(countLine);
  function _updateCountLine() {
    countLine.textContent = `${bboxUuids.length.toLocaleString()} candidate bbox(es) across ${imageUuids.length.toLocaleString()} image(s).`;
  }
  _updateCountLine();

  // Neither scope is capped here. The large-run prompt fires once, on Run,
  // against whichever scope is selected by then — switching scopes is free and
  // reversible, so a prompt on toggle would be asking about a set the user is
  // still browsing.
  async function _onScopeChange(kind) {
    activeScopeKind = kind;
    if (kind === 'bboxes') {
      bboxUuids = bboxScope.bboxUuids;
      imageUuids = bboxScope.imageUuids;
      bboxImageUuids = bboxScope.bboxImageUuids ?? [];
      cropModeBboxOpt.hidden = false;
      _updateCountLine();
      _syncCropModeUI();
      return;
    }

    // kind === 'images'
    if (imageScope === null) {
      imageScopeRadio.radio.disabled = true;
      imageScopeRadio.text.textContent = 'Filtered images (loading…)';
      try {
        let resolved = await resolveImageScope(ctx);
        if (resolved.imageUuids.length === 0) {
          toast('No images in the current selection/filter.', 4000, 'warning');
          imageScopeRadio.radio.checked = false;
          bboxScopeRadio.radio.checked = true;
          activeScopeKind = 'bboxes';
          imageScopeRadio.text.textContent = 'Filtered images (0)';
          return;
        }
        imageScope = resolved;
      } catch (err) {
        Logger.error('Failed to resolve image scope:', err);
        toast(`Failed to resolve filtered images: ${err.message}`, 5000, 'error');
        imageScopeRadio.radio.checked = false;
        bboxScopeRadio.radio.checked = true;
        activeScopeKind = 'bboxes';
        imageScopeRadio.text.textContent = 'Filtered images (…)';
        return;
      } finally {
        imageScopeRadio.radio.disabled = false;
      }
    }
    imageScopeRadio.text.textContent =
      `Filtered images (${imageScope.imageUuids.length.toLocaleString()}) — run on every matching image, including ones with no bboxes`;
    bboxUuids = imageScope.bboxUuids;
    imageUuids = imageScope.imageUuids;
    bboxImageUuids = imageScope.bboxImageUuids ?? [];
    // Bbox-crop mode has nothing to crop on a boxless image — the whole point
    // of this scope — so it's not offered here; whole-image + IoU match is the
    // only mode, and adding new boxes is the only way a boxless image can
    // produce anything.
    cropModeSelect.value = 'image';
    cropModeBboxOpt.hidden = true;
    if (newBoxModeSelect.value === 'none') newBoxModeSelect.value = 'also';
    _updateCountLine();
    _syncCropModeUI();
  }
  bboxScopeRadio.radio.addEventListener('change', () => { if (bboxScopeRadio.radio.checked) _onScopeChange('bboxes'); });
  imageScopeRadio.radio.addEventListener('change', () => { if (imageScopeRadio.radio.checked) _onScopeChange('images'); });

  // ── Model + inference-input pickers, side by side to keep this compact ──
  const pickerRow = document.createElement('div');
  pickerRow.style.cssText = 'display:flex;gap:1em;';
  dialog.appendChild(pickerRow);

  const modelLabelRow = document.createElement('div');
  modelLabelRow.style.cssText = 'display:flex;align-items:center;justify-content:space-between;gap:0.5em;';
  const modelLabel = document.createElement('div');
  modelLabel.textContent = 'Model(s):';
  modelLabel.style.cssText = 'font-size:0.85em;color:#aaa;';
  const modelRefreshBtn = document.createElement('button');
  modelRefreshBtn.type = 'button';
  modelRefreshBtn.className = 'secondary run-model-on-results-refresh';
  modelRefreshBtn.textContent = 'Refresh';
  modelRefreshBtn.title = 'Reload the model/group list (picks up newly imported models or saved groups)';
  modelRefreshBtn.style.cssText = 'padding:0.1em 0.6em;font-size:0.8em;';
  modelLabelRow.append(modelLabel, modelRefreshBtn);

  const modelRow = document.createElement('div');
  modelRow.style.cssText = 'display:flex;flex-direction:column;gap:0.3em;flex:1;';
  const modelListEl = document.createElement('div');
  modelListEl.className = 'inference-model-list run-model-on-results-model-list';
  modelListEl.style.cssText = 'max-height:180px;overflow-y:auto;border:1px solid #333;border-radius:4px;padding:0.3em;';
  // ── Multi-model conflict policy ──────────────────────────────────────
  // Two models running over the same scope can each propose on the same bbox.
  // Until now the last one to run silently won (a bare `.set()` in the merge
  // loop below) with no way to see or change it. Model order is the check
  // order: `selectedItems` is a Set, which preserves insertion order, and
  // `modelNames` is iterated in that order.
  const conflictRow = document.createElement('div');
  conflictRow.className = 'run-model-on-results-conflict-row';
  conflictRow.style.cssText = 'display:flex;align-items:center;gap:0.4em;font-size:0.85em;';
  const conflictLabel = document.createElement('span');
  conflictLabel.textContent = 'When models disagree on a box:';
  conflictLabel.style.cssText = 'color:#aaa;white-space:nowrap;';
  const conflictSelect = document.createElement('select');
  conflictSelect.className = 'run-model-on-results-conflict-policy';
  conflictSelect.style.cssText = 'flex:1;padding:0.1em 0.3em;font-size:0.9em;';
  for (const [value, text] of [['last', 'Last model wins'], ['first', 'First model wins']]) {
    const opt = document.createElement('option');
    opt.value = value;
    opt.textContent = text;
    conflictSelect.appendChild(opt);
  }
  conflictSelect.setAttribute('aria-label', 'When models disagree on a box');
  conflictRow.append(conflictLabel, conflictSelect);

  /** Only meaningful with more than one detector in play. */
  function _syncConflictRow() {
    const multi = _resolveSelectedModelNames().length > 1;
    conflictRow.style.display = multi ? 'flex' : 'none';
  }

  modelRow.append(modelLabelRow, modelListEl, conflictRow);
  pickerRow.appendChild(modelRow);

  let models = [];
  /** @type {Map<string, string>} model name -> modelKind */
  let modelKinds = new Map();
  /** @type {Map<string, Object>} model group uuid -> group */
  let groupsMap = new Map();
  /** Multi-select detector/group ids ("model:<name>" / "group:<uuid>"); Set preserves check order. */
  let selectedItems = new Set();
  /** Single-select classifier name, mutually exclusive with `selectedItems` (same rule as inferencePanel.js). */
  let selectedClassifier = null;
  // Declared here (assigned below near the other dialog buttons) so
  // `_renderModelList` can toggle it whether it runs at initial load or
  // from a later refresh click, without a temporal-dead-zone reference error.
  let runBtn = null;

  async function _loadModels() {
    models = [];
    modelKinds = new Map();
    try {
      const allModels = await fetchModelsList();
      modelKinds = await fetchModelsKinds(allModels);
      models = allModels;
      groupsMap = await fetchModelGroups();
    } catch (err) {
      Logger.error('Failed to load models:', err);
    }
  }

  function _resolveSelectedModelNames() {
    const names = [];
    for (const id of selectedItems) {
      if (id.startsWith(MODEL_ID_PREFIX)) names.push(id.slice(MODEL_ID_PREFIX.length));
      else if (id.startsWith(GROUP_ID_PREFIX)) names.push(...expandModelGroupMembers(id.slice(GROUP_ID_PREFIX.length), groupsMap));
    }
    return names;
  }

  function _makeListItem({ id, label, badgeText, isSelected, onToggle }) {
    const item = document.createElement('label');
    item.className = 'inference-model-item' + (isSelected ? ' selected' : '');
    item.title = label;

    const cb = document.createElement('input');
    cb.type = 'checkbox';
    cb.checked = isSelected;
    cb.addEventListener('change', () => onToggle(cb.checked));

    const nameSpan = document.createElement('span');
    nameSpan.className = 'inference-model-name';
    nameSpan.textContent = label;

    const badge = document.createElement('span');
    badge.className = 'inference-model-badge';
    badge.textContent = badgeText;

    item.append(cb, nameSpan, badge);
    return item;
  }

  function _makeSectionEl(title, items, makeItem) {
    if (items.length === 0) return null;
    const section = document.createElement('div');
    section.className = 'inference-model-section';
    const header = document.createElement('div');
    header.className = 'inference-model-section-header';
    header.textContent = title;
    section.appendChild(header);
    for (const it of items) section.appendChild(makeItem(it));
    return section;
  }

  function _renderModelList() {
    modelListEl.innerHTML = '';
    if (models.length === 0 && groupsMap.size === 0) {
      const empty = document.createElement('div');
      empty.className = 'inference-model-empty';
      empty.textContent = '(no models found)';
      modelListEl.appendChild(empty);
      if (runBtn) runBtn.disabled = true;
      return;
    }

    const detectorNames = models.filter(m => !_CLASSIFIER_KINDS.has(modelKinds.get(m)));
    const classifierNames = models.filter(m => _CLASSIFIER_KINDS.has(modelKinds.get(m)));
    const groups = [...groupsMap.values()].sort((a, b) => a.name.localeCompare(b.name));

    const detSection = _makeSectionEl('Detectors', detectorNames, m => _makeListItem({
      id: MODEL_ID_PREFIX + m,
      label: m,
      badgeText: 'yolo',
      isSelected: selectedItems.has(MODEL_ID_PREFIX + m),
      onToggle: (checked) => {
        if (checked) selectedItems.add(MODEL_ID_PREFIX + m);
        else selectedItems.delete(MODEL_ID_PREFIX + m);
        selectedClassifier = null;
        _renderModelList();
        _syncCropModeUI();
      },
    }));
    const groupSection = _makeSectionEl('Groups', groups, g => _makeListItem({
      id: GROUP_ID_PREFIX + g.uuid,
      label: g.name,
      badgeText: 'group',
      isSelected: selectedItems.has(GROUP_ID_PREFIX + g.uuid),
      onToggle: (checked) => {
        if (checked) selectedItems.add(GROUP_ID_PREFIX + g.uuid);
        else selectedItems.delete(GROUP_ID_PREFIX + g.uuid);
        selectedClassifier = null;
        _renderModelList();
        _syncCropModeUI();
      },
    }));
    const clfSection = _makeSectionEl('Classifiers', classifierNames, m => _makeListItem({
      id: 'classifier:' + m,
      label: m,
      badgeText: modelKinds.get(m) === 'image_classifier' ? 'binary' : 'multi',
      isSelected: selectedClassifier === m,
      onToggle: (checked) => {
        selectedClassifier = checked ? m : null;
        selectedItems.clear();
        _renderModelList();
        _syncCropModeUI();
      },
    }));

    if (detSection) modelListEl.appendChild(detSection);
    if (groupSection) modelListEl.appendChild(groupSection);
    if (clfSection) modelListEl.appendChild(clfSection);

    if (runBtn) runBtn.disabled = models.length === 0;
  }

  modelRefreshBtn.addEventListener('click', async () => {
    modelRefreshBtn.disabled = true;
    try {
      clearModelsCache();
      clearModelGroupsCache();
      await _loadModels();
      _renderModelList();
      _syncCropModeUI();
    } finally {
      modelRefreshBtn.disabled = false;
    }
  });

  await _loadModels();
  _renderModelList();

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

  // ── New-box mode (whole-image mode only) ─────────────────────────────
  // Three-way rather than the old "also add new boxes" checkbox: relabelling
  // and adding are independent decisions, and "only add" is the one a user
  // wants when the existing boxes are already right (or there are none).
  const NEW_BOX_MODES = [
    ['none', 'Relabel existing boxes only'],
    ['also', 'Also add new boxes (detections overlapping nothing)'],
    ['only', 'Only add new boxes (leave existing boxes untouched)'],
  ];
  const addNewRow = document.createElement('div');
  addNewRow.style.cssText = 'display:flex;align-items:center;gap:0.4em;font-size:0.9em;';
  const addNewLabel = document.createElement('span');
  addNewLabel.textContent = 'New boxes:';
  addNewLabel.style.cssText = 'color:#aaa;white-space:nowrap;';
  const newBoxModeSelect = document.createElement('select');
  newBoxModeSelect.className = 'run-model-on-results-new-box-mode';
  newBoxModeSelect.setAttribute('aria-label', 'New boxes');
  newBoxModeSelect.style.cssText = 'flex:1;padding:0.1em 0.3em;font-size:0.9em;';
  for (const [value, text] of NEW_BOX_MODES) {
    const opt = document.createElement('option');
    opt.value = value;
    opt.textContent = text;
    newBoxModeSelect.appendChild(opt);
  }
  addNewRow.append(addNewLabel, newBoxModeSelect);
  dialog.appendChild(addNewRow);

  const _addNew = () => newBoxModeSelect.value !== 'none';
  const _relabelExisting = () => newBoxModeSelect.value !== 'only';

  function _syncCropModeUI() {
    cropMode = cropModeSelect.value;
    const isClassifier = selectedClassifier !== null;
    // Crop mode, IoU matching, and "add new boxes" are all detector-only
    // concepts (whole-image classification has no geometry to match/refine).
    cropModeRow.style.display = isClassifier ? 'none' : '';
    iouRow.style.display = (isClassifier || cropMode === 'bbox') ? 'none' : '';
    // Adding new boxes needs whole-image detections to reason about overlap, so
    // the mode picker is only meaningful in image crop_mode.
    addNewRow.style.display = (isClassifier || cropMode === 'bbox') ? 'none' : 'flex';
    if (isClassifier || cropMode === 'bbox') newBoxModeSelect.value = 'none';
    _syncConflictRow();
  }
  cropModeSelect.addEventListener('change', _syncCropModeUI);
  newBoxModeSelect.addEventListener('change', _syncCropModeUI);
  _syncCropModeUI();

  // Images view (and an empty bbox scope) starts on the image scope, so apply
  // it now rather than waiting for a toggle the user has no reason to touch.
  // Runs down here, not at the scope row: `_onScopeChange` touches the crop
  // mode and new-box controls, which are only constructed above this line. Not
  // awaited — the rest of the dialog should render immediately, and
  // `_onScopeChange` owns its own loading label and error fallback.
  if (activeScopeKind === 'images') _onScopeChange('images');

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

  function _setModelListDisabled(disabled) {
    modelListEl.querySelectorAll('input[type=checkbox]').forEach(cb => { cb.disabled = disabled; });
  }

  function _resetRunUI() {
    runBtn.disabled = false;
    cancelBtn.disabled = false;
    _setModelListDisabled(false);
    cropModeSelect.disabled = false;
    bboxScopeRadio.radio.disabled = noCandidateBboxes;
    imageScopeRadio.radio.disabled = false;
    progressWrap.style.display = 'none';
  }

  runBtn.addEventListener('click', async () => {
    const isClassifier = selectedClassifier !== null;
    const modelNames = isClassifier ? [selectedClassifier] : _resolveSelectedModelNames();

    if (modelNames.length === 0) {
      toast(isClassifier ? 'Please select a classifier' : 'Please select at least one model or group', 3000, 'warning');
      return;
    }

    if (imageUuids.length === 0) {
      toast('No images in the current selection/filter.', 4000, 'warning');
      return;
    }
    if (!isClassifier && bboxUuids.length === 0) {
      // Only a problem if there's also no way to produce a candidate without
      // an existing bbox — the image scope's add_new path is exactly that.
      const canFindNew = activeScopeKind === 'images' && cropModeSelect.value === 'image' && _addNew();
      if (!canFindNew) {
        toast(
          'No candidate bboxes in the current selection/filter. Switch to "Filtered images" scope and set '
            + 'New boxes to "Also add" or "Only add" to find objects on images with no bboxes.',
          5000,
          'warning',
        );
        return;
      }
    }

    // The one large-run prompt, raised here rather than before the dialog
    // opens: only now is the unit settled. Image scope (and any classifier
    // run, which is keyed off images regardless of scope) counts images; bbox
    // scope counts candidate bboxes.
    const capByImage = isClassifier || activeScopeKind === 'images';
    const capTotal = capByImage ? imageUuids.length : bboxUuids.length;
    let runBboxUuids = bboxUuids;
    let runImageUuids = imageUuids;
    if (capTotal > WARN_THRESHOLD) {
      const limit = await confirmLargeRun(capTotal, capByImage ? 'image(s)' : 'candidate bbox(es)');
      if (limit === null) return; // user cancelled/dismissed the warning
      if (Number.isFinite(limit)) {
        if (capByImage) {
          // Drop whole images, then the bboxes belonging to them, so a capped
          // run never sends a bbox whose image isn't in the run.
          const kept = new Set(imageUuids.slice(0, limit));
          runImageUuids = [...kept];
          runBboxUuids = bboxUuids.filter((_, i) => kept.has(bboxImageUuids[i]));
        } else {
          const capped = truncateScope({ bboxUuids, imageUuids, bboxImageUuids }, limit);
          runBboxUuids = capped.bboxUuids;
          runImageUuids = capped.imageUuids;
        }
      }
    }

    runBtn.disabled = true;
    cancelBtn.disabled = true;
    _setModelListDisabled(true);
    cropModeSelect.disabled = true;
    bboxScopeRadio.radio.disabled = true;
    imageScopeRadio.radio.disabled = true;
    progressWrap.style.display = 'flex';
    const totalImages = runImageUuids.length;

    // Sequential run: each model in `modelNames` (raw picks + expanded group
    // members, in selection order) launches its own PreviewModelRelabelTask
    // against the same candidate scope. Results merge into a single final
    // preview instead of one modal per model — which model wins a contested
    // bbox is the user's choice (`conflictPolicy`), defaulting to the last in
    // the sequence, which is what this always did before the control existed.
    // "New box" candidates (bboxUuid === null, from add_new) have no stable
    // identity to merge on, so they're appended from every model's run rather
    // than deduped.
    const conflictPolicy = conflictSelect.value;
    const mergedByBbox = new Map();
    const newBoxCandidates = [];
    let aggImagesFailed = 0;
    let aggNoDetectionCount = 0;
    let aggNoOpCount = 0;
    let aggNoMatchCount = 0;
    const aggSampleErrors = [];

    // Detector runs are persisted: the run row is created up front and every
    // model in the sequence writes its candidates straight into it, so the
    // result survives this modal being closed and can be triaged over several
    // sittings. Classifier runs still use the in-memory confirmation modal —
    // their "apply" is a free-form bulk *image* action chosen afterwards, which
    // has no representation in a candidate's persisted decision.
    // The two halves of the new-box mode. `relabel_existing` is only allowed to
    // be false when new boxes are actually being collected (the task rejects
    // the combination that could propose nothing at all).
    const runAddNew = cropMode === 'image' && _addNew();
    const runRelabelExisting = runAddNew ? _relabelExisting() : true;

    let previewRun = null;
    if (!isClassifier) {
      try {
        previewRun = await createPreviewRun({
          modelName: modelNames.join(' → '),
          taskTypename: 'PreviewModelRelabelTask',
          params: {
            crop_mode: cropMode,
            add_new: runAddNew,
            relabel_existing: runRelabelExisting,
            conf_thresh: confThresh,
            iou_thresh: iouThresh,
            conflict_policy: conflictPolicy,
            model_names: modelNames,
          },
          scope: {
            kind: activeScopeKind,
            bbox_uuid_count: runBboxUuids.length,
            image_uuid_count: runImageUuids.length,
          },
        });
      } catch (err) {
        // Not fatal: fall back to the in-memory preview rather than refusing to
        // run a detector the user already waited on a dialog for.
        Logger.error('Failed to create preview run; falling back to an unsaved preview:', err);
        toast('Could not save this preview run; results will not persist.', 5000, 'warning');
      }
    }

    try {
      for (let i = 0; i < modelNames.length; i++) {
        if (cancelled) break;
        const modelName = modelNames[i];
        const stepPrefix = modelNames.length > 1 ? `Model ${i + 1}/${modelNames.length} (${modelName}): ` : '';
        progressBar.value = 0;
        progressLabel.textContent = `${stepPrefix}Starting… (0/${totalImages} images)`;

        const params = isClassifier
          ? {
              model_name: modelName,
              conf_thresh: confThresh,
              image_uuids: runImageUuids,
            }
          : {
              model_name: modelName,
              crop_mode: cropMode,
              add_new: runAddNew,
              relabel_existing: runRelabelExisting,
              conf_thresh: confThresh,
              iou_thresh: iouThresh,
              image_uuids: runImageUuids,
              bbox_uuids: runBboxUuids,
              // All models in the sequence share one run, so their proposals
              // merge server-side under the chosen conflict policy.
              ...(previewRun ? { preview_run_uuid: previewRun.uuid, conflict_policy: conflictPolicy } : {}),
            };

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
          progressLabel.textContent = `${stepPrefix}Processing… (${processed}/${totalImages} images)`;
        });

        if (cancelled) break;

        const status = finalInfo?.instance?.status;
        if (status !== TaskStatus.COMPLETED) {
          const msg = finalInfo?.instance?.errorMessage || `Task ended with status: ${status}`;
          toast(`Run failed on ${modelName}: ${msg}`, 5000, 'error');
          await deleteTasks({ taskUuids: [taskUuid] }).catch(() => {});
          if (previewRun) await deletePreviewRun(previewRun.uuid).catch(() => {});
          _resetRunUI();
          return;
        }

        progressLabel.textContent = `${stepPrefix}Fetching results…`;
        const resultsMap = await fetchTasksResult({ taskUuids: [taskUuid], cacheResults: false });
        const result = resultsMap.get(taskUuid);
        const data = result?.data ?? {};
        const candidates = data.candidates ?? [];
        // TaskResult.data is a raw passthrough dict (no camelCase alias conversion,
        // unlike most other API responses) — read the task's snake_case keys directly.

        await deleteTasks({ taskUuids: [taskUuid] }).catch((e) => Logger.error('Cleanup failed', e));
        taskUuid = null;

        for (const c of candidates) {
          if (isClassifier || c.isNew || !c.bboxUuid) {
            // Two models proposing new boxes are two detections, not a
            // conflict — there is no shared bbox to disagree about.
            newBoxCandidates.push(c);
          } else if (conflictPolicy === 'first') {
            if (!mergedByBbox.has(c.bboxUuid)) mergedByBbox.set(c.bboxUuid, c);
          } else {
            mergedByBbox.set(c.bboxUuid, c);
          }
        }
        aggImagesFailed += data.images_failed ?? 0;
        aggNoDetectionCount += data.no_detection_count ?? 0;
        aggNoOpCount += data.no_op_count ?? 0;
        aggNoMatchCount += data.no_match_count ?? 0;
        if (data.sample_errors?.length) aggSampleErrors.push(...data.sample_errors);
      }

      if (cancelled) return;

      close();
      const displayName = modelNames.join(' → ');
      const finalCandidates = isClassifier ? newBoxCandidates : [...mergedByBbox.values(), ...newBoxCandidates];

      // Persisted run: the candidates are already in the DB (the task wrote them
      // as it went and returned only the uuid), so hand off to the triage panel
      // rather than the in-memory confirmation modal.
      if (previewRun) {
        try {
          await updatePreviewRun(previewRun.uuid, {
            stats: {
              images_failed: aggImagesFailed,
              no_detection_count: aggNoDetectionCount,
              no_op_count: aggNoOpCount,
              sample_errors: aggSampleErrors.slice(0, 5),
            },
          });
        } catch (err) {
          Logger.error('Failed to record preview run stats:', err);
        }
        const { openPreviewRunModal } = await import('/app-static/js/app/review/previewPanel.js');
        await openPreviewRunModal({
          runUuid: previewRun.uuid,
          labels: ctx.labels,
          onApplied: ctx.refresh,
        });
        return;
      }

      if (isClassifier) {
        await openClassifierTagConfirmationModal(finalCandidates, {
          modelName: displayName,
          refresh: ctx.refresh,
          labels: ctx.labels,
          stats: {
            imagesFailed: aggImagesFailed,
            noMatchCount: aggNoMatchCount,
            sampleErrors: aggSampleErrors.slice(0, 5),
          },
        });
      } else {
        await openRelabelConfirmationModal(finalCandidates, {
          modelName: displayName,
          refresh: ctx.refresh,
          labels: ctx.labels,
          stats: {
            imagesFailed: aggImagesFailed,
            noDetectionCount: aggNoDetectionCount,
            noOpCount: aggNoOpCount,
            sampleErrors: aggSampleErrors.slice(0, 5),
          },
        });
      }
    } catch (err) {
      Logger.error('Run Model on Results failed:', err);
      toast(`Run failed: ${err.message}`, 5000, 'error');
      if (previewRun) await deletePreviewRun(previewRun.uuid).catch(() => {});
      _resetRunUI();
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

  // An empty bbox scope is NOT an empty run: the image scope can still find
  // objects on images that have no bboxes at all, which is exactly what a user
  // whose filter matched only boxless images is asking for. Resolve it here so
  // the dialog can open on it, and only refuse when there is no image either.
  let imageScope = null;
  if (scope.bboxUuids.length === 0) {
    try {
      imageScope = await resolveImageScope(ctx);
    } catch (err) {
      Logger.error('Failed to resolve image scope:', err);
      toast(`Failed to resolve filtered images: ${err.message}`, 5000, 'error');
      return;
    }
    if (imageScope.imageUuids.length === 0) {
      toast('No candidates in the current selection/filter.', 4000, 'warning');
      return;
    }
  }

  // No cap prompt here: which scope is actually being run (bboxes vs images)
  // isn't decided until the run dialog's scope toggle, so asking "how many of
  // these N bboxes?" up front asks about a unit the user may never process.
  // The dialog raises it on Run, against whichever scope is selected.
  await _showRunDialog(scope, ctx, imageScope);
}
