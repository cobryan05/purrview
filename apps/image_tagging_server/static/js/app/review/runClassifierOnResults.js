/**
 * "Run Model on Results" — classifier confirmation modal.
 *
 * Sibling of runModelOnResults.js's `openRelabelConfirmationModal`, for the
 * classifier branch: shows a card grid of image-level match candidates
 * (predicted label + confidence, no bbox geometry), lets the user check/uncheck
 * candidates, and applies via the same generic `showImageBulkActionModal` /
 * `executeBulkImageAction` the review toolbar's own "Bulk Apply"/"Apply to All"
 * buttons use — so a classifier match can be tagged, label-marked, or even
 * deleted, with full parity with the rest of the app (not just "add a tag").
 */

import { getImageThumbnailUrl } from '/app-static/js/shared/api/review.js';
import { labelChip as _labelChip } from '/app-static/js/app/review/labelChip.js';
import { showImageBulkActionModal, executeBulkImageAction, isRemovalAction, confirmRemovalAction, removeSingleImage, removalVerb } from '/app-static/js/app/review/imageBulkActionModal.js';
import { createModalShell } from '/app-static/js/ui/utils/modalShell.js';
import { createHoverPreview } from '/app-static/js/ui/thumbnailHoverPreview.js';
import { toast, Logger } from '/app-static/js/ui/utils/logging.js';
import { createCardGrid } from '/app-static/js/app/review/cardGrid.js';
import { attachContext } from '/app-static/js/ui/contextMenu/index.js';

/**
 * Per-card adapter for an image-level classifier preview candidate — the object
 * the `candidateImage` context-menu provider drives off. Mirrors
 * runModelOnResults.js's `makeCandidateBboxAdapter`, but IMAGE-LEVEL: a
 * classifier proposes a whole-image label, not a box, so there is NO geometry,
 * verification, tags, or bbox uuid. It mutates the in-memory candidate / preview
 * state ONLY; nothing hits the DB until the user hits Apply.
 *
 * @param {object} candidate - the transient preview candidate object.
 * @param {HTMLElement} cardEl - the candidate's card element.
 * @param {{
 *   labels: Map<string, object>,
 *   removeCandidate: (candidate: object) => void,
 *   rerenderProposedChip: (candidate: object) => void,
 * }} previewController
 */
export function makeCandidateImageAdapter(candidate, cardEl, previewController) {
  const { labels, removeCandidate, rerenderProposedChip } = previewController;
  return {
    // ── Display getters ────────────────────────────────────────────────────
    getImageUuid() { return candidate.imageUuid ?? null; },
    getLabelUuid() { return candidate.predictedLabelUuid ?? null; },
    getLabelName() { return candidate.predictedLabelName ?? null; },
    getLabelColor() { return labels.get(candidate.predictedLabelUuid)?.color ?? null; },
    getConfidence() { return candidate.confidence ?? null; },
    // No box exists — return null so any box-only menu item (similar-search,
    // edit-geometry) self-disables. The `candidateImage` provider omits those
    // entirely, but this keeps the adapter safe if ever pointed at a box menu.
    getBboxUuid() { return null; },
    getBboxCoords() { return null; },

    // ── Relabel (no DB) ────────────────────────────────────────────────────
    // Sets the proposed image label on the candidate and re-renders its chip in
    // place. The name comes from the labels Map (fetchLabels() returns a Map).
    async applyLabel(labelUuid) {
      candidate.predictedLabelUuid = labelUuid;
      candidate.predictedLabelName = labelUuid ? (labels.get(labelUuid)?.name ?? null) : null;
      rerenderProposedChip(candidate);
    },

    // ── Remove from preview (no DB) ────────────────────────────────────────
    // Drops the candidate so Apply won't include it (unchecks + removes the row).
    async remove() {
      removeCandidate(candidate);
    },
  };
}

/**
 * Confirmation modal for "Run Model on Results" (classifier branch). Lets the
 * user review the classifier's match candidates, check/uncheck individual
 * ones, and apply any image-level bulk action to the checked set via the
 * shared `showImageBulkActionModal`.
 *
 * @param {Array<{imageUuid: string, imagePath: string, predictedLabelUuid: string,
 *   predictedLabelName: ?string, confidence: number}>} candidates
 * @param {Object} ctx
 * @param {string} ctx.modelName - classifier used to generate the candidates.
 * @param {() => void} ctx.refresh - reload the current review-panel page (e.g. `render()`).
 * @param {Map<string, object>} ctx.labels - label lookup, for display and the bulk-action modal.
 * @param {{imagesFailed: number, noMatchCount: number, sampleErrors: string[]}} [ctx.stats]
 *   - diagnostic breakdown from the task result, shown when there are zero candidates.
 */
export async function openClassifierTagConfirmationModal(candidates, ctx) {
  const hoverPreview = createHoverPreview();
  const { dialog, open, close } = createModalShell({ onClose: () => hoverPreview.destroy() });
  dialog.classList.add('app-modal-dialog--lg', 'classifier-match-confirm-modal');

  const sorted = [...candidates].sort((a, b) => (b.confidence ?? 0) - (a.confidence ?? 0));

  const title = document.createElement('h3');
  title.className = 'app-modal-title';
  title.style.flexShrink = '0';
  title.textContent = sorted.length === 0
    ? `No matches from ${ctx.modelName}`
    : `${sorted.length} match${sorted.length === 1 ? '' : 'es'} from ${ctx.modelName}`;
  dialog.appendChild(title);

  if (sorted.length === 0) {
    const empty = document.createElement('div');
    empty.className = 'relabel-modal-empty';
    const stats = ctx.stats;
    if (stats) {
      const parts = [];
      if (stats.noMatchCount > 0) parts.push(`${stats.noMatchCount} image(s) below threshold`);
      if (stats.imagesFailed > 0) parts.push(`${stats.imagesFailed} image(s) failed to load/process`);
      empty.textContent = parts.length > 0
        ? `No matches: ${parts.join('; ')}.`
        : 'No matches above threshold.';
      if (stats.imagesFailed > 0 && stats.sampleErrors?.length) {
        const errDetail = document.createElement('div');
        errDetail.style.cssText = 'font-size:0.8em;color:#e88;margin-top:0.5em;';
        errDetail.textContent = `Sample error: ${stats.sampleErrors[0]}`;
        empty.appendChild(document.createElement('br'));
        empty.appendChild(errDetail);
      }
    } else {
      empty.textContent = 'No matches above threshold.';
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

  /** @type {Array<{id: string, candidate: object, card: HTMLElement, checkbox: HTMLInputElement, proposedChip: HTMLElement}>} */
  const rows = [];

  // ── Shared selection controller (click / shift-range select) ────────────
  // Image-level candidate cards get real review-card selection semantics via the
  // shared cardGrid controller — distinct from the per-card "will Apply" checkbox
  // (which stays the Apply bit). `_visibleEntries` mirrors `rows` in order so
  // shift-range select iterates the same list. Mirrors runModelOnResults.js.
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
    const fresh = _labelChip(candidate.predictedLabelName, candidate.predictedLabelUuid, ctx.labels);
    fresh.classList.add('relabel-modal-proposed-chip');
    row.proposedChip.replaceWith(fresh);
    row.proposedChip = fresh;
  }

  // Drop a candidate from the preview so Apply won't include it. Uncheck (so the
  // "will Apply" bit is off even if the row lingered) and remove the row/card.
  function _removeCandidate(candidate) {
    const idx = rows.findIndex(r => r.candidate === candidate);
    if (idx === -1) return;
    const { card, checkbox, id } = rows[idx];
    checkbox.checked = false;
    cardGrid.selected.delete(id);
    rows.splice(idx, 1);
    const veIdx = _visibleEntries.findIndex(e => e.candidate === candidate);
    if (veIdx !== -1) _visibleEntries.splice(veIdx, 1);
    card.remove();
    _updateApplyBtn();
  }

  const _previewController = {
    labels: ctx.labels,
    removeCandidate: _removeCandidate,
    rerenderProposedChip: _rerenderProposedChip,
  };

  let _cardIndex = 0;
  for (const candidate of sorted) {
    const card = document.createElement('div');
    card.className = 'review-card relabel-modal-card';
    // Stable id for the selection controller. Classifier candidates are
    // image-level (one per image), so the image uuid is a natural card id; the
    // card carries data-kind/data-bbox-uuid so the shared applySelect shift-range
    // selectors resolve (the controller keys bbox-kind cards off data-bbox-uuid).
    const cardId = candidate.imageUuid;
    card.dataset.imageUuid = candidate.imageUuid;
    card.dataset.bboxUuid = cardId;
    card.dataset.kind = 'bbox';

    // Whole-image candidate — no bbox to highlight, so just show the image
    // (mirrors runModelOnResults.js's `isNew`-candidate hover preview, minus
    // the geometry overlay since a classifier proposes no new box).
    hoverPreview.attach(card, {
      imagePath: candidate.imagePath,
      imageUuid: candidate.imageUuid,
      allBboxes: [],
    });

    const checkbox = document.createElement('input');
    checkbox.type = 'checkbox';
    checkbox.checked = true;
    checkbox.addEventListener('change', _updateApplyBtn);
    card.appendChild(checkbox);

    const thumb = document.createElement('img');
    thumb.className = 'thumb';
    thumb.loading = 'lazy';
    thumb.src = getImageThumbnailUrl(candidate.imagePath, 300);
    card.appendChild(thumb);

    const info = document.createElement('div');
    info.className = 'card-info';

    const labelsRow = document.createElement('div');
    labelsRow.className = 'relabel-modal-labels-row';
    labelsRow.style.cssText = 'display:flex;align-items:center;gap:0.4em;flex-wrap:wrap;';
    const proposedChip = _labelChip(candidate.predictedLabelName, candidate.predictedLabelUuid, ctx.labels);
    proposedChip.classList.add('relabel-modal-proposed-chip');
    labelsRow.appendChild(proposedChip);
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

    // ── Right-click context menu (Relabel / Remove-from-preview only) ──────
    // Image-level candidates: NO geometry / similar-search / verification / tags.
    attachContext(card, () => {
      const _selected = cardGrid.selected;
      // Multi-select: act on every selected candidate.
      const targets = (_selected.has(cardId) && _selected.size > 1)
        ? _visibleEntries.filter(en => _selected.has(en.id))
        : [entry];
      return targets.map(en => ({
        adapters: { candidateImage: makeCandidateImageAdapter(en.candidate, en.card, _previewController) },
      }));
    });

    rows.push({ id: cardId, candidate, card, checkbox, proposedChip });
    _cardIndex += 1;
  }

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
    applyBtn.textContent = `Apply to ${n} checked…`;
    applyBtn.disabled = n === 0;
  }
  _updateApplyBtn();

  applyBtn.addEventListener('click', async () => {
    const checkedRows = rows.filter(r => r.checkbox.checked);
    if (checkedRows.length === 0) return;

    // Shape entries the way showImageBulkActionModal/executeBulkImageAction
    // expect — mirrors review/main.js's `_visibleEntries` image entries.
    const entries = checkedRows.map(r => ({
      img: { uuid: r.candidate.imageUuid, filename: r.candidate.imagePath, tags: [] },
    }));

    await showImageBulkActionModal(entries, ctx.labels, async (action, labelUuids, tagUuids) => {
      try {
        if (isRemovalAction(action)) {
          const confirmed = await confirmRemovalAction(action, entries.length);
          if (!confirmed) return;

          let failCount = 0;
          for (const { img } of entries) {
            try {
              await removeSingleImage(action, img.uuid);
            } catch (err) {
              Logger.error(`${action} image failed for ${img.uuid}: ${err.message}`);
              failCount++;
            }
          }
          const succeeded = entries.length - failCount;
          const verb = removalVerb(action);
          toast(
            failCount > 0 ? `${verb} ${succeeded}/${entries.length} image(s), ${failCount} failed` : `${verb} ${succeeded} image(s)`,
            4000,
            failCount > 0 ? 'warning' : 'success',
          );
        } else {
          const { failCount, succeededEntries } = await executeBulkImageAction(action, entries, labelUuids, tagUuids);
          const summary = failCount > 0
            ? `Applied to ${succeededEntries.length}/${entries.length} image(s), ${failCount} failed`
            : `Applied to ${succeededEntries.length} image(s)`;
          toast(summary, 4000, failCount > 0 ? 'warning' : 'success');
        }
        close();
        ctx.refresh?.();
      } catch (err) {
        Logger.error('Failed to apply classifier match action:', err);
        toast(`Apply failed: ${err.message}`, 5000, 'error');
      }
    });
  });

  open();
}
