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
import { showImageBulkActionModal, executeBulkImageAction } from '/app-static/js/app/review/imageBulkActionModal.js';
import { createModalShell } from '/app-static/js/ui/utils/modalShell.js';
import { createHoverPreview } from '/app-static/js/ui/thumbnailHoverPreview.js';
import { showConfirmModal } from '/app-static/js/ui/utils/confirmModal.js';
import { deleteImage } from '/app-static/js/shared/api/images.js';
import { toast, Logger } from '/app-static/js/ui/utils/logging.js';

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
  dialog.style.cssText = 'max-width:900px;width:90vw;max-height:85vh;display:flex;flex-direction:column;';
  dialog.classList.add('classifier-match-confirm-modal');

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

  /** @type {Array<{candidate: object, checkbox: HTMLInputElement}>} */
  const rows = [];

  for (const candidate of sorted) {
    const card = document.createElement('div');
    card.className = 'review-card relabel-modal-card';
    card.dataset.imageUuid = candidate.imageUuid;

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
    labelsRow.appendChild(_labelChip(candidate.predictedLabelName, candidate.predictedLabelUuid, ctx.labels));
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
    rows.push({ candidate, checkbox });
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

    await showImageBulkActionModal(entries, ctx.labels, async (action, labelUuid, tagUuids) => {
      try {
        if (action === 'delete') {
          const n = entries.length;
          const noun = n === 1 ? 'image' : 'images';
          const confirmed = await showConfirmModal({
            title: `Delete ${n} ${noun}`,
            message: `Permanently delete ${n} selected ${noun}? This removes the files and all their annotations from disk. This cannot be undone.`,
            confirmText: 'Delete',
            danger: true,
          });
          if (!confirmed) return;

          let failCount = 0;
          for (const { img } of entries) {
            try {
              await deleteImage(img.uuid);
            } catch (err) {
              Logger.error(`Delete image failed for ${img.uuid}: ${err.message}`);
              failCount++;
            }
          }
          const succeeded = entries.length - failCount;
          toast(
            failCount > 0 ? `Deleted ${succeeded}/${entries.length} image(s), ${failCount} failed` : `Deleted ${succeeded} image(s)`,
            4000,
            failCount > 0 ? 'warning' : 'success',
          );
        } else {
          const { failCount, succeededEntries } = await executeBulkImageAction(action, entries, labelUuid, tagUuids);
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
