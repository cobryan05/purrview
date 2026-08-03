import {
  UNVERIFIED_TAG_UUID,
  VERIFIED_TAG_UUID,
  FALSE_POS_TAG_UUID,
  getBboxThumbnailUrl,
  getImageThumbnailUrl,
} from '/app-static/js/shared/api/review.js';
import { attachContext } from '/app-static/js/ui/contextMenu/index.js';
import { drawLetterboxedBboxes } from '/app-static/js/ui/utils/bboxThumbOverlay.js';

// Verification cycle advanced by double-clicking a card's verify chip:
// Unverified → Verified → False-Pos → Unverified. Keyed by the chip's current
// verification tag so the next state derives purely from what's rendered.
const VERIFY_CYCLE_NEXT = {
  [UNVERIFIED_TAG_UUID]: VERIFIED_TAG_UUID,
  [VERIFIED_TAG_UUID]: FALSE_POS_TAG_UUID,
  [FALSE_POS_TAG_UUID]: UNVERIFIED_TAG_UUID,
};

/**
 * Build and return a verify-status chip span for a bbox's tag UUIDs.
 *
 * When `onToggle` is supplied, a `dblclick` handler advances the verification
 * state in the cycle Unverified → Verified → False-Pos → Unverified (based on
 * the chip's current verification tag) and calls `onToggle(nextUuid)`. Without
 * `onToggle` the chip is display-only, exactly as before.
 *
 * @param {string[]} tagUuids
 * @param {{ onToggle?: (nextUuid: string) => void }} [opts]
 */
export function buildVerifyChip(tagUuids, { onToggle } = {}) {
  const isFalsePos = tagUuids.includes(FALSE_POS_TAG_UUID);
  const isVerified = tagUuids.includes(VERIFIED_TAG_UUID);
  const chip = document.createElement('span');
  chip.className = 'verify-chip';
  chip.textContent = isFalsePos ? '✗ False Pos.' : isVerified ? '✓ Verified' : '◌ Unverified';
  chip.classList.toggle('verify-chip--verified', isVerified);
  chip.classList.toggle('verify-chip--false-pos', isFalsePos);
  chip.title = isFalsePos ? 'False Positive' : isVerified ? 'Verified' : 'Not yet verified';
  if (onToggle) {
    const currentUuid = isFalsePos ? FALSE_POS_TAG_UUID : isVerified ? VERIFIED_TAG_UUID : UNVERIFIED_TAG_UUID;
    chip.addEventListener('dblclick', ev => {
      ev.stopPropagation();
      onToggle(VERIFY_CYCLE_NEXT[currentUuid]);
    });
  }
  return chip;
}

/**
 * Create a card-grid controller that OWNS the card selection state and the
 * click / shift-range / select-all selection logic shared by the review grid
 * and (later) the Run-Model-on-Results preview.
 *
 * The controller owns:
 *   - `selected`       — Set of selected entry IDs (bbox.uuid or img.uuid)
 *   - `selectionKind`  — 'bbox' | 'image' | null — locked type of the selection
 *   - `lastClickedIndex` — index in visibleEntries of the last checkbox click
 *   - `deletedUuids`   — entry IDs deleted this page (disabled/unselectable)
 *   - `staleUuids`     — entry IDs modified this page (for status display)
 *
 * `ctx` injects the host-specific bits the moved logic needs but that stay in
 * the host (main.js):
 *   - `gridEl`             — the live container holding the cards/rows; all
 *                            selection/stale DOM queries are scoped to it.
 *   - `getVisibleEntries()` — returns the host's live visible-entries array
 *                            (shift-range + select-all iterate it).
 *   - `entryId(entry)`     — extracts an entry's ID (bbox.uuid or img.uuid).
 *   - `onSelectionChange()` — host callback invoked whenever the selection
 *                            changes (main.js wires this to `updateBulkBtn`).
 *   - `hooks`              — { toast, resetSelectAll }:
 *       - `toast(msg)`        — surface the "one type at a time" message.
 *       - `resetSelectAll()`  — reset the host's Select-all checkbox (clear
 *                               `checked`/`indeterminate`) during clearSelection.
 *
 * The controller only reads/writes its own selection state plus `ctx`; it never
 * closes over host closure vars directly. This is a behavior-preserving
 * extraction of main.js's former `applySelect` / `attachSelectListeners` /
 * `markCardStale` / `clearSelection` / `updateTypeFading`.
 *
 * @param {HTMLElement} gridEl
 * @param {{
 *   getVisibleEntries: () => Array,
 *   entryId: (entry: any) => string,
 *   onSelectionChange: () => void,
 *   hooks?: { toast?: (msg: string) => void, resetSelectAll?: () => void },
 * }} ctx
 */
export function createCardGrid(gridEl, ctx) {
  const { getVisibleEntries, entryId, onSelectionChange, hooks = {} } = ctx;
  const toast = hooks.toast ?? (() => {});
  const resetSelectAll = hooks.resetSelectAll ?? (() => {});

  // ── Selection state (owned here) ──────────────────────────────────
  let _selected = new Set();      // selected IDs (bbox.uuid or img.uuid depending on kind)
  let _selectionKind = null;      // 'bbox' | 'image' | null — type of currently selected cards
  let _staleUuids = new Set();    // entry IDs modified on this page (for status display)
  let _deletedUuids = new Set();  // entry IDs deleted this page (disabled/unselectable)
  let _lastClickedIndex = -1;     // index in visibleEntries of the last checkbox click

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

  function updateTypeFading() {
    gridEl.querySelectorAll('.review-card').forEach(card => {
      const shouldFade = _selectionKind !== null && card.dataset.kind !== _selectionKind;
      card.classList.toggle('type-faded', shouldFade);
    });
  }

  function clearSelection() {
    _selected.clear();
    _selectionKind = null;
    resetSelectAll();
    gridEl.querySelectorAll('.review-card').forEach(c => {
      c.classList.remove('selected');
      const cb = c.querySelector('input[type=checkbox]');
      if (cb && !cb.disabled) cb.checked = false;
    });
    onSelectionChange();
    updateTypeFading();
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
      const visibleEntries = getVisibleEntries();
      const lo = Math.min(_lastClickedIndex, index);
      const hi = Math.max(_lastClickedIndex, index);
      for (let i = lo; i <= hi; i++) {
        const entry = visibleEntries[i];
        if (!entry) continue;
        const eid = entryId(entry);
        if (_deletedUuids.has(eid)) continue;
        // Only select entries of the same kind
        if (entry.kind !== entryKind) continue;
        const selector = entry.kind === 'image'
          ? `[data-image-uuid="${eid}"][data-kind="image"]`
          : `[data-bbox-uuid="${eid}"]`;
        const c = gridEl.querySelector(selector);
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

    onSelectionChange();
    updateTypeFading();
  }

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

  return {
    // ── Selection state accessors ───────────────────────────────────
    // `selected`, `staleUuids`, `deletedUuids` are the live Set instances so
    // callers that read membership / size see current state. `resetSelected`
    // etc. below re-init them (matching main.js's former `_x = new Set()`).
    get selected() { return _selected; },
    get staleUuids() { return _staleUuids; },
    get deletedUuids() { return _deletedUuids; },
    get selectionKind() { return _selectionKind; },
    set selectionKind(v) { _selectionKind = v; },
    get lastClickedIndex() { return _lastClickedIndex; },
    set lastClickedIndex(v) { _lastClickedIndex = v; },

    // Re-initialise the per-page state (used by render / view-switch resets,
    // mirroring the former `_selected.clear()` / `_x = new Set()` in main.js).
    resetForRender() {
      _selected.clear();
      _staleUuids = new Set();
      _deletedUuids = new Set();
      _lastClickedIndex = -1;
      _selectionKind = null;
    },

    // ── Moved selection logic ───────────────────────────────────────
    markCardStale,
    updateTypeFading,
    clearSelection,
    applySelect,
    attachSelectListeners,
  };
}

// ── Shared card-interaction helpers (moved from main.js) ─────────────────────

/**
 * Attach the hover group-highlight behaviour to a card/row. On mouseenter the
 * hovered card gets a bold highlight and every other card sharing the same
 * `imageUuid` (scoped to `gridEl`) gets a softer highlight; mouseleave clears
 * them. Behaviour-preserving move of main.js's former `attachGroupHighlight`.
 */
export function attachGroupHighlight(gridEl, card, imageUuid) {
  card.addEventListener('mouseenter', () => {
    // Bold highlight on the hovered card itself
    card.classList.add(card.dataset.kind === 'image' ? 'group-highlight-self-image' : 'group-highlight-self-bbox');
    // Softer highlight on related cards
    gridEl.querySelectorAll(`[data-image-uuid="${imageUuid}"]`).forEach(c => {
      if (c === card) return;
      c.classList.add(c.dataset.kind === 'image' ? 'group-highlight-image' : 'group-highlight-bbox');
    });
  });
  card.addEventListener('mouseleave', () => {
    card.classList.remove('group-highlight-self-image', 'group-highlight-self-bbox');
    gridEl.querySelectorAll('.group-highlight-image,.group-highlight-bbox').forEach(c =>
      c.classList.remove('group-highlight-image', 'group-highlight-bbox'));
  });
}

/**
 * Populate `container` with deduplicated label chips for `bboxes`.
 * Falls back to an "unlabeled" chip when boxes exist but none have a label.
 * `labels` is the fetchLabels() Map (uuid → {name,color}).
 */
export function appendLabelChips(container, bboxes, labels) {
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

/**
 * Pure card/row builders — moved out of main.js so the review grid and (later)
 * the Run-Model-on-Results preview share them. Each builder takes the shared
 * `controller` (createCardGrid, for selection/stale), the entry data, and an
 * `opts` bag carrying the host display bits + injected host callbacks. No
 * builder reads a main.js closure directly.
 *
 * Common `opts` keys (subset used per builder):
 *   - `gridEl`             — live container (group-highlight / editor refresh scope).
 *   - `labels`             — fetchLabels() Map (uuid → {name,color}).
 *   - `index`              — entry index in visibleEntries (shift-range select).
 *   - `referenceBboxUuids` / `referenceImageUuids` — Sets flagging reference cards.
 *   - `hoverPreview`       — createHoverPreview() instance (`.attach`).
 *   - `thumbLoader`        — ThumbnailLoader instance (`.load`).
 *   - `makeAdapter(entry, card)` — realized card adapter factory (review grid:
 *                            makeReviewBboxAdapter; DB writes + markCardStale +
 *                            EventBus). Used for the verify-chip onToggle.
 *   - `resolveContext(entry, card, kind)` — returns the context-menu resolver's
 *                            objects array (host's _resolveContextObjects).
 *   - `onOpenEditor(entry)` — double-click open-editor path (host wires
 *                            _buildNavContext + showCanvasModal + refresh hooks).
 *   - `registerImageRefresher(uuid, fn)` — record the overlay-redraw fn so bulk
 *                            actions can trigger it (image card only).
 *   - `getPageImageLabels(uuid)` — page-level image-label map for a uuid.
 *   - `subscribeImageLabelChanges(uuid, filename, render)` — wire the
 *                            IMAGE_LABELS_CHANGED / IMAGE_ANNOTATIONS_SAVED
 *                            per-card subscriptions (image card only).
 */

// ── Bbox card builder ─────────────────────────────────────────────
export function buildBboxCard(controller, entry, opts) {
  const { bbox, img } = entry;
  const {
    gridEl, labels, index, referenceBboxUuids, hoverPreview, thumbLoader,
    makeAdapter, resolveContext, onOpenEditor,
  } = opts;

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
  card.dataset.testid = 'review-card';
  card.dataset.bboxUuid = bbox.uuid;
  card.dataset.imageUuid = img.uuid;
  card.dataset.kind = 'bbox';

  const cb = document.createElement('input');
  cb.type = 'checkbox';
  const clearTimer = controller.attachSelectListeners(card, cb, bbox.uuid, 'bbox', index);

  const thumbImg = document.createElement('img');
  thumbImg.className = 'thumb';
  thumbImg.alt = labelName;
  thumbImg.title = 'Double-click to edit in canvas';
  thumbLoader.load(thumbImg, getBboxThumbnailUrl(img.filename, bbox.uuid, 300));
  thumbImg.addEventListener('dblclick', () => {
    clearTimer();
    onOpenEditor(entry);
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

  // Double-clicking the chip advances verification via the same adapter the
  // right-click menu uses; markCardStale/_refreshBboxCard handle the refresh.
  const verifyChip = buildVerifyChip(bboxTagUuids, {
    onToggle: nextUuid => makeAdapter({ bbox, img }, card).setVerification(nextUuid),
  });

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

  attachGroupHighlight(gridEl, card, img.uuid);
  hoverPreview.attach(card, {
    imagePath: img.filename,
    imageUuid: img.uuid,
    allBboxes: img.bboxes,
    hoveredBboxUuid: bbox.uuid,
  });

  attachContext(card, ev => resolveContext({ kind: 'bbox', bbox, img }, card, 'bbox'));

  return card;
}

// ── Image card builder ────────────────────────────────────────────
export function buildImageCard(controller, entry, opts) {
  const { img } = entry;
  const {
    gridEl, labels, index, referenceImageUuids, hoverPreview, thumbLoader,
    resolveContext, onOpenEditor,
    registerImageRefresher, getPageImageLabels, subscribeImageLabelChanges,
  } = opts;

  const card = document.createElement('div');
  card.className = 'review-card review-card--image';
  card.dataset.testid = 'review-card';
  card.dataset.imageUuid = img.uuid;
  card.dataset.kind = 'image';

  const cb = document.createElement('input');
  cb.type = 'checkbox';
  const clearTimer = controller.attachSelectListeners(card, cb, img.uuid, 'image', index);

  // Double-click thumbnail → open canvas modal on first available bbox
  const thumbImg = document.createElement('img');
  thumbImg.className = 'thumb';
  thumbImg.alt = img.filename.split('/').pop();
  thumbImg.title = 'Double-click to edit in canvas';
  thumbLoader.load(thumbImg, getImageThumbnailUrl(img.filename, 200));
  thumbImg.addEventListener('dblclick', () => {
    clearTimer();
    const firstBbox = (img.bboxes ?? [])[0];
    if (firstBbox) onOpenEditor({ kind: 'bbox', bbox: firstBbox, img, imageInitiated: true });
  });

  // Bbox canvas overlay — draws normalized rects over the thumbnail
  const thumbCanvas = document.createElement('canvas');
  thumbCanvas.className = 'thumb-overlay';

  function drawBboxOverlay() {
    drawLetterboxedBboxes(thumbImg, thumbCanvas, (img.bboxes ?? []).map(bbox => ({
      x: bbox.x, y: bbox.y, width: bbox.width, height: bbox.height, color: bbox.label?.color,
    })));
  }

  if (thumbImg.complete && thumbImg.naturalWidth) {
    drawBboxOverlay();
  } else {
    thumbImg.onload = drawBboxOverlay;
  }

  // Register so bulk bbox actions can trigger a redraw
  registerImageRefresher(img.uuid, drawBboxOverlay);

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

  function _renderImageLabelChips(labelMap) {
    imageLabelChipsRow.innerHTML = '';
    for (const [luuid, status] of (labelMap ?? [])) {
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

  // Initial chips come from the page-level batch fetch — no per-card GET.
  _renderImageLabelChips(getPageImageLabels(img.uuid));

  // Keep chips up to date when labels change from context menu or canvas save.
  // Only then is a single-image fetch warranted (rare, user-triggered).
  // Subscriptions are drained by _drainCardUnsubs() before each page render and in destroy().
  subscribeImageLabelChanges(img.uuid, img.filename, _renderImageLabelChips);

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

  attachGroupHighlight(gridEl, card, img.uuid);
  hoverPreview.attach(card, {
    imagePath: img.filename,
    imageUuid: img.uuid,
    allBboxes: img.bboxes,
    hoveredBboxUuid: null,
  });

  attachContext(card, ev => resolveContext({ kind: 'image', img }, card, 'image'));

  return card;
}

// ── Bbox list-row builder (compact card for list view) ────────────
export function buildBboxRow(controller, entry, opts) {
  const { bbox, img } = entry;
  const {
    gridEl, labels, index, referenceBboxUuids, hoverPreview, thumbLoader,
    makeAdapter, resolveContext, onOpenEditor,
  } = opts;

  const label = bbox.label ?? (bbox.labelUuid ? labels.get(bbox.labelUuid) : null);
  const labelName = label?.name ?? '(unknown)';
  const labelColor = label?.color ?? '#888';
  const rowTagUuids = (bbox.tags ?? []).map(t => t.uuid);
  const confidence = bbox.metadataJson?.confidence
    ? `${(parseFloat(bbox.metadataJson.confidence) * 100).toFixed(0)}%`
    : '';

  const row = document.createElement('li');
  row.className = 'review-row review-card';
  row.dataset.testid = 'review-row';
  row.dataset.bboxUuid = bbox.uuid;
  row.dataset.imageUuid = img.uuid;
  row.dataset.kind = 'bbox';

  const cb = document.createElement('input');
  cb.type = 'checkbox';
  const clearTimer = controller.attachSelectListeners(row, cb, bbox.uuid, 'bbox', index);

  const thumbImg = document.createElement('img');
  thumbImg.className = 'thumb';
  thumbImg.alt = labelName;
  thumbImg.title = 'Double-click to edit in canvas';
  thumbLoader.load(thumbImg, getBboxThumbnailUrl(img.filename, bbox.uuid, 96));
  thumbImg.addEventListener('dblclick', () => {
    clearTimer();
    onOpenEditor(entry);
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

  // Double-clicking the chip advances verification via the same adapter the
  // right-click menu uses; markCardStale/_refreshBboxCard handle the refresh.
  const verifyChip = buildVerifyChip(rowTagUuids, {
    onToggle: nextUuid => makeAdapter({ bbox, img }, row).setVerification(nextUuid),
  });

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

  attachGroupHighlight(gridEl, row, img.uuid);
  hoverPreview.attach(row, {
    imagePath: img.filename,
    imageUuid: img.uuid,
    allBboxes: img.bboxes,
    hoveredBboxUuid: bbox.uuid,
  });

  attachContext(row, ev => resolveContext({ kind: 'bbox', bbox, img }, row, 'bbox'));

  return row;
}

// ── Image list-row builder (compact card for list view) ───────────
export function buildImageRow(controller, entry, opts) {
  const { img } = entry;
  const {
    gridEl, labels, index, referenceImageUuids, hoverPreview, thumbLoader,
    onOpenEditor,
  } = opts;

  const row = document.createElement('li');
  row.className = 'review-row review-card review-card--image';
  row.dataset.testid = 'review-row';
  row.dataset.imageUuid = img.uuid;
  row.dataset.kind = 'image';

  const cb = document.createElement('input');
  cb.type = 'checkbox';
  const clearTimer = controller.attachSelectListeners(row, cb, img.uuid, 'image', index);

  // Double-click thumbnail → open canvas modal on first available bbox
  const thumbImg = document.createElement('img');
  thumbImg.className = 'thumb';
  thumbImg.alt = img.filename.split('/').pop();
  thumbImg.title = 'Double-click to edit in canvas';
  thumbLoader.load(thumbImg, getImageThumbnailUrl(img.filename, 96));
  thumbImg.addEventListener('dblclick', () => {
    clearTimer();
    const firstBbox = (img.bboxes ?? [])[0];
    if (firstBbox) onOpenEditor({ kind: 'bbox', bbox: firstBbox, img, imageInitiated: true });
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

  attachGroupHighlight(gridEl, row, img.uuid);
  hoverPreview.attach(row, {
    imagePath: img.filename,
    imageUuid: img.uuid,
    allBboxes: img.bboxes,
    hoveredBboxUuid: null,
  });

  return row;
}
