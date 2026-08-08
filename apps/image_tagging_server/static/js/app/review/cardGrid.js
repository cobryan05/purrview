import {
  UNVERIFIED_TAG_UUID,
  VERIFIED_TAG_UUID,
  getBboxThumbnailUrl,
  getImageThumbnailUrl,
} from '/app-static/js/shared/api/review.js';
import { attachContext } from '/app-static/js/ui/contextMenu/index.js';
import { drawLetterboxedBboxes } from '/app-static/js/ui/utils/bboxThumbOverlay.js';

// Verification toggle advanced by double-clicking a card's verify chip:
// Unverified ↔ Verified. Keyed by the chip's current verification tag so the
// next state derives purely from what's rendered.
const VERIFY_CYCLE_NEXT = {
  [UNVERIFIED_TAG_UUID]: VERIFIED_TAG_UUID,
  [VERIFIED_TAG_UUID]: UNVERIFIED_TAG_UUID,
};

/**
 * Build and return a verify-status chip span for a bbox's tag UUIDs.
 *
 * When `onToggle` is supplied, a `dblclick` handler flips the verification
 * state between Unverified and Verified (based on the chip's current
 * verification tag) and calls `onToggle(nextUuid)`. Without `onToggle` the
 * chip is display-only, exactly as before.
 *
 * @param {string[]} tagUuids
 * @param {{ onToggle?: (nextUuid: string) => void }} [opts]
 */
export function buildVerifyChip(tagUuids, { onToggle } = {}) {
  const isVerified = tagUuids.includes(VERIFIED_TAG_UUID);
  const chip = document.createElement('span');
  chip.className = 'verify-chip';
  chip.textContent = isVerified ? '✓ Verified' : '? Unverified';
  chip.classList.toggle('verify-chip--verified', isVerified);
  chip.title = isVerified ? 'Verified' : 'Not yet verified';
  if (onToggle) {
    const currentUuid = isVerified ? VERIFIED_TAG_UUID : UNVERIFIED_TAG_UUID;
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
    card.classList.remove('stale-verified', 'stale-relabeled', 'stale-deleted', 'stale-tagged');
    card.classList.add(`stale-${state}`);
    const stateLabels = {
      verified: '✓ Verified',
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
 *
 * With `onToggleVerify(labelBboxes, nextUuid)`, each chip's ✓/? mark becomes a
 * double-click target that flips every box behind that chip — mirroring the
 * per-bbox `.verify-chip`. Without it the marks are display-only.
 *
 * @param {HTMLElement} container
 * @param {object[]} bboxes
 * @param {Map<string, {name: string, color: string}>} labels
 * @param {{ onToggleVerify?: (labelBboxes: object[], nextUuid: string) => void }} [opts]
 */
export function appendLabelChips(container, bboxes, labels, { onToggleVerify } = {}) {
  // Group first: a chip is per-label, but its verify marker summarises every
  // box carrying that label, so we can't emit on first sight of the label.
  const byLabel = new Map();
  for (const bbox of bboxes) {
    const luuid = bbox.labelUuid;
    if (!luuid) continue;
    if (!byLabel.has(luuid)) byLabel.set(luuid, []);
    byLabel.get(luuid).push(bbox);
  }

  let emitted = 0;
  for (const [luuid, labelBboxes] of byLabel) {
    const lbl = labels.get(luuid);
    if (!lbl) continue;
    emitted++;
    const chip = document.createElement('span');
    chip.className = 'label-chip';
    chip.textContent = lbl.name;
    chip.style.cssText = `background:${lbl.color}33;border:1px solid ${lbl.color};`;
    appendChipVerifyMark(chip, labelBboxes, lbl.name, onToggleVerify);
    container.appendChild(chip);
  }
  if (emitted === 0 && bboxes.length > 0) {
    const chip = document.createElement('span');
    chip.className = 'label-chip';
    chip.textContent = 'unlabeled';
    chip.style.cssText = 'background:#88888833;border:1px solid #888;';
    appendChipVerifyMark(chip, bboxes, 'unlabeled', onToggleVerify);
    container.appendChild(chip);
  }
}

/**
 * Append the verified/unverified marker to a label chip, summarising the
 * verification state of every box that carries that label:
 *   all verified → ✓ · none verified → ? · mixed → ✓n ?m
 * Uses the same glyph vocabulary and colours as the per-bbox `.verify-chip`.
 *
 * When `onToggleVerify` is supplied the mark becomes a double-click target:
 * anything not fully verified goes Verified, fully verified goes back to
 * Unverified — so the mixed state resolves upward in one gesture rather than
 * needing a pass per box.
 */
function appendChipVerifyMark(chip, labelBboxes, labelName, onToggleVerify) {
  let verified = 0;
  for (const bbox of labelBboxes) {
    if ((bbox.tags ?? []).some(t => t.uuid === VERIFIED_TAG_UUID)) verified++;
  }
  const unverified = labelBboxes.length - verified;

  const mark = document.createElement('span');
  mark.className = 'label-chip__verify';
  if (unverified === 0) {
    mark.classList.add('label-chip__verify--verified');
    mark.textContent = '✓';
    chip.title = `${labelName}: all ${verified} box${verified !== 1 ? 'es' : ''} verified`;
  } else if (verified === 0) {
    mark.classList.add('label-chip__verify--unverified');
    mark.textContent = '?';
    chip.title = `${labelName}: ${unverified} box${unverified !== 1 ? 'es' : ''} not yet verified`;
  } else {
    mark.classList.add('label-chip__verify--mixed');
    const ok = document.createElement('span');
    ok.className = 'label-chip__verify--verified';
    ok.textContent = `✓${verified}`;
    const bad = document.createElement('span');
    bad.className = 'label-chip__verify--unverified';
    bad.textContent = `?${unverified}`;
    mark.append(ok, ' ', bad);
    chip.title = `${labelName}: ${verified} verified, ${unverified} not yet verified`;
  }
  if (onToggleVerify) {
    const nextUuid = unverified === 0 ? UNVERIFIED_TAG_UUID : VERIFIED_TAG_UUID;
    mark.classList.add('label-chip__verify--clickable');
    chip.title += nextUuid === VERIFIED_TAG_UUID
      ? ' — double-click the mark to verify all'
      : ' — double-click the mark to un-verify all';
    mark.addEventListener('dblclick', ev => {
      ev.stopPropagation();
      onToggleVerify(labelBboxes, nextUuid);
    });
    // The card's own click handler toggles selection; a double-click on the
    // mark must not also select the card.
    mark.addEventListener('click', ev => ev.stopPropagation());
  }
  chip.appendChild(mark);
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
    makeAdapter, resolveContext, onOpenEditor,
    registerImageRefresher, getPageImageLabels, subscribeImageLabelChanges,
  } = opts;

  // Read `img.bboxes` fresh on every render rather than snapshotting it: a save
  // from the canvas modal REPLACES the array on the entry, so a captured
  // reference would keep rendering the pre-save boxes.
  const currentBboxes = () => img.bboxes ?? [];

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
    drawLetterboxedBboxes(thumbImg, thumbCanvas, currentBboxes().map(bbox => ({
      x: bbox.x, y: bbox.y, width: bbox.width, height: bbox.height, color: bbox.label?.color,
    })));
  }

  if (thumbImg.complete && thumbImg.naturalWidth) {
    drawBboxOverlay();
  } else {
    thumbImg.onload = drawBboxOverlay;
  }

  // Bbox count badge
  const countBadge = document.createElement('div');
  countBadge.className = 'bbox-count-badge';

  function renderCountBadge() {
    const bc = currentBboxes().length;
    countBadge.textContent = `${bc} box${bc !== 1 ? 'es' : ''}`;
  }
  renderCountBadge();

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
  const chipsRow = document.createElement('div');
  chipsRow.style.cssText = 'display:flex;flex-wrap:wrap;gap:2px;margin-bottom:0.2em;';

  // Double-clicking a chip's ✓/? mark flips every box behind that chip through
  // the same adapter the right-click menu uses. The adapter mutates the bbox
  // objects in place, so re-rendering from `bboxes` picks up the new state.
  //
  // The adapter is built for BBOX cards: it refreshes `card.querySelector(
  // '.thumb')` with a bbox-crop URL. On an image card `.thumb` is the whole
  // image, so hand it a detached node and let that refresh no-op — the overlay
  // redraw it triggers via _imageCardRefreshers covers this card anyway.
  async function _toggleChipVerification(labelBboxes, nextUuid) {
    const scratch = document.createElement('div');
    for (const b of labelBboxes) {
      // Per-box card refresh is suppressed: a chip can cover dozens of boxes,
      // and refreshCard() below does the same work once at the end.
      await makeAdapter({ bbox: b, img }, scratch).setVerification(nextUuid, { refreshImageCard: false });
    }
    // Only the verify direction earns the "✓ Verified" badge; un-verifying has
    // no honest badge state, and the chip already shows the new state.
    if (nextUuid === VERIFIED_TAG_UUID) controller.markCardStale(card, 'verified');
    refreshCard();
  }

  function renderChips() {
    chipsRow.innerHTML = '';
    appendLabelChips(chipsRow, currentBboxes(), labels, {
      onToggleVerify: makeAdapter ? _toggleChipVerification : undefined,
    });
  }
  renderChips();

  // Whole-image label chips — fetched async, distinct from bbox chips.
  // Image-label presence is largely DERIVED from verified bboxes (see
  // computeDerivedPresent in ui/labelList.js), so a label with boxes on this
  // image would otherwise render twice: once solid, once dashed. Suppress the
  // dashed duplicate — the solid chip already carries the ✓/? verify mark. Only
  // image labels with no box on this image (true whole-image classification)
  // still get a dashed chip.
  const imageLabelChipsRow = document.createElement('div');
  imageLabelChipsRow.style.cssText = 'display:flex;flex-wrap:wrap;gap:2px;margin-bottom:0.2em;';

  // Remembered so a bbox-side refresh can re-run the dedup against the new
  // boxes without re-fetching the image labels.
  let _lastImageLabelMap = null;

  function _renderImageLabelChips(labelMap) {
    _lastImageLabelMap = labelMap ?? _lastImageLabelMap;
    const bboxLabelUuids = new Set(currentBboxes().map(b => b.labelUuid).filter(Boolean));
    imageLabelChipsRow.innerHTML = '';
    for (const [luuid, status] of (_lastImageLabelMap ?? [])) {
      if (status !== 'present') continue;
      if (bboxLabelUuids.has(luuid)) continue;
      const lbl = labels.get(luuid);
      if (!lbl) continue;
      const chip = document.createElement('span');
      chip.className = 'image-label-chip';
      chip.textContent = lbl.name;
      chip.title = `Whole-image label "${lbl.name}" — present on the image with no bounding box of its own`;
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

  function renderStatusRow() {
    let verified = 0;
    let unverified = 0;
    for (const bbox of currentBboxes()) {
      const bboxTags = bbox.tags ?? [];
      if (bboxTags.some(t => t.uuid === VERIFIED_TAG_UUID)) verified++;
      else if (bboxTags.some(t => t.uuid === UNVERIFIED_TAG_UUID)) unverified++;
    }
    const parts = [];
    if (verified > 0) parts.push(`✓ ${verified}`);
    if (unverified > 0) parts.push(`? ${unverified}`);
    statusRow.textContent = parts.join('  ');
    // Always in the DOM (hidden when empty) so a save that adds the first box
    // to a previously empty image can fill it in without a re-render.
    statusRow.hidden = parts.length === 0;
  }
  renderStatusRow();

  const meta = document.createElement('div');
  meta.style.cssText = 'color:#aaa;font-size:0.75em;';
  meta.textContent = img.filename.split('/').slice(-2).join('/');

  // Full card refresh — everything that reads `img.bboxes`. Registered below so
  // bulk actions AND canvas-modal saves bring the whole card up to date, not
  // just the overlay.
  function refreshCard() {
    drawBboxOverlay();
    renderCountBadge();
    renderChips();
    _renderImageLabelChips(null);
    renderStatusRow();
  }

  // Register so bulk bbox actions and canvas saves can refresh this card.
  registerImageRefresher(img.uuid, refreshCard);

  info.appendChild(chipsRow);
  info.appendChild(imageLabelChipsRow);
  info.appendChild(statusRow);
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
