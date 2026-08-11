import {
  UNVERIFIED_TAG_UUID,
  VERIFIED_TAG_UUID,
  getBboxThumbnailUrl,
  getImageThumbnailUrl,
} from '/app-static/js/shared/api/review.js';
import { attachContext } from '/app-static/js/ui/contextMenu/index.js';
import { drawLetterboxedBboxes } from '/app-static/js/ui/utils/bboxThumbOverlay.js';
import { Logger } from '/app-static/js/ui/utils/logging.js';

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
 *   - `gapsConfig`         — {uuid, name} training run the gap row is judged
 *                            against, or null to render no gap row.
 *   - `getPageImageGaps(uuid)` — page-level gap list for a uuid (image card only).
 *   - `refreshImageGaps(uuid)` — re-read one image's gaps after a resolve; returns
 *                            the fresh list (image card only).
 *   - `makeImageAdapter(entry)` — image-level adapter (setLabelPresence) used by the
 *                            gap chips (image card only).
 */

// How many gap chips render on a card before the rest collapse into a "+N" chip.
// The full list is always available in the hover preview's side panel.
const MAX_GAP_CHIPS = 3;

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

/**
 * Build the popover side panel listing every label this image still needs — the
 * uncapped counterpart of the card's gap chip row.
 *
 * @param {{labelUuid: string, labelName: string, labelScope: string}[]} gaps
 * @param {{name: string}} config
 */
function buildGapSidePanel(gaps, config, labels) {
  if (!gaps.length) return null;
  const panel = document.createElement('div');

  const heading = document.createElement('div');
  heading.style.cssText = 'color:var(--pico-muted-color,#888);margin-bottom:4px;line-height:1.3;';
  heading.textContent = `Needs ${gaps.length} verification${gaps.length !== 1 ? 's' : ''} for "${config.name}"`;
  panel.appendChild(heading);

  const list = document.createElement('div');
  list.style.cssText = 'display:flex;flex-direction:column;gap:2px;';
  for (const gap of gaps) {
    const row = document.createElement('div');
    row.style.cssText = 'display:flex;align-items:center;gap:4px;';
    const color = labels.get(gap.labelUuid)?.color;
    if (color) {
      const dot = document.createElement('span');
      dot.style.cssText = `flex:0 0 auto;width:7px;height:7px;border-radius:50%;background:${color};`;
      row.appendChild(dot);
    }
    const name = document.createElement('span');
    name.textContent = gap.labelName;
    row.appendChild(name);
    const scope = document.createElement('span');
    scope.style.cssText = 'color:var(--pico-muted-color,#888);margin-left:auto;';
    // Says what would actually resolve it — the two scopes clear differently.
    scope.textContent = gap.labelScope === 'image' ? 'present/absent' : 'box or absent';
    row.appendChild(scope);
    list.appendChild(row);
  }
  panel.appendChild(list);
  return panel;
}

// ── Image card builder ────────────────────────────────────────────
export function buildImageCard(controller, entry, opts) {
  const { img } = entry;
  const {
    gridEl, labels, index, referenceImageUuids, hoverPreview, thumbLoader,
    makeAdapter, resolveContext, onOpenEditor,
    registerImageRefresher, getPageImageLabels, subscribeImageLabelChanges,
    gapsConfig = null, getPageImageGaps, refreshImageGaps, makeImageAdapter,
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
    // An image with no boxes yet still opens — that's the case where the editor
    // is most needed (draw the first box). It used to bail out silently, which
    // read as a dead double-click on every un-annotated search result.
    // bbox is deliberately null, not img.bboxes[0]: this is an image-level
    // open, not a specific-bbox one, so nothing should arrive pre-selected
    // with transformer handles on it — that's reserved for double-clicking
    // a bbox's own card (the sibling handler below, kind: 'bbox').
    onOpenEditor({ kind: 'bbox', bbox: null, img, imageInitiated: true });
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

  // ── Training-run gap row ────────────────────────────────────────
  // What this image still needs before the active/selected training run counts it.
  // Chips are actionable: resolving one here is the whole point of a gaps view.
  const gapsRow = document.createElement('div');
  gapsRow.className = 'card-gaps-row';
  gapsRow.dataset.testid = 'card-gaps-row';

  // Live list — re-read after every resolve so the hover panel (built at show
  // time) and the chip row never disagree.
  let _gaps = gapsConfig ? (getPageImageGaps?.(img.uuid) ?? []) : [];

  async function _resolveGap(gap, status) {
    // Absent on the top-level label is enough even under rollup — the server
    // counts an Absent on the parent as covering its rollup children, so this
    // must NOT fan out to children.
    await makeImageAdapter?.({ img })?.setLabelPresence(gap.labelUuid, status);
    _gaps = (await refreshImageGaps?.(img.uuid)) ?? [];
    renderGapsRow();
    if (_gaps.length === 0) controller.markCardStale(card, 'verified');
  }

  function _makeGapChip(gap) {
    const chip = document.createElement('span');
    chip.className = 'gap-chip';
    const color = labels.get(gap.labelUuid)?.color;
    if (color) chip.style.cssText = `--gap-chip-color:${color};`;

    const name = document.createElement('span');
    name.className = 'gap-chip__name';
    name.textContent = gap.labelName;
    chip.appendChild(name);

    // Image-scope labels are cleared by EITHER present or absent, so both are
    // offered. Object-scope ones can only be cleared here by absent — the other
    // route (a verified box) needs the canvas, which the card's dblclick opens.
    if (gap.labelScope === 'image') {
      const presentBtn = document.createElement('button');
      presentBtn.type = 'button';
      presentBtn.className = 'gap-chip__act';
      presentBtn.textContent = '✓';
      presentBtn.title = `Mark "${gap.labelName}" present on this image`;
      presentBtn.addEventListener('click', ev => { ev.stopPropagation(); _resolveGap(gap, 'present'); });
      chip.appendChild(presentBtn);
    }

    const absentBtn = document.createElement('button');
    absentBtn.type = 'button';
    absentBtn.className = 'gap-chip__act';
    absentBtn.textContent = '✕';
    absentBtn.title = `Mark "${gap.labelName}" absent from this image`;
    absentBtn.addEventListener('click', ev => { ev.stopPropagation(); _resolveGap(gap, 'absent'); });
    chip.appendChild(absentBtn);

    chip.title = gap.labelScope === 'image'
      ? `"${gap.labelName}" has no present/absent status on this image`
      : `"${gap.labelName}" needs a verified box or an explicit absent`;
    return chip;
  }

  function renderGapsRow() {
    gapsRow.innerHTML = '';
    gapsRow.hidden = _gaps.length === 0;
    if (!_gaps.length) return;

    const lead = document.createElement('span');
    lead.className = 'card-gaps-row__lead';
    lead.textContent = 'Needs:';
    // The run doing the judging is never implicit — the active run and the run a
    // gaps modal was opened from can differ.
    lead.title = `Unresolved for training run "${gapsConfig.name}" — hover the card for the full list`;
    gapsRow.appendChild(lead);

    for (const gap of _gaps.slice(0, MAX_GAP_CHIPS)) gapsRow.appendChild(_makeGapChip(gap));

    const overflow = _gaps.length - MAX_GAP_CHIPS;
    if (overflow > 0) {
      const more = document.createElement('span');
      more.className = 'gap-chip gap-chip--more';
      more.textContent = `+${overflow}`;
      more.title = _gaps.slice(MAX_GAP_CHIPS).map(g => g.labelName).join(', ');
      gapsRow.appendChild(more);
    }
  }
  renderGapsRow();

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
    // Verifying a box can close a gap, so a canvas save has to re-read them.
    // Deliberately not awaited: the rest of the card must repaint immediately
    // rather than block on a round trip.
    if (gapsConfig && refreshImageGaps) {
      refreshImageGaps(img.uuid).then(gaps => {
        _gaps = gaps ?? [];
        renderGapsRow();
      }).catch(err => Logger.error('review: image gap refresh failed', err));
    }
  }

  // Register so bulk bbox actions and canvas saves can refresh this card.
  registerImageRefresher(img.uuid, refreshCard);

  info.appendChild(chipsRow);
  info.appendChild(imageLabelChipsRow);
  info.appendChild(gapsRow);
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
    // Uncapped gap list beside the enlarged image — where a 12-item list belongs.
    buildSidePanel: gapsConfig ? () => buildGapSidePanel(_gaps, gapsConfig, labels) : undefined,
  });

  attachContext(card, ev => resolveContext({ kind: 'image', img }, card, 'image'));

  return card;
}

// ── Preview-candidate card variant ───────────────────────────────────────────

/** Per-decision badge text/class, so the three surfaces that render one agree. */
const DECISION_BADGE = {
  accepted: { text: '✓ Accepted', cls: 'state-verified' },
  rejected: { text: '✕ Rejected', cls: 'state-deleted' },
  applied: { text: '● Applied', cls: 'state-applied' },
  stale: { text: '⚠ Stale', cls: 'state-stale' },
};

/**
 * Build a card for a **persisted preview candidate** — a model's proposal, not
 * an annotation.
 *
 * A deliberate opt-in variant rather than a mode of `buildBboxCard`: a candidate
 * carries `proposedLabelUuid` / `confidence` / a proposed `geometry`, none of
 * which exist on a real bbox, and a `isNew` candidate has no bbox row at all —
 * so collapsing the two builders would either leak preview-only fields into
 * ordinary review cards or make every review card carry dead branches. Callers
 * must ask for this one by name.
 *
 * @param {object} controller - createCardGrid controller (selection).
 * @param {{id: string, candidate: object}} entry
 * @param {{
 *   gridEl: HTMLElement,
 *   labels: Map<string, {name: string, color: string}>,
 *   index: number,
 *   hoverPreview: object,
 *   labelChip: (name: ?string, uuid: ?string, labels: Map) => HTMLElement,
 *   resolveContext: (entry: object, card: HTMLElement) => object[],
 *   onOpenEditor: (entry: object) => void,
 * }} opts
 */
export function buildCandidateCard(controller, entry, opts) {
  const { candidate } = entry;
  const { gridEl, labels, index, hoverPreview, labelChip, resolveContext, onOpenEditor } = opts;

  const card = document.createElement('div');
  card.className = 'review-card relabel-modal-card';
  card.dataset.testid = 'preview-candidate-card';
  card.dataset.candidateUuid = candidate.uuid;
  // The selection controller and its shift-range selectors resolve off
  // data-bbox-uuid; a new-box candidate has no bbox, so the entry id (which is
  // the candidate uuid) stands in for one.
  card.dataset.bboxUuid = entry.id;
  card.dataset.imageUuid = candidate.imageUuid;
  card.dataset.kind = 'bbox';
  card.dataset.decision = candidate.decision;

  const cb = document.createElement('input');
  cb.type = 'checkbox';
  const clearTimer = controller.attachSelectListeners(card, cb, entry.id, 'bbox', index);

  const proposedColor = labels.get(candidate.proposedLabelUuid)?.color ?? '#5cc45c';

  if (candidate.isNew) {
    // No bbox to crop yet — show the whole image with the proposed geometry
    // drawn over it (same letterbox-aware overlay as buildImageCard).
    const thumbWrap = document.createElement('div');
    thumbWrap.style.cssText = 'position:relative;';
    const thumbImg = document.createElement('img');
    thumbImg.className = 'thumb';
    thumbImg.loading = 'lazy';
    thumbImg.title = 'Double-click to edit in canvas';
    thumbImg.src = getImageThumbnailUrl(candidate.imageFilename, 300);
    const overlay = document.createElement('canvas');
    overlay.className = 'thumb-overlay';
    overlay.style.cssText = 'position:absolute;inset:0;width:100%;height:100%;pointer-events:none;';
    const drawOverlay = () => {
      if (!candidate.geometry) return;
      drawLetterboxedBboxes(thumbImg, overlay, [{ ...candidate.geometry, color: proposedColor }]);
    };
    if (thumbImg.complete && thumbImg.naturalWidth) drawOverlay();
    else thumbImg.onload = drawOverlay;
    thumbImg.addEventListener('dblclick', () => { clearTimer(); onOpenEditor(entry); });
    thumbWrap.append(thumbImg, overlay);
    card.append(cb, thumbWrap);
  } else {
    const thumbImg = document.createElement('img');
    thumbImg.className = 'thumb';
    thumbImg.loading = 'lazy';
    thumbImg.title = 'Double-click to edit in canvas';
    thumbImg.src = getBboxThumbnailUrl(candidate.imageFilename, candidate.bboxUuid, 300);
    thumbImg.addEventListener('dblclick', () => { clearTimer(); onOpenEditor(entry); });
    card.append(cb, thumbImg);
  }

  const info = document.createElement('div');
  info.className = 'card-info';

  const badgeSpec = DECISION_BADGE[candidate.decision];
  if (badgeSpec) {
    const badge = document.createElement('div');
    badge.className = `card-state-badge ${badgeSpec.cls}`;
    badge.dataset.testid = 'preview-decision-badge';
    badge.textContent = badgeSpec.text;
    info.appendChild(badge);
  }

  const labelsRow = document.createElement('div');
  labelsRow.className = 'relabel-modal-labels-row';
  labelsRow.style.cssText = 'display:flex;align-items:center;gap:0.4em;flex-wrap:wrap;';
  const proposedChip = labelChip(labels.get(candidate.proposedLabelUuid)?.name, candidate.proposedLabelUuid, labels);
  proposedChip.classList.add('relabel-modal-proposed-chip');
  if (candidate.isNew) {
    const newBadge = document.createElement('span');
    newBadge.className = 'label-chip relabel-modal-new-badge';
    newBadge.textContent = '＋ NEW';
    labelsRow.append(newBadge, proposedChip);
  } else {
    const currentChip = labelChip(labels.get(candidate.currentLabelUuid)?.name, candidate.currentLabelUuid, labels);
    const arrow = document.createElement('span');
    arrow.textContent = '→';
    labelsRow.append(currentChip, arrow, proposedChip);
  }
  // A relabel is still applicable when the box moved off the label the run
  // snapshotted — but the user should see that before accepting it.
  if (candidate.labelChangedSinceRun) {
    const changed = document.createElement('span');
    changed.className = 'label-chip preview-changed-badge';
    changed.dataset.testid = 'preview-changed-badge';
    changed.textContent = '⟳ changed since run';
    changed.title = "This box's label changed after the model ran.";
    labelsRow.appendChild(changed);
  }
  info.appendChild(labelsRow);

  // A candidate whose decision is Delete has no proposed label to speak of —
  // dim the chip rather than hide it, so the row still reads as one thing.
  if (candidate.decisionAction === 'delete') proposedChip.style.opacity = '0.35';

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

  // Which model won this box, so a multi-model run's merge is inspectable
  // rather than something the user has to infer.
  if (candidate.sourceModel) {
    const src = document.createElement('div');
    src.className = 'preview-card-source';
    src.style.cssText = 'margin-top:0.2em;color:#aaa;font-size:0.75em;';
    src.textContent = candidate.sourceModel;
    info.appendChild(src);
  }

  card.appendChild(info);

  attachGroupHighlight(gridEl, card, candidate.imageUuid);
  if (candidate.isNew) {
    hoverPreview.attach(card, {
      imagePath: candidate.imageFilename,
      imageUuid: candidate.imageUuid,
      allBboxes: candidate.geometry
        ? [{ uuid: 'new', ...candidate.geometry, label: { color: proposedColor } }]
        : [],
      hoveredBboxUuid: 'new',
    });
  } else {
    hoverPreview.attach(card, {
      imagePath: candidate.imageFilename,
      imageUuid: candidate.imageUuid,
      hoveredBboxUuid: candidate.bboxUuid,
    });
  }

  attachContext(card, () => resolveContext(entry, card));

  return card;
}
