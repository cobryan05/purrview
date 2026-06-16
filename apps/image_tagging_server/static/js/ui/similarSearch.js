/**
 * Similar-search helpers — open a nested review modal showing results of a
 * perceptual-hash similarity query.
 *
 * Each call creates an independent showReviewModal overlay stacked on top of
 * whatever is currently open. Closing one modal returns to the previous one;
 * the user can nest arbitrarily (e.g. search → result image → right-click →
 * search again). Each modal gets its own independent canvas runtime.
 *
 * Usage:
 *   openSimilarImagesModal({ imageUuid, threshold? })
 *   openSimilarBboxesModal({ bboxUuid, threshold?, labelUuid? })
 *
 * Exported helpers (for settings page / context-menu providers):
 *   getImageHashMax()          → Promise<number>  (n² for current hash size n)
 *   getBboxHashMax()           → Promise<number>
 *   getImageDefaultThreshold() → Promise<number>  (stored setting or 20% of max)
 *   getBboxDefaultThreshold()  → Promise<number>
 */

import { showReviewModal } from '/app-static/js/app/review/main.js';
import {
  findSimilarImagesByUuid,
  findSimilarBboxesByUuid,
  findSimilarBboxesByCoords,
  fetchImagesByUuids,
} from '/app-static/js/shared/api/images.js';
import { getSetting } from '/app-static/js/shared/api/settings.js';
import { fetchTags } from '/app-static/js/shared/api/tags.js';
import { fetchLabels } from '/app-static/js/shared/api/labels.js';
import { createMultiSelectDropdown } from '/app-static/js/ui/utils/index.js';
import { pageStartIdx } from '/app-static/js/ui/searchUtils.js';

/** @returns {Promise<number>} */
export async function getImageHashMax() {
  const v = await getSetting('dedup_image_hash_size').catch(() => null);
  return (parseInt(v, 10) || 8) ** 2;
}

/** @returns {Promise<number>} */
export async function getBboxHashMax() {
  const v = await getSetting('dedup_bbox_hash_size').catch(() => null);
  return (parseInt(v, 10) || 16) ** 2;
}

// ── Default threshold helpers ──────────────────────────────────────────────────

/** Read stored threshold setting clamped to max, falling back to 20% of max. */
async function _readThreshold(key, max) {
  try {
    const stored = await getSetting(key);
    if (stored !== null && stored !== '') {
      const n = parseInt(stored, 10);
      if (!isNaN(n) && n >= 1) return Math.min(n, max);
    }
  } catch (_) {}
  return Math.round(max * 0.2);
}

/**
 * @param {number|null} [hashMax]  Pass a pre-fetched hash max to avoid a
 *   redundant getSetting call when the caller already has it.
 * @returns {Promise<number>}
 */
export async function getImageDefaultThreshold(hashMax = null) {
  return _readThreshold('dedup_image_threshold', hashMax ?? await getImageHashMax());
}

/**
 * @param {number|null} [hashMax]  Pass a pre-fetched hash max to avoid a
 *   redundant getSetting call when the caller already has it.
 * @returns {Promise<number>}
 */
export async function getBboxDefaultThreshold(hashMax = null) {
  return _readThreshold('dedup_bbox_threshold', hashMax ?? await getBboxHashMax());
}

// ── Shared slider builder ──────────────────────────────────────────────────────

/**
 * Build a threshold slider control element.
 * Returns { element, getValue } so callers can read the current value.
 * `onChange` is called whenever the value changes (debounced to 'change' event
 * on the range input, not 'input', to avoid hammering the server on drag).
 */
function buildThresholdSlider(initial, { min = 0, max = 64, onChange }) {
  const el = document.createElement('div');
  el.style.cssText = 'display:flex;align-items:center;gap:0.5em;font-size:0.85em;color:#aaa;margin-left:0.5em;';

  const label = document.createElement('span');
  label.style.cssText = 'white-space:nowrap;';
  label.textContent = 'Threshold:';

  const slider = document.createElement('input');
  slider.type = 'range';
  slider.min = String(min);
  slider.max = String(max);
  slider.step = '1';
  slider.value = String(initial);
  slider.style.cssText = 'width:120px;accent-color:#c9a227;cursor:pointer;';
  slider.title = `Similarity threshold (Hamming distance 0–${max}) — lower = stricter match`;

  const valueEl = document.createElement('span');
  valueEl.style.cssText = 'min-width:2.5em;text-align:right;white-space:nowrap;';
  valueEl.textContent = String(initial);

  // Update value label live while dragging, re-query only on release
  slider.addEventListener('input', () => {
    valueEl.textContent = slider.value;
  });
  slider.addEventListener('change', () => {
    onChange(parseInt(slider.value, 10));
  });

  el.append(label, slider, valueEl);
  return { element: el, getValue: () => parseInt(slider.value, 10) };
}

// ── Public API ─────────────────────────────────────────────────────────────────

/**
 * Open a nested review modal showing images similar to the given image UUID.
 * The modal appears immediately; results load asynchronously inside the panel.
 *
 * @param {{ imageUuid: string, threshold: number }} params
 */
export async function openSimilarImagesModal({ imageUuid, threshold = null }) {
  const sliderMax = await getImageHashMax();
  let _threshold = threshold !== null ? threshold : await _readThreshold('dedup_image_threshold', sliderMax);

  // Sorted match list — [{imageUuid, distance}] ascending by distance.
  // Null until the first query resolves; reset when threshold changes.
  let _sortedMatches = null;
  let _queryPromise = findSimilarImagesByUuid(imageUuid, _threshold);
  let _refresh = null;

  async function _ensureMatches() {
    if (_sortedMatches !== null) return;
    // Backend already returns results sorted by distance ascending.
    _sortedMatches = await _queryPromise;
  }

  const { element: sliderEl } = buildThresholdSlider(_threshold, {
    max: sliderMax,
    onChange(newThreshold) {
      _threshold = newThreshold;
      _sortedMatches = null;
      _queryPromise = findSimilarImagesByUuid(imageUuid, _threshold);
      _refresh?.();
    },
  });

  _refresh = await showReviewModal({
    title: 'Similar Images',
    toolbarControls: sliderEl,
    initialViewFilter: 'images',
    referenceImageUuids: new Set([imageUuid]),
    fetchImages: async ({ afterUuid = null, limit }) => {
      await _ensureMatches();
      if (!_sortedMatches.length) return [];

      const startIdx = pageStartIdx(_sortedMatches, afterUuid);
      const page = _sortedMatches.slice(startIdx, startIdx + limit);
      if (!page.length) return [];

      const images = await fetchImagesByUuids(page.map(m => m.imageUuid));
      // Re-order to match distance sort (fetchImagesByUuids returns in arbitrary order)
      const orderMap = new Map(page.map((m, i) => [m.imageUuid, i]));
      return images.sort((a, b) => (orderMap.get(a.uuid) ?? 9999) - (orderMap.get(b.uuid) ?? 9999));
    },
  });
}

/**
 * Open a nested review modal showing images/bboxes similar to the given bbox UUID.
 * The modal appears immediately; results load asynchronously inside the panel.
 *
 * @param {{ bboxUuid: string, threshold: number, labelUuid?: string|null }} params
 */
export async function openSimilarBboxesModal({ bboxUuid, bboxCoords, imageUuid, threshold = null, labelUuid = null }) {
  const [sliderMax, tagsMap, labelsMap] = await Promise.all([
    getBboxHashMax(),
    fetchTags().catch(() => new Map()),
    fetchLabels().catch(() => new Map()),
  ]);
  let _threshold = threshold !== null ? threshold : await _readThreshold('dedup_bbox_threshold', sliderMax);

  // Label + tag filter state
  let _labelUuids = labelUuid ? [labelUuid] : [];
  let _tagUuids = [];
  let _excludeTagUuids = [];
  let _tagMatchMode = 'any';

  // Use coords only when the bbox is dirty (moved/resized since last save, or brand-new).
  // getBboxCoords() returns null for clean (saved, unmodified) bboxes, which forces the
  // UUID path and guarantees the source bbox appears at distance 0 in results.
  // For dirty bboxes the coords path searches by the current crop position, which is correct.
  const _query = (t) => bboxCoords && imageUuid
    ? findSimilarBboxesByCoords(imageUuid, bboxCoords.x, bboxCoords.y, bboxCoords.w, bboxCoords.h, t, _labelUuids, _tagUuids, _excludeTagUuids, _tagMatchMode)
    : findSimilarBboxesByUuid(bboxUuid, t, _labelUuids, _tagUuids, _excludeTagUuids, _tagMatchMode);

  let _queryPromise = _query(_threshold);
  let _refresh = null;

  // Derived from the match list; reset when threshold or tag filter changes.
  // _imageOrder: [{imageUuid, minDist}] sorted ascending by minDist
  // _imageBboxes: Map<imageUuid, Set<bboxUuid>>
  let _imageOrder = null;
  let _imageBboxes = null;

  function _invalidate() {
    _imageOrder = null;
    _imageBboxes = null;
    _queryPromise = _query(_threshold);
    _refresh?.();
  }

  async function _ensureMatches() {
    if (_imageOrder !== null) return;
    const matches = await _queryPromise;

    const imageMinDist = new Map();   // imageUuid → minimum distance across its matching bboxes
    const imageBboxes = new Map();    // imageUuid → Set<bboxUuid>

    for (const { bboxUuid: bu, imageUuid: iu, distance } of matches) {
      if (!iu) continue;
      if (!imageBboxes.has(iu)) {
        imageBboxes.set(iu, new Set());
        imageMinDist.set(iu, distance);
      } else {
        imageMinDist.set(iu, Math.min(imageMinDist.get(iu), distance));
      }
      imageBboxes.get(iu).add(bu);
    }

    _imageBboxes = imageBboxes;
    _imageOrder = [...imageMinDist.entries()]
      .sort((a, b) => a[1] - b[1])
      .map(([iu, minDist]) => ({ imageUuid: iu, minDist }));
  }

  const { element: sliderEl } = buildThresholdSlider(_threshold, {
    max: sliderMax,
    onChange(newThreshold) {
      _threshold = newThreshold;
      _invalidate();
    },
  });

  // ── Label filter dropdown ──────────────────────────────────────────────────
  const labelItems = [...labelsMap.values()].map(l => ({ uuid: l.uuid, name: l.name, color: l.color ?? null }));
  const labelsDropdown = createMultiSelectDropdown({
    label: 'Labels',
    items: labelItems,
    defaultSelected: _labelUuids,
    showModeToggle: false,  // label filter is always OR (each label maps to its own BK-tree)
    onChange(selected) {
      _labelUuids = selected;
      _invalidate();
    },
  });

  // ── Tag filter dropdown ────────────────────────────────────────────────────
  const tagItems = [...tagsMap.values()].map(t => ({ uuid: t.uuid, name: t.name, color: t.color ?? null }));
  const tagsDropdown = createMultiSelectDropdown({
    label: 'Tags',
    items: tagItems,
    showModeToggle: true,
    onChange(selected, excluded, mode) {
      _tagUuids = selected;
      _excludeTagUuids = excluded;
      _tagMatchMode = mode;
      _invalidate();
    },
  });

  // Wrap slider + label + tag dropdowns side-by-side
  const headerEl = document.createElement('div');
  headerEl.style.cssText = 'display:flex;align-items:center;gap:0.75em;flex-wrap:wrap;';
  headerEl.append(sliderEl, labelsDropdown.element, tagsDropdown.element);

  _refresh = await showReviewModal({
    title: 'Similar Bboxes',
    toolbarControls: headerEl,
    referenceBboxUuids: new Set([bboxUuid]),
    fetchImages: async ({ afterUuid = null, limit }) => {
      await _ensureMatches();
      if (!_imageOrder.length) return [];

      const startIdx = pageStartIdx(_imageOrder, afterUuid);
      const page = _imageOrder.slice(startIdx, startIdx + limit);
      if (!page.length) return [];

      const images = await fetchImagesByUuids(page.map(e => e.imageUuid));

      // Re-order by distance and narrow each image's bboxes to only matched ones
      const orderMap = new Map(page.map((e, i) => [e.imageUuid, i]));
      return images
        .sort((a, b) => (orderMap.get(a.uuid) ?? 9999) - (orderMap.get(b.uuid) ?? 9999))
        .map(img => ({
          ...img,
          bboxes: (img.bboxes || []).filter(b => _imageBboxes.get(img.uuid)?.has(b.uuid)),
        }));
    },
  });
}

// ── Legacy shim ────────────────────────────────────────────────────────────────

/** @deprecated Use openSimilarImagesModal / openSimilarBboxesModal directly. */
export async function openSimilarSearch({ kind, imageUuid, bboxUuid, threshold }) {
  if (kind === 'image') return openSimilarImagesModal({ imageUuid, threshold });
  return openSimilarBboxesModal({ bboxUuid, threshold });
}
