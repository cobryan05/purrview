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

async function _imageHashMax() {
  const v = await getSetting('dedup_image_hash_size').catch(() => null);
  const n = parseInt(v, 10) || 8;
  return n * n;
}

/** @returns {Promise<number>} */
export async function getImageHashMax() { return _imageHashMax(); }

async function _bboxHashMax() {
  const v = await getSetting('dedup_bbox_hash_size').catch(() => null);
  const n = parseInt(v, 10) || 16;
  return n * n;
}

/** @returns {Promise<number>} */
export async function getBboxHashMax() { return _bboxHashMax(); }

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

/** @returns {Promise<number>} */
export async function getImageDefaultThreshold() {
  return _readThreshold('dedup_image_threshold', await _imageHashMax());
}

/** @returns {Promise<number>} */
export async function getBboxDefaultThreshold() {
  return _readThreshold('dedup_bbox_threshold', await _bboxHashMax());
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
  const sliderMax = await _imageHashMax();
  let _threshold = threshold !== null ? threshold : await _readThreshold('dedup_image_threshold', sliderMax);
  let _cachedImages = null;
  let _queryPromise = findSimilarImagesByUuid(imageUuid, _threshold);
  let _refresh = null;

  const { element: sliderEl } = buildThresholdSlider(_threshold, {
    max: sliderMax,
    onChange(newThreshold) {
      _threshold = newThreshold;
      _cachedImages = null;
      _queryPromise = findSimilarImagesByUuid(imageUuid, _threshold);
      _refresh?.();
    },
  });

  _refresh = await showReviewModal({
    title: 'Similar Images',
    headerControls: sliderEl,
    initialViewFilter: 'images',
    referenceImageUuids: new Set([imageUuid]),
    fetchImages: async () => {
      if (_cachedImages !== null) return _cachedImages;
      const matches = await _queryPromise;
      if (!matches.length) { _cachedImages = []; return []; }
      const imageUuids = matches.map(m => m.imageUuid);
      const images = await fetchImagesByUuids(imageUuids);
      // Sort by similarity distance (most similar first — reference at distance 0 will be first)
      const orderMap = new Map(matches.map((m, i) => [m.imageUuid, i]));
      images.sort((a, b) => (orderMap.get(a.uuid) ?? 9999) - (orderMap.get(b.uuid) ?? 9999));
      _cachedImages = images;
      return _cachedImages;
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
  const sliderMax = await _bboxHashMax();
  let _threshold = threshold !== null ? threshold : await _readThreshold('dedup_bbox_threshold', sliderMax);
  let _cachedImages = null;

  // If live coordinates are provided (e.g. bbox moved on canvas but not yet saved),
  // compute the phash server-side from those coords; otherwise fall back to UUID lookup.
  const _query = (t) => bboxCoords && imageUuid
    ? findSimilarBboxesByCoords(imageUuid, bboxCoords.x, bboxCoords.y, bboxCoords.w, bboxCoords.h, t, labelUuid)
    : findSimilarBboxesByUuid(bboxUuid, t, labelUuid);

  let _queryPromise = _query(_threshold);
  let _refresh = null;

  const { element: sliderEl } = buildThresholdSlider(_threshold, {
    max: sliderMax,
    onChange(newThreshold) {
      _threshold = newThreshold;
      _cachedImages = null;
      _queryPromise = _query(_threshold);
      _refresh?.();
    },
  });

  _refresh = await showReviewModal({
    title: 'Similar Bboxes',
    headerControls: sliderEl,
    referenceBboxUuids: new Set([bboxUuid]),
    fetchImages: async () => {
      if (_cachedImages !== null) return _cachedImages;
      const matches = await _queryPromise;
      if (!matches.length) { _cachedImages = []; return []; }

      const imageUuidSet = new Set(matches.filter(m => m.imageUuid).map(m => m.imageUuid));
      if (!imageUuidSet.size) { _cachedImages = []; return []; }

      const matchedBboxUuids = new Set(matches.map(m => m.bboxUuid));
      const images = await fetchImagesByUuids([...imageUuidSet]);

      // Narrow each image's bbox list to only the matched bboxes so the review
      // panel highlights exactly the relevant annotations.
      _cachedImages = images.map(img => ({
        ...img,
        bboxes: (img.bboxes || []).filter(b => matchedBboxUuids.has(b.uuid)),
      }));
      return _cachedImages;
    },
  });
}

// ── Legacy shim ────────────────────────────────────────────────────────────────

/** @deprecated Use openSimilarImagesModal / openSimilarBboxesModal directly. */
export async function openSimilarSearch({ kind, imageUuid, bboxUuid, threshold }) {
  if (kind === 'image') return openSimilarImagesModal({ imageUuid, threshold });
  return openSimilarBboxesModal({ bboxUuid, threshold });
}
