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
 * *   openSimilarImagesModal({ imageUuids, threshold? })
 *   openSimilarBboxesModal({ bboxes, threshold? })
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
import { pageStartIdx, buildSwitchModeButton } from '/app-static/js/ui/searchUtils.js';
import { openDissimilarImagesModal, openDissimilarBboxesModal } from '/app-static/js/ui/dissimilarSearch.js';

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
  el.className = 'modal-toolbar-control';

  const label = document.createElement('span');
  label.className = 'modal-toolbar-control__label';
  label.textContent = 'Threshold:';

  const slider = document.createElement('input');
  slider.type = 'range';
  slider.min = String(min);
  slider.max = String(max);
  slider.step = '1';
  slider.value = String(initial);
  slider.className = 'modal-threshold-slider';
  slider.title = `Similarity threshold (Hamming distance 0–${max}) — lower = stricter match`;

  const valueEl = document.createElement('span');
  valueEl.className = 'modal-threshold-value';
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
 * @param {{ imageUuid: string, threshold: number, candidateUuids?: string[]|null }} params
 *   `candidateUuids`, when given, restricts results to that UUID set (the similar-search
 *   backend has no candidate-set param, so this is a client-side post-filter). The
 *   anchor images are always kept even if absent from the set — see NOTES.md: excluding
 *   the query image at the search layer breaks the single-item-dataset case.
 */
export async function openSimilarImagesModal({ imageUuids, threshold = null, candidateUuids = null }) {
  if (!imageUuids?.length) return;

  const candidateSet = candidateUuids ? new Set(candidateUuids) : null;
  const sliderMax = await getImageHashMax();
  let _threshold = threshold !== null ? threshold : await _readThreshold('dedup_image_threshold', sliderMax);

  // Sorted match list — [{imageUuid, distance}] ascending by distance.
  // Null until the first query resolves; reset when threshold changes.
  let _sortedMatches = null;

  function _runQuery(t) {
    return Promise.all(imageUuids.map(uuid => findSimilarImagesByUuid(uuid, t))).then(allResults => {
      // Union by imageUuid, keeping minimum distance across all anchor queries.
      const byUuid = new Map();
      for (const results of allResults) {
        for (const { imageUuid: mu, distance } of results) {
          if (!byUuid.has(mu) || byUuid.get(mu) > distance) byUuid.set(mu, distance);
        }
      }
      let entries = [...byUuid.entries()].map(([iu, distance]) => ({ imageUuid: iu, distance }));
      if (candidateSet) {
        entries = entries.filter(e => candidateSet.has(e.imageUuid) || imageUuids.includes(e.imageUuid));
      }
      return entries.sort((a, b) => a.distance - b.distance);
    });
  }

  let _queryGen = 0;
  let _queryPromise = _runQuery(_threshold);
  let _refresh = null;

  // Loops rather than awaiting once so a query that's superseded mid-flight
  // (onChange fired while we were awaiting) doesn't clobber a fresher result
  // with stale data — we keep retrying until we land on the current generation.
  async function _ensureMatches() {
    while (_sortedMatches === null) {
      const gen = _queryGen;
      const result = await _queryPromise;
      if (gen === _queryGen) _sortedMatches = result;
    }
  }

  const { element: sliderEl } = buildThresholdSlider(_threshold, {
    max: sliderMax,
    onChange(newThreshold) {
      _threshold = newThreshold;
      _sortedMatches = null;
      _queryGen++;
      _queryPromise = _runQuery(_threshold);
      _refresh?.();
    },
  });

  const dissimilarBtn = buildSwitchModeButton('Most Distinct →', () => {
    openDissimilarImagesModal({ imageUuids, candidateUuids });
  });

  const imgHeaderEl = document.createElement('div');
  imgHeaderEl.className = 'modal-toolbar-row';
  imgHeaderEl.append(sliderEl, dissimilarBtn);

  _refresh = await showReviewModal({
    title: candidateSet ? 'Similar Images (filtered)' : 'Similar Images',
    toolbarControls: imgHeaderEl,
    initialViewFilter: 'images',
    referenceImageUuids: new Set(imageUuids),
    fetchImages: async ({ afterUuid = null, limit }) => {
      await _ensureMatches();
      // Snapshot locally: a second await below yields control back to the
      // event loop, where onChange could reset _sortedMatches out from under us.
      const matches = _sortedMatches;
      if (!matches.length) return [];

      const startIdx = pageStartIdx(matches, afterUuid);
      const page = matches.slice(startIdx, startIdx + limit);
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
 * Label/Tag filtering used to be pushed server-side into the BK-tree query
 * here (`findSimilarBboxesByUuid(..., labelUuids, tagUuids, ...)`). As of M1
 * ("unify Review filters") that filtering moved to the panel's shared curated
 * result-filter surface (client-side predicate over the fetched candidates —
 * see `createReviewPanel`'s `_curated` state), so this modal no longer owns
 * or accepts label/tag state; only the threshold slider (a real query param,
 * not a result-filter) remains here.
 *
 * @param {{ bboxUuid: string, threshold: number }} params
 */
export async function openSimilarBboxesModal({ bboxes, threshold = null }) {
  // bboxes: [{ bboxUuid, bboxCoords, imageUuid }]
  // bboxCoords present means the bbox is dirty (moved/resized since save); use coords path.
  // bboxUuid-only uses the saved UUID path, which ensures the source appears at distance 0.
  if (!bboxes?.length) return;

  const sliderMax = await getBboxHashMax();
  let _threshold = threshold !== null ? threshold : await _readThreshold('dedup_bbox_threshold', sliderMax);

  function _runQuery(t) {
    return Promise.all(bboxes.map(({ bboxUuid, bboxCoords, imageUuid }) =>
      bboxCoords && imageUuid
        ? findSimilarBboxesByCoords(imageUuid, bboxCoords.x, bboxCoords.y, bboxCoords.w, bboxCoords.h, t)
        : findSimilarBboxesByUuid(bboxUuid, t)
    )).then(allResults => {
      // Union by bboxUuid, keeping minimum distance across all anchor queries.
      const byBboxUuid = new Map(); // bboxUuid → { imageUuid, distance }
      for (const results of allResults) {
        for (const { bboxUuid: bu, imageUuid: iu, distance } of results) {
          if (!byBboxUuid.has(bu) || byBboxUuid.get(bu).distance > distance) {
            byBboxUuid.set(bu, { imageUuid: iu, distance });
          }
        }
      }
      return [...byBboxUuid.entries()].map(([bu, { imageUuid: iu, distance }]) => ({
        bboxUuid: bu, imageUuid: iu, distance,
      }));
    });
  }

  let _queryGen = 0;
  let _queryPromise = _runQuery(_threshold);
  let _refresh = null;

  // Derived from the match list; reset when the threshold changes.
  // _imageOrder: [{imageUuid, minDist}] sorted ascending by minDist
  // _imageBboxes: Map<imageUuid, Set<bboxUuid>>
  let _imageOrder = null;
  let _imageBboxes = null;

  function _invalidate() {
    _imageOrder = null;
    _imageBboxes = null;
    _queryGen++;
    _queryPromise = _runQuery(_threshold);
    _refresh?.();
  }

  // Loops rather than awaiting once so a query superseded mid-flight by
  // _invalidate() doesn't clobber fresher state with stale results.
  async function _ensureMatches() {
    while (_imageOrder === null) {
      const gen = _queryGen;
      const matches = await _queryPromise;
      if (gen !== _queryGen) continue;

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
  }

  const { element: sliderEl } = buildThresholdSlider(_threshold, {
    max: sliderMax,
    onChange(newThreshold) {
      _threshold = newThreshold;
      _invalidate();
    },
  });

  // ── Switch-to-dissimilar button ────────────────────────────────────────────
  const bboxAnchorUuids = bboxes.map(b => b.bboxUuid).filter(Boolean);
  const dissimilarBboxBtn = buildSwitchModeButton('Most Distinct →', () => {
    openDissimilarBboxesModal({ bboxes });
  });
  // Only show the button when there is at least one saved bbox UUID to anchor from
  if (!bboxAnchorUuids.length) dissimilarBboxBtn.disabled = true;

  // Wrap slider + switch button side-by-side. Label/Tag filtering now comes
  // from the panel's own curated result-filter surface (M1) — see the module
  // doc comment on openSimilarBboxesModal above.
  const headerEl = document.createElement('div');
  headerEl.className = 'modal-toolbar-row';
  headerEl.append(sliderEl, dissimilarBboxBtn);

  const referenceBboxUuids = new Set(bboxes.map(b => b.bboxUuid).filter(Boolean));

  _refresh = await showReviewModal({
    title: 'Similar Bboxes',
    toolbarControls: headerEl,
    initialViewFilter: 'bboxes',
    referenceBboxUuids,
    fetchImages: async ({ afterUuid = null, limit }) => {
      await _ensureMatches();
      // Snapshot locally: the await below can yield to an _invalidate() call
      // that resets the outer _imageOrder/_imageBboxes to null mid-flight.
      const order = _imageOrder;
      const bboxesByImage = _imageBboxes;
      if (!order.length) return [];

      const startIdx = pageStartIdx(order, afterUuid);
      const page = order.slice(startIdx, startIdx + limit);
      if (!page.length) return [];

      const images = await fetchImagesByUuids(page.map(e => e.imageUuid));

      // Re-order by distance and narrow each image's bboxes to only matched ones
      const orderMap = new Map(page.map((e, i) => [e.imageUuid, i]));
      return images
        .sort((a, b) => (orderMap.get(a.uuid) ?? 9999) - (orderMap.get(b.uuid) ?? 9999))
        .map(img => ({
          ...img,
          bboxes: (img.bboxes || []).filter(b => bboxesByImage.get(img.uuid)?.has(b.uuid)),
        }));
    },
  });
}

// ── Legacy shim ────────────────────────────────────────────────────────────────

/** @deprecated Use openSimilarImagesModal / openSimilarBboxesModal directly. */
export async function openSimilarSearch({ kind, imageUuids, bboxes, threshold }) {
  if (kind === 'image') return openSimilarImagesModal({ imageUuids, threshold });
  return openSimilarBboxesModal({ bboxes, threshold });
}
