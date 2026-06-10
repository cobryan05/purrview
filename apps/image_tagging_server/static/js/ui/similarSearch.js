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
 *   openSimilarImagesModal({ imageUuid, threshold })
 *   openSimilarBboxesModal({ bboxUuid, threshold, labelUuid? })
 */

import { showReviewModal } from '/app-static/js/app/review/main.js';
import {
  findSimilarImagesByUuid,
  findSimilarBboxesByUuid,
  fetchImagesByUuids,
} from '/app-static/js/shared/api/images.js';

// ── Public API ─────────────────────────────────────────────────────────────────

/**
 * Open a nested review modal showing images similar to the given image UUID.
 * The modal appears immediately; results load asynchronously inside the panel.
 *
 * @param {{ imageUuid: string, threshold: number }} params
 */
export async function openSimilarImagesModal({ imageUuid, threshold }) {
  // Kick off the search now; fetchImages will await it (cached after first call).
  const matchesPromise = findSimilarImagesByUuid(imageUuid, threshold);
  let cachedImages = null;

  await showReviewModal({
    title: `Similar Images (threshold ≤ ${threshold})`,
    fetchImages: async () => {
      if (cachedImages !== null) return cachedImages;
      const matches = await matchesPromise;
      if (!matches.length) { cachedImages = []; return []; }
      const imageUuids = matches.map(m => m.imageUuid);
      const images = await fetchImagesByUuids(imageUuids);
      // Sort by similarity distance (most similar first)
      const orderMap = new Map(matches.map((m, i) => [m.imageUuid, i]));
      images.sort((a, b) => (orderMap.get(a.uuid) ?? 9999) - (orderMap.get(b.uuid) ?? 9999));
      cachedImages = images;
      return cachedImages;
    },
  });
}

/**
 * Open a nested review modal showing images/bboxes similar to the given bbox UUID.
 * The modal appears immediately; results load asynchronously inside the panel.
 *
 * @param {{ bboxUuid: string, threshold: number, labelUuid?: string|null }} params
 */
export async function openSimilarBboxesModal({ bboxUuid, threshold, labelUuid = null }) {
  const matchesPromise = findSimilarBboxesByUuid(bboxUuid, threshold, labelUuid);
  let cachedImages = null;

  await showReviewModal({
    title: `Similar Bboxes (threshold ≤ ${threshold})`,
    fetchImages: async () => {
      if (cachedImages !== null) return cachedImages;
      const matches = await matchesPromise;
      if (!matches.length) { cachedImages = []; return []; }

      const imageUuidSet = new Set(matches.filter(m => m.imageUuid).map(m => m.imageUuid));
      if (!imageUuidSet.size) { cachedImages = []; return []; }

      const matchedBboxUuids = new Set(matches.map(m => m.bboxUuid));
      const images = await fetchImagesByUuids([...imageUuidSet]);

      // Narrow each image's bbox list to only the matched bboxes so the review
      // panel highlights exactly the relevant annotations.
      cachedImages = images.map(img => ({
        ...img,
        bboxes: (img.bboxes || []).filter(b => matchedBboxUuids.has(b.uuid)),
      }));
      return cachedImages;
    },
  });
}

// ── Legacy shim ────────────────────────────────────────────────────────────────

/** @deprecated Use openSimilarImagesModal / openSimilarBboxesModal directly. */
export async function openSimilarSearch({ kind, imageUuid, bboxUuid, threshold }) {
  if (kind === 'image') return openSimilarImagesModal({ imageUuid, threshold });
  return openSimilarBboxesModal({ bboxUuid, threshold });
}
