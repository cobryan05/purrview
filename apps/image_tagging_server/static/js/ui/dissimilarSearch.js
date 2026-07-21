/**
 * Dissimilar-search helpers — open a nested review modal showing the most
 * visually distinct images/bboxes relative to an anchor, using greedy
 * farthest-point sampling on perceptual hashes.
 *
 * Usage:
 *   openDissimilarImagesModal({ imageUuid })              — single anchor (backward-compat)
 *   openDissimilarImagesModal({ imageUuids: [...] })      — multi-anchor
 *   openDissimilarBboxesModal({ bboxUuid, labelUuid? })   — single anchor (backward-compat)
 *   openDissimilarBboxesModal({ bboxes: [...], labelUuids?: [...] }) — multi-anchor
 *   openDissimilarGlobalModal()
 */

import { showReviewModal } from '/app-static/js/app/review/main.js';
import {
  findDissimilarImagesByUuid,
  findDissimilarBboxesByUuid,
  findDissimilarGlobal,
  fetchImagesByUuids,
} from '/app-static/js/shared/api/images.js';
import { fetchLabels } from '/app-static/js/shared/api/labels.js';
import { createMultiSelectDropdown } from '/app-static/js/ui/utils/index.js';
import { pageStartIdx, buildCountControl, buildSwitchModeButton } from '/app-static/js/ui/searchUtils.js';

// ── Multi-anchor query helpers ─────────────────────────────────────────────────

/**
 * Fetch dissimilar images for one or more anchor UUIDs.
 * For multiple anchors: queries each anchor in parallel, then merges by minimum rank
 * (images that rank near 0 for any anchor are considered most distinct from all anchors).
 *
 * @param {string[]} anchorUuids
 * @param {number} count
 * @param {string[]|null} [candidateUuids] Restrict the diversity walk to this UUID set.
 */
function _fetchDissimilarImages(anchorUuids, count, candidateUuids = null) {
  if (anchorUuids.length === 1) return findDissimilarImagesByUuid(anchorUuids[0], count, candidateUuids);
  return Promise.all(anchorUuids.map(uuid => findDissimilarImagesByUuid(uuid, count, candidateUuids))).then(allResults => {
    const bestRank = new Map(); // imageUuid → best (min) rank across all queries
    for (const results of allResults) {
      for (const { imageUuid: mu, rank } of results) {
        if (!bestRank.has(mu) || bestRank.get(mu) > rank) bestRank.set(mu, rank);
      }
    }
    return [...bestRank.entries()]
      .sort((a, b) => a[1] - b[1])
      .map(([iu], i) => ({ imageUuid: iu, rank: i }));
  });
}

/**
 * Fetch dissimilar bboxes for one or more anchor bbox UUIDs.
 * Merges results the same way as _fetchDissimilarImages.
 */
function _fetchDissimilarBboxes(anchorBboxUuids, count, labelUuids) {
  if (anchorBboxUuids.length === 1) return findDissimilarBboxesByUuid(anchorBboxUuids[0], count, labelUuids);
  return Promise.all(anchorBboxUuids.map(uuid => findDissimilarBboxesByUuid(uuid, count, labelUuids))).then(allResults => {
    const bestEntry = new Map(); // bboxUuid → {rank, imageUuid, labelUuid}
    for (const results of allResults) {
      for (const { bboxUuid: bu, imageUuid: iu, rank, labelUuid: lu } of results) {
        if (!bestEntry.has(bu) || bestEntry.get(bu).rank > rank) {
          bestEntry.set(bu, { rank, imageUuid: iu, labelUuid: lu });
        }
      }
    }
    return [...bestEntry.entries()]
      .sort((a, b) => a[1].rank - b[1].rank)
      .map(([bu, { imageUuid: iu, labelUuid: lu }], i) => ({ bboxUuid: bu, imageUuid: iu, rank: i, labelUuid: lu }));
  });
}

// ── Shared ranked-images modal ─────────────────────────────────────────────────

/**
 * Open a review modal showing a ranked list of images from a diversity query.
 *
 * @param {object} opts
 * @param {string} opts.title
 * @param {function(number): Promise<Array<{imageUuid: string, rank: number}>>} opts.makeFetchFn
 *   Called with the current count; returns a promise for the sorted match list.
 * @param {Set<string>|null} [opts.referenceImageUuids]
 * @param {Element|null} [opts.extraControls]  Additional toolbar controls to append after the count input.
 */
async function _openRankedImagesModal({ title, makeFetchFn, referenceImageUuids = null, extraControls = null }) {
  let _count = 100;
  let _sortedMatches = null;
  let _queryGen = 0;
  let _queryPromise = makeFetchFn(_count);
  let _refresh = null;

  // Loops rather than awaiting once so a query superseded mid-flight by a
  // count change doesn't clobber fresher state with stale results.
  async function _ensureMatches() {
    while (_sortedMatches === null) {
      const gen = _queryGen;
      const result = await _queryPromise;  // already ordered by rank ascending
      if (gen === _queryGen) _sortedMatches = result;
    }
  }

  const { element: countEl } = buildCountControl(_count, {
    onChange(newCount) {
      _count = newCount;
      _sortedMatches = null;
      _queryGen++;
      _queryPromise = makeFetchFn(_count);
      _refresh?.();
    },
  });

  let toolbarEl;
  if (extraControls) {
    toolbarEl = document.createElement('div');
    toolbarEl.className = 'modal-toolbar-row';
    toolbarEl.append(countEl, extraControls);
  } else {
    toolbarEl = countEl;
  }

  _refresh = await showReviewModal({
    title,
    toolbarControls: toolbarEl,
    initialViewFilter: 'images',
    ...(referenceImageUuids ? { referenceImageUuids } : {}),
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
      // Re-order to match rank (fetchImagesByUuids returns in arbitrary order)
      const orderMap = new Map(page.map((m, i) => [m.imageUuid, i]));
      return images.sort((a, b) => (orderMap.get(a.uuid) ?? 9999) - (orderMap.get(b.uuid) ?? 9999));
    },
  });
}

// ── Public API ─────────────────────────────────────────────────────────────────

/**
 * Open a review modal showing images most visually distinct from the given anchor(s).
 *
 * @param {{ imageUuid?: string, imageUuids?: string[], candidateUuids?: string[]|null }} params
 *   Pass `imageUuid` (string) for a single anchor or `imageUuids` (array) for multiple.
 *   `candidateUuids`, when given, restricts the diversity walk to that UUID set.
 */
export async function openDissimilarImagesModal({ imageUuid = null, imageUuids = null, candidateUuids = null }) {
  const anchorUuids = imageUuids ?? (imageUuid ? [imageUuid] : []);
  if (!anchorUuids.length) return;

  const similarBtn = buildSwitchModeButton('← Most Similar', async () => {
    const { openSimilarImagesModal } = await import('/app-static/js/ui/similarSearch.js');
    openSimilarImagesModal({ imageUuids: anchorUuids, candidateUuids });
  });

  return _openRankedImagesModal({
    title: candidateUuids ? 'Most Distinct Images (filtered)' : 'Most Distinct Images',
    makeFetchFn: count => _fetchDissimilarImages(anchorUuids, count, candidateUuids),
    referenceImageUuids: new Set(anchorUuids),
    extraControls: similarBtn,
  });
}

/**
 * Open a review modal showing bboxes most visually distinct from the given anchor(s).
 *
 * @param {{ bboxUuid?: string, bboxes?: Array<{bboxUuid: string}>, labelUuid?: string, labelUuids?: string[] }} params
 *   Pass `bboxUuid` (string) for a single anchor or `bboxes` (array) for multiple.
 *   `labelUuid` / `labelUuids` seed the label filter (backward-compat: prefer `labelUuids`).
 */
export async function openDissimilarBboxesModal({ bboxUuid = null, bboxes = null, labelUuid = null, labelUuids = null }) {
  const anchorBboxUuids = bboxes
    ? bboxes.map(b => b.bboxUuid).filter(Boolean)
    : (bboxUuid ? [bboxUuid] : []);
  if (!anchorBboxUuids.length) return;

  const labelsMap = await fetchLabels().catch(() => new Map());

  let _count = 100;
  let _labelUuids = labelUuids ?? (labelUuid ? [labelUuid] : []);
  let _refresh = null;

  // Derived from match list; reset when count or label filter changes.
  // _imageOrder: [{imageUuid, minRank}] sorted ascending by minRank
  // _imageBboxes: Map<imageUuid, Set<bboxUuid>>
  let _imageOrder = null;
  let _imageBboxes = null;
  let _queryGen = 0;
  let _queryPromise = _fetchDissimilarBboxes(anchorBboxUuids, _count, _labelUuids);

  function _invalidate() {
    _imageOrder = null;
    _imageBboxes = null;
    _queryGen++;
    _queryPromise = _fetchDissimilarBboxes(anchorBboxUuids, _count, _labelUuids);
    _refresh?.();
  }

  // Loops rather than awaiting once so a query superseded mid-flight by
  // _invalidate() doesn't clobber fresher state with stale results.
  async function _ensureMatches() {
    while (_imageOrder === null) {
      const gen = _queryGen;
      const matches = await _queryPromise;
      if (gen !== _queryGen) continue;

      const imageMinRank = new Map();   // imageUuid → minimum rank across its matching bboxes
      const imageBboxes = new Map();    // imageUuid → Set<bboxUuid>

      for (const { bboxUuid: bu, imageUuid: iu, rank } of matches) {
        if (!iu) continue;
        if (!imageBboxes.has(iu)) {
          imageBboxes.set(iu, new Set());
          imageMinRank.set(iu, rank);
        } else {
          imageMinRank.set(iu, Math.min(imageMinRank.get(iu), rank));
        }
        imageBboxes.get(iu).add(bu);
      }

      _imageBboxes = imageBboxes;
      _imageOrder = [...imageMinRank.entries()]
        .sort((a, b) => a[1] - b[1])
        .map(([iu, minRank]) => ({ imageUuid: iu, minRank }));
    }
  }

  // ── Count control ──────────────────────────────────────────────────────────
  const { element: countEl } = buildCountControl(_count, {
    onChange(newCount) {
      _count = newCount;
      _invalidate();
    },
  });

  // ── Label filter dropdown ──────────────────────────────────────────────────
  const labelItems = [...labelsMap.values()].map(l => ({ uuid: l.uuid, name: l.name, color: l.color ?? null }));
  const labelsDropdown = createMultiSelectDropdown({
    label: 'Labels',
    items: labelItems,
    defaultSelected: _labelUuids,
    showModeToggle: false,
    onChange(selected) {
      _labelUuids = selected;
      _invalidate();
    },
  });

  // ── Switch-to-similar button ───────────────────────────────────────────────
  const similarBtn = buildSwitchModeButton('← Most Similar', async () => {
    const { openSimilarBboxesModal } = await import('/app-static/js/ui/similarSearch.js');
    openSimilarBboxesModal({
      bboxes: anchorBboxUuids.map(uuid => ({ bboxUuid: uuid })),
      labelUuids: _labelUuids,
    });
  });

  const headerEl = document.createElement('div');
  headerEl.className = 'modal-toolbar-row';
  headerEl.append(countEl, labelsDropdown.element, similarBtn);

  _refresh = await showReviewModal({
    title: 'Most Distinct Bboxes',
    toolbarControls: headerEl,
    initialViewFilter: 'bboxes',
    referenceBboxUuids: new Set(anchorBboxUuids),
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

      // Re-order by rank and narrow each image's bboxes to only matched ones
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

/**
 * Open a review modal showing the most visually distinct unverified images.
 * Uses server-side unverified-tag scoping; no anchor image needed.
 */
export async function openDissimilarGlobalModal() {
  return _openRankedImagesModal({
    title: 'Label Most Unique First',
    makeFetchFn: count => findDissimilarGlobal(count),
  });
}
