/**
 * Dissimilar-search helpers — open a nested review modal showing the most
 * visually distinct images/bboxes relative to an anchor, using greedy
 * farthest-point sampling on perceptual hashes.
 *
 * Usage:
 *   openDissimilarImagesModal({ imageUuid })
 *   openDissimilarBboxesModal({ bboxUuid, labelUuid? })
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
import { pageStartIdx } from '/app-static/js/ui/searchUtils.js';

// ── Count control ──────────────────────────────────────────────────────────────

/**
 * Build a "Top N" number input control.
 * Returns { element, getValue }.
 * `onChange` is called when the value changes (on blur / Enter).
 */
function buildCountControl(initial, { min = 10, max = 500, onChange }) {
  const el = document.createElement('div');
  el.style.cssText = 'display:flex;align-items:center;gap:0.5em;font-size:0.85em;color:#aaa;margin-left:0.5em;';

  const label = document.createElement('span');
  label.style.cssText = 'white-space:nowrap;';
  label.textContent = 'Top N:';

  const input = document.createElement('input');
  input.type = 'number';
  input.min = String(min);
  input.max = String(max);
  input.step = '10';
  input.value = String(initial);
  input.style.cssText = 'width:70px;accent-color:#c9a227;';
  input.title = `Number of most-distinct results to return (${min}–${max})`;

  function _emit() {
    const v = Math.max(min, Math.min(max, parseInt(input.value, 10) || initial));
    input.value = String(v);
    onChange(v);
  }
  input.addEventListener('change', _emit);
  input.addEventListener('keydown', e => { if (e.key === 'Enter') { e.preventDefault(); _emit(); } });

  el.append(label, input);
  return { element: el, getValue: () => parseInt(input.value, 10) || initial };
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
  let _queryPromise = makeFetchFn(_count);
  let _refresh = null;

  async function _ensureMatches() {
    if (_sortedMatches !== null) return;
    _sortedMatches = await _queryPromise;  // already ordered by rank ascending
  }

  const { element: countEl } = buildCountControl(_count, {
    onChange(newCount) {
      _count = newCount;
      _sortedMatches = null;
      _queryPromise = makeFetchFn(_count);
      _refresh?.();
    },
  });

  let toolbarEl;
  if (extraControls) {
    toolbarEl = document.createElement('div');
    toolbarEl.style.cssText = 'display:flex;align-items:center;gap:0.75em;flex-wrap:wrap;';
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
      if (!_sortedMatches.length) return [];

      const startIdx = pageStartIdx(_sortedMatches, afterUuid);
      const page = _sortedMatches.slice(startIdx, startIdx + limit);
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
 * Open a review modal showing images most visually distinct from the given image.
 */
export async function openDissimilarImagesModal({ imageUuid }) {
  return _openRankedImagesModal({
    title: 'Most Distinct Images',
    makeFetchFn: count => findDissimilarImagesByUuid(imageUuid, count),
    referenceImageUuids: new Set([imageUuid]),
  });
}

/**
 * Open a review modal showing bboxes most visually distinct from the given bbox.
 */
export async function openDissimilarBboxesModal({ bboxUuid, labelUuid = null }) {
  const labelsMap = await fetchLabels().catch(() => new Map());

  let _count = 100;
  let _labelUuids = labelUuid ? [labelUuid] : [];
  let _refresh = null;

  // Derived from match list; reset when count or label filter changes.
  // _imageOrder: [{imageUuid, minRank}] sorted ascending by minRank
  // _imageBboxes: Map<imageUuid, Set<bboxUuid>>
  let _imageOrder = null;
  let _imageBboxes = null;
  let _queryPromise = findDissimilarBboxesByUuid(bboxUuid, _count, _labelUuids);

  function _invalidate() {
    _imageOrder = null;
    _imageBboxes = null;
    _queryPromise = findDissimilarBboxesByUuid(bboxUuid, _count, _labelUuids);
    _refresh?.();
  }

  async function _ensureMatches() {
    if (_imageOrder !== null) return;
    const matches = await _queryPromise;

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

  const headerEl = document.createElement('div');
  headerEl.style.cssText = 'display:flex;align-items:center;gap:0.75em;flex-wrap:wrap;';
  headerEl.append(countEl, labelsDropdown.element);

  _refresh = await showReviewModal({
    title: 'Most Distinct Bboxes',
    toolbarControls: headerEl,
    referenceBboxUuids: new Set([bboxUuid]),
    fetchImages: async ({ afterUuid = null, limit }) => {
      await _ensureMatches();
      if (!_imageOrder.length) return [];

      const startIdx = pageStartIdx(_imageOrder, afterUuid);
      const page = _imageOrder.slice(startIdx, startIdx + limit);
      if (!page.length) return [];

      const images = await fetchImagesByUuids(page.map(e => e.imageUuid));

      // Re-order by rank and narrow each image's bboxes to only matched ones
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
