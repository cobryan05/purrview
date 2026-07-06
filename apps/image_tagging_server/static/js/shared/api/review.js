import { apiFetch } from './apiUtils.js';
// System tag UUIDs (must match constants in apps/helpers/db/types.py)
export const UNVERIFIED_TAG_UUID = '00000000-0000-0000-0000-000000000001';
// Keep in sync with apps/helpers/db/types.py
export const FALSE_POS_TAG_UUID = '00000000-0000-0000-0000-000000000003';
export const REJECTION_TAG_UUID = '00000000-0000-0000-0000-000000000004';
export const VERIFIED_TAG_UUID = '00000000-0000-0000-0000-000000000005';

// All three are in the verification exclusive group — mutually exclusive per bbox
export const VERIFICATION_TAG_UUIDS = new Set([UNVERIFIED_TAG_UUID, FALSE_POS_TAG_UUID, VERIFIED_TAG_UUID]);

/**
 * Search the review queue with flexible filters and cursor-based pagination.
 *
 * @param {{
 *   tagUuids?: string[],          // bbox tags to include; default [UNVERIFIED_TAG_UUID]
 *   showAllBboxes?: boolean,       // ignore tagUuids, show all bboxes
 *   tagMatchMode?: 'any'|'all',   // OR (default) or AND across tagUuids
 *   labelUuids?: string[],         // bbox label filter (multi-select)
 *   labelMatchMode?: 'any'|'all', // OR (default) or AND across labelUuids
 *   labelIncludeChildren?: boolean, // also match child labels in the hierarchy
 *   labelUuid?: string,            // legacy single-label filter (still accepted by backend)
 *   sortBy?: 'created_at'|'updated_at'|'uuid',
 *   sortOrder?: 'asc'|'desc',
 *   filenameSearch?: string,
 *   dateFrom?: string|null,        // ISO datetime (filters on images.created_at)
 *   dateTo?: string|null,
 *   afterUuid?: string|null,       // keyset cursor: UUID of last image on previous page
 *   afterSortValue?: string|null,  // ISO datetime of last image's sort field (when sortBy != 'uuid')
 *   imageLabelFilters?: Array<{labelUuid: string, status: string, exclude: boolean}>,
 *   imageLabelConditions?: {labelUuid: string, state: 'present'|'absent'|'uncertain'}[],
 *   limit?: number,
 * }} opts
 */
export async function fetchReviewQueue({
  tagUuids = [UNVERIFIED_TAG_UUID],
  tagExcludeUuids = [],
  showAllBboxes = false,
  tagMatchMode = 'any',
  labelUuids = [],
  labelExcludeUuids = [],
  labelMatchMode = 'any',
  labelIncludeChildren = false,
  labelUuid,
  sortBy = 'created_at',
  sortOrder = 'desc',
  filenameSearch,
  dateFrom = null,
  dateTo = null,
  afterUuid = null,
  afterSortValue = null,
  imageLabelFilters = [],
  imageLabelConditions = [],
  verificationCheck = null,
  requireBboxes = false,
  limit = 50,
} = {}) {
  const body = {
    tagUuids,
    showAllBboxes,
    tagMatchMode,
    sortBy,
    sortOrder,
    limit,
  };
  if (requireBboxes) body.requireBboxes = true;
  if (labelUuids.length > 0) {
    body.labelUuids = labelUuids;
    body.labelMatchMode = labelMatchMode;
    if (labelIncludeChildren) body.labelIncludeChildren = true;
  }
  if (tagExcludeUuids.length > 0) {
    body.tagPresenceFilters = tagExcludeUuids.map(uuid => ({ tagUuid: uuid, exclude: true }));
  }
  if (labelExcludeUuids.length > 0) {
    body.labelExclusions = labelExcludeUuids;
  }
  if (labelUuid) body.labelUuid = labelUuid;
  if (filenameSearch) body.filenameSearch = filenameSearch;
  if (dateFrom) body.dateFrom = dateFrom;
  if (dateTo) body.dateTo = dateTo;
  if (afterUuid) body.afterUuid = afterUuid;
  if (afterSortValue) body.afterSortValue = afterSortValue;
  if (verificationCheck) body.labelVerificationCheck = verificationCheck;

  // Explicit imageLabelFilters (direct format) take precedence over conditions.
  // imageLabelConditions (from the filter panel) are mapped to the same wire format.
  const allLabelFilters = [
    ...imageLabelFilters.map(f => ({ labelUuid: f.labelUuid, status: f.status, exclude: f.exclude ?? false })),
    ...imageLabelConditions.map(c => ({ labelUuid: c.labelUuid, status: c.state, exclude: false })),
  ];
  if (allLabelFilters.length > 0) body.imageLabelFilters = allLabelFilters;

  const res = await apiFetch('/api/images/review', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  const data = await res.json();
  if (!res.ok || data.status !== 'success') throw new Error(data.message || 'Failed to fetch review queue');
  return {
    images: data.images ?? [],
    hasMore: data.hasMore ?? false,
    totalCount: data.totalCount ?? 0,
  };
}

/**
 * Replace all bboxes on an image. Each bbox in the list is a plain object:
 * { uuid, labelUuid, x, y, width, height, tagUuids }
 * @param {string} imagePath
 * @param {Array<object>} bboxes
 */
export async function updateImageBboxes(imagePath, bboxes) {
  const payload = { imagePath, boxes: bboxes };
  const res = await apiFetch('/api/images/metadata/update', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  const data = await res.json();
  if (!res.ok || data.status !== 'success') throw new Error(data.message || 'Failed to update image');
}



/**
 * Return a copy of tagUuids with the verification tag replaced by newTagUuid.
 * @param {string[]} tagUuids
 * @param {string} newTagUuid
 * @returns {string[]}
 */
/**
 * Generic exclusive-group swap: remove all tags that belong to the same exclusive
 * group (identified by groupUuids) and append newTagUuid.
 *
 * @param {string[]} tagUuids      - current tag UUID list
 * @param {Set<string>} groupUuids - the full set of UUIDs in the exclusive group
 * @param {string} newTagUuid      - the new group member to set
 * @returns {string[]}
 */
export function swapExclusiveGroupTag(tagUuids, groupUuids, newTagUuid) {
  return [...tagUuids.filter(t => !groupUuids.has(t)), newTagUuid];
}

/**
 * Return a copy of tagUuids with the verification tag replaced by newTagUuid.
 * @param {string[]} tagUuids
 * @param {string} newTagUuid
 */
export function swapVerificationTag(tagUuids, newTagUuid) {
  return swapExclusiveGroupTag(tagUuids, VERIFICATION_TAG_UUIDS, newTagUuid);
}

/**
 * Return a URL suitable for use as <img src> for a bbox crop thumbnail.
 * @param {string} imagePath
 * @param {string} bboxUuid
 * @param {number} [width=200]
 * @returns {string}
 */
export function getBboxThumbnailUrl(imagePath, bboxUuid, width = 200) {
  const encodedPath = imagePath.split('/').map(encodeURIComponent).join('/');
  return `/api/images/${encodedPath}/bbox/${bboxUuid}/thumbnail?width=${width}`;
}

export function getImageThumbnailUrl(imagePath, width = 200) {
  const encodedPath = imagePath.split('/').map(encodeURIComponent).join('/');
  return `/api/images/${encodedPath}/thumbnail?width=${width}`;
}

/**
 * Update a single bbox's label and tags.
 * @param {string} bboxUuid
 * @param {{ labelUuid: string, tagUuids: string[] }} opts
 */
export async function updateBbox(bboxUuid, { labelUuid, tagUuids }) {
  const res = await apiFetch(`/api/images/bbox/${bboxUuid}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ labelUuid, tagUuids }),
  });
  if (!res.ok) throw new Error(`updateBbox failed: ${res.status}`);
}

/**
 * Delete a single bbox.
 * @param {string} bboxUuid
 */
export async function deleteBbox(bboxUuid) {
  const res = await apiFetch(`/api/images/bbox/${bboxUuid}`, { method: 'DELETE' });
  if (!res.ok) throw new Error(`deleteBbox failed: ${res.status}`);
}

/**
 * Apply an action to all bboxes matching the current filter, across all pages.
 * The filter params mirror fetchReviewQueue; bboxTagFilter / bboxLabelFilter
 * are the client-computed bbox-level subset filters.
 */
export async function bulkApplyAll({
  // Filter params (same shape as fetchReviewQueue, no pagination)
  tagUuids = [UNVERIFIED_TAG_UUID],
  tagExcludeUuids = [],
  showAllBboxes = false,
  tagMatchMode = 'any',
  labelUuids = [],
  labelExcludeUuids = [],
  labelMatchMode = 'any',
  labelIncludeChildren = false,
  sortBy = 'created_at',
  sortOrder = 'desc',
  filenameSearch,
  dateFrom = null,
  dateTo = null,
  imageLabelFilters = [],
  verificationCheck = null,
  requireBboxes = false,
  // Bbox-level subset filters (client-computed)
  bboxTagFilter = [],
  bboxLabelFilter = [],
  // Action params
  action,
  labelUuid = null,
  selectedTagUuids = [],
} = {}) {
  const filterBody = {
    tagUuids: showAllBboxes ? [] : tagUuids,
    showAllBboxes,
    tagMatchMode,
    sortBy,
    sortOrder,
    limit: 200,
  };
  if (requireBboxes) filterBody.requireBboxes = true;
  if (labelUuids.length > 0) {
    filterBody.labelUuids = labelUuids;
    filterBody.labelMatchMode = labelMatchMode;
    if (labelIncludeChildren) filterBody.labelIncludeChildren = true;
  }
  if (tagExcludeUuids.length > 0) {
    filterBody.tagPresenceFilters = tagExcludeUuids.map(uuid => ({ tagUuid: uuid, exclude: true }));
  }
  if (labelExcludeUuids.length > 0) filterBody.labelExclusions = labelExcludeUuids;
  if (filenameSearch) filterBody.filenameSearch = filenameSearch;
  if (dateFrom) filterBody.dateFrom = dateFrom;
  if (dateTo) filterBody.dateTo = dateTo;
  if (imageLabelFilters.length > 0) {
    filterBody.imageLabelFilters = imageLabelFilters.map(f => ({
      labelUuid: f.labelUuid,
      status: f.status,
      exclude: f.exclude,
    }));
  }
  if (verificationCheck) filterBody.labelVerificationCheck = verificationCheck;

  const res = await apiFetch('/api/images/bbox/bulk-apply', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      filter: filterBody,
      action,
      labelUuid: labelUuid ?? null,
      tagUuids: selectedTagUuids,
      bboxTagFilter,
      bboxLabelFilter,
      showAllBboxes,
    }),
  });
  const data = await res.json();
  if (!res.ok || data.status !== 'success') throw new Error(data.message || 'Bulk apply failed');
  return { modifiedCount: data.modifiedCount ?? 0, failCount: data.failCount ?? 0 };
}

/**
 * Apply an action to an explicit list of bboxes in one server-side pass.
 * Used by candidate-list-driven modals (similarity/diversity search) that have
 * no server-side filter equivalent for their already-known result set.
 *
 * @param {Object} opts
 * @param {?Record<string, {x: number, y: number, width: number, height: number}>}
 *   [opts.bboxGeometry] - bboxUuid -> replacement coordinates, applied on relabel
 *   (e.g. "Run Model on Results" swapping in a refinement model's suggested box).
 */
export async function bulkApplyBboxesByUuids({ bboxUuids, action, labelUuid = null, tagUuids = [], bboxGeometry = null }) {
  const res = await apiFetch('/api/images/bbox/bulk-apply-by-uuids', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ bboxUuids, action, labelUuid, tagUuids, bboxGeometry }),
  });
  const data = await res.json();
  if (!res.ok || data.status !== 'success') throw new Error(data.message || 'Bulk apply failed');
  return { modifiedCount: data.modifiedCount ?? 0, failCount: data.failCount ?? 0 };
}

/**
 * Apply an image-level action to an explicit list of images in one server-side
 * pass — the image-kind counterpart of bulkApplyBboxesByUuids.
 */
export async function bulkApplyImagesByUuids({ imageUuids, action, labelUuid = null, tagUuids = [] }) {
  const res = await apiFetch('/api/images/bulk-apply-by-uuids', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ imageUuids, action, labelUuid, tagUuids }),
  });
  const data = await res.json();
  if (!res.ok || data.status !== 'success') throw new Error(data.message || 'Bulk apply failed');
  return { modifiedCount: data.modifiedCount ?? 0, failCount: data.failCount ?? 0 };
}

/**
 * Invalidate cached thumbnails for a list of bbox UUIDs.
 * @param {string[]} bboxUuids
 */
export async function invalidateBboxThumbnails(bboxUuids) {
  if (!bboxUuids.length) return;
  const res = await apiFetch('/api/images/bbox/thumbnails/invalidate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ bboxUuids }),
  });
  if (!res.ok) console.warn(`invalidateBboxThumbnails failed: ${res.status}`);
}
