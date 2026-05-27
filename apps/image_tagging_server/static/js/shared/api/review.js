// System tag UUIDs (must match constants in apps/helpers/db/db_client.py)
export const UNVERIFIED_TAG_UUID = '00000000-0000-0000-0000-000000000001';
export const VERIFIED_TAG_UUID = '00000000-0000-0000-0000-000000000002';
export const FALSE_POS_TAG_UUID = '00000000-0000-0000-0000-000000000003';

const VERIFICATION_TAG_UUIDS = new Set([UNVERIFIED_TAG_UUID, VERIFIED_TAG_UUID, FALSE_POS_TAG_UUID]);

/**
 * Search the review queue with flexible filters and cursor-based pagination.
 *
 * @param {{
 *   tagUuids?: string[],          // bbox tags to include (OR); default [UNVERIFIED_TAG_UUID]
 *   showAllBboxes?: boolean,       // ignore tagUuids, show all bboxes
 *   labelUuid?: string,            // optional bbox label filter
 *   sortBy?: 'created_at'|'updated_at'|'uuid',
 *   sortOrder?: 'asc'|'desc',
 *   filenameSearch?: string,
 *   dateFrom?: string|null,        // ISO datetime (filters on images.created_at)
 *   dateTo?: string|null,
 *   afterUuid?: string|null,       // keyset cursor: UUID of last image on previous page
 *   afterSortValue?: string|null,  // ISO datetime of last image's sort field (when sortBy != 'uuid')
 *   imageLabelFilters?: Array<{labelUuid: string, status: string, exclude: boolean}>,
 *   limit?: number,
 * }} opts
 */
export async function fetchReviewQueue({
  tagUuids = [UNVERIFIED_TAG_UUID],
  showAllBboxes = false,
  labelUuid,
  sortBy = 'created_at',
  sortOrder = 'desc',
  filenameSearch,
  dateFrom = null,
  dateTo = null,
  afterUuid = null,
  afterSortValue = null,
  imageLabelFilters = [],
  limit = 50,
} = {}) {
  const body = {
    tag_uuids: tagUuids,
    show_all_bboxes: showAllBboxes,
    sort_by: sortBy,
    sort_order: sortOrder,
    limit,
  };
  if (labelUuid) body.label_uuid = labelUuid;
  if (filenameSearch) body.filename_search = filenameSearch;
  if (dateFrom) body.date_from = dateFrom;
  if (dateTo) body.date_to = dateTo;
  if (afterUuid) body.after_uuid = afterUuid;
  if (afterSortValue) body.after_sort_value = afterSortValue;
  if (imageLabelFilters.length > 0) {
    body.image_label_filters = imageLabelFilters.map(f => ({
      label_uuid: f.labelUuid,
      status: f.status,
      exclude: f.exclude,
    }));
  }

  const res = await fetch('/api/images/review', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  const data = await res.json();
  if (!res.ok || data.status !== 'success') throw new Error(data.message || 'Failed to fetch review queue');
  return data.images ?? [];
}

/**
 * Replace all bboxes on an image. Each bbox in the list is a plain object:
 * { uuid, labelUuid, x, y, width, height, tagUuids }
 * @param {string} imagePath
 * @param {Array<object>} bboxes
 */
export async function updateImageBboxes(imagePath, bboxes) {
  const payload = { imagePath, boxes: bboxes };
  const res = await fetch('/api/images/metadata/update', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  const data = await res.json();
  if (!res.ok || data.status !== 'success') throw new Error(data.message || 'Failed to update image');
}

/**
 * Explicitly set a label's status on an image (e.g. absent with no bboxes).
 * @param {string} imagePath
 * @param {string} labelUuid
 * @param {'uncertain'|'present'|'absent'} status
 */
export async function setImageLabel(imagePath, labelUuid, status) {
  const res = await fetch('/api/images/labels/set', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ imagePath, labelUuid, status }),
  });
  const data = await res.json();
  if (!res.ok || data.status !== 'success') throw new Error(data.message || 'Failed to set image label');
}

/**
 * Return a copy of tagUuids with the verification tag replaced by newTagUuid.
 * @param {string[]} tagUuids
 * @param {string} newTagUuid
 * @returns {string[]}
 */
export function swapVerificationTag(tagUuids, newTagUuid) {
  return [...tagUuids.filter(t => !VERIFICATION_TAG_UUIDS.has(t)), newTagUuid];
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

/**
 * Update a single bbox's label and tags.
 * @param {string} bboxUuid
 * @param {{ labelUuid: string, tagUuids: string[] }} opts
 */
export async function updateBbox(bboxUuid, { labelUuid, tagUuids }) {
  const res = await fetch(`/api/images/bbox/${bboxUuid}`, {
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
  const res = await fetch(`/api/images/bbox/${bboxUuid}`, { method: 'DELETE' });
  if (!res.ok) throw new Error(`deleteBbox failed: ${res.status}`);
}

/**
 * Invalidate cached thumbnails for a list of bbox UUIDs.
 * @param {string[]} bboxUuids
 */
export async function invalidateBboxThumbnails(bboxUuids) {
  if (!bboxUuids.length) return;
  const res = await fetch('/api/images/bbox/thumbnails/invalidate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ bboxUuids }),
  });
  if (!res.ok) console.warn(`invalidateBboxThumbnails failed: ${res.status}`);
}
