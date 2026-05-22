// System tag UUIDs (must match constants in apps/helpers/db/db_client.py)
export const UNVERIFIED_TAG_UUID = '00000000-0000-0000-0000-000000000001';
export const VERIFIED_TAG_UUID = '00000000-0000-0000-0000-000000000002';
export const FALSE_POS_TAG_UUID = '00000000-0000-0000-0000-000000000003';

const VERIFICATION_TAG_UUIDS = new Set([UNVERIFIED_TAG_UUID, VERIFIED_TAG_UUID, FALSE_POS_TAG_UUID]);

/**
 * Fetch images that have at least one bbox tagged with the given tag.
 * @param {{ tagUuid?: string, labelUuid?: string, limit?: number, offset?: number }} opts
 */
export async function fetchReviewQueue({ tagUuid = UNVERIFIED_TAG_UUID, labelUuid, limit = 50, offset = 0 } = {}) {
  const params = new URLSearchParams({ tag_uuid: tagUuid, limit, offset });
  if (labelUuid) params.set('label_uuid', labelUuid);

  const res = await fetch(`/api/images/review?${params}`);
  const data = await res.json();
  if (!res.ok || data.status !== 'success') throw new Error(data.message || 'Failed to fetch review queue');
  return data.images || [];
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
