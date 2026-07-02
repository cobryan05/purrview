import { apiFetch } from './apiUtils.js';

/**
 * Batch variant of fetchImageLabels — one POST for a whole page of images.
 * @param {string[]} imageUuids
 * @returns {Promise<Map<string, Map<string, 'present'|'absent'>>>}
 *   imageUuid → (labelUuid → status). Images with no rows are omitted.
 */
export async function fetchImageLabelsBatch(imageUuids) {
  if (!imageUuids.length) return new Map();
  const res = await apiFetch('/api/images/labels/batch', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ imageUuids }),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok || data.status !== 'success') {
    throw new Error(data.message || 'Failed to fetch image labels batch');
  }
  const out = new Map();
  for (const [imageUuid, rows] of Object.entries(data.imageLabels ?? {})) {
    out.set(imageUuid, new Map(rows.map(r => [r.labelUuid, r.status])));
  }
  return out;
}

/**
 * @param {string} imagePath
 * @returns {Promise<Map<string, 'present'|'absent'>>}
 *   Map from labelUuid → status.  Labels absent from the map are UNCERTAIN.
 */
export async function fetchImageLabels(imagePath) {
  const res = await apiFetch(`/api/images/${encodeURIComponent(imagePath)}/labels`, {
    method: 'GET',
    headers: { 'Accept': 'application/json' },
  });

  /** @type {{ status: string, labels?: Array<{ labelUuid: string, status: string }> }} */
  const data = await res.json().catch(() => ({}));

  if (!res.ok || data.status !== 'success') {
    throw new Error(data.message || 'Failed to fetch image labels');
  }

  /** @type {Map<string, 'present'|'absent'>} */
  const map = new Map();
  for (const row of (data.labels ?? [])) {
    map.set(row.labelUuid, row.status);
  }
  return map;
}

/**
 * @param {string} imagePath
 * @param {string} labelUuid
 * @param {'present'|'absent'|'uncertain'} status
 *   'present'   - include as a positive training example for this label
 *   'absent'    - confirmed not present; clean negative training example
 *   'uncertain' - unreviewed (clears any existing present/absent row)
 */
export async function setImageLabel(imagePath, labelUuid, status) {
  const res = await apiFetch('/api/images/labels/set', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ imagePath, labelUuid, status }),
  });

  const data = await res.json().catch(() => ({}));
  if (!res.ok || data.status !== 'success') {
    throw new Error(data.message || 'Failed to set image label');
  }
}
