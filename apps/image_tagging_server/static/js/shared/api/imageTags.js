import { apiFetch } from './apiUtils.js';

/**
 * @param {string} imagePath
 * @returns {Promise<string[]>} tag UUIDs currently linked to the whole image.
 */
export async function fetchImageTags(imagePath) {
  const res = await apiFetch(`/api/images/${encodeURIComponent(imagePath)}/tags`, {
    method: 'GET',
    headers: { 'Accept': 'application/json' },
  });

  const data = await res.json().catch(() => ({}));
  if (!res.ok || data.status !== 'success') {
    throw new Error(data.message || 'Failed to fetch image tags');
  }
  return (data.tags ?? []).map(t => t.uuid);
}

/**
 * @param {string} imagePath
 * @param {string} tagUuid
 * @param {'add'|'remove'} action
 */
export async function setImageTag(imagePath, tagUuid, action) {
  const res = await apiFetch('/api/images/tags/set', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ imagePath, tagUuid, action }),
  });

  const data = await res.json().catch(() => ({}));
  if (!res.ok || data.status !== 'success') {
    throw new Error(data.message || 'Failed to set image tag');
  }
}

/**
 * @param {string} imagePath
 * @returns {Promise<Map<string, 'present'|'absent'>>}
 *   Map from tagUuid → status. Tags absent from the map are UNCERTAIN (never
 *   applied, never verified absent). PRESENT tags are also reachable via
 *   fetchImageTags(); this map only ever has 'absent' entries in practice
 *   since PRESENT is normally read via the tags list, but the server returns
 *   both for symmetry with fetchImageLabels.
 */
export async function fetchImageTagStatuses(imagePath) {
  const res = await apiFetch(`/api/images/${encodeURIComponent(imagePath)}/tags/statuses`, {
    method: 'GET',
    headers: { 'Accept': 'application/json' },
  });

  const data = await res.json().catch(() => ({}));
  if (!res.ok || data.status !== 'success') {
    throw new Error(data.message || 'Failed to fetch image tag statuses');
  }

  /** @type {Map<string, 'present'|'absent'>} */
  const map = new Map();
  for (const row of (data.tagStatuses ?? [])) {
    map.set(row.tagUuid, row.status);
  }
  return map;
}

/**
 * @param {string} imagePath
 * @param {string} tagUuid
 * @param {'present'|'absent'|'uncertain'} status
 *   'present'   - tag applies to this image
 *   'absent'    - explicitly verified the tag does NOT apply to this image
 *   'uncertain' - clear any explicit status (reverts to the default)
 */
export async function setImageTagStatus(imagePath, tagUuid, status) {
  const res = await apiFetch('/api/images/tags/set-status', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ imagePath, tagUuid, status }),
  });

  const data = await res.json().catch(() => ({}));
  if (!res.ok || data.status !== 'success') {
    throw new Error(data.message || 'Failed to set image tag status');
  }
}
