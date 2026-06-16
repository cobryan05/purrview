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
