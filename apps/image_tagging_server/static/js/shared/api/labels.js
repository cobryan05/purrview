import { createCachedFetcher } from '/app-static/js/ui/utils/data/createCachedFetcher.js';
import { ignoreKeyReturn } from './apiUtils.js';

/** -----------------------------
 * LABEL CACHE
 * -----------------------------
 */
export const labelsFetcher = createCachedFetcher(async (keys) => {
  const res = await fetch('/api/labels');

  /** @type {import('@web_api').ListLabelsResponse} */
  const data = await res.json();

  if (!res.ok || data.status !== 'success') {
    throw new Error(data.message || 'Failed to fetch labels');
  }

  const map = new Map(data.labels?.map(l => [l.uuid, l]));

  return ignoreKeyReturn(keys, map);
});


/** -----------------------------
 * CACHE CLEAR
 * -----------------------------
 */
export function clearLabelsCache() {
  labelsFetcher.clear();
}


/** -----------------------------
 * FETCH FUNCTIONS
 * -----------------------------
 */
export async function fetchLabels() {
  return await labelsFetcher.fetch('labelsMap');
}

export async function fetchLabelByUuid(uuid) {
  const labels = await fetchLabels();
  return labels.get(uuid) || null;
}


/** -----------------------------
 * CRUD FUNCTIONS
 * -----------------------------
 */
export async function createNewLabel(name, color, parent_uuid = null) {
  const res = await fetch('/api/labels', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name, color, parent_uuid }),
  });

  const data = await res.json();

  if (!res.ok || data.status !== 'success') {
    throw new Error(data.message || 'Failed to add label');
  }

  labelsFetcher.clear();
}


export async function updateLabel(uuid, name, color) {
  const res = await fetch(`/api/labels/${uuid}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name, color }),
  });

  const data = await res.json();

  if (!res.ok || data.status !== 'success') {
    throw new Error(data.message || 'Failed to update label');
  }

  labelsFetcher.clear();
}


export async function deleteLabel(uuid) {
  const res = await fetch(`/api/labels/${uuid}`, {
    method: 'DELETE',
  });

  const data = await res.json();

  if (!res.ok || data.status !== 'success') {
    throw new Error(data.message || 'Failed to delete label');
  }

  labelsFetcher.clear();
}