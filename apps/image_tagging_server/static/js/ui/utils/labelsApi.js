import { createCachedFetcher } from './createCachedFetcher.js';

/** -----------------------------
 * LABEL CACHE
 * -----------------------------
 */
export const labelsFetcher = createCachedFetcher(async () => {
  const res = await fetch('/api/labels/list');
  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.message || 'Failed to fetch labels');
  }
  const { labels } = await res.json();
  // Store as Map keyed by UUID
  const map = new Map(labels.map(l => [l.metadata.uuid, l]));
  return map;
});


/** -----------------------------
 * CACHE CLEAR
 * -----------------------------
 */
export function clearLabelCache() {
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
export async function createNewLabel(name, color, parentUuid = null) {
  const res = await fetch('/api/labels/add', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name, color, parent_uuid: parentUuid }),
  });

  if (!res.ok) {
    const { message } = await res.json();
    throw new Error(message || 'Failed to add label');
  }

  // Invalidate cache after successful creation
  labelsFetcher.clear();
}

export async function updateLabel(uuid, name, color) {
  const res = await fetch('/api/labels/update', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ label_uuid: uuid, name, color }),
  });

  if (!res.ok) {
    const { message } = await res.json();
    throw new Error(message || 'Failed to update label');
  }

  // Invalidate cache after successful update
  labelsFetcher.clear();
}

export async function deleteLabel(uuid) {
  const res = await fetch('/api/labels/delete', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ label_uuid: uuid }),
  });

  if (!res.ok) {
    const { message } = await res.json();
    throw new Error(message || 'Failed to delete label');
  }

  // Invalidate cache after successful deletion
  labelsFetcher.clear();
}
