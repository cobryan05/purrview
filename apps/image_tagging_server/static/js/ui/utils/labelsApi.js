let _cache = new Map();

/**
 * Refresh the label cache by fetching the latest labels from the server.
 * Stores them in a Map keyed by uuid.
 */
export async function refreshLabelCache() {
  const labelRes = await fetch('/api/labels/list');
  const { labels } = await labelRes.json();
  _cache = new Map(labels.map(l => [l.metadata.uuid, l]));
}

/**
 * Fetch a label's metadata from the cache using its UUID.
 */
export function getLabelByUuid(uuid) {
  return _cache.get(uuid) || null;
}

/**
 * Return the labels map
 */
export function getAllLabels() {
  return new Map(_cache);
}

/**
 * Add a new label to the server and refresh the cache.
 * @param {string} name - The name of the new label.
 * @param {string} color - The color of the new label.
 * @param {string|null} parentUuid - The UUID of the parent label (or null for root labels).
 */
export async function createNewLabel(name, color, parentUuid = null) {
  const response = await fetch('/api/labels/add', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name, color, parent_uuid: parentUuid }),
  });

  if (!response.ok) {
    const { message } = await response.json();
    throw new Error(message || 'Failed to add label');
  }

  await refreshLabelCache();
}

/**
 * Update an existing label on the server and refresh the cache.
 * @param {string} uuid - The UUID of the label to update.
 * @param {string} name - The new name of the label.
 * @param {string} color - The new color of the label.
 */
export async function updateLabel(uuid, name, color) {
  const response = await fetch('/api/labels/update', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ label_uuid: uuid, name, color }),
  });

  if (!response.ok) {
    const { message } = await response.json();
    throw new Error(message || 'Failed to update label');
  }

  await refreshLabelCache();
}

/**
 * Delete a label from the server and refresh the cache.
 * @param {string} uuid - The UUID of the label to delete.
 */
export async function deleteLabel(uuid) {
  const response = await fetch('/api/labels/delete', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ label_uuid: uuid }),
  });

  if (!response.ok) {
    const { message } = await response.json();
    throw new Error(message || 'Failed to delete label');
  }

  await refreshLabelCache();
}
