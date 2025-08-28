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
 * Fetch a label's metadata from the cache using its name.
 */
export function getLabelByName(name) {
  for (const label of _cache.values()) {
    if (label.name === name) return label;
  }
  return null;
}

/**
 * Return all labels as an array.
 */
export function getAllLabels() {
  return Array.from(_cache.values());
}

/**
 * Return all label names (for dropdowns).
 */
export function getAllLabelNames() {
  return Array.from(_cache.values()).map(l => l.name);
}
