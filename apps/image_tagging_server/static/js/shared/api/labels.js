import { createCachedFetcher } from '/app-static/js/ui/utils/data/createCachedFetcher.js';
import { apiFetch, ignoreKeyReturn } from './apiUtils.js';
import { events, EventTypes } from '/app-static/js/shared/events/index.js';

/** -----------------------------
 * LABEL CACHE
 * -----------------------------
 */
export const labelsFetcher = createCachedFetcher(async (keys) => {
  const res = await apiFetch('/api/labels');

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
export async function createNewLabel(name, color, parentUuid = null, labelType = 'standard', scope = 'object') {
  const res = await apiFetch('/api/labels', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name, color, parentUuid, labelType, scope }),
  });

  const data = await res.json();

  if (!res.ok || data.status !== 'success') {
    throw new Error(data.message || 'Failed to add label');
  }

  labelsFetcher.clear();
  events.publish(EventTypes.LABELS_UPDATED);
  return data.label ?? null;
}


export async function updateLabel(uuid, name, color, { labelType, scope } = {}) {
  const body = { name, color };
  if (labelType !== undefined) body.labelType = labelType;
  if (scope !== undefined) body.scope = scope;

  const res = await apiFetch(`/api/labels/${uuid}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });

  const data = await res.json();

  if (!res.ok || data.status !== 'success') {
    throw new Error(data.message || 'Failed to update label');
  }

  labelsFetcher.clear();
  events.publish(EventTypes.LABELS_UPDATED);
}


export async function deleteLabel(uuid) {
  const res = await apiFetch(`/api/labels/${uuid}`, {
    method: 'DELETE',
  });

  const data = await res.json();

  if (!res.ok || data.status !== 'success') {
    throw new Error(data.message || 'Failed to delete label');
  }

  labelsFetcher.clear();
  events.publish(EventTypes.LABELS_UPDATED);
}