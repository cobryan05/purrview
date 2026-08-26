import { createCachedFetcher } from '/app-static/js/ui/utils/data/createCachedFetcher.js';
import { apiFetch, ignoreKeyReturn } from './apiUtils.js';

/** -----------------------------
 * CACHE
 * -----------------------------
 */
const collectorsFetcher = createCachedFetcher(async (keys) => {
  const res = await apiFetch('/api/training-collectors');
  const data = await res.json();
  if (!res.ok || data.status !== 'success') {
    throw new Error(data.message || 'Failed to fetch collectors');
  }
  const map = new Map(data.collectors.map(c => [c.uuid, c]));
  return ignoreKeyReturn(keys, map);
});

/** The collect knobs are defined once, in Python (`collector_params_schema()`),
 *  and rendered by the editor here — never restated in JS. Cached separately
 *  because it is static for the life of the server. */
const schemaFetcher = createCachedFetcher(async (keys) => {
  const res = await apiFetch('/api/training-collectors/schema');
  const data = await res.json();
  if (!res.ok || data.status !== 'success') {
    throw new Error(data.message || 'Failed to fetch collector schema');
  }
  return ignoreKeyReturn(keys, data.paramsSchema);
});

export function clearTrainingCollectorsCache() {
  collectorsFetcher.clear();
}

/** -----------------------------
 * FETCH
 * -----------------------------
 */
export async function fetchTrainingCollectors() {
  return collectorsFetcher.fetch('collectorsMap');
}

export async function fetchCollectorParamsSchema() {
  return schemaFetcher.fetch('collectorSchema');
}

/** -----------------------------
 * CRUD
 * -----------------------------
 */
export async function createTrainingCollector({ name, description = '', enabled = true, params = {} }) {
  const res = await apiFetch('/api/training-collectors', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name, description, enabled, params }),
  });
  const data = await res.json();
  if (!res.ok || data.status !== 'success') {
    throw new Error(data.message || 'Failed to create collector');
  }
  collectorsFetcher.clear();
  return data.collector;
}

export async function updateTrainingCollector(uuid, { name, description, enabled, params }) {
  const body = {};
  if (name !== undefined) body.name = name;
  if (description !== undefined) body.description = description;
  if (enabled !== undefined) body.enabled = enabled;
  if (params !== undefined) body.params = params;

  const res = await apiFetch(`/api/training-collectors/${uuid}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  const data = await res.json();
  if (!res.ok || data.status !== 'success') {
    throw new Error(data.message || 'Failed to update collector');
  }
  collectorsFetcher.clear();
  return data.collector;
}

export async function deleteTrainingCollector(uuid) {
  const res = await apiFetch(`/api/training-collectors/${uuid}`, { method: 'DELETE' });
  const data = await res.json();
  if (!res.ok || data.status !== 'success') {
    throw new Error(data.message || 'Failed to delete collector');
  }
  collectorsFetcher.clear();
}
