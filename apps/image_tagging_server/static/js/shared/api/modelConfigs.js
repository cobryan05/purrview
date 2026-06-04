import { createCachedFetcher } from '/app-static/js/ui/utils/data/createCachedFetcher.js';
import { apiFetch, ignoreKeyReturn } from './apiUtils.js';

/** -----------------------------
 * CACHE
 * -----------------------------
 */
const modelConfigsFetcher = createCachedFetcher(async (keys) => {
  const res = await apiFetch('/api/model-configs');
  const data = await res.json();
  if (!res.ok || data.status !== 'success') {
    throw new Error(data.message || 'Failed to fetch model configs');
  }
  const map = new Map(data.configs.map(c => [c.uuid, c]));
  return ignoreKeyReturn(keys, map);
});

const typenamesFetcher = createCachedFetcher(async (keys) => {
  const res = await apiFetch('/api/model-configs/typenames');
  const data = await res.json();
  if (!res.ok || data.status !== 'success') {
    throw new Error(data.message || 'Failed to fetch model typenames');
  }
  return ignoreKeyReturn(keys, data.typenames);
});

export function clearModelConfigsCache() {
  modelConfigsFetcher.clear();
}

/** -----------------------------
 * FETCH
 * -----------------------------
 */
export async function fetchModelConfigs() {
  return modelConfigsFetcher.fetch('configsMap');
}

export async function fetchModelTypenames() {
  return typenamesFetcher.fetch('typenames');
}

/** -----------------------------
 * CRUD
 * -----------------------------
 */
export async function createModelConfig({ name, description = '', nodes = [] }) {
  const res = await apiFetch('/api/model-configs', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name, description, nodes }),
  });
  const data = await res.json();
  if (!res.ok || data.status !== 'success') {
    throw new Error(data.message || 'Failed to create model config');
  }
  modelConfigsFetcher.clear();
  return data.config;
}

export async function updateModelConfig(uuid, { name, description, nodes }) {
  const body = {};
  if (name !== undefined) body.name = name;
  if (description !== undefined) body.description = description;
  if (nodes !== undefined) body.nodes = nodes;

  const res = await apiFetch(`/api/model-configs/${uuid}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  const data = await res.json();
  if (!res.ok || data.status !== 'success') {
    throw new Error(data.message || 'Failed to update model config');
  }
  modelConfigsFetcher.clear();
  return data.config;
}

export async function deleteModelConfig(uuid) {
  const res = await apiFetch(`/api/model-configs/${uuid}`, { method: 'DELETE' });
  const data = await res.json();
  if (!res.ok || data.status !== 'success') {
    throw new Error(data.message || 'Failed to delete model config');
  }
  modelConfigsFetcher.clear();
}
