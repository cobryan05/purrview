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
    throw new Error(data.message || 'Failed to fetch model shapes');
  }
  const map = new Map(data.configs.map(c => [c.uuid, c]));
  return ignoreKeyReturn(keys, map);
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
    throw new Error(data.message || 'Failed to create model shape');
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
    throw new Error(data.message || 'Failed to update model shape');
  }
  modelConfigsFetcher.clear();
  return data.config;
}

export async function deleteModelConfig(uuid) {
  const res = await apiFetch(`/api/model-configs/${uuid}`, { method: 'DELETE' });
  const data = await res.json();
  if (!res.ok || data.status !== 'success') {
    throw new Error(data.message || 'Failed to delete model shape');
  }
  modelConfigsFetcher.clear();
}

/**
 * The authoritative list of training targets (top model + each sub-model) for
 * a model config. Single authority on sub-model boundaries — ui/modelSpec.js
 * mirrors the boundary rule for display only.
 * Not cached: callers use this right before a train action where freshness
 * matters more than avoiding a refetch.
 * @returns {Promise<Array<{nodeUuid:string, role:'top'|'submodel', labelUuid:?string,
 *   modelType:string, architecture:string, labelUuids:string[], classNames:string[],
 *   classRemap:Object}>>}
 */
export async function fetchModelConfigTrainingTargets(modelConfigUuid) {
  const res = await apiFetch(`/api/model-configs/${modelConfigUuid}/training-targets`);
  const data = await res.json();
  if (!res.ok || data.status !== 'success') {
    throw new Error(data.message || 'Failed to fetch training targets');
  }
  return data.targets ?? [];
}
