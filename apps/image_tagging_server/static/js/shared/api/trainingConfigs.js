import { createCachedFetcher } from '/app-static/js/ui/utils/data/createCachedFetcher.js';
import { apiFetch, ignoreKeyReturn } from './apiUtils.js';

/** -----------------------------
 * CACHE
 * -----------------------------
 */
const trainingConfigsFetcher = createCachedFetcher(async (keys) => {
  const res = await apiFetch('/api/training-configs');
  const data = await res.json();
  if (!res.ok || data.status !== 'success') {
    throw new Error(data.message || 'Failed to fetch training configs');
  }
  const map = new Map(data.configs.map(c => [c.uuid, c]));
  return ignoreKeyReturn(keys, map);
});

export function clearTrainingConfigsCache() {
  trainingConfigsFetcher.clear();
}

/** -----------------------------
 * FETCH
 * -----------------------------
 */
export async function fetchTrainingConfigs() {
  return trainingConfigsFetcher.fetch('configsMap');
}

/** -----------------------------
 * CRUD
 * -----------------------------
 */
export async function createTrainingConfig({ name, description = '', trainerType, modelConfigUuid = null, datasetUuid, hyperparamsJson = {}, labelUuids = [] }) {
  const res = await apiFetch('/api/training-configs', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name, description, trainerType, modelConfigUuid, datasetUuid, hyperparamsJson, labelUuids }),
  });
  const data = await res.json();
  if (!res.ok || data.status !== 'success') {
    throw new Error(data.message || 'Failed to create training config');
  }
  trainingConfigsFetcher.clear();
  return data.config;
}

export async function updateTrainingConfig(uuid, fields) {
  const res = await apiFetch(`/api/training-configs/${uuid}`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(fields),
  });
  const data = await res.json();
  if (!res.ok || data.status !== 'success') {
    throw new Error(data.message || 'Failed to update training config');
  }
  trainingConfigsFetcher.clear();
  return data.config;
}

export async function deleteTrainingConfig(uuid) {
  const res = await apiFetch(`/api/training-configs/${uuid}`, { method: 'DELETE' });
  const data = await res.json();
  if (!res.ok || data.status !== 'success') {
    throw new Error(data.message || 'Failed to delete training config');
  }
  trainingConfigsFetcher.clear();
}

export async function fetchTrainingConfigReadiness(uuid) {
  const res = await apiFetch(`/api/training-configs/${uuid}/readiness`);
  const data = await res.json();
  if (!res.ok || data.status !== 'success') {
    throw new Error(data.message || 'Failed to fetch training config readiness');
  }
  return { readyCount: data.readyCount, needsReviewCount: data.needsReviewCount, datasetUuid: data.datasetUuid, labelUuids: data.labelUuids };
}
