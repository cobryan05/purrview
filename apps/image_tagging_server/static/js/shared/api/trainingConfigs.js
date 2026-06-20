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
export async function createTrainingConfig({ name, description = '', modelConfigNodeUuid = null, datasetUuids = [], hyperparamsJson = {} }) {
  const res = await apiFetch('/api/training-configs', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name, description, modelConfigNodeUuid, datasetUuids, hyperparamsJson }),
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
  return {
    readyCount: data.readyCount,
    needsReviewCount: data.needsReviewCount,
    datasetUuids: data.datasetUuids ?? [],
    labelUuids: data.labelUuids ?? [],
    rollUpActive: data.rollUpActive ?? false,
  };
}

export async function fetchTrainingConfigNeedsReview({
  uuid,
  labelUuids = [],
  afterUuid = null,
  afterSortValue = null,
  sortBy = 'created_at',
  sortOrder = 'desc',
  limit = 50,
}) {
  const res = await apiFetch(`/api/training-configs/${uuid}/needs-review-images`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ labelUuids, afterUuid, afterSortValue, sortBy, sortOrder, limit }),
  });
  const data = await res.json();
  if (!res.ok || data.status !== 'success') {
    throw new Error(data.message || 'Failed to fetch needs-review images');
  }
  return data.images ?? [];
}

export async function fetchTrainingConfigImagesByLabelStatus(uuid, {
  labelUuid,
  rollupLabelUuids = undefined,
  status = null,
  afterUuid = null,
  afterSortValue = null,
  sortBy = 'created_at',
  sortOrder = 'desc',
  limit = 50,
} = {}) {
  const res = await apiFetch(`/api/training-configs/${uuid}/images-by-label-status`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ labelUuid, rollupLabelUuids, status, afterUuid, afterSortValue, sortBy, sortOrder, limit }),
  });
  const data = await res.json();
  if (!res.ok || data.status !== 'success') {
    throw new Error(data.message || 'Failed to fetch images by label status');
  }
  return data.images ?? [];
}

export async function fetchTrainingConfigLabelStats(uuid) {
  const res = await apiFetch(`/api/training-configs/${uuid}/label-stats`);
  const data = await res.json();
  if (!res.ok || data.status !== 'success') {
    throw new Error(data.message || 'Failed to fetch training config label stats');
  }
  return { labelStats: data.labelStats ?? [], totalImages: data.totalImages ?? 0 };
}
