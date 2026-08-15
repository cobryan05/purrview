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
    throw new Error(data.message || 'Failed to fetch training runs');
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
    throw new Error(data.message || 'Failed to create training run');
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
    throw new Error(data.message || 'Failed to update training run');
  }
  trainingConfigsFetcher.clear();
  return data.config;
}

export async function deleteTrainingConfig(uuid) {
  const res = await apiFetch(`/api/training-configs/${uuid}`, { method: 'DELETE' });
  const data = await res.json();
  if (!res.ok || data.status !== 'success') {
    throw new Error(data.message || 'Failed to delete training run');
  }
  trainingConfigsFetcher.clear();
}

export async function fetchTrainingConfigReadiness(uuid) {
  const res = await apiFetch(`/api/training-configs/${uuid}/readiness`);
  const data = await res.json();
  if (!res.ok || data.status !== 'success') {
    throw new Error(data.message || 'Failed to fetch training run readiness');
  }
  return {
    readyCount: data.readyCount,
    readyPositiveCount: data.readyPositiveCount ?? 0,
    readyNegativeCount: data.readyNegativeCount ?? 0,
    needsReviewCount: data.needsReviewCount,
    notStartedCount: data.notStartedCount ?? 0,
    datasetUuids: data.datasetUuids ?? [],
    labelUuids: data.labelUuids ?? [],
    rollUpActive: data.rollUpActive ?? false,
  };
}

/**
 * The same readiness counts, over only `imageUuids` — a previous run's recorded
 * manifest, so the positive/negative split of "then" and "now" is produced by
 * one counting rule. Images no longer in the config's datasets drop out, so the
 * counts can total less than imageUuids.length.
 *
 * @param {string} uuid
 * @param {string[]} imageUuids
 */
export async function fetchTrainingConfigReadinessForImages(uuid, imageUuids) {
  const res = await apiFetch(`/api/training-configs/${uuid}/readiness-for-images`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ imageUuids }),
  });
  const data = await res.json();
  if (!res.ok || data.status !== 'success') {
    throw new Error(data.message || 'Failed to fetch previous-run readiness');
  }
  return {
    readyCount: data.readyCount ?? 0,
    readyPositiveCount: data.readyPositiveCount ?? 0,
    readyNegativeCount: data.readyNegativeCount ?? 0,
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

export async function fetchTrainingConfigReadyImages({
  uuid,
  afterUuid = null,
  afterSortValue = null,
  sortBy = 'created_at',
  sortOrder = 'desc',
  limit = 50,
}) {
  const res = await apiFetch(`/api/training-configs/${uuid}/ready-images`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ afterUuid, afterSortValue, sortBy, sortOrder, limit }),
  });
  const data = await res.json();
  if (!res.ok || data.status !== 'success') {
    throw new Error(data.message || 'Failed to fetch ready images');
  }
  return data.images ?? [];
}

export async function fetchTrainingConfigReadyPositiveImages({
  uuid,
  afterUuid = null,
  afterSortValue = null,
  sortBy = 'created_at',
  sortOrder = 'desc',
  limit = 50,
}) {
  const res = await apiFetch(`/api/training-configs/${uuid}/ready-positive-images`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ afterUuid, afterSortValue, sortBy, sortOrder, limit }),
  });
  const data = await res.json();
  if (!res.ok || data.status !== 'success') {
    throw new Error(data.message || 'Failed to fetch positive-sample images');
  }
  return data.images ?? [];
}

export async function fetchTrainingConfigReadyNegativeImages({
  uuid,
  afterUuid = null,
  afterSortValue = null,
  sortBy = 'created_at',
  sortOrder = 'desc',
  limit = 50,
}) {
  const res = await apiFetch(`/api/training-configs/${uuid}/ready-negative-images`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ afterUuid, afterSortValue, sortBy, sortOrder, limit }),
  });
  const data = await res.json();
  if (!res.ok || data.status !== 'success') {
    throw new Error(data.message || 'Failed to fetch background-sample images');
  }
  return data.images ?? [];
}

export async function fetchTrainingConfigPartialReviewImages({
  uuid,
  afterUuid = null,
  afterSortValue = null,
  sortBy = 'created_at',
  sortOrder = 'desc',
  limit = 50,
}) {
  const res = await apiFetch(`/api/training-configs/${uuid}/partial-review-images`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ afterUuid, afterSortValue, sortBy, sortOrder, limit }),
  });
  const data = await res.json();
  if (!res.ok || data.status !== 'success') {
    throw new Error(data.message || 'Failed to fetch partial-review images');
  }
  return data.images ?? [];
}

export async function fetchTrainingConfigNotStartedImages({
  uuid,
  afterUuid = null,
  afterSortValue = null,
  sortBy = 'created_at',
  sortOrder = 'desc',
  limit = 50,
}) {
  const res = await apiFetch(`/api/training-configs/${uuid}/not-started-images`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ afterUuid, afterSortValue, sortBy, sortOrder, limit }),
  });
  const data = await res.json();
  if (!res.ok || data.status !== 'success') {
    throw new Error(data.message || 'Failed to fetch not-started images');
  }
  return data.images ?? [];
}

export async function fetchTrainingConfigImagesByLabelStatus(uuid, {
  labelUuid,
  rollupLabelUuids = undefined,
  status = null,
  missingLabelUuid = null,
  afterUuid = null,
  afterSortValue = null,
  sortBy = 'created_at',
  sortOrder = 'desc',
  limit = 50,
} = {}) {
  const res = await apiFetch(`/api/training-configs/${uuid}/images-by-label-status`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      labelUuid, rollupLabelUuids, status, missingLabelUuid, afterUuid, afterSortValue, sortBy, sortOrder, limit,
    }),
  });
  const data = await res.json();
  if (!res.ok || data.status !== 'success') {
    throw new Error(data.message || 'Failed to fetch images by label status');
  }
  return data.images ?? [];
}

/**
 * Per-image list of the labels a training config still needs resolved.
 *
 * Batched on purpose — pass one rendered page of image uuids, never one image at a time.
 * Images with nothing outstanding are omitted from the returned map.
 *
 * @returns {Promise<Map<string, {labelUuid: string, labelName: string, labelScope: string}[]>>}
 */
export async function fetchTrainingConfigImageGaps(uuid, imageUuids) {
  if (!uuid || !imageUuids?.length) return new Map();
  const res = await apiFetch(`/api/training-configs/${uuid}/image-gaps`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ imageUuids }),
  });
  const data = await res.json();
  if (!res.ok || data.status !== 'success') {
    throw new Error(data.message || 'Failed to fetch image gaps');
  }
  return new Map(Object.entries(data.gaps ?? {}));
}

export async function fetchTrainingConfigLabelStats(uuid) {
  const res = await apiFetch(`/api/training-configs/${uuid}/label-stats`);
  const data = await res.json();
  if (!res.ok || data.status !== 'success') {
    throw new Error(data.message || 'Failed to fetch training run label stats');
  }
  return { labelStats: data.labelStats ?? [], totalImages: data.totalImages ?? 0 };
}

/**
 * The same per-label stats as fetchTrainingConfigLabelStats, but counted over
 * only `imageUuids` — used to replay a previously-trained model's recorded
 * manifest so the "last run" numbers come from the same counting rules as the
 * current ones. `totalImages` comes back as the number of those images still in
 * the config's datasets, which can be lower than imageUuids.length.
 *
 * @param {string} uuid
 * @param {string[]} imageUuids
 */
export async function fetchTrainingConfigLabelStatsForImages(uuid, imageUuids) {
  const res = await apiFetch(`/api/training-configs/${uuid}/label-stats-for-images`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ imageUuids }),
  });
  const data = await res.json();
  if (!res.ok || data.status !== 'success') {
    throw new Error(data.message || 'Failed to fetch previous-run label stats');
  }
  return { labelStats: data.labelStats ?? [], totalImages: data.totalImages ?? 0 };
}
