import { createCachedFetcher } from '/app-static/js/ui/utils/data/createCachedFetcher.js';
import { apiFetch, ignoreKeyReturn } from './apiUtils.js';
import { events, EventTypes } from '/app-static/js/shared/events/index.js';

/** -----------------------------
 * DATASETS CACHE
 * -----------------------------
 */
const datasetsFetcher = createCachedFetcher(async (keys) => {
  const res = await apiFetch('/api/datasets');
  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.message || 'Failed to fetch datasets');
  }
  const data = await res.json();
  if (data.status !== 'success') {
    throw new Error(data.message || 'Failed to fetch datasets');
  }
  const map = new Map(data.datasets.map(d => [d.uuid, d]));
  return ignoreKeyReturn(keys, map);
});

export function clearDatasetsCache() {
  datasetsFetcher.clear();
}

export async function fetchDatasets() {
  return datasetsFetcher.fetch('datasetsMap');
}

export async function fetchDatasetByUuid(uuid) {
  const map = await fetchDatasets();
  return map.get(uuid) ?? null;
}

/** GET /api/datasets/{uuid}/labels */
export async function fetchDatasetLabels(datasetUuid) {
  const res = await apiFetch(`/api/datasets/${datasetUuid}/labels`);
  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.message || 'Failed to fetch dataset labels');
  }
  const data = await res.json();
  if (data.status !== 'success') {
    throw new Error(data.message || 'Failed to fetch dataset labels');
  }
  return data.labels;
}

/** POST /api/datasets/{uuid}/images */
export async function fetchDatasetImages(datasetUuid, params = {}) {
  const query = new URLSearchParams();
  if (params.limit != null) query.set('limit', params.limit);
  if (params.sort_by != null) query.set('sort_by', params.sort_by);
  if (params.sort_order != null) query.set('sort_order', params.sort_order);
  if (params.after_uuid != null) query.set('after_uuid', params.after_uuid);
  if (params.after_sort_value != null) query.set('after_sort_value', params.after_sort_value);

  const url = `/api/datasets/${datasetUuid}/images${query.toString() ? '?' + query.toString() : ''}`;
  const res = await apiFetch(url, { method: 'POST' });
  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.message || 'Failed to fetch dataset images');
  }
  const data = await res.json();
  if (data.status !== 'success') {
    throw new Error(data.message || 'Failed to fetch dataset images');
  }
  return data.images;
}

/** POST /api/datasets */
export async function createDataset({ name, description = '', filterJson = null, labelUuids = [] }) {
  const res = await apiFetch('/api/datasets', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name, description, filterJson, labelUuids }),
  });

  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.message || 'Failed to create dataset');
  }
  const data = await res.json();
  if (data.status !== 'success') {
    throw new Error(data.message || 'Failed to create dataset');
  }

  datasetsFetcher.clear();
  events.publish(EventTypes.DATASETS_UPDATED, {});
  return data.dataset;
}

/** PATCH /api/datasets/{uuid} */
export async function updateDataset(uuid, patch) {
  const body = {};
  if (patch.name != null) body.name = patch.name;
  if (patch.description !== undefined) body.description = patch.description;
  if (patch.filterJson !== undefined) body.filterJson = patch.filterJson;
  if (patch.labelUuids != null) body.labelUuids = patch.labelUuids;

  const res = await apiFetch(`/api/datasets/${uuid}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });

  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.message || 'Failed to update dataset');
  }
  const data = await res.json();
  if (data.status !== 'success') {
    throw new Error(data.message || 'Failed to update dataset');
  }

  datasetsFetcher.clear();
  events.publish(EventTypes.DATASETS_UPDATED, {});
  return data.dataset;
}

/** POST /api/datasets/preview — count images matching unsaved filter state */
export async function previewDatasetImages({ filterJson, labelUuids = [], limit = 500 } = {}) {
  const query = new URLSearchParams({ limit });
  const url = `/api/datasets/preview?${query}`;
  const res = await apiFetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      filterJson: filterJson || { conditions: [] },
      labelUuids,
    }),
  });
  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.message || 'Failed to preview dataset');
  }
  const data = await res.json();
  if (data.status !== 'success') {
    throw new Error(data.message || 'Failed to preview dataset');
  }
  return { images: data.images, hasMore: data.hasMore };
}

/** DELETE /api/datasets/{uuid} */
export async function deleteDataset(uuid) {
  const res = await apiFetch(`/api/datasets/${uuid}`, { method: 'DELETE' });

  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.message || 'Failed to delete dataset');
  }

  datasetsFetcher.clear();
  events.publish(EventTypes.DATASETS_UPDATED, {});
}
