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

// Separate cache for `include_views=true`, which returns Views alongside
// promoted Datasets in one list (decision register #10: one table, one flag).
// Kept apart from datasetsFetcher so the five "give me trainable Datasets"
// call sites can't accidentally pick up this query's Views.
const allRowsFetcher = createCachedFetcher(async (keys) => {
  const res = await apiFetch('/api/datasets?include_views=true');
  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.message || 'Failed to fetch views');
  }
  const data = await res.json();
  if (data.status !== 'success') {
    throw new Error(data.message || 'Failed to fetch views');
  }
  const map = new Map(data.datasets.map(d => [d.uuid, d]));
  return ignoreKeyReturn(keys, map);
});

export function clearDatasetsCache() {
  datasetsFetcher.clear();
  allRowsFetcher.clear();
}

export async function fetchDatasets() {
  return datasetsFetcher.fetch('datasetsMap');
}

/** Saved Views (unpromoted rows) — for the Review tab's View picker / manager. */
export async function fetchViews() {
  const map = await allRowsFetcher.fetch('allRowsMap');
  return new Map([...map].filter(([, d]) => !d.promoted));
}

/**
 * Datasets *and* Views in one map — for the management surface, which is the
 * only screen that shows both and is where promotion happens.
 *
 * ⚠️ Never substitute this for fetchDatasets() in a training/model picker:
 * those five call sites all mean "a real, trainable Dataset".
 */
export async function fetchDatasetsAndViews() {
  return allRowsFetcher.fetch('allRowsMap');
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

/** POST /api/datasets. `promoted: false` creates a View instead of a Dataset. */
export async function createDataset({ name, description = '', filterJson = null, labelUuids = [], promoted } = {}) {
  const body = { name, description, filterJson, labelUuids };
  if (promoted !== undefined) body.promoted = promoted;
  const res = await apiFetch('/api/datasets', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });

  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.message || 'Failed to create dataset');
  }
  const data = await res.json();
  if (data.status !== 'success') {
    throw new Error(data.message || 'Failed to create dataset');
  }

  clearDatasetsCache();
  events.publish(EventTypes.DATASETS_UPDATED, {});
  return data.dataset;
}

/** PATCH /api/datasets/{uuid}. `promoted` flips a row between View and Dataset. */
export async function updateDataset(uuid, patch) {
  const body = {};
  if (patch.name != null) body.name = patch.name;
  if (patch.description !== undefined) body.description = patch.description;
  if (patch.filterJson !== undefined) body.filterJson = patch.filterJson;
  if (patch.labelUuids != null) body.labelUuids = patch.labelUuids;
  if (patch.promoted !== undefined) body.promoted = patch.promoted;

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

  clearDatasetsCache();
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
      // `{}` is a valid empty DatasetFilterCriteria (all fields optional).
      // The old `{ conditions: [] }` default here was already-stale legacy
      // shape from before terms existed; as of PR 1.4b DatasetFilterCriteria
      // rejects unknown fields (extra='forbid'), so it would now 422.
      filterJson: filterJson || {},
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

  clearDatasetsCache();
  events.publish(EventTypes.DATASETS_UPDATED, {});
}

/**
 * GET /api/datasets/export-formats — the export layouts the server can write.
 *
 * Fetched rather than hardcoded: a copy of this list in JS would go stale the
 * first time a format is added on the Python side, and the failure (a picker
 * offering a format the server rejects) would only show up on click.
 */
const exportFormatsFetcher = createCachedFetcher(async (keys) => {
  const res = await apiFetch('/api/datasets/export-formats');
  if (!res.ok) throw new Error('Failed to fetch export formats');
  const data = await res.json();
  if (data.status !== 'success') {
    throw new Error(data.message || 'Failed to fetch export formats');
  }
  return ignoreKeyReturn(keys, data.formats);
});

export async function fetchDatasetExportFormats() {
  return exportFormatsFetcher.fetch('exportFormats');
}

/**
 * URL for the export download. Not fetched through `apiFetch`: the archive is
 * streamed straight to disk by the browser, so the page navigates to this URL
 * (the `Content-Disposition: attachment` response downloads without unloading
 * the SPA) instead of buffering an entire dataset into a Blob in memory.
 */
export function datasetExportUrl(uuid, { format = 'yolo', valFraction = 0.2, requireVerification = true } = {}) {
  const query = new URLSearchParams({
    format,
    val_fraction: String(valFraction),
    require_verification: String(requireVerification),
  });
  return `/api/datasets/${uuid}/export?${query.toString()}`;
}
