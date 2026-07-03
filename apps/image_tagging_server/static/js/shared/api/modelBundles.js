import { createCachedFetcher } from '/app-static/js/ui/utils/data/createCachedFetcher.js';
import { apiFetch, ignoreKeyReturn } from './apiUtils.js';

/** -----------------------------
 * CACHE
 * -----------------------------
 */
const modelBundlesFetcher = createCachedFetcher(async (keys) => {
  const res = await apiFetch('/api/model-bundles');
  const data = await res.json();
  if (!res.ok || data.status !== 'success') {
    throw new Error(data.message || 'Failed to fetch model bundles');
  }
  const map = new Map(data.bundles.map(b => [b.uuid, b]));
  return ignoreKeyReturn(keys, map);
});

export function clearModelBundlesCache() {
  modelBundlesFetcher.clear();
}

/** -----------------------------
 * FETCH
 * -----------------------------
 */

/** @returns {Promise<Map<string, object>>} uuid → ModelBundleRead */
export async function fetchModelBundles() {
  return modelBundlesFetcher.fetch('bundlesMap');
}

/**
 * Convenience: bundles trained from the given model config, newest first.
 * @param {string} modelConfigUuid
 * @returns {Promise<object[]>}
 */
export async function fetchModelBundlesForConfig(modelConfigUuid) {
  const map = await fetchModelBundles();
  return [...map.values()]
    .filter(b => b.modelConfigUuid === modelConfigUuid)
    .sort((a, b) => (b.createdAt ?? '').localeCompare(a.createdAt ?? ''));
}

/** -----------------------------
 * CRUD
 * -----------------------------
 */

/**
 * Create a new ModelBundle. Used both by TrainBundleTask (server-side) and by
 * the "New bundle from artifacts" compose flow, which reuses artifacts already
 * on disk from earlier bundle/training runs (no new training happens here —
 * bundle_path stays the flat models root, manifest_json just points at the
 * chosen artifacts).
 * @param {{name:string, description?:string, modelConfigUuid:string, bundlePath:string,
 *   manifestJson?:object, trainingConfigUuids?:string[]}} data
 * @returns {Promise<object>} the created ModelBundleRead
 */
export async function createModelBundle({
  name, description = '', modelConfigUuid, bundlePath, manifestJson = {}, trainingConfigUuids = [],
}) {
  const res = await apiFetch('/api/model-bundles', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      name, description, modelConfigUuid, bundlePath, manifestJson, trainingConfigUuids,
    }),
  });
  const data = await res.json();
  if (!res.ok || data.status !== 'success') {
    throw new Error(data.message || 'Failed to create model bundle');
  }
  modelBundlesFetcher.clear();
  return data.bundle;
}
