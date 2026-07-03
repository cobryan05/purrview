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
