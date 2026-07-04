import { createCachedFetcher } from '/app-static/js/ui/utils/data/createCachedFetcher.js';
import { apiFetch, ignoreKeyReturn } from './apiUtils.js';

/** -----------------------------
 * CACHE
 * -----------------------------
 */
const filterChainsFetcher = createCachedFetcher(async (keys) => {
  const res = await apiFetch('/api/filter-chains');
  const data = await res.json();
  if (!res.ok || data.status !== 'success') {
    throw new Error(data.message || 'Failed to fetch filter chains');
  }
  const map = new Map(data.chains.map(c => [c.uuid, c]));
  return ignoreKeyReturn(keys, map);
});

export function clearFilterChainsCache() {
  filterChainsFetcher.clear();
}

/** -----------------------------
 * FETCH
 * -----------------------------
 */
export async function fetchFilterChains() {
  return filterChainsFetcher.fetch('chainsMap');
}

/** -----------------------------
 * CRUD
 * -----------------------------
 */
export async function createFilterChain({ name, description = '', steps = [] }) {
  const res = await apiFetch('/api/filter-chains', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name, description, steps }),
  });
  const data = await res.json();
  if (!res.ok || data.status !== 'success') {
    throw new Error(data.message || 'Failed to create filter chain');
  }
  filterChainsFetcher.clear();
  return data.chain;
}

export async function updateFilterChain(uuid, { name, description, steps }) {
  const body = {};
  if (name !== undefined) body.name = name;
  if (description !== undefined) body.description = description;
  if (steps !== undefined) body.steps = steps;

  const res = await apiFetch(`/api/filter-chains/${uuid}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  const data = await res.json();
  if (!res.ok || data.status !== 'success') {
    throw new Error(data.message || 'Failed to update filter chain');
  }
  filterChainsFetcher.clear();
  return data.chain;
}

export async function deleteFilterChain(uuid) {
  const res = await apiFetch(`/api/filter-chains/${uuid}`, { method: 'DELETE' });
  const data = await res.json();
  if (!res.ok || data.status !== 'success') {
    throw new Error(data.message || 'Failed to delete filter chain');
  }
  filterChainsFetcher.clear();
}
