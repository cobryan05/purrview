import { createCachedFetcher } from '/app-static/js/ui/utils/data/createCachedFetcher.js';
import { apiFetch, ignoreKeyReturn } from './apiUtils.js';

/** -----------------------------
 * CACHE
 * -----------------------------
 */
const modelGroupsFetcher = createCachedFetcher(async (keys) => {
  const res = await apiFetch('/api/model-groups');
  const data = await res.json();
  if (!res.ok || data.status !== 'success') {
    throw new Error(data.message || 'Failed to fetch model groups');
  }
  const map = new Map(data.groups.map(g => [g.uuid, g]));
  return ignoreKeyReturn(keys, map);
});

export function clearModelGroupsCache() {
  modelGroupsFetcher.clear();
}

/** -----------------------------
 * FETCH
 * -----------------------------
 */
export async function fetchModelGroups() {
  return modelGroupsFetcher.fetch('groupsMap');
}

/** -----------------------------
 * CRUD
 * -----------------------------
 */
export async function createModelGroup({ name, description = '', members = [] }) {
  const res = await apiFetch('/api/model-groups', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name, description, members }),
  });
  const data = await res.json();
  if (!res.ok || data.status !== 'success') {
    throw new Error(data.message || 'Failed to create model group');
  }
  modelGroupsFetcher.clear();
  return data.group;
}

export async function updateModelGroup(uuid, { name, description, members }) {
  const body = {};
  if (name !== undefined) body.name = name;
  if (description !== undefined) body.description = description;
  if (members !== undefined) body.members = members;

  const res = await apiFetch(`/api/model-groups/${uuid}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  const data = await res.json();
  if (!res.ok || data.status !== 'success') {
    throw new Error(data.message || 'Failed to update model group');
  }
  modelGroupsFetcher.clear();
  return data.group;
}

export async function deleteModelGroup(uuid) {
  const res = await apiFetch(`/api/model-groups/${uuid}`, { method: 'DELETE' });
  const data = await res.json();
  if (!res.ok || data.status !== 'success') {
    throw new Error(data.message || 'Failed to delete model group');
  }
  modelGroupsFetcher.clear();
}
