import { createCachedFetcher } from '/app-static/js/ui/utils/data/createCachedFetcher.js';
import { ignoreKeyReturn } from './apiUtils.js';

/** -----------------------------
 * TAGS CACHE
 * -----------------------------
 */
export const tagsFetcher = createCachedFetcher(async (keys) => {
  const res = await fetch('/api/tags/list');
  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.message || 'Failed to fetch tags');
  }
  const { tags } = await res.json();
  // Store as Map keyed by UUID
  const ret = new Map(tags.map(l => [l.uuid, l]));
  return ignoreKeyReturn(keys, ret);
});


/** -----------------------------
 * CACHE CLEAR
 * -----------------------------
 */
export function clearTagCache() {
  tagsFetcher.clear();
}


/** -----------------------------
 * FETCH FUNCTIONS
 * -----------------------------
 */
export async function fetchTags() {
  return await tagsFetcher.fetch('tagsMap');
}

export async function fetchTagByUuid(uuid) {
  const tags = await fetchTags();
  return tags.get(uuid) || null;
}


/** -----------------------------
 * CRUD FUNCTIONS
 * -----------------------------
 */
export async function createNewTag(name, color) {
  const res = await fetch('/api/tags/add', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name, color }),
  });

  if (!res.ok) {
    const { message } = await res.json();
    throw new Error(message || 'Failed to add tag');
  }

  const data = await res.json();
  if (data.status != 'success') {
    throw new Error(`Creating tag failed: ${data.message}`);
  }

  // Invalidate cache after successful creation
  tagsFetcher.clear();
}

export async function updateTag(uuid, name, color) {
  const res = await fetch('/api/tags/update', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ tagUuid: uuid, data: { name, color } }),
  });

  if (!res.ok) {
    const { message } = await res.json();
    throw new Error(message || 'Failed to update tag');
  }

  const data = await res.json();
  if (data.status != 'success') {
    throw new Error(`Updating tag failed: ${data.message}`);
  }

  // Invalidate cache after successful update
  tagsFetcher.clear();
}

export async function deleteTag(uuid) {
  const res = await fetch('/api/tags/delete', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ tagUuid: uuid }),
  });

  if (!res.ok) {
    const { message } = await res.json();
    throw new Error(message || 'Failed to delete tag');
  }

  // Invalidate cache after successful deletion
  tagsFetcher.clear();
}
