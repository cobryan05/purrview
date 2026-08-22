import { createCachedFetcher } from '/app-static/js/ui/utils/data/createCachedFetcher.js';
import { apiFetch, ignoreKeyReturn } from './apiUtils.js';
import { events, EventTypes } from '/app-static/js/shared/events/index.js';

/** -----------------------------
 * TAGS CACHE
 * -----------------------------
 */
export const tagsFetcher = createCachedFetcher(async (keys) => {
  const res = await apiFetch('/api/tags/list');
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
  // A uuid that was missing may exist again (re-created, or restored); an
  // explicit invalidation is exactly the moment to stop suppressing its lookup.
  _knownMissing.clear();
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
  if (tags.has(uuid)) return tags.get(uuid);
  return (await refreshTagsIfUnknown([uuid])).get(uuid) || null;
}

/**
 * Tag uuids we have already refetched for and still do not know about.
 *
 * Without this, a genuinely deleted tag whose uuid is still referenced by some
 * saved filter would trigger a fresh `/api/tags/list` on every render.
 */
const _knownMissing = new Set();
let _missRefresh = null;

/**
 * Heal the cache when the app meets a tag uuid it has never heard of.
 *
 * The client only invalidates its tag cache on its *own* CRUD, but tags are
 * also minted server-side — `CollectTrainingImagesTask` creates a TASK_GENERATED
 * tag named after the task, and renaming that task mints another one. That tag
 * exists in the DB and comes back attached to images, but the running SPA's map
 * has never seen it, so every `tags.get(uuid)?.name ?? uuid` fallback rendered
 * the raw uuid until a page reload.
 *
 * Refetches once (coalesced across the many chips that render at the same
 * moment) and publishes TAGS_UPDATED only if something new was actually
 * learned, so a re-render cannot loop.
 *
 * @param {Iterable<string>} uuids
 * @returns {Promise<Map<string, Object>>} the (possibly refreshed) tags map
 */
export async function refreshTagsIfUnknown(uuids) {
  const tags = await fetchTags();
  const unknown = [...uuids].filter(u => u && !tags.has(u) && !_knownMissing.has(u));
  if (unknown.length === 0) return tags;

  _missRefresh ??= (async () => {
    tagsFetcher.clear();
    return await fetchTags();
  })().finally(() => { _missRefresh = null; });

  const fresh = await _missRefresh;
  const learned = unknown.filter(u => fresh.has(u));
  unknown.filter(u => !fresh.has(u)).forEach(u => _knownMissing.add(u));
  if (learned.length) events.publish(EventTypes.TAGS_UPDATED, {});
  return fresh;
}


/** -----------------------------
 * CRUD FUNCTIONS
 * -----------------------------
 */
export async function createNewTag(name, color, icon = null) {
  const res = await apiFetch('/api/tags/add', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name, color, icon }),
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
  events.publish(EventTypes.TAGS_UPDATED, {});
  return data.tag;
}

export async function updateTag(uuid, name, color, icon = undefined) {
  const res = await apiFetch('/api/tags/update', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ tagUuid: uuid, data: { name, color, icon } }),
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
  events.publish(EventTypes.TAGS_UPDATED, {});
}

export async function deleteTag(uuid) {
  const res = await apiFetch('/api/tags/delete', {
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
  events.publish(EventTypes.TAGS_UPDATED, {});
}
