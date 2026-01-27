import { createCachedFetcher } from '/app-static/js/ui/utils/data/createCachedFetcher.js';
import { ignoreKeyReturn } from './apiUtils.js';

/** -----------------------------
 * CLASS CACHE
 * -----------------------------
 */
export const classesFetcher = createCachedFetcher(async (keys) => {
  /** @type {import('@web_api').ListClassesPayload} */
  const payload = {};

  const res = await fetch('/api/classes/list');

  /** @type {import('@web_api').ListClassesResponse} */
  const data = await res.json();

  if (!res.ok || data.status !== 'success') {
    throw new Error(data.message || 'Failed to fetch classes');
  }
  // Store as Map keyed by UUID
  const map = new Map(data.classes.map(l => [l.metadata.uuid, l]));

  return ignoreKeyReturn(keys, map);
});


/** -----------------------------
 * CACHE CLEAR
 * -----------------------------
 */
export function clearClassCache() {
  classesFetcher.clear();
}


/** -----------------------------
 * FETCH FUNCTIONS
 * -----------------------------
 */
export async function fetchClasses() {
  return await classesFetcher.fetch('classesMap');
}

export async function fetchClassByUuid(uuid) {
  const classes = await fetchClasses();
  return classes.get(uuid) || null;
}


/** -----------------------------
 * CRUD FUNCTIONS
 * -----------------------------
 */
export async function createNewClass(name, color, parentUuid = null) {
  const res = await fetch('/api/classes/add', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name, color, parent_uuid: parentUuid }),
  });

  if (!res.ok) {
    const { message } = await res.json();
    throw new Error(message || 'Failed to add class');
  }

  // Invalidate cache after successful creation
  classesFetcher.clear();
}

export async function updateClass(uuid, name, color) {
  const res = await fetch('/api/classes/update', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ class_uuid: uuid, name, color }),
  });

  if (!res.ok) {
    const { message } = await res.json();
    throw new Error(message || 'Failed to update class');
  }

  // Invalidate cache after successful update
  classesFetcher.clear();
}

export async function deleteClass(uuid) {
  const res = await fetch('/api/classes/delete', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ class_uuid: uuid }),
  });

  if (!res.ok) {
    const { message } = await res.json();
    throw new Error(message || 'Failed to delete class');
  }

  // Invalidate cache after successful deletion
  classesFetcher.clear();
}
