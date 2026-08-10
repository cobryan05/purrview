import { createCachedFetcher } from '/app-static/js/ui/utils/data/createCachedFetcher.js';
import { apiFetch, ignoreKeyReturn } from './apiUtils.js';

/** -----------------------------
 * CACHE
 * -----------------------------
 */
const watchProfilesFetcher = createCachedFetcher(async (keys) => {
  const res = await apiFetch('/api/watch-profiles');
  const data = await res.json();
  if (!res.ok || data.status !== 'success') {
    throw new Error(data.message || 'Failed to fetch watchers');
  }
  // NOTE: a Map, not an array — same convention as fetchLabels(). Iterate with
  // .values(); a bare for...of yields [key, value] pairs.
  const map = new Map((data.profiles || []).map(p => [p.uuid, p]));
  return ignoreKeyReturn(keys, map);
});

export function clearWatchProfilesCache() {
  watchProfilesFetcher.clear();
}

/** -----------------------------
 * FETCH
 * -----------------------------
 */
export async function fetchWatchProfiles() {
  return watchProfilesFetcher.fetch('watchProfilesMap');
}

export async function fetchWatchProfile(uuid) {
  const res = await apiFetch(`/api/watch-profiles/${uuid}`);
  const data = await res.json();
  if (!res.ok || data.status !== 'success') {
    throw new Error(data.message || 'Failed to fetch watcher');
  }
  return data.profile;
}

/**
 * Configuration problems that would otherwise only show up at run time — and
 * silently, which is why this is worth a round trip rather than a client-side guess:
 * a rule with a missing zone simply never fires.
 */
export async function validateWatchProfile(uuid) {
  const res = await apiFetch(`/api/watch-profiles/${uuid}/validate`);
  const data = await res.json();
  if (!res.ok || data.status !== 'success') {
    throw new Error(data.message || 'Failed to validate watcher');
  }
  return { valid: data.valid, errors: data.errors || [], warnings: data.warnings || [] };
}

/** -----------------------------
 * CRUD
 * -----------------------------
 */
export async function createWatchProfile(payload) {
  const res = await apiFetch('/api/watch-profiles', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  const data = await res.json();
  if (!res.ok || data.status !== 'success') {
    throw new Error(data.message || 'Failed to create watcher');
  }
  watchProfilesFetcher.clear();
  return data.profile;
}

export async function updateWatchProfile(uuid, payload) {
  const res = await apiFetch(`/api/watch-profiles/${uuid}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  const data = await res.json();
  if (!res.ok || data.status !== 'success') {
    throw new Error(data.message || 'Failed to update watcher');
  }
  watchProfilesFetcher.clear();
  return data.profile;
}

export async function deleteWatchProfile(uuid) {
  const res = await apiFetch(`/api/watch-profiles/${uuid}`, { method: 'DELETE' });
  const data = await res.json();
  if (!res.ok || data.status !== 'success') {
    throw new Error(data.message || 'Failed to delete watcher');
  }
  watchProfilesFetcher.clear();
}

/** -----------------------------
 * EVENTS / RECORDINGS
 * -----------------------------
 */
export async function fetchWatchEvents({ profileUuid, sourceUuid, limit = 100 } = {}) {
  const params = new URLSearchParams();
  if (profileUuid) params.set('profile_uuid', profileUuid);
  if (sourceUuid) params.set('source_uuid', sourceUuid);
  params.set('limit', String(limit));
  const res = await apiFetch(`/api/watch-events/?${params}`);
  const data = await res.json();
  if (!res.ok || data.status !== 'success') {
    throw new Error(data.message || 'Failed to fetch events');
  }
  return data.events || [];
}

export async function fetchRecordings({ profileUuid, sourceUuid, limit = 100 } = {}) {
  const params = new URLSearchParams();
  if (profileUuid) params.set('profile_uuid', profileUuid);
  if (sourceUuid) params.set('source_uuid', sourceUuid);
  params.set('limit', String(limit));
  const res = await apiFetch(`/api/recordings/?${params}`);
  const data = await res.json();
  if (!res.ok || data.status !== 'success') {
    throw new Error(data.message || 'Failed to fetch recordings');
  }
  return data.recordings || [];
}
