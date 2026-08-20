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

/**
 * Available trackers and their parameter schemas.
 *
 * Registry-driven so a new tracker appears in the picker with no change here: the
 * backend returns `paramsSchema` verbatim from the tracker class, and the editor
 * renders whatever fields it declares.
 */
export async function fetchTrackers() {
  const res = await apiFetch('/api/watch-profiles/trackers');
  const data = await res.json();
  if (!res.ok || data.status !== 'success') {
    throw new Error(data.message || 'Failed to fetch trackers');
  }
  return data.trackers || [];
}

/**
 * Available actions and their parameter schemas.
 *
 * The same registry-driven arrangement the trackers have, and for a reason the actions
 * learned the hard way: the Watchers pane used to hold a hand-written form per action
 * kind, so a parameter added on the Python side never appeared, and one removed from it
 * carried on being written into saved config by a form that still offered it.
 *
 * @returns {Promise<Array<{kind: string, displayName: string, paramsSchema: object}>>}
 */
export async function fetchWatchActionSchemas() {
  const res = await apiFetch('/api/watch-profiles/actions');
  const data = await res.json();
  if (!res.ok || data.status !== 'success') {
    throw new Error(data.message || 'Failed to fetch action schemas');
  }
  return data.actions || [];
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
/**
 * Search the recorded event history.
 *
 * Returns `{events, total}` rather than a bare array: `total` counts the whole filtered
 * set, not the page, and the review page's "N events" and paging both need it. Callers
 * that only want the rows should destructure — a `.forEach` on the result would
 * silently do nothing.
 *
 * @param {Object} [opts]
 * @param {?string} [opts.profileUuid]
 * @param {?string} [opts.sourceUuid] - the camera
 * @param {?string} [opts.ruleUuid] - the event type
 * @param {?string} [opts.subjectRef] - the acting thing
 * @param {?string} [opts.targetRef] - the thing acted on
 * @param {?string} [opts.startFrom] - ISO, inclusive
 * @param {?string} [opts.startTo] - ISO, inclusive
 * @param {?boolean} [opts.hasRecording]
 * @param {number} [opts.limit]
 * @param {number} [opts.offset]
 * @returns {Promise<{events: Array<Object>, total: number}>}
 */
export async function fetchWatchEvents({
  profileUuid,
  sourceUuid,
  ruleUuid,
  subjectRef,
  targetRef,
  startFrom,
  startTo,
  hasRecording,
  limit = 60,
  offset = 0,
} = {}) {
  const params = new URLSearchParams();
  if (profileUuid) params.set('profile_uuid', profileUuid);
  if (sourceUuid) params.set('source_uuid', sourceUuid);
  if (ruleUuid) params.set('rule_uuid', ruleUuid);
  if (subjectRef) params.set('subject_ref', subjectRef);
  if (targetRef) params.set('target_ref', targetRef);
  if (startFrom) params.set('start_from', startFrom);
  if (startTo) params.set('start_to', startTo);
  // Explicit null check: `false` is a real filter ("events with no clip") and must not
  // be dropped the way a falsy guard would drop it.
  if (hasRecording != null) params.set('has_recording', String(hasRecording));
  params.set('limit', String(limit));
  params.set('offset', String(offset));
  const res = await apiFetch(`/api/watch-events/?${params}`);
  const data = await res.json();
  if (!res.ok || data.status !== 'success') {
    throw new Error(data.message || 'Failed to fetch events');
  }
  return { events: data.events || [], total: data.total ?? 0 };
}

/**
 * The values each filter dropdown can offer, taken from history rather than config.
 *
 * @param {Object} [opts]
 * @param {?string} [opts.profileUuid]
 * @returns {Promise<{rules: Array, subjects: Array, targets: Array, sources: Array}>}
 */
export async function fetchWatchEventFacets({ profileUuid } = {}) {
  const params = new URLSearchParams();
  if (profileUuid) params.set('profile_uuid', profileUuid);
  const res = await apiFetch(`/api/watch-events/facets?${params}`);
  const data = await res.json();
  if (!res.ok || data.status !== 'success') {
    throw new Error(data.message || 'Failed to fetch event filters');
  }
  return {
    rules: data.rules || [],
    subjects: data.subjects || [],
    targets: data.targets || [],
    sources: data.sources || [],
  };
}

/** @param {string} uuid @returns {Promise<Object>} */
export async function fetchWatchEvent(uuid) {
  const res = await apiFetch(`/api/watch-events/${uuid}`);
  const data = await res.json();
  if (!res.ok || data.status !== 'success') {
    throw new Error(data.message || 'Failed to fetch event');
  }
  return data.event;
}

export async function deleteWatchEvent(uuid) {
  const res = await apiFetch(`/api/watch-events/${uuid}`, { method: 'DELETE' });
  const data = await res.json();
  if (!res.ok || data.status !== 'success') {
    throw new Error(data.message || 'Failed to delete event');
  }
}

/**
 * Delete a selection of events in one request.
 *
 * One round trip rather than a DELETE per card: a full page is 60 events, and 60
 * sequential requests each unlinking clip files is slow enough that the grid sits
 * visibly half-deleted. The server deletes what it can and reports the count, so a
 * uuid that has already gone (deleted in another tab) is not an error here either.
 *
 * @param {string[]} uuids
 * @returns {Promise<number>} how many were actually deleted
 */
export async function deleteWatchEvents(uuids) {
  const res = await apiFetch('/api/watch-events/delete', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ uuids }),
  });
  const data = await res.json();
  if (!res.ok || data.status !== 'success') {
    throw new Error(data.message || 'Failed to delete events');
  }
  return data.deleted ?? 0;
}

/**
 * One clip's metadata — needed for its real start time, which is earlier than its
 * event's by however much pre-roll the rewind buffer actually had.
 *
 * @param {string} uuid
 * @returns {Promise<?Object>} null when the row is gone
 */
export async function fetchRecording(uuid) {
  const res = await apiFetch(`/api/recordings/${uuid}`);
  const data = await res.json();
  if (!res.ok || data.status !== 'success') return null;
  return data.recording ?? null;
}

/**
 * The event's card art. A URL, not a fetch: an <img> loads it itself, and routing it
 * through fetch would buy nothing but a blob URL to revoke.
 *
 * @param {string} uuid
 * @returns {string}
 */
export function watchEventSnapshotUrl(uuid) {
  return `/api/watch-events/${uuid}/snapshot`;
}

/**
 * The clip, as a `<video>` src. Served through this port like everything else — the
 * proxy forwards Range requests, which is what makes seeking work.
 *
 * @param {string} recordingUuid
 * @returns {string}
 */
export function recordingFileUrl(recordingUuid) {
  return `/api/recordings/${recordingUuid}/file`;
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
