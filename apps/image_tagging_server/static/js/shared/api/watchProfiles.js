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
 * @param {string[]} [opts.sourceUuids] - cameras to include; any of
 * @param {string[]} [opts.ruleUuids] - event types to include; any of
 * @param {string[]} [opts.subjectRefs] - acting things to include; any of
 * @param {string[]} [opts.targetRefs] - things acted on to include; any of
 * @param {string[]} [opts.sourceExcludeUuids] - cameras to exclude
 * @param {string[]} [opts.ruleExcludeUuids] - event types to exclude
 * @param {string[]} [opts.subjectExcludeRefs] - acting things to exclude
 * @param {string[]} [opts.targetExcludeRefs] - things acted on to exclude
 * @param {?string} [opts.startFrom] - ISO, inclusive
 * @param {?string} [opts.startTo] - ISO, inclusive
 * @param {?boolean} [opts.hasRecording] - owns any media, or none
 * @param {?string} [opts.recordingKind] - owns one of a specific kind: 'clip' | 'timelapse'
 * @param {number} [opts.limit]
 * @param {number} [opts.offset]
 * @returns {Promise<{events: Array<Object>, total: number}>}
 */
/**
 * Turn an event filter state into the query string every event endpoint understands.
 *
 * The grid, the grouped view and the timeline all ask the *same* question and differ
 * only in what comes back, so they must serialise the filters identically — a filter
 * spelled here and forgotten in one of the other two would show a group count that
 * disagreed with the grid it drills into. The backend holds the matching halves of this
 * contract in `WatchEventQuery` and `DbClient._apply_watch_event_filters`.
 *
 * @param {Object} [state] - an EventFilterState (see app/nvr/events/filters.js)
 * @returns {URLSearchParams}
 */
export function watchEventFilterParams({
  profileUuid,
  sourceUuid,
  ruleUuid,
  subjectRef,
  targetRef,
  sourceUuids,
  ruleUuids,
  subjectRefs,
  targetRefs,
  sourceExcludeUuids,
  ruleExcludeUuids,
  subjectExcludeRefs,
  targetExcludeRefs,
  startFrom,
  startTo,
  hasRecording,
  recordingKind,
  hasBlips,
  reviewState,
} = {}) {
  const params = new URLSearchParams();
  if (profileUuid) params.set('profile_uuid', profileUuid);
  if (sourceUuid) params.set('source_uuid', sourceUuid);
  if (ruleUuid) params.set('rule_uuid', ruleUuid);
  if (subjectRef) params.set('subject_ref', subjectRef);
  if (targetRef) params.set('target_ref', targetRef);
  // Repeated rather than comma-joined: a rule name is free text and may contain a
  // comma, which a joined list would silently split into two filters.
  for (const [key, values] of [
    ['source_uuids', sourceUuids],
    ['rule_uuids', ruleUuids],
    ['subject_refs', subjectRefs],
    ['target_refs', targetRefs],
    ['exclude_source_uuids', sourceExcludeUuids],
    ['exclude_rule_uuids', ruleExcludeUuids],
    ['exclude_subject_refs', subjectExcludeRefs],
    ['exclude_target_refs', targetExcludeRefs],
  ]) {
    for (const value of values ?? []) params.append(key, value);
  }
  if (startFrom) params.set('start_from', startFrom);
  if (startTo) params.set('start_to', startTo);
  // Explicit null checks: `false` is a real filter ("events with no clip") and must not
  // be dropped the way a falsy guard would drop it.
  if (hasRecording != null) params.set('has_recording', String(hasRecording));
  if (recordingKind) params.set('recording_kind', recordingKind);
  if (hasBlips != null) params.set('has_blips', String(hasBlips));
  if (reviewState) params.set('review_state', reviewState);
  return params;
}

export async function fetchWatchEvents({ limit = 60, offset = 0, ...filters } = {}) {
  const params = watchEventFilterParams(filters);
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
 * The filtered events collapsed to one row per identity, biggest first.
 *
 * Tens of rows for any filter, because it aggregates server-side — the counts are
 * properties of the whole filtered set and the grid only ever holds a page of it.
 * Deliberately not answerable by fetching every event with a huge `limit`: this log is
 * append-only and the list endpoint joins recordings and blip counts per row.
 *
 * @param {Object} [filters] - an EventFilterState
 * @returns {Promise<{groups: Array<Object>, total: number}>}
 */
export async function fetchWatchEventGroups(filters = {}) {
  const res = await apiFetch(`/api/watch-events/groups?${watchEventFilterParams(filters)}`);
  const data = await res.json();
  if (!res.ok || data.status !== 'success') {
    throw new Error(data.message || 'Failed to fetch event groups');
  }
  return { groups: data.groups || [], total: data.total ?? 0 };
}

/**
 * The flat grid's events, collapsed into the visits they describe.
 *
 * ⚠️ Paged over *sessions*, not events, and computed server-side — which is the whole
 * reason this is an endpoint rather than a `reduce` over the page. Grouping in the
 * browser would split every session that straddles a page boundary and would group
 * differently depending on which page you landed on.
 *
 * Expect roughly a halving, not the order-of-magnitude the grouped view gives: on live
 * history a 120s gap turns 1277 events into 653 sessions. It is what stops one drink at
 * the bowl looking like seven visits, not what makes the page comprehensible.
 *
 * @param {Object} [opts]
 * @param {number} [opts.gapS] - seconds of quiet that end a session
 * @returns {Promise<{sessions: Array<Object>, total: number, gapS: number}>}
 */
export async function fetchWatchEventSessions({ gapS = null, limit = 60, offset = 0, ...filters } = {}) {
  const params = watchEventFilterParams(filters);
  if (gapS != null) params.set('gap_s', String(gapS));
  params.set('limit', String(limit));
  params.set('offset', String(offset));
  const res = await apiFetch(`/api/watch-events/sessions?${params}`);
  const data = await res.json();
  if (!res.ok || data.status !== 'success') {
    throw new Error(data.message || 'Failed to fetch event sessions');
  }
  return { sessions: data.sessions || [], total: data.total ?? 0, gapS: data.gapS ?? gapS };
}

/**
 * Move a selection of events to one triage state.
 *
 * @param {string[]} uuids
 * @param {string} state - 'unreviewed' | 'confirmed' | 'dismissed' | 'flagged'
 * @returns {Promise<number>} how many rows actually moved
 */
export async function setWatchEventReviewStates(uuids, state) {
  const res = await apiFetch('/api/watch-events/review-states', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ uuids, state }),
  });
  const data = await res.json();
  if (!res.ok || data.status !== 'success') {
    throw new Error(data.message || 'Failed to update those events');
  }
  return data.updated ?? 0;
}

/**
 * Event counts per time bucket per identity — the timeline's data source.
 *
 * ⚠️ `bucketStart` is **UTC** and carries no offset. Ask for hour buckets and compose
 * local days from them (`groupBucketsByLocalDay`); the server has no timezone parameter
 * on purpose. `bucket` in the response is the resolution actually used, which is `day`
 * when an hour-resolution answer would have been too large — check it rather than
 * assuming you got what you asked for.
 *
 * @param {Object} [opts]
 * @param {string} [opts.bucket] - 'hour' | 'day'
 * @returns {Promise<{bucket: string, buckets: Array<Object>}>}
 */
export async function fetchWatchEventHistogram({ bucket = 'hour', ...filters } = {}) {
  const params = watchEventFilterParams(filters);
  params.set('bucket', bucket);
  const res = await apiFetch(`/api/watch-events/histogram?${params}`);
  const data = await res.json();
  if (!res.ok || data.status !== 'success') {
    throw new Error(data.message || 'Failed to fetch the event timeline');
  }
  return { bucket: data.bucket || bucket, buckets: data.buckets || [] };
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

/**
 * Correct one event's own record of what it saw.
 *
 * Deliberately narrow: this is the human-correction path, not a general row editor.
 * The server accepts the subject/target pair (ref *and* name — see `WatchEventUpdate`,
 * which explains why a ref without its name is worse than neither) and rejects the
 * facts about what ran, which nobody should be rewriting after the fact.
 *
 * @param {string} uuid
 * @param {{subjectRef?: ?string, subjectName?: ?string, targetRef?: ?string, targetName?: ?string}} patch
 * @returns {Promise<Object>} the updated event, as the grid's rows are shaped
 */
export async function updateWatchEvent(uuid, patch) {
  const res = await apiFetch(`/api/watch-events/${uuid}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(patch),
  });
  const data = await res.json();
  if (!res.ok || data.status !== 'success') {
    throw new Error(data.message || 'Failed to update that event');
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
 * The event's *cropped* card art — the same moment as the snapshot, cut to the boxes
 * the event fired on and taken from the camera's own frame.
 *
 * A separate URL rather than a parameter on the snapshot so the browser caches the two
 * independently: the grid asks for crops all day and the detail modal asks for the wide
 * shot occasionally.
 *
 * 404s are ordinary — an event with nothing to frame has no crop. Use `setEventCardArt`
 * rather than this directly unless you are handling that fallback yourself.
 *
 * @param {string} uuid
 * @returns {string}
 */
export function watchEventCropUrl(uuid) {
  return `/api/watch-events/${uuid}/crop`;
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
