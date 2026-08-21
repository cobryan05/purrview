/**
 * Tracking blips — what the analyser found wrong in a clip's own detection sidecar.
 *
 * Its own module rather than another section of `watchProfiles.js` because blips are
 * queried *across* events as well as within one, which is why db_server gives them a
 * top-level `/api/event-blips` prefix instead of nesting them under an event.
 *
 * Query params are **snake_case**; only response bodies are camelCased. A misspelt
 * param is silently ignored by FastAPI and the endpoint answers with the *unfiltered*
 * set — so a filter that is quietly wrong looks like a filter that matches everything.
 */
import { apiFetch } from '/app-static/js/shared/api/apiUtils.js';

/**
 * One event's findings, or a slice across all of them. Worst first — ordered by how
 * sure the detector was about the thing it then got wrong.
 *
 * @param {Object} [opts]
 * @param {?string} [opts.eventUuid]
 * @param {?string} [opts.recordingUuid]
 * @param {string[]} [opts.uuids] - fetch specific blips, e.g. to re-read after a run
 * @param {string[]} [opts.kinds]
 * @param {string[]} [opts.states]
 * @param {?boolean} [opts.harvestable]
 * @param {number} [opts.limit]
 * @param {number} [opts.offset]
 * @returns {Promise<{blips: Array<Object>, total: number}>}
 */
export async function fetchEventBlips({
  eventUuid = null,
  recordingUuid = null,
  uuids = null,
  kinds = null,
  states = null,
  harvestable = null,
  limit = 200,
  offset = 0,
} = {}) {
  const params = new URLSearchParams();
  if (eventUuid) params.set('event_uuid', eventUuid);
  if (recordingUuid) params.set('recording_uuid', recordingUuid);
  for (const uuid of uuids || []) params.append('uuids', uuid);
  for (const kind of kinds || []) params.append('kinds', kind);
  for (const state of states || []) params.append('states', state);
  // Explicit null check: `false` is a real filter — "show me only the ones that can
  // never be harvested" — and a falsy guard would drop it.
  if (harvestable != null) params.set('harvestable', String(harvestable));
  params.set('limit', String(limit));
  params.set('offset', String(offset));
  const res = await apiFetch(`/api/event-blips/?${params}`);
  const data = await res.json();
  if (!res.ok || data.status !== 'success') {
    throw new Error(data.message || 'Failed to fetch tracking blips');
  }
  return { blips: data.blips || [], total: data.total ?? 0 };
}

/**
 * Move a selection of blips to one triage state.
 *
 * Note that `harvested` is not set from here: the harvest task sets it itself, once
 * frames have actually landed. Marking them harvested at launch would claim an outcome
 * for a run that can still fail, and the record would say frames exist that do not.
 *
 * @param {string[]} uuids
 * @param {'new'|'confirmed'|'dismissed'} state
 * @returns {Promise<number>} how many rows actually moved
 */
export async function setEventBlipStates(uuids, state) {
  const res = await apiFetch('/api/event-blips/states', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ uuids, state }),
  });
  const data = await res.json();
  if (!res.ok || data.status !== 'success') {
    throw new Error(data.message || 'Failed to update tracking blips');
  }
  return data.updated ?? 0;
}
