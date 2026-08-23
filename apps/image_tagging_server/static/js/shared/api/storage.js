/**
 * Disk accounting for the recording side of the app, and the ways to reclaim it.
 *
 * Deliberately uncached. Every one of these is either a measurement the user just
 * asked for or an act that invalidates the previous measurement, so a cache would
 * only ever be showing them a number from before they pressed the button.
 */
import { apiFetch } from './apiUtils.js';

/**
 * @typedef {Object} StorageBucket
 * @property {number} bytes
 * @property {number} files
 * @property {number} maxBytes  0 = no size limit
 * @property {?number} retentionDays  null when the bucket has no age bound
 * @property {string[]} roots
 * @property {?string} dir
 */

/**
 * Current usage per bucket. Walks directories server-side, so treat it as a
 * user-initiated request, not something to poll.
 *
 * @returns {Promise<{recordings: StorageBucket, timelapse: StorageBucket, thumbnailCache: StorageBucket}>}
 */
export async function fetchStorageUsage() {
  const res = await apiFetch('/api/storage/usage');
  const data = await res.json();
  if (!res.ok || data.status === 'failure') {
    throw new Error(data.message || 'Failed to read storage usage');
  }
  return {
    recordings: data.recordings ?? {},
    timelapse: data.timelapse ?? {},
    thumbnailCache: data.thumbnailCache ?? {},
  };
}

/** @typedef {{files: number, bytes: number, events: number, recordings: number}} PurgeResult */

async function _purge(path, method) {
  const res = await apiFetch(path, { method });
  const data = await res.json();
  if (!res.ok || data.status === 'failure') {
    throw new Error(data.message || 'The clean-up failed');
  }
  return {
    files: data.files ?? 0,
    bytes: data.bytes ?? 0,
    events: data.events ?? 0,
    recordings: data.recordings ?? 0,
  };
}

/**
 * Apply the configured retention bounds now rather than at the next hourly sweep.
 * Removes exactly what the sweep would have — it changes when, not what.
 * @returns {Promise<PurgeResult>}
 */
export const runStorageSweep = () => _purge('/api/storage/sweep', 'POST');

/**
 * Delete every timelapse frame, bounds or no bounds. Scoped to the day folders
 * the save_timelapse action writes; never touches an image with a JSON sidecar.
 * @returns {Promise<PurgeResult>}
 */
export const purgeTimelapseFrames = () => _purge('/api/storage/timelapse', 'DELETE');

/**
 * Drop every cached thumbnail. They are derived from images still on disk and
 * regenerate on the next request, so this is the one purge that loses nothing.
 * @returns {Promise<PurgeResult>}
 */
export const clearThumbnailCache = () => _purge('/api/storage/thumbnail-cache', 'DELETE');
