/**
 * Source regions — the zones and masks drawn on a camera's frame.
 *
 * Regions belong to the *source*, not to a watcher's use of it: the geometry describes
 * the camera's view, so it is drawn once and every watcher on that camera sees it.
 *
 * Two roles share one shape because the geometry is identical; only what consumes them
 * differs. A **zone** is inert until a rule names it by uuid. A **mask** acts on every
 * detection whether or not anything references it — which is exactly why a watcher can
 * opt out of one (`disabledMaskUuidsJson` on its source row) and cannot "opt in" to a
 * zone.
 */
import { apiFetch } from './apiUtils.js';

export const REGION_ROLE_ZONE = 'zone';
export const REGION_ROLE_MASK = 'mask';

/**
 * Regions for one source, or for every source when `sourceUuid` is null.
 *
 * The rule editor uses the all-sources form: it offers zones across every source a
 * watcher uses, and fanning out one request per source would be a request storm on a
 * watcher with eight cameras.
 *
 * @param {?string} sourceUuid
 * @param {?string} role  'zone' | 'mask' | null for both
 * @returns {Promise<Array>} region rows; empty on failure is *not* returned — it throws
 */
export async function fetchSourceRegions(sourceUuid = null, role = null) {
  const res = await apiFetch('/api/sources/regions/get', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ sourceUuid, role }),
  });
  const data = await res.json();
  if (!res.ok || data.status !== 'success') {
    throw new Error(data.message || 'Failed to fetch source regions');
  }
  return data.regions || [];
}

/**
 * Whole-list replace of one source's regions.
 *
 * Uuids are round-tripped deliberately: a rule points at a zone by uuid from another
 * table entirely, so dropping them here would break every zone rule at once with no
 * error anywhere. `orphanedRules` names rules whose zone this write removed — a warning
 * to surface, not an error, since deleting a zone alongside its rule is legitimate.
 *
 * @param {string} sourceUuid
 * @param {Array} regions
 * @returns {Promise<{regions: Array, orphanedRules: string[]}>}
 */
export async function saveSourceRegions(sourceUuid, regions) {
  const res = await apiFetch('/api/sources/regions/set', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ sourceUuid, regions }),
  });
  const data = await res.json();
  if (!res.ok || data.status !== 'success') {
    throw new Error(data.message || 'Failed to save source regions');
  }
  return { regions: data.regions || [], orphanedRules: data.orphanedRules || [] };
}

/**
 * Grab one still from a source, to draw regions against.
 *
 * Served from a running watcher's latest analysed frame when there is one, which avoids
 * opening a second RTSP session against a camera that may cap concurrent connections.
 * That fallback is why `source` comes back on the response: a watcher's frame is
 * downscaled to its analysis width, so the still can be lower resolution than the
 * camera's own — worth telling the user, since they are about to draw on it.
 *
 * Slow by nature (a cold camera needs a handshake and a keyframe), so callers should
 * show progress rather than assume this returns promptly.
 *
 * @param {string} sourceUuid
 * @param {number} [quality]
 * @returns {Promise<{frame: string, width: number, height: number, source: string}>}
 */
export async function grabSourceFrame(sourceUuid, quality = 85) {
  const res = await apiFetch(
    `/api/tasks/sources/${encodeURIComponent(sourceUuid)}/grab-frame?quality=${quality}`
  );
  const data = await res.json();
  if (!res.ok || data.status !== 'success') {
    throw new Error(data.message || 'Could not grab a frame from this source');
  }
  return data;
}
