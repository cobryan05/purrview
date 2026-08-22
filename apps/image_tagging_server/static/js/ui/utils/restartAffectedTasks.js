/**
 * "You just changed something a running task is using — restart it?"
 *
 * Tasks read their configuration exactly once, in `_init`. Editing a watch
 * profile, a filter chain, a source or a model stack therefore has *no effect*
 * on an instance that is already running: the UI says "Saved", the watcher
 * keeps using the rules it loaded at start, and the two silently disagree
 * until someone restarts it by hand. This module closes that gap by offering
 * the restart at the moment of the save.
 *
 * ## Why a deep scan rather than a per-type registry
 *
 * The obvious design is a table mapping each config kind to the task params
 * that reference it (`watch_profile_uuid`, `filter_chain_uuid`,
 * `source_uuids`, …). That table is a maintenance burden and is wrong by
 * omission the moment someone adds a task that references a config through a
 * new param name — and being wrong by omission here is invisible, because the
 * symptom is a prompt that *doesn't* appear.
 *
 * Instead this scans a running task's params for the saved config's UUID
 * anywhere in the structure. UUIDs are unique, so a hit is a genuine
 * reference, not a coincidence; and a new task type that stores a config UUID
 * under any name is covered on the day it is written, with no change here.
 */

import { cancelTasks, fetchTasksStatus, startTask } from '/app-static/js/shared/api/tasks.js';
import { fetchWatchProfiles } from '/app-static/js/shared/api/watchProfiles.js';
import { showConfirmModal } from './confirmModal.js';
// Note the argument order: toast(message, duration, type). Several older call
// sites pass the type as the second argument, where it lands on `duration`.
import { toast } from './logging.js';

/** Statuses whose instances are actually using the config right now. */
const LIVE_STATUSES = new Set(['running', 'paused', 'pending', 'new']);

/** How long to wait for a cancelled instance to actually stop, before starting the new one. */
const STOP_TIMEOUT_MS = 15000;
const STOP_POLL_MS = 300;

/**
 * True if `uuid` appears anywhere in `value` (object, array or scalar).
 *
 * Depth-limited rather than fully recursive-safe: task params are plain JSON
 * from the server, so cycles are impossible, but a depth cap keeps a
 * pathological payload from freezing the save handler.
 *
 * @param {any} value
 * @param {string} uuid
 * @param {number} [depth]
 * @returns {boolean}
 */
function referencesUuid(value, uuid, depth = 0) {
  if (depth > 8 || value == null) return false;
  if (typeof value === 'string') return value === uuid;
  if (Array.isArray(value)) return value.some(v => referencesUuid(v, uuid, depth + 1));
  if (typeof value === 'object') {
    // Keys matter as well as values: some params are keyed *by* uuid
    // (`labelValues: { "<labelUuid>": {...} }` in the collector, for one).
    for (const [k, v] of Object.entries(value)) {
      if (k === uuid) return true;
      if (referencesUuid(v, uuid, depth + 1)) return true;
    }
  }
  return false;
}

/**
 * UUIDs to look for, given the one that was actually saved.
 *
 * The params scan only sees what a task stores *directly*, and a watcher stores
 * one thing: `watch_profile_uuid`. Its sources and its actions' filter chains
 * live in the profile row, so editing a camera's URL while a watcher was running
 * matched nothing and offered no restart — the exact scenario this module opens
 * by describing.
 *
 * Rather than teach the scan about watcher params, this widens the *needle*: a
 * profile that references the saved uuid anywhere makes that profile's uuid
 * worth searching for too. Same deep, name-agnostic match as the params scan, so
 * a new field on a profile is covered without a change here.
 *
 * @param {string} configUuid
 * @returns {Promise<string[]>} the saved uuid, plus any profile that holds it
 */
async function _uuidsToMatch(configUuid) {
  let profiles;
  try {
    profiles = await fetchWatchProfiles();
  } catch {
    return [configUuid];  // Prompt on what we can see rather than not at all.
  }
  const uuids = [configUuid];
  // ⚠️ A Map keyed by uuid, not an array (see fetchWatchProfiles). Iterate .values().
  for (const profile of profiles.values()) {
    if (profile?.uuid && profile.uuid !== configUuid && referencesUuid(profile, configUuid)) {
      uuids.push(profile.uuid);
    }
  }
  return uuids;
}

/**
 * Running/queued task instances whose params reference `configUuid`, directly or
 * through a watch profile that holds it.
 *
 * @param {string} configUuid
 * @returns {Promise<Array<{taskUuid: string, configUuid: string, name: string}>>}
 */
export async function findTasksUsing(configUuid) {
  if (!configUuid) return [];
  let tasks;
  let needles;
  try {
    // In parallel: the profile lookup is usually a cache hit, and when it is not
    // there is no reason for it to sit behind the task poll.
    [{ tasks }, needles] = await Promise.all([fetchTasksStatus(), _uuidsToMatch(configUuid)]);
  } catch {
    // A restart prompt is a convenience. If the task server can't be reached,
    // the save itself still succeeded — say nothing rather than surfacing a
    // second, confusing error on top of a successful save.
    return [];
  }
  const affected = [];
  for (const [taskUuid, info] of Object.entries(tasks || {})) {
    const status = info?.instance?.status;
    if (!LIVE_STATUSES.has(status)) continue;
    const params = info?.config?.params ?? {};
    if (!needles.some(uuid => referencesUuid(params, uuid))) continue;
    affected.push({
      taskUuid,
      configUuid: info?.config?.uuid,
      name: info?.config?.name || info?.config?.typename || 'task',
    });
  }
  return affected;
}

/**
 * Waits for a cancelled instance to leave the live set.
 *
 * Starting the replacement before the old one has released its sources gives
 * two instances competing for the same camera — which for an RTSP source means
 * both get a degraded stream rather than one getting a good one.
 *
 * @param {string} taskUuid
 * @returns {Promise<void>}
 */
async function waitForStop(taskUuid) {
  const deadline = Date.now() + STOP_TIMEOUT_MS;
  while (Date.now() < deadline) {
    let tasks;
    try {
      ({ tasks } = await fetchTasksStatus({ taskUuids: [taskUuid] }));
    } catch {
      return; // Can't tell; don't block the restart on a transient poll failure.
    }
    const status = tasks?.[taskUuid]?.instance?.status;
    if (!status || !LIVE_STATUSES.has(status)) return;
    await new Promise(r => setTimeout(r, STOP_POLL_MS));
  }
}

/**
 * Offers to restart any running task affected by a config that was just saved.
 *
 * Call this *after* the save succeeds. It is a no-op — no modal, no toast —
 * when nothing is running that uses the config, which is the common case, so
 * it is safe to call unconditionally from any save handler.
 *
 * @param {string} configUuid — UUID of the entity just saved (watch profile,
 *        filter chain, source, model stack, …).
 * @param {Object} [opts]
 * @param {string} [opts.label='configuration'] — what was saved, for the prompt text.
 * @returns {Promise<number>} how many tasks were restarted (0 if declined/none).
 */
export async function promptRestartAffectedTasks(configUuid, { label = 'configuration' } = {}) {
  const affected = await findTasksUsing(configUuid);
  if (!affected.length) return 0;

  const names = affected.map(a => `"${a.name}"`).join(', ');
  const plural = affected.length === 1 ? 'task is' : 'tasks are';
  return _confirmAndRestart(affected,
    `${affected.length} running ${plural} using this ${label}: ${names}. ` +
    `A running task keeps the settings it started with — restart ${affected.length === 1 ? 'it' : 'them'} to pick up your changes.`);
}

/**
 * Live instances started from `configUuid` itself.
 *
 * The sibling of `findTasksUsing`, and deliberately not the same query. That one
 * asks "whose params mention this uuid?", which is the right question for a
 * source or a filter chain but never matches a task config: a config's params
 * do not contain its own uuid. Editing a running task's own settings is the
 * *most* common version of this gap and was the one case the deep scan could
 * not see.
 *
 * @param {string} configUuid
 * @returns {Promise<Array<{taskUuid: string, configUuid: string, name: string}>>}
 */
export async function findRunningInstancesOfConfig(configUuid) {
  if (!configUuid) return [];
  let tasks;
  try {
    ({ tasks } = await fetchTasksStatus());
  } catch {
    // Same reasoning as findTasksUsing: the save succeeded, so a failed status
    // poll must not turn into a second error on top of it.
    return [];
  }
  const affected = [];
  for (const [taskUuid, info] of Object.entries(tasks || {})) {
    if (!LIVE_STATUSES.has(info?.instance?.status)) continue;
    if (info?.config?.uuid !== configUuid) continue;
    affected.push({
      taskUuid,
      configUuid,
      name: info?.config?.name || info?.config?.typename || 'task',
    });
  }
  return affected;
}

/**
 * Offers to restart the task whose own config was just saved.
 *
 * Call this after a task-config *update* succeeds. Like
 * `promptRestartAffectedTasks` it is a no-op when nothing is live, so it is safe
 * to call unconditionally — and there is nothing to prompt about on a create.
 *
 * @param {string} configUuid — UUID of the task config just saved.
 * @returns {Promise<number>} how many instances were restarted (0 if declined/none).
 */
export async function promptRestartOwnTask(configUuid) {
  const affected = await findRunningInstancesOfConfig(configUuid);
  if (!affected.length) return 0;

  const one = affected.length === 1;
  return _confirmAndRestart(affected,
    `${one ? 'This task is' : `${affected.length} instances of this task are`} still running with the ` +
    `settings ${one ? 'it' : 'they'} started with. Your changes are saved, but they will not take effect ` +
    `until ${one ? 'it is' : 'they are'} restarted.`);
}

/**
 * Shared tail of both prompts: confirm, then cancel-and-restart each instance.
 *
 * @param {Array<{taskUuid: string, configUuid: string, name: string}>} affected
 * @param {string} message
 * @returns {Promise<number>}
 */
async function _confirmAndRestart(affected, message) {
  const one = affected.length === 1;
  const confirmed = await showConfirmModal({
    title: 'Restart to apply changes?',
    message,
    confirmText: one ? 'Restart it' : 'Restart them',
    cancelText: 'Keep running',
  });
  if (!confirmed) return 0;

  let restarted = 0;
  for (const { taskUuid, configUuid: cfgUuid, name } of affected) {
    try {
      await cancelTasks({ taskUuids: [taskUuid] });
      await waitForStop(taskUuid);
      // Restarting means starting a fresh instance of the same *config*; the
      // cancelled instance keeps its own history rather than being reused.
      await startTask({ configUuid: cfgUuid });
      restarted += 1;
    } catch (e) {
      toast(`Could not restart "${name}": ${e.message || e}`, 5000, 'error');
    }
  }
  if (restarted) {
    toast(`Restarted ${restarted} ${restarted === 1 ? 'task' : 'tasks'}`, 5000, 'success');
  }
  return restarted;
}
