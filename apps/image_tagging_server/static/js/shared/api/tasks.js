import { createCachedFetcher } from '/app-static/js/ui/utils/data/createCachedFetcher.js';
import { apiFetch, ignoreKeyReturn } from './apiUtils.js';

export const TaskStatus = Object.freeze({
  NEW: "new",
  PENDING: "pending",
  RUNNING: "running",
  PAUSED: "paused",
  COMPLETED: "completed",
  ERROR: "error"
});


/* ---- Task type list --- */

/**
 * typename -> card accent name, from the same response as the type list.
 *
 * Static per task type (it is a class attribute), so it rides along with the
 * already-cached type list rather than costing a request of its own — every
 * task card drawn in the Jobs pane needs it.
 */
let _accentsByTypename = new Map();

/**
 * Typenames a dedicated screen starts, which the "new task config" form hides
 * unless "Show system" is ticked. Rides along with the type list for the same
 * reason the accents do — it is the same static fact about a task type.
 * @type {Set<string>}
 */
let _systemTypenames = new Set();

async function fetchTaskTypeListFromServer(keys) {
  const res = await apiFetch('/api/tasks/types/list');
  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.message || 'Failed to fetch task types');
  }
  const { types, accents, systemTypes } = await res.json();
  _accentsByTypename = new Map(Object.entries(accents || {}));
  _systemTypenames = new Set(systemTypes || []);
  return ignoreKeyReturn(keys, types);
}
const taskTypeFetcher = createCachedFetcher(fetchTaskTypeListFromServer);
export async function fetchTaskTypeList() {
  return taskTypeFetcher.fetch('taskTypes'); // single key for list
}

export function clearTaskTypeCache() {
  taskTypeFetcher.clear();
}

/**
 * Card accent name per task typename, for the types that declare one.
 *
 * Typenames the server did not name are absent from the Map — "no accent" is a
 * real answer, and callers render an untinted card for it.
 *
 * @returns {Promise<Map<string, string>>}
 */
export async function fetchTaskTypeAccents() {
  await fetchTaskTypeList(); // populates _accentsByTypename (cached after the first call)
  return _accentsByTypename;
}

/**
 * The typenames that dedicated UI starts. Anything not in here is a type a user
 * can meaningfully build a saved config for.
 *
 * @returns {Promise<Set<string>>}
 */
export async function fetchSystemTaskTypenames() {
  await fetchTaskTypeList(); // populates _systemTypenames (cached after the first call)
  return _systemTypenames;
}

/* ---- Task type schemas --- */
/**
 * Gets the schemas for a list of task typenames
 *
 * @param {import('@web_api').TasksTypeSchemaPayload} payload
 * @returns {Promise<Map<string, any>>}
 * @throws {Error} If the fetch fails or the API response is invalid.
 */
async function fetchTasksSchemaFromServer(typenames) {
  const res = await apiFetch('/api/tasks/types/schema', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ taskTypenames: typenames }),
  });
  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.message || 'Failed to fetch task schema');
  }
  /** @type {import('@web_api').TasksTypeSchemaResponse} */
  const response = await res.json();
  const { schemas } = response;
  return new Map(Object.entries(schemas));
}

const taskSchemaFetcher = createCachedFetcher(fetchTasksSchemaFromServer);
export async function fetchTaskTypeSchemas(typenames) {
  const types = await fetchTaskTypeList();
  for (const t of typenames) {
    if (!types.includes(t)) throw new Error(`Unknown task type: ${t}`);
  }

  const result = new Map();
  for (const t of typenames) {
    const schema = await taskSchemaFetcher.fetch(t);
    result.set(t, schema);
  }

  return result;
}

export function clearTaskSchemaCache() {
  taskSchemaFetcher.clear();
}

/* ---- Task status schemas (Phase 3: task status schema framework) --- */
/**
 * Gets the status-display schemas (Task.status_schema()) for a list of task
 * typenames. Static per typename — fetched once and cached, mirroring
 * fetchTaskTypeSchemas / the existing get_task_type_schema path.
 *
 * @param {string[]} typenames
 * @returns {Promise<Map<string, any[]>>}
 * @throws {Error} If the fetch fails.
 */
async function fetchTaskStatusSchemasFromServer(typenames) {
  const res = await apiFetch('/api/tasks/types/status-schema', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ taskTypenames: typenames }),
  });
  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.message || 'Failed to fetch task status schema');
  }
  const response = await res.json();
  // Defensive: an older/not-yet-landed backend may respond with
  // {status: "success"} and no `schemas` key at all.
  const schemas = response.schemas || {};
  return new Map(Object.entries(schemas));
}

const taskStatusSchemaFetcher = createCachedFetcher(fetchTaskStatusSchemasFromServer);

export async function fetchTaskStatusSchemas(typenames) {
  const result = new Map();
  for (const t of typenames) {
    const schema = await taskStatusSchemaFetcher.fetch(t);
    result.set(t, schema);
  }
  return result;
}

export function clearTaskStatusSchemaCache() {
  taskStatusSchemaFetcher.clear();
}

/* ---- Task status data (Phase 3) --- */
/**
 * Gets live status-data values for a set of running task instances.
 *
 * Deliberately NOT cached via createCachedFetcher: callers (expanded task
 * cards) own their own polling cadence, and caching here would either go
 * stale or defeat the whole point of splitting this out of the cheap
 * all-tasks summary poll. Only call this while a card is actually expanded.
 *
 * Unknown/missing uuids are simply omitted from the returned Map — not an error.
 *
 * @param {string[]} taskUuids
 * @returns {Promise<Map<string, any>>}
 * @throws {Error} If the fetch fails.
 */
export async function fetchTaskStatusData(taskUuids) {
  if (!taskUuids?.length) return new Map();
  const res = await apiFetch('/api/tasks/instances/status-data', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ taskUuids }),
  });
  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.message || 'Failed to fetch task status data');
  }
  const data = await res.json();
  const statusData = data.statusData || {};
  return new Map(Object.entries(statusData));
}

/**
 * The frame a running task is currently working on, plus what it sees in it.
 *
 * Only some task types have a live view (today: StreamWatcherTask); the rest
 * answer "no live view", which is a normal answer and not an error. A stopped
 * task answers the same way, which is exactly what a poll running when the user
 * stops the watcher will hit — so this resolves to `null` in both cases rather
 * than throwing, and only a transport failure throws.
 *
 * Pass `sinceSeq` with the last `frameSeq` you received: a poll that outruns the
 * camera then comes back with fresh metadata and `frameUnchanged: true`, having
 * re-encoded and re-sent nothing.
 *
 * @param {string} taskUuid
 * @param {{sourceUuid?: ?string, sinceSeq?: ?number, quality?: ?number}} [opts]
 * @returns {Promise<?object>} the preview, or null when there is none to show
 * @throws {Error} only on a transport failure
 */
export async function fetchTaskLivePreview(
  taskUuid,
  { sourceUuid = null, sinceSeq = null, quality = null, full = false } = {},
) {
  const params = new URLSearchParams();
  if (sourceUuid) params.set('source_uuid', sourceUuid);
  // 0 is a legitimate seq (no frame yet), so test for null rather than falsiness.
  if (sinceSeq != null) params.set('since_seq', String(sinceSeq));
  if (quality != null) params.set('quality', String(quality));
  // The camera's own resolution instead of the downscaled frame inference ran on.
  // Costs a bigger encode and transfer per frame, so it is only ever sent when asked.
  if (full) params.set('full', 'true');
  const query = params.toString();

  const res = await apiFetch(`/api/tasks/instances/${encodeURIComponent(taskUuid)}/live-preview${query ? `?${query}` : ''}`);
  if (!res.ok) throw new Error(`Live preview request failed (${res.status})`);
  const data = await res.json();
  return data.status === 'success' ? (data.preview ?? null) : null;
}

/** Task typename of the watcher, which is what owns a live event's frame buffers. */
const WATCHER_TYPENAME = 'StreamWatcherTask';

/** Statuses in which a watcher still holds its buffers. A stopped one has none. */
const LIVE_WATCHER_STATUSES = new Set(['running', 'pending', 'paused']);

/**
 * The running watcher instance for one watch profile, or null if none is live.
 *
 * A watcher carries its profile uuid in its config params rather than in a column, so
 * this reads params rather than joining. The instance filter matters: finished
 * instances are retained after a watcher stops, and one of those holds no buffer at
 * all — asking it for a preview would 404 in a way that reads as "nothing recorded"
 * rather than "not running".
 *
 * @param {string} profileUuid
 * @returns {Promise<?string>} the task instance uuid
 */
export async function fetchRunningWatcherUuid(profileUuid) {
  if (!profileUuid) return null;
  const { tasks } = await fetchTasksStatus();
  for (const [taskUuid, task] of Object.entries(tasks ?? {})) {
    if (task?.config?.typename !== WATCHER_TYPENAME) continue;
    if (!LIVE_WATCHER_STATUSES.has(task?.instance?.status)) continue;
    if (task?.config?.params?.watch_profile_uuid === profileUuid) return taskUuid;
  }
  return null;
}

/**
 * The video an in-progress event timelapse has collected so far.
 *
 * An event timelapse only becomes a recording when its event closes, and the events
 * worth watching this way run for hours — so this asks the running watcher to compile
 * what it is holding right now. The session keeps sampling; nothing here interrupts it.
 *
 * Returns an object URL rather than the bytes: the caller hands it to a `<video>`, and
 * a blob URL means the browser never has to hold the whole file as a JS string. **The
 * caller owns it and must `URL.revokeObjectURL` when the player goes away**, or a modal
 * opened repeatedly on a long event leaks a video per open.
 *
 * @param {string} taskUuid - the running watcher instance
 * @param {{sourceUuid: string, ruleUuid: string, subjectRef?: string, targetRef?: string}} event
 * @returns {Promise<?{url: string, frames: number, durationS: number, sourceSpanS: number,
 *   partial: boolean}>} null when nothing is buffered yet, or the watcher has stopped
 * @throws {Error} only on a transport failure
 */
export async function fetchTimelapsePreview(taskUuid, { sourceUuid, ruleUuid, subjectRef = '', targetRef = '' }) {
  const params = new URLSearchParams({ source_uuid: sourceUuid, rule_uuid: ruleUuid });
  if (subjectRef) params.set('subject_ref', subjectRef);
  if (targetRef) params.set('target_ref', targetRef);

  const res = await apiFetch(`/api/tasks/instances/${encodeURIComponent(taskUuid)}/timelapse-preview?${params}`);
  // 404 is the ordinary "not yet" — too few frames, or the event closed while the modal
  // was open — and must read as "nothing to show", not as a failure.
  if (res.status === 404) return null;
  if (!res.ok) throw new Error(`Timelapse preview failed (${res.status})`);
  const blob = await res.blob();
  return {
    url: URL.createObjectURL(blob),
    frames: Number(res.headers.get('X-Timelapse-Frames') || 0),
    durationS: Number(res.headers.get('X-Timelapse-Duration-S') || 0),
    sourceSpanS: Number(res.headers.get('X-Timelapse-Source-Span-S') || 0),
    partial: res.headers.get('X-Timelapse-Partial') === 'true',
  };
}

/**
 * Sends a command to a running task instance — the `task_command` status-action
 * back-channel. `Task.handle_command()` on the backend is only supposed to set
 * a flag or push to a queue (never perform I/O), so this call is expected to
 * resolve quickly regardless of what the task itself is doing.
 *
 * @param {{taskUuid: string, command: string, params?: any}} payload
 * @returns {Promise<{status: string, message?: string}>}
 * @throws {Error} If the fetch fails.
 */
export async function sendTaskCommand({ taskUuid, command, params = null }) {
  const res = await apiFetch('/api/tasks/instances/command', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ taskUuid, command, params }),
  });
  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.message || 'Failed to send task command');
  }
  return res.json();
}

/** -----------------------------
 * TASK CONFIGS
 * -----------------------------
 */
/**
 * Gets the list of task configurations
 *
 * @param {import('@web_api').GetTaskConfigsPayload} payload
 * @returns {Promise<Map<string, any>>}
 * @throws {Error} If the fetch fails or the API response is invalid.
 */
export async function fetchTaskConfigs(uuids = null, { excludeTemporary = false } = {}) {
  const fetchAll = uuids == null;
  const res = await apiFetch('/api/tasks/configs/get', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ configUuids: fetchAll ? null : uuids, excludeTemporary }),
  });

  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.message || 'Failed to fetch task configs');
  }
  /** @type {import('@web_api').GetTaskConfigResponse} */
  const result = await res.json();
  const { configs } = result;
  return new Map(Object.entries(configs));
}

export const taskConfigFetcher = createCachedFetcher(fetchTaskConfigs);

export async function deleteTaskConfigs({ uuids }) {
  const res = await apiFetch('/api/tasks/configs/delete', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ configUuids: uuids }),
  });

  if (res.ok) {
    uuids.forEach(uuid => taskConfigFetcher.delete(uuid));
  } else {
    const err = await res.json();
    throw new Error(err.message || 'Failed to delete task configs');
  }
}

/**
 * Creates a new task config.
 *
 * `persistent` controls whether the task auto-restarts after a server restart/reboot.
 * Pass it explicitly when the caller has an opinion — a preview/one-shot runner
 * should pass `false`, and the task-config management form passes whatever the user
 * ticked. **Omit it** to let the task type decide via its `DEFAULT_PERSISTENT`
 * (StreamWatcherTask defaults to persistent; batch work does not).
 *
 * This used to default to `false` here, which meant a watcher launched from the UI
 * stopped silently on every service restart no matter what the task type wanted —
 * the frontend was overriding a decision that belongs to the task.
 */
export async function createTaskConfig({ persistent, ...config }) {
  const res = await apiFetch('/api/tasks/configs/create', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    // Omitted entirely when undefined, so the server sees "no opinion" rather
    // than an explicit null it would have to distinguish from false.
    body: JSON.stringify(persistent === undefined ? config : { persistent, ...config }),
  });

  if (!res.ok) {
    const errJson = await res.json();
    const error = new Error(errJson.message || 'Failed to create task config');
    error.data = errJson;
    throw error;
  }

  const data = await res.json();
  if (data.status !== 'success') {
    throw new Error(data.message || 'Failed to fetch task status');
  }

  const { configUuid } = data;
  return configUuid;
}


export async function updateTaskConfig(config) {
  const res = await apiFetch('/api/tasks/configs/update', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ configUuid: config.uuid, ...config }),
  });

  if (!res.ok) {
    const error = await res.json();
    throw new Error(error.message || 'Failed to update task config');
  }
}

/** -----------------------------
 * TASK RESULTS
 * -----------------------------
 */

/**
 * Gets the results of a task from its uuid
 *
 * @param {import('@web_api').TasksResultPayload} payload
 * @returns {Promise<import('@web_api').TaskResultModel>}
 * @throws {Error} If the fetch fails or the API response is invalid.
 */
async function fetchTaskResultFromServer(taskUuids) {
  const res = await apiFetch('/api/tasks/instances/result', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ taskUuids }),
  });

  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.message || 'Failed to fetch task result');
  }

  const { results } = await res.json();
  const resultMap = new Map(Object.entries(results));
  return resultMap;
}

const taskResultFetcher = createCachedFetcher(fetchTaskResultFromServer);

/**
 * The payload a task returned, unwrapped from its envelope.
 *
 * `Task._results` is always `{ status, data }` (or `{ status, message }` on
 * error) — never the bare dict `_run` returned. Reading `instance.result.foo`
 * therefore silently yields `undefined` for every task in the app, which is
 * how a finished harvest could report "no frames to review" while holding a
 * preview run full of them.
 *
 * @param {object} [instanceOrResult] - a TaskInstanceRead, or its `result`.
 * @returns {object} the task's own result dict; `{}` when there is none.
 */
export function taskResultData(instanceOrResult) {
  const result = instanceOrResult?.result ?? instanceOrResult;
  return result?.data ?? {};
}

export async function fetchTasksResult({ taskUuids, cacheResults = true }) {
  const results = new Map();
  for (const t of taskUuids) {
    const r = await taskResultFetcher.fetch(t);
    if (!cacheResults) taskResultFetcher.delete(t);
    results.set(t, r);
  }

  return results;
}

/** -----------------------------
 * TASK INSTANCE OPERATIONS
 * -----------------------------
 */
/**
 * Starts a task from a given task config
 *
 * @param {import('@web_api').StartTasksPayload} payload - uuid of configuration to create instance of
 * @returns {Promise<string>} - uuid of the started task
 * @throws {Error} If the fetch fails or the API response is invalid.
 */
export async function startTask({ configUuid }) {
  const res = await apiFetch('/api/tasks/instances/start', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ configUuid }),
  });
  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.message || 'Failed to start task');
  }
  /** @type {import('@web_api').StartTasksResponse} */
  const response = await res.json();
  const { taskUuid } = response;
  return taskUuid;
}


/**
 * Gets the status of a task from its uuid
 *
 * @param {import('@web_api').TasksInfoPayload} payload
 * @returns {Promise<import('@web_api').TasksInfoResponse>}
 * @throws {Error} If the fetch fails or the API response is invalid.
 */
export async function fetchTasksStatus({ taskUuids } = {}) {
  const res = await apiFetch('/api/tasks/instances/get', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ taskUuids }),
  });

  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.message || 'Failed to fetch task status');
  }

  const data = await res.json();
  if (data.status !== 'success') {
    throw new Error(data.message || 'Failed to fetch task status');
  }

  return data;
}
/** POST taskUuids to a task-instance operation endpoint; throws on failure.
 *
 * Returns the parsed body, so a caller can tell what the server actually did rather
 * than only that the request arrived. The `status === 'failure'` check matters as
 * much as `res.ok`: these endpoints catch their own exceptions and answer HTTP 200
 * with a failure body, so checking the status code alone reported every server-side
 * error as a successful action.
 */
async function _taskOp(endpoint, taskUuids) {
  const res = await apiFetch(endpoint, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ taskUuids }),
  });
  let data = null;
  try { data = await res.json(); } catch { /* an empty or non-JSON body is handled below */ }
  if (!res.ok || data?.status === 'failure') {
    throw new Error(data?.message || `Task operation failed: ${endpoint}`);
  }
  return data ?? {};
}

/** Stop running tasks.
 *
 * Resolves to `{ stopped, stopping, notRunning }` — three outcomes, because they need
 * three different things said to the user. `stopping` is a task that accepted the
 * cancel but is still winding down (a watcher closing its streams, say); `notRunning`
 * is a task the server had nothing to stop, which previously came back looking
 * exactly like success.
 */
export async function cancelTasks({ taskUuids }) {
  const data = await _taskOp('/api/tasks/instances/cancel', taskUuids);
  return {
    stopped: data.taskUuids ?? [],
    stopping: data.stopping ?? [],
    notRunning: data.notRunning ?? [],
  };
}

export const deleteTasks        = ({ taskUuids }) => _taskOp('/api/tasks/instances/delete',           taskUuids);
export const pauseTasks         = ({ taskUuids }) => _taskOp('/api/tasks/instances/pause',            taskUuids);
export const resumeTasks        = ({ taskUuids }) => _taskOp('/api/tasks/instances/resume',           taskUuids);
export const pauseTasksInPlace  = ({ taskUuids }) => _taskOp('/api/tasks/instances/pause-in-place',  taskUuids);
export const resumeTasksInPlace = ({ taskUuids }) => _taskOp('/api/tasks/instances/resume-in-place', taskUuids);

