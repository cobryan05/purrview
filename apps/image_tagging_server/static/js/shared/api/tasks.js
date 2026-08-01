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
async function fetchTaskTypeListFromServer(keys) {
  const res = await apiFetch('/api/tasks/types/list');
  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.message || 'Failed to fetch task types');
  }
  const { types } = await res.json();
  return ignoreKeyReturn(keys, types);
}
const taskTypeFetcher = createCachedFetcher(fetchTaskTypeListFromServer);
export async function fetchTaskTypeList() {
  return taskTypeFetcher.fetch('taskTypes'); // single key for list
}

export function clearTaskTypeCache() {
  taskTypeFetcher.clear();
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
 * It must be explicit opt-in: default it to `false` here rather than relying on the
 * backend's default, and only surface a way to set it to `true` in UI where restarting
 * on its own makes sense (e.g. the task-config management form). Transient/preview
 * one-shot task runners should never set it.
 */
export async function createTaskConfig({ persistent = false, ...config }) {
  const res = await apiFetch('/api/tasks/configs/create', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ persistent, ...config }),
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
/** POST taskUuids to a task-instance operation endpoint; throws on failure. */
async function _taskOp(endpoint, taskUuids) {
  const res = await apiFetch(endpoint, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ taskUuids }),
  });
  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.message || `Task operation failed: ${endpoint}`);
  }
}

export const cancelTasks        = ({ taskUuids }) => _taskOp('/api/tasks/instances/cancel',           taskUuids);
export const deleteTasks        = ({ taskUuids }) => _taskOp('/api/tasks/instances/delete',           taskUuids);
export const pauseTasks         = ({ taskUuids }) => _taskOp('/api/tasks/instances/pause',            taskUuids);
export const resumeTasks        = ({ taskUuids }) => _taskOp('/api/tasks/instances/resume',           taskUuids);
export const pauseTasksInPlace  = ({ taskUuids }) => _taskOp('/api/tasks/instances/pause-in-place',  taskUuids);
export const resumeTasksInPlace = ({ taskUuids }) => _taskOp('/api/tasks/instances/resume-in-place', taskUuids);

