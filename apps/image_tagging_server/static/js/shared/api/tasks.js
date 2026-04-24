import { createCachedFetcher } from '/app-static/js/ui/utils/data/createCachedFetcher.js';
import { ignoreKeyReturn } from './apiUtils.js';

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
  const res = await fetch('/api/tasks/types/list');
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
  const res = await fetch('/api/tasks/types/schema', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ task_typenames: typenames }),
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
export async function fetchTaskConfigs(uuids = null) {
  const fetchAll = uuids == null;
  const res = await fetch('/api/tasks/configs/get', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ config_uuids: fetchAll ? null : uuids }),
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
  const res = await fetch('/api/tasks/configs/delete', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ config_uuids: uuids }),
  });

  if (res.ok) {
    uuids.forEach(uuid => taskConfigFetcher.delete(uuid));
  } else {
    const err = await res.json();
    throw new Error(err.message || 'Failed to delete task configs');
  }
}

export async function createTaskConfig(config) {
  const res = await fetch('/api/tasks/configs/create', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(config),
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

  const { config_uuid } = data;
  return config_uuid;
}


export async function updateTaskConfig(config) {
  const res = await fetch('/api/tasks/configs/update', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ config_uuid: config.uuid, ...config }),
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
async function fetchTaskResultFromServer(task_uuids) {
  const res = await fetch('/api/tasks/instances/result', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ task_uuids }),
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

export async function fetchTasksResult({ task_uuids, cacheResults = true }) {
  const results = new Map();
  for (const t of task_uuids) {
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
export async function startTask({ config_uuid }) {
  const res = await fetch('/api/tasks/instances/start', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ config_uuid }),
  });
  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.message || 'Failed to start task');
  }
  /** @type {import('@web_api').StartTasksResponse} */
  const response = await res.json();
  const { status, task_uuid } = response;
  return task_uuid;
}


/**
 * Gets the status of a task from its uuid
 *
 * @param {import('@web_api').TasksInfoPayload} payload
 * @returns {Promise<import('@web_api').TasksInfoResponse>}
 * @throws {Error} If the fetch fails or the API response is invalid.
 */
export async function fetchTasksStatus({ task_uuids } = {}) {
  const res = await fetch('/api/tasks/instances/get', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ task_uuids }),
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
export async function cancelTasks({ task_uuids }) {
  const res = await fetch('/api/tasks/instances/cancel', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ task_uuids }),
  });
  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.message || 'Failed to cancel task');
  }
}

export async function deleteTasks({ task_uuids }) {
  const res = await fetch('/api/tasks/instances/delete', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ task_uuids }),
  });
  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.message || 'Failed to delete task');
  }
}

export async function pauseTasks({ task_uuids }) {
  const res = await fetch('/api/tasks/instances/pause', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ task_uuids }),
  });
  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.message || 'Failed to pause task');
  }
}

export async function resumeTasks({ task_uuids }) {
  const res = await fetch('/api/tasks/instances/resume', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ task_uuids }),
  });
  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.message || 'Failed to resume task');
  }
}

/**
 * Wait for a task to complete and return its result
 */
export async function waitForTaskResult(task_uuid, { intervalMs = 1000, timeoutMs = 60000 } = {}) {
  const startTime = Date.now();

  while (true) {
    const status = await fetchTasksStatus({ task_uuids: [task_uuid] });

    if (status === TaskStatus.COMPLETED) return await fetchTasksResult({ task_uuids: [task_uuid] });
    if (status === TaskStatus.ERROR) throw new Error(`Task ${task_uuid} failed`);

    if (Date.now() - startTime > timeoutMs) throw new Error(`Timeout waiting for task ${task_uuid}`);

    await new Promise(r => setTimeout(r, intervalMs));
  }
}
