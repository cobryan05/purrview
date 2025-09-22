import { createCachedFetcher } from "./createCachedFetcher.js";
import { wrapSingleKey } from './apiUtils.js';

export const TaskStatus = Object.freeze({
  NEW: "new",
  PENDING: "pending",
  RUNNING: "running",
  PAUSED: "paused",
  COMPLETED: "completed",
  ERROR: "error"
});


/* ---- Task type list --- */
async function fetchTaskTypeListFromServer() {
  const res = await fetch('/api/tasks/types/list');
  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.message || 'Failed to fetch task types');
  }
  const { types } = await res.json();
  return types;
}
const taskTypeFetcher = createCachedFetcher(fetchTaskTypeListFromServer);
export async function fetchTaskTypeList() {
  return taskTypeFetcher.fetch('taskTypes'); // single key for list
}

export function clearTaskTypeCache() {
  taskTypeFetcher.clear();
}

/* ---- Task type schemas --- */
async function _fetchTasksSchemaFromServer(typenames) {
  const res = await fetch('/api/tasks/types/schema', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ task_typenames: typenames }),
  });
  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.message || 'Failed to fetch task schema');
  }
  const { schemas } = await res.json();
  return schemas;
}
const fetchTasksSchemaFromServer = wrapSingleKey(_fetchTasksSchemaFromServer);

const taskSchemaFetcher = createCachedFetcher(fetchTasksSchemaFromServer);
async function _fetchTaskTypeSchema(typenames) {
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
export const fetchTaskTypeSchema = wrapSingleKey(_fetchTaskTypeSchema);

export function clearTaskSchemaCache() {
  taskSchemaFetcher.clear();
}

/** -----------------------------
 * TASK CONFIGS
 * -----------------------------
 */
async function _fetchTaskConfigs(uuids = null) {
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

  const { configs } = await res.json();
  return new Map(Object.entries(configs));
}

export const fetchTaskConfigs = wrapSingleKey(_fetchTaskConfigs);
export const taskConfigFetcher = createCachedFetcher(fetchTaskConfigs);

export async function _deleteTaskConfigs({ uuids }) {
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
export const deleteTaskConfigs = wrapSingleKey(_deleteTaskConfigs);

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

  const { config_uuid } = await res.json();
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
async function fetchTaskResultFromServer(taskId) {
  const res = await fetch('/api/tasks/instances/result', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ task_ids: [taskId] }),
  });

  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.message || 'Failed to fetch task result');
  }

  const { results } = await res.json();
  return results[taskId];
}

const taskResultFetcher = createCachedFetcher(fetchTaskResultFromServer);

export async function _fetchTasksResult({ taskIds, cacheResults = true }) {
  const results = {};
  for (const t of taskIds) {
    const r = await taskResultFetcher.fetch(t);
    if (!cacheResults) taskResultFetcher.delete(t);
    results[t] = r;
  }

  return results;
}
export const fetchTasksResult = wrapSingleKey(_fetchTasksResult, "taskIds");


/** -----------------------------
 * TASK INSTANCE OPERATIONS
 * -----------------------------
 */
export async function startTask({ uuid }) {
  const res = await fetch('/api/tasks/instances/start', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ config_uuid: uuid }),
  });
  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.message || 'Failed to start task');
  }
  const { task_id } = await res.json();
  return task_id;
}

async function _fetchTasksStatus({ taskIds } = {}) {
  const res = await fetch('/api/tasks/instances/get', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ task_ids: taskIds }),
  });

  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.message || 'Failed to fetch task status');
  }

  const data = await res.json();
  if (data.status !== 'success') {
    throw new Error(data.message || 'Failed to fetch task status');
  }

  return data.tasks;
}
export const fetchTasksStatus = wrapSingleKey(_fetchTasksStatus);

async function _cancelTasks({ taskIds }) {
  const res = await fetch('/api/tasks/instances/cancel', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ task_ids: taskIds }),
  });
  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.message || 'Failed to cancel task');
  }
}
export const cancelTasks = wrapSingleKey(_cancelTasks);

async function _deleteTasks({ taskIds }) {
  const res = await fetch('/api/tasks/instances/delete', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ task_ids: taskIds }),
  });
  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.message || 'Failed to delete task');
  }
}
export const deleteTasks = wrapSingleKey(_deleteTasks);

async function _pauseTasks({ taskIds }) {
  const res = await fetch('/api/tasks/instances/pause', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ task_ids: taskIds }),
  });
  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.message || 'Failed to pause task');
  }
}
export const pauseTasks = wrapSingleKey(_pauseTasks);

async function _resumeTasks({ taskIds }) {
  const res = await fetch('/api/tasks/instances/resume', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ task_ids: taskIds }),
  });
  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.message || 'Failed to resume task');
  }
}
export const resumeTasks = wrapSingleKey(_resumeTasks);

/**
 * Wait for a task to complete and return its result
 */
export async function waitForTaskResult(taskId, { intervalMs = 1000, timeoutMs = 60000 } = {}) {
  const startTime = Date.now();

  while (true) {
    const status = await fetchTasksStatus({ taskIds: taskId });

    if (status === TaskStatus.COMPLETED) return await fetchTasksResult({ taskIds: taskId });
    if (status === TaskStatus.ERROR) throw new Error(`Task ${taskId} failed`);

    if (Date.now() - startTime > timeoutMs) throw new Error(`Timeout waiting for task ${taskId}`);

    await new Promise(r => setTimeout(r, intervalMs));
  }
}
