import { createCachedFetcher } from "./createCachedFetcher.js";

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
async function fetchTaskSchemaFromServer(typename) {
  const res = await fetch('/api/tasks/types/schema', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ task_typenames: [typename] }),
  });
  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.message || 'Failed to fetch task schema');
  }
  const { schemas } = await res.json();
  return schemas[typename];
}

const taskSchemaFetcher = createCachedFetcher(fetchTaskSchemaFromServer);

export async function fetchTaskTypeSchema(typenames) {
  if (!Array.isArray(typenames)) typenames = [typenames];

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
async function fetchTaskConfigFromServer(uuid) {
  const res = await fetch('/api/tasks/configs/get', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ config_uuids: [uuid] }),
  });

  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.message || 'Failed to fetch task config');
  }

  const { configs } = await res.json();
  return configs[uuid];
}

export const taskConfigFetcher = createCachedFetcher(fetchTaskConfigFromServer);

export async function fetchTaskConfigs(uuids) {
  const fetchAll = uuids == null;

  if (fetchAll) {
    const res = await fetch('/api/tasks/configs/get', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ config_uuids: null }),
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.message || 'Failed to fetch task configs');
    }

    const { configs } = await res.json();
    for (const [uuid, config] of Object.entries(configs)) {
      taskConfigFetcher.set(uuid, config);
    }

    return new Map(Object.entries(configs));
  }

  if (!Array.isArray(uuids)) uuids = [uuids];

  const result = new Map();
  for (const uuid of uuids) {
    const config = await taskConfigFetcher.fetch(uuid);
    result.set(uuid, config);
  }

  return result;
}

export async function deleteTaskConfigs({ uuids }) {
  if (!Array.isArray(uuids)) uuids = [uuids];
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
    const err = await res.json();
    throw new Error(err.message || 'Failed to create task config');
  }

  const { config_uuid } = await res.json();
  return config_uuid;
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

export const taskResultFetcher = createCachedFetcher(fetchTaskResultFromServer);

export async function fetchTasksResult({ taskIds, cacheResults = true }) {
  if (!Array.isArray(taskIds)) taskIds = [taskIds];

  const results = {};
  for (const t of taskIds) {
    const r = await taskResultFetcher.fetch(t);
    if (!cacheResults) taskResultFetcher.delete(t);
    results[t] = r;
  }

  return results;
}


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

export async function fetchTasksStatus({ taskIds }) {
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

export async function cancelTasks({ taskIds }) {
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

export async function deleteTasks({ taskIds }) {
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

export async function pauseTasks({ taskIds }) {
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

export async function resumeTasks({ taskIds }) {
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

/**
 * Wait for a task to complete and return its result
 */
export async function waitForTaskResult(taskId, { intervalMs = 1000, timeoutMs = 60000 } = {}) {
  const startTime = Date.now();

  while (true) {
    const statusMap = await fetchTasksStatus({ taskIds: [taskId] });
    const status = statusMap[taskId]?.status;

    if (status === TaskStatus.COMPLETED) return await fetchTasksResult({ taskIds: taskId });
    if (status === TaskStatus.ERROR) throw new Error(`Task ${taskId} failed`);

    if (Date.now() - startTime > timeoutMs) throw new Error(`Timeout waiting for task ${taskId}`);

    await new Promise(r => setTimeout(r, intervalMs));
  }
}
