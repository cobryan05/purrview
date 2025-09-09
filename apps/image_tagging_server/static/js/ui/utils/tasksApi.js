let _taskTypeCache = null;
let _taskSchemaCache = new Map();
let _taskConfigs = new Map();

export const TaskStatus = Object.freeze({
  NEW: "new",
  PENDING: "pending",
  RUNNING: "running",
  PAUSED: "paused",
  COMPLETED: "completed",
  ERROR: "error"
});


export function clearTaskTypeCache() {
  _taskTypeCache = null;
}

export function clearTaskSchemaCache() {
  _taskSchemaCache.clear();
}

export async function fetchTaskTypeList() {
  if (!_taskTypeCache) {
    const res = await fetch('/api/tasks/types/list');
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.message || 'Failed to fetch task types');
    }
    const { types } = await res.json();
    _taskTypeCache = new Map();
    for (const type of types) {
      _taskTypeCache.set(type, null);
    }
  }
  return Array.from(_taskTypeCache.keys());
}

/**
 * Fetch the schema for a specific task type, with caching.
 */
export async function fetchTaskTypeSchema(typenames) {
  if (!Array.isArray(typenames)) {
    typenames = [typenames];
  }

  const types = await fetchTaskTypeList();
  for (const t of typenames) {
    if (!types.includes(t)) {
      throw new Error(`Unknown task type: ${t}`);
    }
  }

  // Gather cached schemas
  const result = new Map();
  const missing = [];
  for (const t of typenames) {
    const cached = _taskSchemaCache.get(t);
    if (cached != null) {
      result.set(t, cached);
    } else {
      missing.push(t);
    }
  }

  if (missing.length > 0) {
    const res = await fetch('/api/tasks/types/schema', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ task_typenames: missing }),
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.message || 'Failed to fetch task schema');
    }

    const { schemas } = await res.json();
    for (const [t, schema] of Object.entries(schemas)) {
      _taskSchemaCache.set(t, schema);
      result.set(t, schema);
    }
  }

  return result;
}

/**
 * Get a task config by uuid.
 */
export async function fetchTaskConfigs(uuids) {
  const fetchAll = ( uuids == null );
  const missing = [];
  const result = new Map();
  if( !fetchAll ) {
    if (!Array.isArray(uuids)) {
      uuids = uuids ? [uuids] : [];
    }

    // Gather cached schemas
    for (const uuid of uuids) {
      const cached = _taskConfigs.get(uuid);
      if (cached != null) {
        result.set(uuid, cached);
      } else {
        missing.push(uuid);
      }
    }
  }

  if (missing.length > 0 || fetchAll) {
    const res = await fetch('/api/tasks/configs/get', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ config_uuids: fetchAll ? null : missing }),
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.message || 'Failed to fetch task config');
    }

    const { configs } = await res.json();
    for (const [t, config] of Object.entries(configs)) {
      _taskConfigs.set(t, config);
      result.set(t, config);
    }
  }

  return result;
}

/**
 * Delete task configs by UUID(s).
 */
export async function deleteTaskConfigs({ uuids }) {
  if (!Array.isArray(uuids)) {
    uuids = [uuids];
  }
  const res = await fetch('/api/tasks/configs/delete', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ config_uuids: uuids }),
  });

  if (res.ok) {
    uuids.forEach((uuid) => _taskSchemas.delete(uuid));
  } else {
    const err = await res.json();
    throw new Error(err.message || 'Failed to delete task configs');
  }
}

/**
 * Create a new task config.
 */
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

/**
 * Start a task from a given config uuid.
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

/**
 * Fetch status for a running task.
 */
export async function fetchTaskStatus(taskId) {
  const res = await fetch('/api/tasks/instances/get', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ task_ids: [taskId] }),
  });

  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.message || 'Failed to fetch task status');
  }

  const data = await res.json();
  if (data.status !== 'success') {
    throw new Error(data.message || 'Failed to fetch task status');
  }

  const taskStatus = data.tasks?.[taskId];
  if (!taskStatus) {
    throw new Error(`Task ${taskId} not found in response`);
  }

  _taskStatuses.set(taskId, taskStatus);
  return taskStatus;
}

/**
 * Get a cached status (if available).
 */
export function getTaskStatus(taskId) {
  return _taskStatuses.get(taskId) || null;
}

/**
 * Fetch result for a completed task.
 */
export async function fetchTaskResult(taskId) {
  const res = await fetch('/api/tasks/instances/result', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ task_ids: [taskId] }),
  });

  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.message || 'Failed to fetch task result');
  }

  const { result } = await res.json();
  const taskRes = result[taskId];
  _taskResults.set(taskId, taskRes);
  return taskRes;
}

/**
 * Get a cached result (if available).
 */
export function getTaskResult(taskId) {
  return _taskResults.get(taskId) || null;
}

/**
 * Cancel a running task.
 */
export async function cancelTask(taskId) {
  const res = await fetch('/api/tasks/instances/cancel', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ task_ids: [taskId] }),
  });

  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.message || 'Failed to cancel task');
  }
}

/**
 * Delete a task
 */
export async function deleteTask(taskId) {
  const res = await fetch('/api/tasks/instances/delete', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ task_ids: [taskId] }),
  });

  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.message || 'Failed to cancel task');
  }
}

/**
 * Pause a running task.
 */
export async function pauseTask(taskId) {
  const res = await fetch('/api/tasks/instances/pause', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ task_ids: [taskId] }),
  });

  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.message || 'Failed to pause task');
  }
}

/**
 * Resume a paused task.
 */
export async function resumeTask(taskId) {
  const res = await fetch('/api/tasks/instances/resume', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ task_ids: [taskId] }),
  });

  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.message || 'Failed to resume task');
  }
}

/**
 * Helper: wait until a task finishes and return its result.
 * Polls /api/tasks/status until status is "completed" or "failed".
 */
export async function waitForTaskResult(taskId, { intervalMs = 1000, timeoutMs = 60000 } = {}) {
  const startTime = Date.now();

  while (true) {
    const status = await fetchTaskStatus(taskId);

    if (status === TaskStatus.COMPLETED) {
      return await fetchTaskResult(taskId);
    }
    if (status === TaskStatus.ERROR) {
      throw new Error(`Task ${taskId} failed`);
    }

    if (Date.now() - startTime > timeoutMs) {
      throw new Error(`Timeout waiting for task ${taskId} to complete`);
    }

    await new Promise((resolve) => setTimeout(resolve, intervalMs));
  }
}
