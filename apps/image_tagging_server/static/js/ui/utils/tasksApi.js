let _taskConfigs = new Map();   // uuid -> task config
let _taskStatuses = new Map();  // taskId -> status
let _taskResults = new Map();   // taskId -> result

/**
 * Refresh the task configs cache from /api/tasks/configs/list.
 * Clears dependent caches.
 */
export async function refreshTaskConfigsCache() {
  const res = await fetch('/api/tasks/configs/list');
  if (!res.ok) {
    const error = await res.json();
    throw new Error(error.message || 'Failed to fetch task configs');
  }
  const { tasks } = await res.json();
  _taskConfigs.clear();
  for (const task of tasks) {
    _taskConfigs.set(task.uuid, task);
  }

  // Clear dependent caches
  _taskStatuses.clear();
  _taskResults.clear();
}

/**
 * Get all known task configs from the cache.
 */
export function getAllTaskConfigs() {
  return Array.from(_taskConfigs.values());
}

/**
 * Get a task config by uuid.
 */
export function getTaskConfig(uuid) {
  return _taskConfigs.get(uuid) || null;
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
    body: JSON.stringify({ task_uuids: uuids }),
  });

  if (res.ok) {
    uuids.forEach((uuid) => _taskConfigs.delete(uuid));
  } else {
    const error = await res.json();
    throw new Error(error.message || 'Failed to delete task configs');
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
    const error = await res.json();
    throw new Error(error.message || 'Failed to create task config');
  }

  const { config_uuid } = await res.json();
  return config_uuid;
}

/**
 * Start a task from a given config uuid.
 */
export async function startTask({ uuid }) {
  const res = await fetch('/api/tasks/start', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ config_uuid: uuid }),
  });

  if (!res.ok) {
    const error = await res.json();
    throw new Error(error.message || 'Failed to start task');
  }

  const { task_id } = await res.json();
  return task_id;
}

/**
 * Fetch status for a running task.
 */
export async function fetchTaskStatus(taskId) {
  const res = await fetch('/api/tasks/status', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ task_id: taskId }),
  });

  if (!res.ok) {
    const error = await res.json();
    throw new Error(error.message || 'Failed to fetch task status');
  }

  const { status } = await res.json();
  _taskStatuses.set(taskId, status);
  return status;
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
  const res = await fetch('/api/tasks/result', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ task_id: taskId }),
  });

  if (!res.ok) {
    const error = await res.json();
    throw new Error(error.message || 'Failed to fetch task result');
  }

  const { result } = await res.json();
  _taskResults.set(taskId, result);
  return result;
}

/**
 * Get a cached result (if available).
 */
export function getTaskResult(taskId) {
  return _taskResults.get(taskId) || null;
}

/**
 * Pause a running task.
 */
export async function pauseTask(taskId) {
  const res = await fetch('/api/tasks/pause', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ task_id: taskId }),
  });

  if (!res.ok) {
    const error = await res.json();
    throw new Error(error.message || 'Failed to pause task');
  }
}

/**
 * Resume a paused task.
 */
export async function resumeTask(taskId) {
  const res = await fetch('/api/tasks/resume', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ task_id: taskId }),
  });

  if (!res.ok) {
    const error = await res.json();
    throw new Error(error.message || 'Failed to resume task');
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

    if (status === 'completed') {
      return await fetchTaskResult(taskId);
    }
    if (status === 'failed') {
      throw new Error(`Task ${taskId} failed`);
    }

    if (Date.now() - startTime > timeoutMs) {
      throw new Error(`Timeout waiting for task ${taskId} to complete`);
    }

    await new Promise((resolve) => setTimeout(resolve, intervalMs));
  }
}
