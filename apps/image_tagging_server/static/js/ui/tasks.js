import { EditableField, TextField } from '/app-static/js/ui/utils/fields/index.js';
import { error, refreshTaskCache, getTaskTypes } from '/app-static/js/ui/utils/index.js'

/**
 * Renders the task configs (left column).
 * @param {object} params
 * @param {string} params.target - ID of the container
 */
export async function renderTaskConfigs({ target = 'task-config-list' }) {
  const container = document.getElementById(target);
  if (!container) return;
  container.innerHTML = '';

  await refreshTaskCache();
  const taskTypes = getTaskTypes();
  taskTypes.forEach(type => {
    const item = document.createElement('div');
    item.textContent = type;
    container.appendChild(item);
  });
}

/**
 * Renders the active tasks (right column).
 * @param {object} params
 * @param {string} params.target - ID of the container
 */
export async function renderActiveTasks({ target = 'active-tasks-list' }) {
  const container = document.getElementById(target);
  if (!container) return;
  container.innerHTML = '';
}
