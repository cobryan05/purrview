import { createGenericRow } from '/app-static/js/ui/utils/createGenericRow.js';
import { ActiveTaskField, EditableField, TaskConfigField } from '/app-static/js/ui/utils/fields/index.js';
import { cancelTasks, deleteTasks, deleteTaskConfigs, fetchTaskConfigs, fetchTasksStatus, startTask, toast } from '/app-static/js/ui/utils/index.js';

/**
 * Renders the task configs (left column).
 * @param {object} params
 * @param {string} params.target - ID of the container
 */
export function renderTaskConfigs({ target = 'task-config-list', onEdit, onDelete, onStartTask }) {
  const container = document.getElementById(target);
  if (!container) return;
  container.innerHTML = '';

  const configListDiv = document.createElement('div');
  container.appendChild(configListDiv);

  const rerender = () => renderTaskConfigs({ target, onEdit, onDelete, onStartTask });

  fetchTaskConfigs().then(configs => {
    configs.forEach((config, uuid) => {
      const deleteButton = {
        text: 'Delete',
        emoji: '🗑️',
        onClick: async ({ field }) => {
          if (!window.confirm(`Are you sure you want to delete "${field.getValue().text}"?`)) return;
          try {
            await deleteTaskConfigs({ uuids: [uuid] });
            onDelete?.();
          } catch (error) {
            toast(error.message || 'Failed to delete task config', 5000, 'error');
          }
        }
      };
      const startButton = {
        text: 'Start',
        emoji: '▶️',
        onClick: async () => {
          try {
            await startTask({ uuid });
            onStartTask?.();
          } catch (error) {
            toast(error.message || 'Failed to start task', 5000, 'error');
          }
        }
      };
      TaskConfigField.create({ value: config.params, ...config }).then(fieldInstance => {
        const row = createGenericRow({
          field: new EditableField({
            field: fieldInstance,
            onSave: async ({ text: name, typename, schema: filled_schema }) => {
              //await updateSource({ uuid, name, providerName: typename, params: filled_schema });
              rerender();
            },
            onCancel: () => {
              rerender();
            },
          }),
          leftButtons: [deleteButton],
          rightButtons: [startButton]
        })
        configListDiv.appendChild(row);
      });
    });
  });


  container.appendChild(document.createElement('hr'));
  const createHeader = document.createElement('h3');
  createHeader.textContent = 'Create New Task Config';
  container.appendChild(createHeader);

  const newConfigDiv = document.createElement('div');
  container.appendChild(newConfigDiv);
  TaskConfigField.create({ editMode: true }).then(fieldInstance => {
    const row = createGenericRow({
      field: fieldInstance,
      editMode: true,
    })
    newConfigDiv.appendChild(row);
  });
}

/**
 * Renders the active tasks (right column).
 * @param {object} params
 * @param {string} params.target - ID of the container
 */
export function renderActiveTasks({ target = 'active-tasks-list' }) {
  const container = document.getElementById(target);
  if (!container) return;
  container.innerHTML = '';

  const configListDiv = document.createElement('div');
  container.appendChild(configListDiv);

  const rerender = () => renderActiveTasks({ target });

  fetchTasksStatus().then(tasks => {
    tasks.forEach((task, taskId) => {
      const deleteButton = {
        text: 'Delete',
        emoji: '🗑️',
        onClick: async ({ field }) => {
          if (!window.confirm(`Are you sure you want to delete "${field.getValue().text}"?`)) return;
          try {
            await deleteTasks({ taskIds: taskId });
            rerender();
          } catch (error) {
            toast(error.message || 'Failed to delete task', 5000, 'error');
          }
        }
      };
      const stopButton = {
        text: 'Stop',
        emoji: '⏹️',
        onClick: async () => {
          try {
            await cancelTasks({ taskIds: taskId });
            rerender();
          } catch (error) {
            toast(error.message || 'Failed to stop task', 5000, 'error');
          }
        }
      };
      ActiveTaskField.create({ value: task.id, ...task }).then(fieldInstance => {
        const row = createGenericRow({
          field: new EditableField({
            field: fieldInstance,
            onSave: async ({ text: name, typename, schema: filled_schema }) => {
              //await updateSource({ uuid, name, providerName: typename, params: filled_schema });
              renderTaskConfigs({ target, onEdit, onDelete, onStartTask });
            },
            onCancel: () => {
              renderTaskConfigs({ target, onEdit, onDelete, onStartTask });
            },
          }),
          leftButtons: [deleteButton],
          rightButtons: [stopButton]
        })
        configListDiv.appendChild(row);
      });
    });
  });
}
