import { cancelTasks, createTaskConfig, deleteTaskConfigs, deleteTasks, fetchTaskConfigs, fetchTasksStatus, startTask, updateTaskConfig } from '/app-static/js/shared/api/tasks.js';
import { ActiveTaskField, EditableField, TaskConfigField } from '/app-static/js/ui/utils/fields/index.js';
import { createGenericRow, Logger, toast } from '/app-static/js/ui/utils/index.js';
/**
 * Renders the task configs (left column).
 * @param {object} params
 * @param {string} params.target - ID of the container
 */
export function renderTaskConfigs({ target = 'task-config-list', onEdit, onDelete, onStartTask, onCreateTask, refresh = null }) {
  const container = document.getElementById(target);
  if (!container) return;
  container.innerHTML = '';

  const configListDiv = document.createElement('div');
  container.appendChild(configListDiv);

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
            refresh?.();
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
            await startTask({ uuid: uuid });
            onStartTask?.();
            refresh?.()
          } catch (error) {
            toast(error.message || 'Failed to start task', 5000, 'error');
          }
        }
      };
      TaskConfigField.create({ value: config.params, ...config }).then(fieldInstance => {
        const row = createGenericRow({
          field: new EditableField({
            field: fieldInstance,
            onSave: async ({ name, typename, params }) => {
              await updateTaskConfig({ uuid, name, typename, params });
              refresh?.();
            },
            onCancel: () => {
              refresh?.();
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

  const saveButton = {
    text: 'Add Config',
    emoji: '✅',
    onClick: async ({ field }) => {
      try {
        const { name, typename, params } = field.getValue();
        if (!name) {
          toast('Error: Name required', 5000, "error");
          return;
        }
        await createTaskConfig({ name, typename, params });
        refresh?.();
      } catch (err) {
        Logger.error('Error creating task config:', err);
      }
    }
  };

  TaskConfigField.create({ editMode: true }).then(fieldInstance => {
    const row = createGenericRow({
      field: fieldInstance,
      editMode: true,
      rightButtons: [saveButton],
    })
    newConfigDiv.appendChild(row);
  });
}

/**
 * Renders the active tasks (right column).
 * @param {object} params
 * @param {string} params.target - ID of the container
 */
export function renderActiveTasks({ target = 'active-tasks-list', refresh = null }) {
  const container = document.getElementById(target);
  if (!container) return;
  container.innerHTML = '';

  const configListDiv = document.createElement('div');
  container.appendChild(configListDiv);

  fetchTasksStatus().then(tasks => {
    tasks.forEach((task, taskId) => {
      const deleteButton = {
        text: 'Delete',
        emoji: '🗑️',
        onClick: async ({ field }) => {
          if (!window.confirm(`Are you sure you want to delete "${taskId}"?`)) return;
          try {
            await deleteTasks({ taskIds: taskId });
            refresh?.();
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
            refresh?.();
          } catch (error) {
            toast(error.message || 'Failed to stop task', 5000, 'error');
          }
        }
      };
      ActiveTaskField.create({ value: { ...task } }).then(fieldInstance => {
        const row = createGenericRow({
          field: fieldInstance,
          leftButtons: [deleteButton],
          rightButtons: [stopButton]
        });
        configListDiv.appendChild(row);
      });
    });
  });
}
