import { cancelTasks, createTaskConfig, deleteTaskConfigs, deleteTasks, fetchTaskConfigs, fetchTasksStatus, pauseTasks, resumeTasks, startTask, updateTaskConfig } from '/app-static/js/shared/api/tasks.js';
import { ActiveTaskField, EditableField, TaskConfigField } from '/app-static/js/ui/utils/fields/index.js';
import { createGenericRow, Logger, toast } from '/app-static/js/ui/utils/index.js';

/**
 * Renders the task configs (left column).
 */
export function renderTaskConfigs({ target = 'task-config-list', onEdit, onDelete, onStartTask, onCreateTask, refresh = null }) {
  const container = document.getElementById(target);
  if (!container) return;
  container.innerHTML = '';

  const configListDiv = document.createElement('div');
  container.appendChild(configListDiv);

  fetchTaskConfigs().then(response => {
    response.forEach(((config, configUuid) => {
      const deleteButton = {
        text: 'Delete',
        emoji: '🗑️',
        onClick: async ({ field }) => {
          if (!window.confirm(`Are you sure you want to delete "${field.getValue().text}"?`)) return;
          try {
            await deleteTaskConfigs({ uuids: [configUuid] });
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
            await startTask({ configUuid });
            onStartTask?.();
            refresh?.()
          } catch (error) {
            toast(error.message || 'Failed to start task', 5000, 'error');
          }
        }
      };
      TaskConfigField.create({ value: config.params_json, ...config }).then(fieldInstance => {
        const row = createGenericRow({
          field: new EditableField({
            field: fieldInstance,
            onSave: async ({ name, typename, params }) => {
              await updateTaskConfig({ uuid: configUuid, name, typename, params });
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
    }));
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
 */
export function renderActiveTasks({ target = 'active-tasks-list', refresh = null }) {
  const container = document.getElementById(target);
  if (!container) return;
  container.innerHTML = '';

  const listDiv = document.createElement('div');
  container.appendChild(listDiv);

  /** @type {Promise<import('@web_api').TasksInfoResponse>} */
  fetchTasksStatus().then(response => {
    const { tasks } = response;

    Object.entries(tasks || {}).forEach(([taskId, task]) => {
      const status = task.instance?.status;

      const deleteButton = {
        text: 'Delete',
        emoji: '🗑️',
        onClick: async () => {
          if (!window.confirm(`Delete task ${task.config?.name ?? taskId}?`)) return;
          try {
            await deleteTasks({ taskUuids: [taskId] });
            refresh?.();
          } catch (error) {
            toast(error.message || 'Failed to delete task', 5000, 'error');
          }
        }
      };

      const cancelButton = {
        text: 'Cancel',
        emoji: '⏹️',
        onClick: async () => {
          try {
            await cancelTasks({ taskUuids: [taskId] });
            refresh?.();
          } catch (error) {
            toast(error.message || 'Failed to cancel task', 5000, 'error');
          }
        }
      };

      const pauseButton = {
        text: 'Pause',
        emoji: '⏸️',
        onClick: async () => {
          try {
            await pauseTasks({ taskUuids: [taskId] });
            refresh?.();
          } catch (error) {
            toast(error.message || 'Failed to pause task', 5000, 'error');
          }
        }
      };

      const resumeButton = {
        text: 'Resume',
        emoji: '▶️',
        onClick: async () => {
          try {
            await resumeTasks({ taskUuids: [taskId] });
            refresh?.();
          } catch (error) {
            toast(error.message || 'Failed to resume task', 5000, 'error');
          }
        }
      };

      let leftButtons = [];
      let rightButtons = [];

      if (status === 'running') {
        leftButtons = [pauseButton];
        rightButtons = [cancelButton];
      } else if (status === 'pending') {
        rightButtons = [cancelButton];
      } else if (status === 'paused') {
        leftButtons = [deleteButton];
        rightButtons = [resumeButton];
      } else {
        // completed, error
        leftButtons = [deleteButton];
      }

      ActiveTaskField.create({ value: { ...task } }).then(fieldInstance => {
        const row = createGenericRow({
          field: fieldInstance,
          leftButtons,
          rightButtons,
        });
        listDiv.appendChild(row);
      });
    });
  });
}
