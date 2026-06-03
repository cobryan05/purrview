import { cancelTasks, createTaskConfig, deleteTaskConfigs, deleteTasks, fetchTaskConfigs, fetchTasksStatus, pauseTasks, pauseTasksInPlace, resumeTasks, resumeTasksInPlace, startTask, updateTaskConfig } from '/app-static/js/shared/api/tasks.js';
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
          if (!window.confirm(`Are you sure you want to delete "${field.getValue().name}"?`)) return;
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
 * Compute a cheap fingerprint over task data so we can skip re-renders when
 * nothing has changed.  Includes status, rounded progress, and the last log line.
 * @param {Record<string, any>} tasks
 * @returns {string}
 */
function _taskFingerprint(tasks) {
  return Object.entries(tasks || {})
    .map(([id, t]) => {
      const logs = t.log_lines ?? [];
      return `${id}:${t.instance?.status}:${t.is_paused_in_place ?? false}:${Math.round(t.progress ?? 0)}:${logs.length}:${logs[logs.length - 1] ?? ''}:${t.instance?.started_at ?? ''}:${t.instance?.ended_at ?? ''}`;
    })
    .sort()
    .join('|');
}

/**
 * Renders the active tasks (right column).
 *
 * Flash-free strategy:
 *   1. Fetch + build all new rows in a detached div while old DOM stays visible.
 *   2. Fingerprint-check: if data hasn't changed, skip the swap entirely.
 *   3. Snapshot open/scroll state from the live DOM.
 *   4. Atomically swap the detached div into the container.
 *
 * This means the pane never goes blank — the old content is replaced only when
 * the new content is fully ready.
 */
export function renderActiveTasks({ target = 'active-tasks-list', refresh = null }) {
  const container = document.getElementById(target);
  if (!container) return;

  // Skip polling when the Configuration main tab or Task Management sub-tab is not visible.
  if (!document.getElementById('tab-configuration')?.classList.contains('active') ||
      !document.getElementById('tab-task-management')?.classList.contains('active')) {
    return;
  }

  // Guard against concurrent renders (e.g. rapid button clicks during a slow fetch).
  if (container._rendering) return;
  container._rendering = true;

  fetchTasksStatus().then(async response => {
    try {
      const { tasks } = response;

      // Skip re-render entirely when nothing meaningful changed.
      const fingerprint = _taskFingerprint(tasks);
      if (container._lastFingerprint === fingerprint) return;

      // Snapshot open/scroll state from the *current* live DOM before we touch it.
      const openState = {};
      container.querySelectorAll('[data-task-uuid]').forEach(row => {
        const uuid = row.dataset.taskUuid;
        if (!uuid) return;
        const details = row.querySelector('details');
        const pre = row.querySelector('pre');
        openState[uuid] = {
          detailsOpen: details?.open ?? false,
          scrollTop: pre?.scrollTop ?? 0,
        };
      });

      // Build every row in parallel inside a detached div.  The live DOM is
      // untouched until everything is ready.
      const newListDiv = document.createElement('div');

      await Promise.all(Object.entries(tasks || {}).map(async ([taskId, task]) => {
        const status = task.instance?.status;

        const deleteButton = {
          text: 'Delete', emoji: '🗑️',
          onClick: async () => {
            if (!window.confirm(`Delete task ${task.config?.name ?? taskId}?`)) return;
            try { await deleteTasks({ taskUuids: [taskId] }); refresh?.(); }
            catch (error) { toast(error.message || 'Failed to delete task', 5000, 'error'); }
          }
        };
        const stopButton = {
          text: 'Stop', emoji: '⏹️',
          onClick: async () => {
            try { await cancelTasks({ taskUuids: [taskId] }); refresh?.(); }
            catch (error) { toast(error.message || 'Failed to stop task', 5000, 'error'); }
          }
        };
        const pauseInPlaceButton = {
          text: 'Pause', emoji: '⏸️',
          onClick: async () => {
            try { await pauseTasksInPlace({ taskUuids: [taskId] }); refresh?.(); }
            catch (error) { toast(error.message || 'Failed to pause task', 5000, 'error'); }
          }
        };
        const resumeInPlaceButton = {
          text: 'Resume', emoji: '▶️',
          onClick: async () => {
            try { await resumeTasksInPlace({ taskUuids: [taskId] }); refresh?.(); }
            catch (error) { toast(error.message || 'Failed to resume task', 5000, 'error'); }
          }
        };
        const suspendButton = {
          text: 'Suspend', emoji: '💾',
          onClick: async () => {
            try { await pauseTasks({ taskUuids: [taskId] }); refresh?.(); }
            catch (error) { toast(error.message || 'Failed to suspend task', 5000, 'error'); }
          }
        };
        const resumeButton = {
          text: 'Resume', emoji: '▶️',
          onClick: async () => {
            try { await resumeTasks({ taskUuids: [taskId] }); refresh?.(); }
            catch (error) { toast(error.message || 'Failed to resume task', 5000, 'error'); }
          }
        };
        const rerunButton = {
          text: 'Rerun', emoji: '🔁',
          onClick: async () => {
            try {
              // Delete the current instance (may be paused, completed, or errored)
              // before creating a fresh one so we don't leave orphaned checkpoints.
              await deleteTasks({ taskUuids: [taskId] });
              await startTask({ configUuid: task.config?.uuid });
              refresh?.();
            } catch (error) { toast(error.message || 'Failed to start task', 5000, 'error'); }
          }
        };

        const isPausedInPlace = task.is_paused_in_place === true;
        let leftButtons = [];
        let rightButtons = [];
        if (status === 'running' && !isPausedInPlace) {
          leftButtons = [pauseInPlaceButton, suspendButton];
          rightButtons = [stopButton];
        } else if (status === 'running' && isPausedInPlace) {
          leftButtons = [resumeInPlaceButton, suspendButton];
          rightButtons = [stopButton];
        } else if (status === 'pending') {
          rightButtons = [stopButton];
        } else if (status === 'paused') {
          // Suspended — has a checkpoint; resume restarts from it
          leftButtons = [deleteButton];
          rightButtons = [resumeButton, rerunButton];
        } else {
          // completed, error
          leftButtons = [deleteButton];
          rightButtons = [rerunButton];
        }

        // Pre-render field content so we get a direct DOM handle for state
        // restoration, and so createGenericRow gets an already-resolved Promise
        // (avoids timing uncertainty about when fieldDiv content is populated).
        const fieldInstance = await ActiveTaskField.create({ value: { ...task } });
        const content = await fieldInstance.renderView();

        // Restore open/scroll state directly on the pre-rendered content node.
        const saved = openState[taskId];
        if (saved?.detailsOpen) {
          const details = content.querySelector('details');
          if (details) {
            details.open = true;
            if (saved.scrollTop) {
              requestAnimationFrame(() => {
                const pre = details.querySelector('pre');
                if (pre) pre.scrollTop = saved.scrollTop;
              });
            }
          }
        }

        // Wrap the pre-rendered node in a minimal field-compatible object so
        // createGenericRow doesn't re-render it.
        const preRendered = {
          getEditMode: () => false,
          renderView: async () => content,
          getValue: () => ({}),
        };

        const row = createGenericRow({ field: preRendered, leftButtons, rightButtons });
        row.dataset.taskUuid = taskId;
        newListDiv.appendChild(row);
      }));

      // Atomic swap: old content stays visible until this exact line.
      container.innerHTML = '';
      container.appendChild(newListDiv);

      // Record fingerprint only after a successful swap.
      container._lastFingerprint = fingerprint;

    } finally {
      container._rendering = false;
    }
  }).catch(() => {
    container._rendering = false;
  });
}
