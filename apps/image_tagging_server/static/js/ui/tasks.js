import { cancelTasks, createTaskConfig, deleteTaskConfigs, deleteTasks, fetchTaskConfigs, fetchTasksStatus, pauseTasks, pauseTasksInPlace, resumeTasks, resumeTasksInPlace, startTask, updateTaskConfig } from '/app-static/js/shared/api/tasks.js';
import { ActiveTaskField, EditableField, TaskConfigField } from '/app-static/js/ui/utils/fields/index.js';
import { createGenericRow, Logger, showPaneError, toast } from '/app-static/js/ui/utils/index.js';

/**
 * Renders the task configs (left column).
 */
export function renderTaskConfigs({ target = 'task-config-list', onEdit, onDelete, onCreateTask, refresh = null }) {
  const container = document.getElementById(target);
  if (!container) return;
  container.innerHTML = '';

  // Toggle: hide temporary/system configs by default
  let showTemporary = false;
  const toggleRow = document.createElement('div');
  toggleRow.style.cssText = 'display:flex;align-items:center;gap:0.5em;margin-bottom:0.5em;font-size:0.875em;';
  const toggleLabel = document.createElement('label');
  toggleLabel.style.cssText = 'display:flex;align-items:center;gap:0.4em;cursor:pointer;';
  const toggleChk = document.createElement('input');
  toggleChk.type = 'checkbox';
  toggleChk.checked = showTemporary;
  toggleLabel.append(toggleChk, 'Show system/temporary configs');
  toggleRow.appendChild(toggleLabel);
  container.appendChild(toggleRow);

  const configListDiv = document.createElement('div');
  container.appendChild(configListDiv);

  function loadConfigs() {
    configListDiv.innerHTML = '';
    fetchTaskConfigs(null, { excludeTemporary: !showTemporary }).then(response => {
      if (response.size === 0) {
        const empty = document.createElement('p');
        empty.textContent = showTemporary ? 'No task configs.' : 'No user-created task configs.';
        empty.style.cssText = 'color:#888;font-style:italic;';
        configListDiv.appendChild(empty);
        return;
      }
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
            toast('Task started', 2000);
            refresh?.();
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
      }).catch(err => {
        Logger.error('Failed to render task config row:', err);
      });
    }));
    }).catch(err => {
      showPaneError(configListDiv, err.message);
    });
  }

  toggleChk.addEventListener('change', () => {
    showTemporary = toggleChk.checked;
    loadConfigs();
  });

  loadConfigs();

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
        const { name, typename, params, persistent } = field.getValue();
        if (!name) {
          toast('Error: Name required', 5000, "error");
          return;
        }
        await createTaskConfig({ name, typename, params, persistent });
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
      const logs = t.logLines ?? [];
      const actionCount = (t.statusActions ?? []).length + (t.instance?.result?.actions ?? []).length;
      const progressPart = (t.progress === null || t.progress === undefined) ? 'ind' : Math.round(t.progress);
      return `${id}:${t.instance?.status}:${t.isPausedInPlace ?? false}:${progressPart}:${logs.length}:${logs[logs.length - 1] ?? ''}:${t.instance?.startedAt ?? ''}:${t.instance?.endedAt ?? ''}:${actionCount}`;
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
export function renderActiveTasks({ target = 'active-tasks-list', refresh = null, tasks: preloadedTasks = null }) {
  const container = document.getElementById(target);
  if (!container) return;

  // Skip when the Configuration main tab or Task Management sub-tab is not visible.
  if (!document.getElementById('tab-configuration')?.classList.contains('active') ||
      !document.getElementById('tab-task-management')?.classList.contains('active')) {
    return;
  }

  // Guard against concurrent renders (e.g. rapid button clicks during a slow fetch).
  if (container._rendering) return;
  container._rendering = true;

  // Use pre-loaded data from SSE if provided, otherwise fetch from the API.
  const dataPromise = preloadedTasks != null
    ? Promise.resolve({ tasks: preloadedTasks })
    : fetchTasksStatus();

  dataPromise.then(async response => {
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

      // Build every row in parallel.  The live DOM is untouched until
      // everything is ready, then partitioned into Active / Completed sections.
      const built = await Promise.all(Object.entries(tasks || {}).map(async ([taskId, task]) => {
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
        const isPausedInPlace = task.isPausedInPlace === true;
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
          // Suspended — has a checkpoint; resume restarts from it.
          // No rerun here: rerun/retry is only offered on terminal tasks.
          leftButtons = [deleteButton];
          rightButtons = [resumeButton];
        } else {
          // completed, error
          leftButtons = [deleteButton];
          rightButtons = [];
        }

        // Pre-render field content so we get a direct DOM handle for state
        // restoration, and so createGenericRow gets an already-resolved Promise
        // (avoids timing uncertainty about when fieldDiv content is populated).
        const fieldInstance = await ActiveTaskField.create({ value: { ...task } });
        const content = await fieldInstance.renderView();

        // Restore open/scroll state directly on the pre-rendered content node.
        // When saved state exists, it wins over the renderView() default so that
        // a user-collapsed completed task stays collapsed on subsequent polls.
        const saved = openState[taskId];
        if (saved !== undefined) {
          const details = content.querySelector('details');
          if (details) {
            details.open = saved.detailsOpen;
            // Explicitly (re-)dispatch 'toggle' rather than relying on the
            // programmatic-open-assignment case of the native event: browser
            // support for firing 'toggle' on a scripted `.open =` set (as
            // opposed to a user click) is a relatively recent addition to the
            // spec. ActiveTaskField's status-data polling (only-while-
            // expanded) is wired entirely off this event, and every poll
            // cycle that changes the fingerprint rebuilds this <details> node
            // from scratch — without this, a card the user left open could
            // silently stop refreshing its status-data widgets after the
            // first SSE-driven rebuild.
            details.dispatchEvent(new Event('toggle'));
            if (saved.detailsOpen && saved.scrollTop) {
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
        return {
          row,
          isCompleted: status === 'completed' || status === 'error',
          endedAt: task.instance?.endedAt ?? null,
        };
      }));

      // Partition into Active (running/pending/paused) and Completed (completed/error)
      // sections. Active preserves server-provided order; Completed is sorted by
      // endedAt descending so the most recently finished task appears first.
      const newListDiv = document.createElement('div');
      const activeRows = built.filter(b => !b.isCompleted).map(b => b.row);
      const completedRows = built.filter(b => b.isCompleted)
        .sort((a, b) => new Date(b.endedAt ?? 0) - new Date(a.endedAt ?? 0))
        .map(b => b.row);

      const appendSection = (title, rows, actionEl = null) => {
        if (!rows.length) return;
        const h4 = document.createElement('h4');
        h4.className = 'task-section-header';
        h4.textContent = title;
        if (actionEl) {
          const headerRow = document.createElement('div');
          Object.assign(headerRow.style, { display: 'flex', alignItems: 'center', gap: '0.5em' });
          h4.style.margin = '0';
          headerRow.appendChild(h4);
          headerRow.appendChild(actionEl);
          newListDiv.appendChild(headerRow);
        } else {
          newListDiv.appendChild(h4);
        }
        rows.forEach(row => newListDiv.appendChild(row));
      };
      appendSection('Active', activeRows);

      if (completedRows.length) {
        const clearBtn = document.createElement('button');
        clearBtn.textContent = 'Clear completed';
        Object.assign(clearBtn.style, { fontSize: '0.75em', padding: '2px 8px' });
        clearBtn.addEventListener('click', async () => {
          const completedIds = completedRows.map(row => row.dataset.taskUuid).filter(Boolean);
          if (!completedIds.length) return;
          try {
            await deleteTasks({ taskUuids: completedIds });
            refresh?.();
          } catch (error) {
            toast(error.message || 'Failed to clear completed tasks', 5000, 'error');
          }
        });
        appendSection('Completed', completedRows, clearBtn);
      }

      // Atomic swap: old content stays visible until this exact line.
      container.innerHTML = '';
      container.appendChild(newListDiv);

      // Record fingerprint only after a successful swap.
      container._lastFingerprint = fingerprint;

    } finally {
      container._rendering = false;
    }
  }).catch(err => {
    container._rendering = false;
    toast(err.message || 'Server is not responding', 5000, 'error');
  });
}
