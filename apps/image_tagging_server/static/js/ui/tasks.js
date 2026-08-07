import {
  cancelTasks, createTaskConfig, deleteTaskConfigs, deleteTasks, fetchTaskConfigs, fetchTasksStatus,
  pauseTasks, pauseTasksInPlace, resumeTasks, resumeTasksInPlace, startTask, updateTaskConfig,
} from '/app-static/js/shared/api/tasks.js';
import { ActiveTaskField, TaskConfigField } from '/app-static/js/ui/utils/fields/index.js';
import { confirmStartIfDirty, createGenericRow, Logger, showPaneError, toast } from '/app-static/js/ui/utils/index.js';
import { buildPanelEditorBar } from '/app-static/js/ui/utils/panelEditorBar.js';
import { createListEditorScreen } from '/app-static/js/ui/utils/listEditorScreen.js';

/** Sub-tab that hosts the config list-editor; used by the nav guard. */
const CONFIG_TAB_ID = 'tab-task-management';
/** Sub-tab that hosts the running/completed instances. */
export const ACTIVITY_TAB_ID = 'tab-task-activity';

let _rendering = false;
let _navGuardController = null;

/**
 * Mark the form dirty on any edit anywhere inside it.
 *
 * See the identical helper in ui/sources.js for why this is delegation rather
 * than a callback threaded through TaskConfigField → SchemaField → each field:
 * there is no single hook that fires for all of them, and the task-type dropdown
 * swaps the whole schema form out mid-edit.
 *
 * @param {HTMLElement} root
 * @param {() => void} onEdit
 */
function watchForEdits(root, onEdit) {
  root.addEventListener('input', onEdit);
  root.addEventListener('change', onEdit);
}

/**
 * Renders the task-config manager as a list-beside-editor screen.
 *
 * Previously this was a `.config-column` holding the config list, a
 * permanently-expanded "Create New Task Config" form, and inline row editing —
 * inside one half of a two-column split, so a task schema was rendered into a
 * quarter of the window and could not be seen without scrolling. Now the list is
 * a narrow column and the schema form owns the rest of the pane.
 *
 * @param {{ target?: string, refresh?: (() => void) | null }} opts
 */
export async function renderTaskConfigs({ target = 'task-config-list', refresh = null } = {}) {
  if (_rendering) return;
  _rendering = true;

  _navGuardController?.abort();
  _navGuardController = new AbortController();

  const container = document.getElementById(target);
  if (!container) { _rendering = false; return; }
  container.innerHTML = '';

  // Temporary/system configs are machinery (preview runs, one-off internal
  // jobs), not things a user maintains — hidden unless asked for.
  let showTemporary = false;

  try {
    const screen = createListEditorScreen({
      container,
      prefix: 'tc',
      title: 'Task Configs',
      itemUuidAttr: 'configUuid',
      withEditorFooter: true,
      searchable: true,
      searchPlaceholder: 'Filter configs…',
      newLabel: '+ New Config',
      onNew: () => _showForm(null),
      ownTabId: CONFIG_TAB_ID,
      navGuardSignal: _navGuardController.signal,
    });
    const { listBody, editorBar, editorBody, editorFooter } = screen;

    // System/temporary toggle sits with the list it filters, between the search
    // box and the rows.
    const toggleRow = document.createElement('label');
    toggleRow.className = 'tc-system-toggle';
    const toggleChk = document.createElement('input');
    toggleChk.type = 'checkbox';
    toggleChk.checked = showTemporary;
    // Short label: the list panel is 240px and the long form wrapped to two lines.
    toggleRow.title = 'Show configs created by the app for preview runs and other internal jobs';
    toggleRow.append(toggleChk, document.createTextNode('Show system'));
    toggleChk.addEventListener('change', () => {
      showTemporary = toggleChk.checked;
      refreshList();
    });
    screen.listPanel.insertBefore(toggleRow, listBody);

    let _selectedUuid = null;
    let _switching = false;

    async function _showForm(config, { force = false } = {}) {
      if (_switching && !force) return;
      _switching = true;
      try {
        if (!force && !await screen.confirmIfDirty()) return;
        screen.setBarHandle(null);
        screen.setDoSave(null);
        _selectedUuid = config?.uuid ?? null;

        editorBody.innerHTML = '';
        editorFooter.innerHTML = '';

        const barHandle = buildPanelEditorBar({
          container: editorBar,
          item: config,
          newTitle: 'New Task Config',
          deleteTitle: 'Delete task config',
          onSave: () => _doSave(true),
          onDelete: config ? async () => {
            await deleteTaskConfigs({ uuids: [config.uuid] });
            toast('Task config deleted', 3000);
            await refreshList();
            await _showForm(null, { force: true });
          } : undefined,
        });
        screen.setBarHandle(barHandle);
        const { setDirty, setSaving, getDirty, updateTitle } = barHandle;

        const field = await TaskConfigField.create(
          config
            ? { value: config.params, name: config.name, typename: config.typename, uuid: config.uuid, editMode: true }
            : { editMode: true },
        );

        const formWrap = document.createElement('div');
        formWrap.className = 'tc-form';
        formWrap.appendChild(await field.renderEdit());
        editorBody.appendChild(formWrap);
        watchForEdits(formWrap, () => setDirty(true));
        setDirty(false);

        // Start is only meaningful for a config that exists server-side; on a new
        // one there is nothing to launch until it has been created.
        if (config) {
          const startBtn = document.createElement('button');
          startBtn.type = 'button';
          startBtn.textContent = '▶ Start';
          startBtn.title = 'Start';
          startBtn.className = 'panel-btn';
          startBtn.dataset.testid = 'task-start';
          startBtn.addEventListener('click', async () => {
            // The server runs the config as stored, so unsaved form edits would
            // silently not be part of the run.
            const proceed = await confirmStartIfDirty({
              getDirty,
              save: () => _doSave(false),
              actionText: 'Start',
            });
            if (!proceed) return;
            startBtn.disabled = true;
            try {
              await startTask({ configUuid: config.uuid });
              toast('Task started — see the Activity tab', 3000);
              refresh?.();
            } catch (error) {
              toast(error.message || 'Failed to start task', 5000, 'error');
            } finally {
              startBtn.disabled = false;
            }
          });
          editorFooter.appendChild(startBtn);
        }

        async function _doSave(navigateAfterCreate = false) {
          const { name, typename, params, persistent } = field.getValue();
          if (!name) { toast('Name is required', 3000, 'error'); return false; }
          if (!typename) { toast('Task type is required', 3000, 'error'); return false; }
          setSaving(true);
          try {
            if (config) {
              await updateTaskConfig({ uuid: config.uuid, name, typename, params });
              setDirty(false);
              setSaving(false);
              updateTitle(name);
              toast('Task config saved', 3000);
              await refreshList();
            } else {
              await createTaskConfig({ name, typename, params, persistent });
              setDirty(false);
              toast(`Task config "${name}" created`, 3000);
              const configs = await refreshList();
              if (navigateAfterCreate) {
                // The create endpoint's response shape isn't a full config
                // record, so re-select from the freshly fetched list by name.
                const created = [...configs.values()].find(c => c.name === name) ?? null;
                await _showForm(created, { force: true });
              } else {
                setSaving(false);
              }
            }
            return true;
          } catch (err) {
            toast(err.message || 'Failed to save task config', 5000, 'error');
            setSaving(false);
            return false;
          }
        }

        screen.setDoSave(() => _doSave(false));
        screen.updateListSelection(config?.uuid ?? null);
      } finally {
        _switching = false;
      }
    }

    /** @returns {Promise<Map<string, any>>} the configs just rendered */
    async function refreshList() {
      listBody.innerHTML = '';
      const configs = await fetchTaskConfigs(null, { excludeTemporary: !showTemporary });
      screen.setCount(configs.size);

      if (configs.size === 0) {
        listBody.appendChild(screen.createListEmpty(
          showTemporary ? 'No task configs.' : 'No user-created task configs.',
        ));
      }

      configs.forEach((config, configUuid) => {
        const chips = [];
        if (config.params?._persistent === true) {
          chips.push({ text: 'persistent', title: 'Restarts automatically after a server restart', variant: 'accent' });
        }
        const item = screen.createListItem({
          uuid: configUuid,
          name: config.name,
          meta: config.typename ?? '',
          chips,
          onClick: () => _showForm({ ...config, uuid: configUuid }),
        });
        item.dataset.testid = 'task-row';
        listBody.appendChild(item);
      });

      screen.applyFilter();
      screen.updateListSelection(_selectedUuid);
      return configs;
    }

    await refreshList();
    await _showForm(null, { force: true });
  } catch (err) {
    showPaneError(container, err.message);
  } finally {
    _rendering = false;
  }
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
 * The element that actually scrolls around `el` — an overflow ancestor if there
 * is one, otherwise null meaning "the page itself".
 *
 * Must be called BEFORE the container is emptied: once its content is gone the
 * ancestor is no longer taller than its box, so the search would find nothing
 * and the scroll position would be dropped exactly when it needs restoring.
 */
function _scrollParent(el) {
  for (let n = el?.parentElement; n; n = n.parentElement) {
    const overflowY = getComputedStyle(n).overflowY;
    if ((overflowY === 'auto' || overflowY === 'scroll') && n.scrollHeight > n.clientHeight) return n;
  }
  return null;
}

/** Statuses that mean "this task is no longer doing anything". */
const TERMINAL_STATUSES = new Set(['completed', 'error']);

/**
 * Count tasks that are still live, for the Activity tab's badge.
 * @param {Record<string, any>} tasks
 * @returns {number}
 */
export function countRunningTasks(tasks) {
  return Object.values(tasks || {})
    .filter(t => !TERMINAL_STATUSES.has(t.instance?.status))
    .length;
}

/**
 * Renders the task instances (its own full-width Activity sub-tab).
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

  // Skip when the Operations main tab or the Activity sub-tab is not visible.
  if (!document.getElementById('tab-operations')?.classList.contains('active') ||
      !document.getElementById(ACTIVITY_TAB_ID)?.classList.contains('active')) {
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

      // Snapshot the page scroll too. A running task appends log lines, which
      // changes the fingerprint and rebuilds every card; emptying the container
      // collapses the document height, so the browser clamps the scroll offset
      // to 0 and the user is thrown back to the top mid-read. Restoring it after
      // the swap is what makes a long expanded card (progress, log tail, dedup
      // calibration tables) readable while the task is actually running.
      const scroller = _scrollParent(container);
      const savedScroll = scroller ? scroller.scrollTop : window.scrollY;

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
        const fieldInstance = await ActiveTaskField.create({
          value: { ...task },
          // Lets the field pre-render its status widgets from cached data when
          // it is replacing a card the user had open, so the rebuild doesn't
          // briefly shrink the page and lose their scroll position.
          expanded: openState[taskId]?.detailsOpen === true,
        });
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
        row.classList.add('task-instance-card');
        row.dataset.taskUuid = taskId;
        return {
          row,
          isCompleted: TERMINAL_STATUSES.has(status),
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

      if (!activeRows.length && !completedRows.length) {
        const empty = document.createElement('p');
        empty.className = 'le-list-empty';
        empty.textContent = 'No tasks have run yet. Start one from the Tasks tab.';
        newListDiv.appendChild(empty);
      }

      const appendSection = (title, rows, actionEl = null) => {
        if (!rows.length) return;
        const h4 = document.createElement('h4');
        h4.className = 'task-section-header';
        h4.textContent = title;
        if (actionEl) {
          const headerRow = document.createElement('div');
          Object.assign(headerRow.style, { display: 'flex', alignItems: 'center', gap: '0.5em' });
          h4.style.margin = '0';
          h4.style.flex = '1';
          headerRow.appendChild(h4);
          headerRow.appendChild(actionEl);
          newListDiv.appendChild(headerRow);
        } else {
          newListDiv.appendChild(h4);
        }
        // Cards flow into as many columns as the (now full-width) pane fits,
        // instead of one per row down half a split.
        const grid = document.createElement('div');
        grid.className = 'task-instance-grid';
        rows.forEach(row => grid.appendChild(row));
        newListDiv.appendChild(grid);
      };
      appendSection('Active', activeRows);

      if (completedRows.length) {
        const clearBtn = document.createElement('button');
        clearBtn.textContent = 'Clear completed';
        clearBtn.className = 'panel-btn secondary';
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

      // Restore the scroll offset synchronously, in the same frame as the swap,
      // so the jump is never painted. Assigning scrollTop/scrollTo forces the
      // layout the new content needs, and the browser clamps for us if the list
      // genuinely got shorter (a task finished and its card collapsed).
      if (savedScroll) {
        if (scroller) scroller.scrollTop = savedScroll;
        else window.scrollTo(0, savedScroll);
      }

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
