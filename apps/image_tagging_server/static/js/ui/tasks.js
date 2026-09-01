import {
  cancelTasks, createTaskConfig, deleteTaskConfigs, deleteTasks, fetchTaskConfigs, fetchTasksStatus,
  fetchTaskTypeAccents, pauseTasks, pauseTasksInPlace, resumeTasks, resumeTasksInPlace, startTask,
  taskResultData, updateTaskConfig,
} from '/app-static/js/shared/api/tasks.js';
import {
  ActiveTaskField, FIELD_EDIT_EVENT, TaskConfigField, TERMINAL_STATUSES,
} from '/app-static/js/ui/utils/fields/index.js';
import { confirmStartIfDirty, createGenericRow, Logger, showPaneError, toast } from '/app-static/js/ui/utils/index.js';
import { buildPanelEditorBar } from '/app-static/js/ui/utils/panelEditorBar.js';
import { createListEditorScreen } from '/app-static/js/ui/utils/listEditorScreen.js';
import { navigateTabs, takePendingSelection } from '/app-static/js/app/utils/tabs.js';
import { configTargetForTask } from '/app-static/js/ui/utils/runKinds.js';
import { openWatcherLiveView } from '/app-static/js/app/nvr/liveViewModal.js';
import { parseApiDateMs } from '/app-static/js/shared/time.js';
import { promptRestartOwnTask } from '/app-static/js/ui/utils/restartAffectedTasks.js';

/** Configuration sub-tab that hosts the config list-editor; used by the nav guard. */
const CONFIG_TAB_ID = 'tab-task-management';
/**
 * Jobs sub-tab that hosts the running/completed instances.
 *
 * This used to be the Operations sub-tab `tab-task-activity`. Jobs is a main
 * tab now, and its Running pane holds the list — its sibling is the launcher.
 */
export const ACTIVITY_TAB_ID = 'tab-runs-active';
/** Main tab enclosing it. */
const RUNS_TAB_ID = 'tab-runs';

let _rendering = false;
let _navGuardController = null;

// ── "show me that run" ───────────────────────────────────────────────────────
//
// Pressing a launch card whose run is already going brings the user here. A
// pane listing every job, three sections deep, is not an answer on its own — the
// run they asked about has to be findable at a glance, so it is scrolled to and
// marked.
//
// The mark has to outlive a re-render: this pane rebuilds every card whenever a
// running task appends a log line (see the flash-free swap below), which would
// otherwise wipe the highlight within a second of arriving. So it is held as
// state with a deadline and re-applied after each swap, rather than being a
// class set once on a node.

/** How long an arriving-from-a-card highlight stays on the row. */
const HIGHLIGHT_MS = 6000;
/** @type {?string} */
let _highlightUuid = null;
let _highlightUntil = 0;
/** Scrolling is a one-off; re-applying the class across re-renders is not. */
let _highlightScrolled = false;
/** @type {?number} pending removal, so re-renders do not stack timers. */
let _highlightTimer = null;

/**
 * Pick up a "open with this run showing" request left by navigateTabs.
 *
 * Consumed as early as the render, not at the end: a fingerprint-unchanged
 * render returns before ever touching the DOM, and the request must not be left
 * in the map to ambush the user's next ordinary visit to this pane.
 */
function _takeHighlightRequest() {
  const uuid = takePendingSelection(ACTIVITY_TAB_ID);
  if (!uuid) return;
  if (_highlightTimer !== null) {
    // A second arrival replaces the first, deadline and all.
    clearTimeout(_highlightTimer);
    _highlightTimer = null;
  }
  _highlightUuid = uuid;
  _highlightUntil = Date.now() + HIGHLIGHT_MS;
  _highlightScrolled = false;
}

/**
 * Mark the requested run's card, for as long as the request is still fresh.
 *
 * @param {HTMLElement} container
 */
function _applyHighlight(container) {
  if (!_highlightUuid) return;
  if (Date.now() > _highlightUntil) {
    _highlightUuid = null;
    return;
  }
  const row = container.querySelector(`[data-task-uuid="${CSS.escape(_highlightUuid)}"]`);
  // Not there yet (or not any more — a cleared task): leave the request armed
  // until it expires, so a card that arrives with the next stream message is
  // still marked.
  if (!row) return;
  row.classList.add('is-highlighted');
  if (!_highlightScrolled) {
    _highlightScrolled = true;
    row.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
  }
  // One timer per request, not per render — this runs after every swap, and a
  // task that keeps logging would otherwise schedule one a second.
  if (_highlightTimer !== null) return;
  _highlightTimer = setTimeout(() => {
    _highlightTimer = null;
    _highlightUuid = null;
    document.querySelectorAll('.task-instance-card.is-highlighted')
      .forEach(el => el.classList.remove('is-highlighted'));
  }, Math.max(0, _highlightUntil - Date.now()) + 50);
}

/**
 * Mark the form dirty on any edit anywhere inside it.
 *
 * See the identical helper in ui/sources.js for why this is delegation rather
 * than a callback threaded through TaskConfigField → SchemaField → each field:
 * there is no single hook that fires for all of them, and the task-type dropdown
 * swaps the whole schema form out mid-edit.
 *
 * Delegation only sees edits that fire a native event, so button-driven ones
 * (ArrayField's add/delete/reorder/group-insert — e.g. rearranging the models on
 * a CollectTrainingImagesTask) dispatch FIELD_EDIT_EVENT instead.
 *
 * @param {HTMLElement} root
 * @param {() => void} onEdit
 */
function watchForEdits(root, onEdit) {
  root.addEventListener('input', onEdit);
  root.addEventListener('change', onEdit);
  root.addEventListener(FIELD_EDIT_EVENT, onEdit);
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
    toggleRow.title = 'Show configs created by the app for preview runs and other internal jobs, '
      + 'and offer their task types when creating one';
    toggleRow.append(toggleChk, document.createTextNode('Show system'));
    toggleChk.addEventListener('change', async () => {
      showTemporary = toggleChk.checked;
      await refreshList();
      // The toggle governs the type dropdown too, and a form that is already open
      // was built with the old list — ticking the box and watching the dropdown not
      // change reads as the toggle being broken. Only an untouched *new* form is
      // rebuilt: doing it to a dirty one would throw away what was typed, and doing
      // it to a saved config would be rebuilding it for no reason at all.
      if (_selectedUuid === null && !screen.isDirty()) await _showForm(null, { force: true });
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

        // The same toggle governs the type dropdown, not just the list: the types
        // hidden from it are hidden for the same reason those configs are, and two
        // separate switches for one idea is one more than the idea deserves.
        const field = await TaskConfigField.create(
          config
            ? {
                value: config.params,
                name: config.name,
                typename: config.typename,
                uuid: config.uuid,
                editMode: true,
                showSystemTypes: showTemporary,
              }
            : { editMode: true, showSystemTypes: showTemporary },
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
              // Only the list needs updating. A full refresh() would re-run
              // renderTaskConfigs, which tears down this editor and drops you
              // back on a blank "New Task Config" form.
              await refreshList();
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
              // Params are read once, in the task's _init — a live instance keeps
              // the ones it started with, so saving alone changes nothing for it.
              // No-op (and no modal) when nothing is running, which is the common
              // case; only an update can have a live instance, never a create.
              await promptRestartOwnTask(config.uuid);
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

    const configs = await refreshList();
    // Arriving from a Jobs launch card's ⚙ means "show me *this* config".
    // The uuid is the map key, not a field on the record — mirror the list
    // rows' click handler and put it back on the object.
    const wantedUuid = takePendingSelection(CONFIG_TAB_ID);
    const wanted = wantedUuid && configs.has(wantedUuid)
      ? { ...configs.get(wantedUuid), uuid: wantedUuid }
      : null;
    await _showForm(wanted, { force: true });
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
      const actionCount = (t.statusActions ?? []).length + (taskResultData(t.instance).actions ?? []).length;
      const progressPart = (t.progress === null || t.progress === undefined) ? 'ind' : Math.round(t.progress);
      // The stopping flag is part of the fingerprint or the card would keep its
      // pre-click face until something else about the task happened to change —
      // which for a task wedged mid-shutdown is never.
      const stoppingPart = _stoppingTasks.has(id) ? 'stopping' : '';
      return `${id}:${t.instance?.status}:${t.isPausedInPlace ?? false}:${progressPart}:${logs.length}:${logs[logs.length - 1] ?? ''}:${t.instance?.startedAt ?? ''}:${t.instance?.endedAt ?? ''}:${actionCount}:${stoppingPart}`;
    })
    .sort()
    .join('|');
}

/** Task uuids the user has asked to stop, that haven't finished stopping yet.
 *
 * Module scope on purpose: the card is destroyed and rebuilt on every poll that
 * changes anything, so per-card state cannot carry "I clicked Stop" across even one
 * refresh. Entries are cleared in _syncStoppingState once the task's own status shows
 * it is no longer running, so nothing here can outlive the thing it describes.
 */
const _stoppingTasks = new Set();

/** Drop stopping-marks for tasks that have since stopped (or were never running). */
function _syncStoppingState(tasks) {
  for (const id of [..._stoppingTasks]) {
    const status = tasks?.[id]?.instance?.status;
    if (status === undefined || (status !== 'running' && status !== 'pending')) {
      _stoppingTasks.delete(id);
    }
  }
}

/** Turn a cancel outcome into something worth telling the user.
 *
 * The three cases are genuinely different and used to look identical from the UI:
 * the task stopped, the task accepted the stop and is still shutting down, or there
 * was no running task for the server to stop (the state a previously-orphaned task
 * left behind, where clicking Stop forever appeared to succeed and changed nothing).
 */
function _describeStopOutcome(outcome, taskId, taskName) {
  if (outcome?.stopping?.includes(taskId)) {
    _stoppingTasks.add(taskId);
    return { message: `Stopping "${taskName}" — it is still shutting down`, level: 'warning' };
  }
  if (outcome?.notRunning?.includes(taskId)) {
    _stoppingTasks.delete(taskId);
    return { message: `"${taskName}" was not running, so there was nothing to stop`, level: 'warning' };
  }
  _stoppingTasks.delete(taskId);
  return { message: `Stopped "${taskName}"`, level: 'success' };
}

/** Run one card action, acknowledging the click whatever the outcome.
 *
 * `describe` returns either a string (an ordinary success message) or
 * `{ message, level }` when the action succeeded but the news isn't simply good —
 * a stop that is still in progress, say. Returning undefined says nothing, which is
 * what a cancelled confirm dialog should do.
 */
async function _runTaskAction({ verb, taskName, run, describe, confirm, refresh }) {
  if (confirm && !confirm()) return;
  let outcome;
  try {
    outcome = await run();
  } catch (error) {
    toast(error.message || `Failed to ${verb} "${taskName}"`, 6000, 'error');
    // Still refresh: the action may have partly landed, and a stale card is exactly
    // what makes a failure hard to reason about.
    await refresh?.();
    return;
  }
  const described = describe?.(outcome);
  if (described) {
    const { message, level } = (typeof described === 'string')
      ? { message: described, level: 'success' }
      : described;
    // A plain confirmation earns less screen time than one carrying a caveat.
    toast(message, level === 'success' ? 3000 : 6000, level);
  }
  await refresh?.();
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

// TERMINAL_STATUSES lives with ActiveTaskField (which also gates its status-data
// polling on it) so the card and this list can't disagree about what "finished" means.

/**
 * Count everything still live, for the Jobs tab's badge.
 *
 * Jobs and continuous runs both count. Excluding continuous work kept the badge
 * transient (a watcher stays armed for weeks and would pin it at ≥1), but it
 * also made the number impossible to explain from the screen it describes:
 * Running lists a running watcher under Continuous and the badge disagreed with
 * no visible rule. The badge answers "how many things are running?", which is
 * the question the tab's own list answers too. `countContinuousTasks` is still
 * here for anyone who needs the split.
 *
 * @param {Record<string, any>} tasks
 * @returns {number}
 */
export function countLiveTasks(tasks) {
  return Object.values(tasks || {})
    .filter(t => !TERMINAL_STATUSES.has(t.instance?.status))
    .length;
}

/**
 * The badge count, split by kind — for the tooltip that explains it.
 *
 * @param {Record<string, any>} tasks
 * @returns {{jobs: number, continuous: number}}
 */
export function countLiveTasksByKind(tasks) {
  const live = Object.values(tasks || {}).filter(t => !TERMINAL_STATUSES.has(t.instance?.status));
  const continuous = live.filter(t => t.continuous).length;
  return { jobs: live.length - continuous, continuous };
}

/**
 * Count jobs that can still change state on their own.
 *
 * Narrower than the badge count, and for a different question. The badge asks
 * "how many things are running?", which a paused job and an armed watcher both
 * belong in. The off-tab stream asks "can anything finish behind the user's
 * back?", which a paused job answers no to (only the user can restart it, and
 * they have to come back to Jobs to do it) and a continuous job answers no to
 * by definition — it does not finish, so there is no completion to miss.
 *
 * Counting paused here is what pinned an EventSource open for the life of every
 * browser tab. A single paused instance is not exotic — restart recovery writes
 * exactly that status for any interrupted task that had a checkpoint.
 *
 * @param {Record<string, any>} tasks
 * @returns {number}
 */
export function countAdvancingTasks(tasks) {
  return Object.values(tasks || {})
    .filter(t => !TERMINAL_STATUSES.has(t.instance?.status) && t.instance?.status !== 'paused' && !t.continuous)
    .length;
}

/**
 * Count armed continuous tasks — watchers and other run-until-stopped work.
 *
 * Separate from the badge so the two can be shown differently: a job count is
 * transient and worth a loud badge, an armed count is steady state.
 *
 * @param {Record<string, any>} tasks
 * @returns {number}
 */
export function countContinuousTasks(tasks) {
  return Object.values(tasks || {})
    .filter(t => !TERMINAL_STATUSES.has(t.instance?.status) && t.continuous)
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

  // Skip when the Jobs tab, or its Running pane, is not visible. The SSE stream
  // stays open across both Jobs sub-tabs to keep the badge counting, so this
  // has to check the pane too rather than just the main tab.
  if (!document.getElementById(RUNS_TAB_ID)?.classList.contains('active') ||
      !document.getElementById(ACTIVITY_TAB_ID)?.classList.contains('active')) {
    return;
  }

  // Guard against concurrent renders (e.g. rapid button clicks during a slow fetch).
  if (container._rendering) return;
  container._rendering = true;

  // Only once this render is actually going ahead: taken earlier, a request
  // would be consumed by a call that returns without ever painting.
  _takeHighlightRequest();

  // Use pre-loaded data from SSE if provided, otherwise fetch from the API.
  const dataPromise = preloadedTasks != null
    ? Promise.resolve({ tasks: preloadedTasks })
    : fetchTasksStatus();

  dataPromise.then(async response => {
    try {
      const { tasks } = response;

      // Before the fingerprint, so a task that has finished stopping drops its
      // marker in the same pass that repaints it without one.
      _syncStoppingState(tasks);

      // Skip re-render entirely when nothing meaningful changed.
      const fingerprint = _taskFingerprint(tasks);
      if (container._lastFingerprint === fingerprint) {
        // Nothing to rebuild, but the user may have just arrived here asking
        // for a particular run — the cards it needs are already on screen.
        _applyHighlight(container);
        return;
      }

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

      // Per-kind card tints, declared by the task types themselves. Awaited
      // before the build (rather than per card) so every card in one pass
      // agrees, and never fatal: a pane full of uncoloured cards is the old
      // behaviour, an empty pane is not.
      let accentsByTypename = new Map();
      try {
        accentsByTypename = await fetchTaskTypeAccents();
      } catch (err) {
        Logger.warn('Could not load task card accents; rendering plain cards', err);
      }

      // Build every row in parallel.  The live DOM is untouched until
      // everything is ready, then partitioned into Continuous / Completed sections.
      const built = await Promise.all(Object.entries(tasks || {}).map(async ([taskId, task]) => {
        const status = task.instance?.status;

        const taskName = task.config?.name ?? taskId;
        // Every action goes through _runTaskAction so that success is as visible as
        // failure. Previously only the error path said anything, which meant the
        // feedback for "it worked" and for "the server did nothing at all" was
        // identical: no feedback.
        const deleteButton = {
          text: 'Delete', emoji: '🗑️', testid: 'task-delete',
          onClick: () => _runTaskAction({
            verb: 'delete', taskName, refresh,
            confirm: () => window.confirm(`Delete task ${taskName}?`),
            run: () => deleteTasks({ taskUuids: [taskId] }),
            describe: () => `Deleted "${taskName}"`,
          }),
        };
        const stopButton = {
          text: 'Stop', emoji: '⏹️', testid: 'task-stop',
          onClick: () => _runTaskAction({
            verb: 'stop', taskName, refresh,
            run: () => cancelTasks({ taskUuids: [taskId] }),
            describe: (outcome) => _describeStopOutcome(outcome, taskId, taskName),
          }),
        };
        const pauseInPlaceButton = {
          text: 'Pause', emoji: '⏸️', testid: 'task-pause',
          onClick: () => _runTaskAction({
            verb: 'pause', taskName, refresh,
            run: () => pauseTasksInPlace({ taskUuids: [taskId] }),
            describe: () => `Paused "${taskName}"`,
          }),
        };
        const resumeInPlaceButton = {
          text: 'Resume', emoji: '▶️', testid: 'task-resume',
          onClick: () => _runTaskAction({
            verb: 'resume', taskName, refresh,
            run: () => resumeTasksInPlace({ taskUuids: [taskId] }),
            describe: () => `Resumed "${taskName}"`,
          }),
        };
        const suspendButton = {
          text: 'Suspend', emoji: '💾', testid: 'task-suspend',
          onClick: () => _runTaskAction({
            verb: 'suspend', taskName, refresh,
            run: () => pauseTasks({ taskUuids: [taskId] }),
            describe: () => `Suspended "${taskName}" — its checkpoint was saved`,
          }),
        };
        const resumeButton = {
          text: 'Resume', emoji: '▶️', testid: 'task-resume',
          onClick: () => _runTaskAction({
            verb: 'resume', taskName, refresh,
            run: () => resumeTasks({ taskUuids: [taskId] }),
            describe: () => `Resumed "${taskName}" from its checkpoint`,
          }),
        };
        // "What is this thing actually running?" — the card names the run, and
        // until now the recipe behind it was only reachable by remembering which
        // Configuration screen defines that task type and finding it by name.
        // Points at the *definition* (training config, watch profile, filter
        // chain) where there is one; see configTargetForTask.
        const configTarget = configTargetForTask(task.config);
        const configButton = configTarget && {
          text: 'Configuration', emoji: '⚙️', testid: 'task-config-jump',
          onClick: () => navigateTabs('tab-configuration', configTarget.tab, { selectUuid: configTarget.uuid }),
        };

        // A watcher is the one task whose output is a live picture, and the
        // drill-down already exists for the NVR pane and the launcher card.
        // Only while running: a queued watcher has no frames yet and a paused
        // one has stopped producing them, so the modal would open empty.
        const liveViewButton = task.config?.typename === 'StreamWatcherTask' && status === 'running' && {
          text: 'Live view', emoji: '📺', testid: 'task-live-view',
          onClick: () => {
            try {
              openWatcherLiveView({
                taskUuid: task.instance?.uuid ?? taskId,
                profileName: task.config?.name ?? '',
              });
            } catch (error) {
              toast(error.message || 'Could not open the live view', 5000, 'error');
            }
          },
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

        // Deliberately NOT folded into leftButtons/rightButtons: those sit in
        // the same flex row as the card's content, and that row's horizontal
        // budget is already spent — an expanded card's sweep tables and
        // histograms have an intrinsic min width that at 760px fits with
        // nothing to spare (test_status_widgets pins it). Two more buttons in
        // that row pushed the widgets 24px past the card's own border. On their
        // own strip they cost the content nothing.
        const navButtons = [liveViewButton, configButton].filter(Boolean);

        // Pre-render field content so we get a direct DOM handle for state
        // restoration, and so createGenericRow gets an already-resolved Promise
        // (avoids timing uncertainty about when fieldDiv content is populated).
        // A stop the server accepted but hasn't finished acting on is stated on the
        // card, not just in a toast that has since faded. Without it the only honest
        // answer the UI could give a user who clicked Stop two minutes ago was the
        // same "Running" it showed before the click.
        const isStopping = _stoppingTasks.has(taskId);
        const cardValue = { ...task };
        if (isStopping) {
          const existing = cardValue.statusMessage ? ` — ${cardValue.statusMessage}` : '';
          cardValue.statusMessage = `Stopping…${existing}`;
        }

        const fieldInstance = await ActiveTaskField.create({
          value: cardValue,
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

        const contentRow = createGenericRow({ field: preRendered, leftButtons, rightButtons });

        // The card is now the wrapper rather than the row itself, so the nav
        // strip can sit under the content instead of beside it. Everything that
        // addresses a card (the open/scroll snapshot, the Clear-completed
        // sweep, the tests) goes through [data-task-uuid] and descendant
        // queries, so moving the hook outwards leaves all of it working.
        const row = document.createElement('div');
        row.className = 'task-instance-card';
        row.dataset.taskUuid = taskId;
        // Only when the type claims one — the attribute's presence is what
        // app.css keys the tint off, so an absent accent must leave it unset
        // rather than write an empty string.
        const accent = accentsByTypename.get(task.config?.typename);
        if (accent) row.dataset.taskAccent = accent;
        // A hook for styling and for tests: "the user asked this to stop" is card
        // state, distinct from any status the server reports.
        if (isStopping) row.dataset.stopping = 'true';
        row.appendChild(contentRow);

        if (navButtons.length) {
          const nav = document.createElement('div');
          nav.className = 'task-instance-card__nav';
          for (const { text, emoji, onClick, testid } of navButtons) {
            const btn = document.createElement('button');
            btn.type = 'button';
            btn.className = 'task-instance-card__nav-btn';
            btn.dataset.testid = testid;
            btn.textContent = `${emoji} ${text}`;
            btn.title = text;
            btn.addEventListener('click', onClick);
            nav.appendChild(btn);
          }
          row.appendChild(nav);
        }

        return {
          row,
          isCompleted: TERMINAL_STATUSES.has(status),
          isContinuous: Boolean(task.continuous),
          endedAt: task.instance?.endedAt ?? null,
        };
      }));

      // Three sections, because two of them are different kinds of thing.
      //
      // Continuous tasks run until stopped (Task.CONTINUOUS) — a watcher has no
      // progress toward a 100% that never arrives, and listing it among jobs
      // makes "what is running?" unanswerable, since the answer is always
      // "at least the watcher". Jobs run to completion; Completed is their
      // history, sorted most-recently-finished first.
      //
      // A continuous task that has stopped is history like any other job, so it
      // falls through to Completed rather than lingering as a dead "service".
      const newListDiv = document.createElement('div');
      const continuousRows = built.filter(b => !b.isCompleted && b.isContinuous).map(b => b.row);
      const activeRows = built.filter(b => !b.isCompleted && !b.isContinuous).map(b => b.row);
      const completedRows = built.filter(b => b.isCompleted)
        .sort((a, b) => (parseApiDateMs(b.endedAt) ?? 0) - (parseApiDateMs(a.endedAt) ?? 0))
        .map(b => b.row);

      if (!continuousRows.length && !activeRows.length && !completedRows.length) {
        const empty = document.createElement('p');
        empty.className = 'le-list-empty';
        empty.textContent = 'No tasks have run yet. Start one from the "Start a job" tab.';
        newListDiv.appendChild(empty);
      }

      // Each section is a real element, not a heading followed by a loose grid.
      // A thin underlined caption between two grids of identically-bordered cards
      // reads as one continuous field of cards, so "what is running right now?"
      // could not be answered at a glance — which is the whole job of this pane.
      // The section owns the boundary (its own surface and spacing) and the
      // `variant` lets finished work recede behind live work in CSS.
      const appendSection = (title, rows, { variant, actionEl = null } = {}) => {
        if (!rows.length) return;
        const section = document.createElement('section');
        section.className = `task-section task-section--${variant}`;
        section.dataset.taskSection = variant;

        const headerRow = document.createElement('div');
        headerRow.className = 'task-section__header';

        const h4 = document.createElement('h4');
        h4.className = 'task-section-header';
        h4.textContent = title;
        headerRow.appendChild(h4);

        // The count belongs in the header, not implied by the grid: "Running 0"
        // never renders (the section is skipped), so a visible number is always
        // the number of things actually there.
        const count = document.createElement('span');
        count.className = 'task-section__count';
        count.textContent = String(rows.length);
        headerRow.appendChild(count);

        if (actionEl) {
          const spacer = document.createElement('span');
          spacer.className = 'task-section__spacer';
          headerRow.append(spacer, actionEl);
        }
        section.appendChild(headerRow);

        // Cards flow into as many columns as the (now full-width) pane fits,
        // instead of one per row down half a split.
        const grid = document.createElement('div');
        grid.className = 'task-instance-grid';
        rows.forEach(row => grid.appendChild(row));
        section.appendChild(grid);
        newListDiv.appendChild(section);
      };
      appendSection('Continuous', continuousRows, { variant: 'continuous' });
      appendSection('Jobs', activeRows, { variant: 'running' });

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
        appendSection('Completed', completedRows, { variant: 'completed', actionEl: clearBtn });
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

      // Re-applied after every swap, not set once: the rebuild above replaces
      // the very node the mark was on.
      _applyHighlight(container);

    } finally {
      container._rendering = false;
    }
  }).catch(err => {
    container._rendering = false;
    toast(err.message || 'Server is not responding', 5000, 'error');
  });
}
