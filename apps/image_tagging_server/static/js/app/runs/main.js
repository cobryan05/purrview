/**
 * Jobs — the app's single runtime surface.
 *
 * Was Operations › Activity. Operations dissolved: its Tasks and Sources panes
 * were definition editors and moved to Configuration, leaving the one thing
 * that was genuinely about *running* behind. It is a main tab now because
 * everything launched anywhere in the app reports here, and because service
 * health belongs beside what is running rather than above two config editors.
 *
 * Two panes: Running (what is live, plus service health) and Start a job
 * (everything launchable). Starting lives here rather than in the Configuration
 * editors, so a config screen defines a recipe and never runs or monitors one.
 */
import { setupWorkspaceTabs } from '/app-static/js/app/utils/tabs.js';
import { events, EventTypes } from '/app-static/js/shared/events/index.js';
import { countAdvancingTasks, renderActiveTasks } from '/app-static/js/ui/tasks.js';
import { seedRunsBadge, updateRunsBadge } from '/app-static/js/ui/runsBadge.js';
import { subscribeTaskStream } from '/app-static/js/shared/api/taskStream.js';
import { notifyModelsChanged } from '/app-static/js/shared/api/models.js';
import { renderServiceStatus } from '/app-static/js/ui/serviceStatus.js';
import { applyRunsLauncherTaskState, renderRunsLauncher } from '/app-static/js/ui/runsLauncher.js';

/**
 * Task types whose successful completion means a new file in the models
 * directory. Training is the one that matters day to day; the imports are here
 * because they finish the same way and the alternative is remembering to add
 * them later.
 */
const MODEL_PRODUCING_TYPENAMES = new Set([
  'TrainModelTask',
  'ConvertModelTask',
  'ImportModelTask',
  'ImportCuratedModelTask',
]);

export async function init() {
  // Service health sits above the task list — it applies to the whole surface.
  renderServiceStatus({ target: 'service-status-box' });

  let unsubscribeTaskStream = null;

  /**
   * Instance uuids already announced, so a task sitting in `completed` for the
   * rest of the session announces itself once rather than on every frame.
   */
  const _announcedModelRuns = new Set();

  /**
   * Tell the model pickers when a run has just produced a model.
   *
   * This lives on the Jobs stream rather than in the model API because the model
   * API never hears about it: the file is written by the tasks server, and the
   * only thing the browser sees is a status going to `completed`. Jobs holds this
   * stream for as long as anything is still advancing, including while the user
   * is on another tab — which is exactly when a training run finishes.
   */
  function announceFinishedModelRuns(tasks) {
    for (const [uuid, task] of Object.entries(tasks || {})) {
      if (task?.instance?.status !== 'completed') continue;
      if (!MODEL_PRODUCING_TYPENAMES.has(task?.config?.typename)) continue;
      if (_announcedModelRuns.has(uuid)) continue;
      _announcedModelRuns.add(uuid);
      notifyModelsChanged();
    }
  }

  // The badge itself lives in ui/runsBadge.js: it is painted at page load,
  // before this init has ever run, so it cannot be owned by this module.

  /** Jobs that can still finish on their own; drives the stream's lifetime. */
  let advancingCount = 0;
  /** Whether Jobs is the main tab on screen. Jobs itself always needs the feed. */
  let onRunsTab = true;

  /**
   * Repaint the badge, and re-derive whether the SSE stream is still worth
   * holding once the user leaves Jobs.
   */
  function updateBadge(tasks) {
    updateRunsBadge(tasks);
    // Deliberately not the badge count: a paused job still counts as running
    // (so the badge shows it) but cannot finish without the user, so it must
    // not hold the stream — nor may an armed watcher, which never finishes at
    // all. See countAdvancingTasks.
    advancingCount = countAdvancingTasks(tasks);
    // The last job finishing is what makes an off-tab stream pointless — drop
    // it at that moment rather than holding it until the user comes back.
    if (advancingCount === 0 && !onRunsTab) closeTaskStream();
  }

  /**
   * Re-poll the badge; SSE keeps it fresh but only pushes *on change*.
   *
   * Routed through this tab's own `updateBadge` rather than the module default,
   * so the poll also refreshes `advancingCount` — otherwise leaving Jobs while
   * a job is running drops the stream on a count that was never filled in.
   */
  const seedBadge = () => seedRunsBadge(updateBadge);

  const refreshTaskTab = () => {
    renderActiveTasks({ target: 'active-tasks-list', refresh: refreshTaskTab });
  };

  function closeTaskStream() {
    if (unsubscribeTaskStream !== null) {
      unsubscribeTaskStream();
      unsubscribeTaskStream = null;
    }
  }

  function openTaskStream() {
    if (unsubscribeTaskStream !== null) return;
    // Shared SSE stream for live task updates — server pushes on change. The
    // handle is the unsubscribe fn; the underlying connection is shared with any
    // other subscriber (e.g. an open preview modal) rather than opened twice.
    unsubscribeTaskStream = subscribeTaskStream({
      onMessage: (data) => {
        if (!data?.tasks) return;
        updateBadge(data.tasks);
        announceFinishedModelRuns(data.tasks);
        // renderActiveTasks no-ops unless its pane is visible, so this costs
        // nothing while the Running pane is not the one on screen.
        renderActiveTasks({ target: 'active-tasks-list', tasks: data.tasks, refresh: refreshTaskTab });
        // The launcher shows "Training in progress" on the card that started a
        // run. It is fed from here rather than subscribing itself: this stream
        // is deliberately the tab's only one, and its lifetime is managed below.
        applyRunsLauncherTaskState(data.tasks);
      },
    });
    // EventSource reconnects automatically on error; no extra handling needed.
  }

  // The stream outlives Jobs-tab focus only while something is actually running.
  //
  // Two failures to avoid at once. Closing it unconditionally froze both badges
  // at whatever they read on the way out — a training that finished while you
  // were annotating still showed as running, and the badge exists precisely to
  // report that from elsewhere. Holding it open unconditionally costs an SSE
  // connection per browser tab for the life of the app, feeding a badge that
  // reads zero and cannot change until the user acts (the leak that
  // test_leaks_and_races.py guards).
  //
  // "Can anything finish behind your back?" is the exact condition that
  // separates them, and it is a narrower question than the badge's — a paused
  // job is unfinished but cannot advance without the user, so it must not hold
  // the connection. `updateBadge` drops it the moment the last live job ends.
  events.subscribe(EventTypes.MAIN_TAB_CHANGED, ({ tabName }) => {
    onRunsTab = tabName === 'tab-runs';
    if (!onRunsTab) {
      if (advancingCount === 0) closeTaskStream();
    } else {
      openTaskStream();
      seedBadge();
      renderActiveTasks({ target: 'active-tasks-list', refresh: refreshTaskTab });
    }
  });

  /**
   * Land the user on Running after starting something, so the run they just
   * kicked off is visible rather than leaving them on the launcher wondering
   * whether the click took.
   */
  function showActiveAfterStart() {
    tabs?.activateTab('tab-runs-active');
    seedBadge();
    renderActiveTasks({ target: 'active-tasks-list', refresh: refreshTaskTab });
  }

  const tabs = setupWorkspaceTabs('#tab-runs', async (tabName) => {
    if (tabName === 'tab-runs-active') {
      renderActiveTasks({ target: 'active-tasks-list', refresh: refreshTaskTab });
    } else if (tabName === 'tab-runs-launch') {
      await renderRunsLauncher({ target: 'runs-launcher-box', onStarted: showActiveAfterStart });
    }
  });

  // activate default Jobs tab
  const defaultTabBtn = tabs.container?.querySelector('.workspace-tab-button');
  if (defaultTabBtn) defaultTabBtn.click();

  openTaskStream();
  seedBadge();
}
