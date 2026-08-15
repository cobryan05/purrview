/**
 * Runs — the app's single runtime surface.
 *
 * Was Operations › Activity. Operations dissolved: its Tasks and Sources panes
 * were definition editors and moved to Configuration, leaving the one thing
 * that was genuinely about *running* behind. It is a main tab now because
 * everything launched anywhere in the app reports here, and because service
 * health belongs beside what is running rather than above two config editors.
 *
 * Two panes: Active (what is running, plus service health) and Start a run
 * (everything launchable). Starting lives here rather than in the Configuration
 * editors, so a config screen defines a recipe and never runs or monitors one.
 */
import { setupWorkspaceTabs } from '/app-static/js/app/utils/tabs.js';
import { events, EventTypes } from '/app-static/js/shared/events/index.js';
import { countAdvancingTasks, countRunningTasks, renderActiveTasks } from '/app-static/js/ui/tasks.js';
import { fetchTasksStatus } from '/app-static/js/shared/api/tasks.js';
import { subscribeTaskStream } from '/app-static/js/shared/api/taskStream.js';
import { renderServiceStatus } from '/app-static/js/ui/serviceStatus.js';
import { applyRunsLauncherTaskState, renderRunsLauncher } from '/app-static/js/ui/runsLauncher.js';

export async function init() {
  // Service health sits above the task list — it applies to the whole surface.
  renderServiceStatus({ target: 'service-status-box' });

  let unsubscribeTaskStream = null;

  // Two badges, one count: the main Runs tab (visible from anywhere in the app)
  // and the Active sub-tab, which is the pane the count is actually about — from
  // inside Runs the main-tab badge is out of the eyeline, and a count next to
  // "Start a run"'s sibling is what says where the running things are.
  const badges = [
    document.getElementById('task-activity-badge'),
    document.getElementById('task-activity-badge-active'),
  ].filter(Boolean);

  /** Jobs that can still finish on their own; drives the stream's lifetime. */
  let advancingCount = 0;
  /** Whether Runs is the main tab on screen. Runs itself always needs the feed. */
  let onRunsTab = true;

  /**
   * Show how many tasks are still live on the Runs tab buttons. Hidden at zero
   * rather than showing "0": a permanent badge stops being a signal.
   */
  function updateBadge(tasks) {
    const n = countRunningTasks(tasks);
    for (const badge of badges) {
      badge.textContent = String(n);
      badge.hidden = n === 0;
    }
    // Deliberately not the badge count: a paused job still counts as unfinished
    // (so the badge shows it) but cannot finish without the user, so it must not
    // hold the stream. See countAdvancingTasks.
    advancingCount = countAdvancingTasks(tasks);
    // The last job finishing is what makes an off-tab stream pointless — drop
    // it at that moment rather than holding it until the user comes back.
    if (advancingCount === 0 && !onRunsTab) closeTaskStream();
  }

  /**
   * Seed the badge from a one-off fetch when the Runs tab opens.
   *
   * SSE keeps it fresh afterwards, but the stream only pushes *on change* — so
   * without this the badge stays blank on arrival until something happens to a
   * task, which is precisely the case where it should already read "2 running".
   */
  async function seedBadge() {
    try {
      const { tasks } = await fetchTasksStatus();
      updateBadge(tasks);
    } catch {
      // A failed poll is not worth a toast here; the SSE stream will correct it.
    }
  }

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
        // renderActiveTasks no-ops unless its pane is visible, so this costs
        // nothing while the Active pane is not the one on screen.
        renderActiveTasks({ target: 'active-tasks-list', tasks: data.tasks, refresh: refreshTaskTab });
        // The launcher shows "Training in progress" on the card that started a
        // run. It is fed from here rather than subscribing itself: this stream
        // is deliberately the tab's only one, and its lifetime is managed below.
        applyRunsLauncherTaskState(data.tasks);
      },
    });
    // EventSource reconnects automatically on error; no extra handling needed.
  }

  // The stream outlives Runs-tab focus only while something is actually running.
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
   * Land the user on Active after starting something, so the run they just
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

  // activate default Runs tab
  const defaultTabBtn = tabs.container?.querySelector('.workspace-tab-button');
  if (defaultTabBtn) defaultTabBtn.click();

  openTaskStream();
  seedBadge();
}
