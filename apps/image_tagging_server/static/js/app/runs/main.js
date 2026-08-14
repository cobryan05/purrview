/**
 * Runs — the app's single runtime surface.
 *
 * Was Operations › Activity. Operations dissolved: its Tasks and Sources panes
 * were definition editors and moved to Configuration, leaving the one thing
 * that was genuinely about *running* behind. It is a main tab now because
 * everything launched anywhere in the app reports here, and because service
 * health belongs beside what is running rather than above two config editors.
 *
 * No sub-tabs: with Tasks and Sources gone there is exactly one pane, so the
 * task list gets the full width the old Activity sub-tab already had.
 */
import { events, EventTypes } from '/app-static/js/shared/events/index.js';
import { countRunningTasks, renderActiveTasks } from '/app-static/js/ui/tasks.js';
import { fetchTasksStatus } from '/app-static/js/shared/api/tasks.js';
import { subscribeTaskStream } from '/app-static/js/shared/api/taskStream.js';
import { renderServiceStatus } from '/app-static/js/ui/serviceStatus.js';

export async function init() {
  // Service health sits above the task list — it applies to the whole surface.
  renderServiceStatus({ target: 'service-status-box' });

  let unsubscribeTaskStream = null;

  const badge = document.getElementById('task-activity-badge');

  /**
   * Show how many tasks are still live on the Runs tab button. Hidden at zero
   * rather than showing "0": a permanent badge stops being a signal.
   */
  function updateBadge(tasks) {
    if (!badge) return;
    const n = countRunningTasks(tasks);
    badge.textContent = String(n);
    badge.hidden = n === 0;
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
        // renderActiveTasks no-ops unless its pane is visible, so this keeps
        // the badge live while the user is on another tab.
        renderActiveTasks({ target: 'active-tasks-list', tasks: data.tasks, refresh: refreshTaskTab });
      },
    });
    // EventSource reconnects automatically on error; no extra handling needed.
  }

  // The stream must not outlive Runs-tab focus: close it whenever another main
  // tab takes over, and reopen it when the user returns.
  events.subscribe(EventTypes.MAIN_TAB_CHANGED, ({ tabName }) => {
    if (tabName !== 'tab-runs') {
      closeTaskStream();
    } else {
      openTaskStream();
      seedBadge();
      renderActiveTasks({ target: 'active-tasks-list', refresh: refreshTaskTab });
    }
  });

  // Initial render via fetch so the pane feels instant on first open.
  renderActiveTasks({ target: 'active-tasks-list', refresh: refreshTaskTab });
  openTaskStream();
  seedBadge();
}
