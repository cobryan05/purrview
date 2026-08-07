import { setupWorkspaceTabs } from '/app-static/js/app/utils/tabs.js';
import { events, EventTypes } from '/app-static/js/shared/events/index.js';
import { renderSourceManager } from '/app-static/js/ui/sources.js';
import { ACTIVITY_TAB_ID, countRunningTasks, renderActiveTasks, renderTaskConfigs } from '/app-static/js/ui/tasks.js';
import { fetchTasksStatus } from '/app-static/js/shared/api/tasks.js';
import { renderServiceStatus } from '/app-static/js/ui/serviceStatus.js';

export async function init() {
  // Service health sits above the sub-tabs — it applies to the whole surface,
  // not to Sources, Tasks or Activity individually.
  renderServiceStatus({ target: 'service-status-box' });

  let taskEventSource = null;
  let activeSubTab = null;

  const badge = document.getElementById('task-activity-badge');

  /**
   * Show how many tasks are still live on the Activity tab button. Hidden at
   * zero rather than showing "0": a permanent badge stops being a signal.
   */
  function updateBadge(tasks) {
    if (!badge) return;
    const n = countRunningTasks(tasks);
    badge.textContent = String(n);
    badge.hidden = n === 0;
  }

  /**
   * Seed the badge from a one-off fetch when the Operations tab opens.
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
    // Only rebuild the config editor when its own pane is showing — it is a
    // full list-editor screen now, and rebuilding it under the user would drop
    // an in-progress edit.
    if (activeSubTab === 'tab-task-management') {
      renderTaskConfigs({ target: 'task-config-list', refresh: refreshTaskTab });
    }
  };

  function closeTaskStream() {
    if (taskEventSource !== null) {
      taskEventSource.close();
      taskEventSource = null;
    }
  }

  function openTaskStream() {
    if (taskEventSource !== null) return;
    // SSE stream for live task updates — server pushes on change.
    taskEventSource = new EventSource('/api/tasks/stream');
    taskEventSource.onmessage = (e) => {
      try {
        const data = JSON.parse(e.data);
        if (data?.tasks) {
          updateBadge(data.tasks);
          // renderActiveTasks no-ops unless its pane is visible, so this keeps
          // the badge live while the user is on Sources or Tasks.
          renderActiveTasks({ target: 'active-tasks-list', tasks: data.tasks, refresh: refreshTaskTab });
        }
      } catch {}
    };
    // EventSource reconnects automatically on error; no extra handling needed.
  }

  // The stream must not outlive Operations-tab focus: close it whenever another
  // main tab takes over, and reopen it when the user returns. It now stays open
  // across all three sub-tabs, because the Activity badge has to keep counting
  // while the user is editing a config or a source.
  events.subscribe(EventTypes.MAIN_TAB_CHANGED, ({ tabName }) => {
    if (tabName !== 'tab-operations') {
      closeTaskStream();
    } else {
      openTaskStream();
      seedBadge();
      if (activeSubTab === ACTIVITY_TAB_ID) renderActiveTasks({ target: 'active-tasks-list', refresh: refreshTaskTab });
    }
  });

  const { container } = setupWorkspaceTabs("#tab-operations", async (tabName, button, pane) => {
    activeSubTab = tabName;

    if (tabName === 'tab-source-management') {
      await renderSourceManager({ target: "source-management-box" });
    } else if (tabName === 'tab-task-management') {
      await renderTaskConfigs({ target: 'task-config-list', refresh: refreshTaskTab });
    } else if (tabName === ACTIVITY_TAB_ID) {
      // Initial render via fetch so the pane feels instant on first open.
      renderActiveTasks({ target: 'active-tasks-list', refresh: refreshTaskTab });
    }
    openTaskStream();
    seedBadge();
  });

  // activate default operations tab
  const defaultTabBtn = container?.querySelector('.workspace-tab-button');
  if (defaultTabBtn) defaultTabBtn.click();
}
