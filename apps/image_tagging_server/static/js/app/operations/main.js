import { setupWorkspaceTabs } from '/app-static/js/app/utils/tabs.js';
import { events, EventTypes } from '/app-static/js/shared/events/index.js';
import { renderSourceManager } from '/app-static/js/ui/sources.js';
import { renderActiveTasks, renderTaskConfigs } from '/app-static/js/ui/tasks.js';

export async function init() {
  let taskEventSource = null;
  let activeSubTab = null;

  const refreshTaskTab = () => {
    renderActiveTasks({ target: 'active-tasks-list', refresh: refreshTaskTab });
    renderTaskConfigs({ target: 'task-config-list', refresh: refreshTaskTab });
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
          renderActiveTasks({ target: 'active-tasks-list', tasks: data.tasks, refresh: refreshTaskTab });
        }
      } catch {}
    };
    // EventSource reconnects automatically on error; no extra handling needed.
  }

  // The stream must not outlive Operations-tab focus: close it whenever
  // another main tab takes over, and reopen it when the user returns while
  // the Tasks sub-tab is still the active one.
  events.subscribe(EventTypes.MAIN_TAB_CHANGED, ({ tabName }) => {
    if (tabName !== 'tab-operations') {
      closeTaskStream();
    } else if (activeSubTab === 'tab-task-management') {
      refreshTaskTab();
      openTaskStream();
    }
  });

  const { container } = setupWorkspaceTabs("#tab-operations", async (tabName, button, pane) => {
    activeSubTab = tabName;
    // Close task SSE stream when leaving the tasks tab.
    closeTaskStream();

    if (tabName === 'tab-source-management') {
      renderSourceManager({ target: "source-management-box" });
    } else if (tabName === 'tab-task-management') {
      // Initial render via fetch so the tab feels instant on first open.
      refreshTaskTab();
      openTaskStream();
    }
  });

  // activate default operations tab
  const defaultTabBtn = container?.querySelector('.workspace-tab-button');
  if (defaultTabBtn) defaultTabBtn.click();
}
