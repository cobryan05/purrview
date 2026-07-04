import { setupWorkspaceTabs } from '/app-static/js/app/utils/tabs.js';
import { events, EventTypes } from '/app-static/js/shared/events/index.js';
import { renderLabelList } from '/app-static/js/ui/labels.js';
import { renderModelLabelAssignments } from '/app-static/js/ui/models.js';
import { renderModelConfigManager } from '/app-static/js/ui/modelConfigs.js';
import { renderSourceManager } from '/app-static/js/ui/sources.js';
import { renderTagList } from '/app-static/js/ui/tags.js';
import { renderActiveTasks, renderTaskConfigs } from '/app-static/js/ui/tasks.js';
import { renderDatasetManager } from '/app-static/js/ui/datasets.js';
import { renderTrainingConfigManager } from '/app-static/js/ui/trainingConfigs.js';
import { renderFilterChainManager } from '/app-static/js/ui/filterChains.js';

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

  // The stream must not outlive Configuration-tab focus: close it whenever
  // another main tab takes over, and reopen it when the user returns while
  // the Task Management sub-tab is still the active one.
  events.subscribe(EventTypes.MAIN_TAB_CHANGED, ({ tabName }) => {
    if (tabName !== 'tab-configuration') {
      closeTaskStream();
    } else if (activeSubTab === 'tab-task-management') {
      refreshTaskTab();
      openTaskStream();
    }
  });

  const { container, activateTab } = setupWorkspaceTabs("#tab-configuration", async (tabName, button, pane) => {
    activeSubTab = tabName;
    // Close task SSE stream when leaving the task management tab.
    closeTaskStream();

    if (tabName === 'tab-tag-management') {
      renderLabelList({ target: "labels-list" });
      renderTagList({ target: "tags-list" });
      renderModelLabelAssignments({ target: "model-labels-box" });
    } else if (tabName === 'tab-source-management') {
      renderSourceManager({ target: "source-management-box" });
    } else if (tabName === 'tab-task-management') {
      // Initial render via fetch so the tab feels instant on first open.
      refreshTaskTab();
      openTaskStream();
    } else if (tabName === 'tab-dataset-management') {
      await renderDatasetManager({ target: 'dataset-management-box' });
    } else if (tabName === 'tab-model-management') {
      await renderModelConfigManager({ target: 'model-config-box' });
    } else if (tabName === 'tab-training-management') {
      await renderTrainingConfigManager({ target: 'training-config-box' });
    } else if (tabName === 'tab-filterchain-management') {
      await renderFilterChainManager({ target: 'filter-chain-management-box' });
    }
  });

  // activate default configuration tab
  const defaultTabBtn = container?.querySelector('.workspace-tab-button');
  if (defaultTabBtn) defaultTabBtn.click();
}
