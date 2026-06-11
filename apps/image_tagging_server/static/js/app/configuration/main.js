import { setupWorkspaceTabs } from '/app-static/js/app/utils/tabs.js';
import { renderLabelList } from '/app-static/js/ui/labels.js';
import { renderModelLabelAssignments } from '/app-static/js/ui/models.js';
import { renderModelConfigManager } from '/app-static/js/ui/modelConfigs.js';
import { renderSourceManager } from '/app-static/js/ui/sources.js';
import { renderTagList } from '/app-static/js/ui/tags.js';
import { renderActiveTasks, renderTaskConfigs } from '/app-static/js/ui/tasks.js';
import { renderDatasetManager } from '/app-static/js/ui/datasets.js';

export async function init() {
  let taskEventSource = null;

  const { container, activateTab } = setupWorkspaceTabs("#tab-configuration", async (tabName, button, pane) => {
    // Close task SSE stream when leaving the task management tab.
    if (taskEventSource !== null) {
      taskEventSource.close();
      taskEventSource = null;
    }

    if (tabName === 'tab-tag-management') {
      renderLabelList({ target: "labels-list" });
      renderTagList({ target: "tags-list" });
      renderModelLabelAssignments({ target: "model-labels-box" });
    } else if (tabName === 'tab-source-management') {
      renderSourceManager({ target: "source-management-box" });
    } else if (tabName === 'tab-task-management') {
      const refreshTaskTab = () => {
        renderActiveTasks({ target: 'active-tasks-list', refresh: refreshTaskTab });
        renderTaskConfigs({ target: 'task-config-list', refresh: refreshTaskTab });
      };

      // Initial render via fetch so the tab feels instant on first open.
      refreshTaskTab();

      // SSE stream for live task updates — no polling, server pushes on change.
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
    } else if (tabName === 'tab-dataset-management') {
      await renderDatasetManager({ target: 'dataset-management-box' });
    } else if (tabName === 'tab-model-management') {
      await renderModelConfigManager({ target: 'model-config-box' });
    }
  });

  // activate default configuration tab
  const defaultTabBtn = container?.querySelector('.workspace-tab-button');
  if (defaultTabBtn) defaultTabBtn.click();
}
