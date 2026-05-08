import { setupWorkspaceTabs } from '/app-static/js/app/utils/tabs.js';
import { renderLabelList } from '/app-static/js/ui/labels.js';
import { renderModelLabelAssignments } from '/app-static/js/ui/models.js';
import { renderSourceManager } from '/app-static/js/ui/sources.js';
import { renderTagList } from '/app-static/js/ui/tags.js';
import { renderActiveTasks, renderTaskConfigs } from '/app-static/js/ui/tasks.js';

export async function init() {
  const { container, activateTab } = setupWorkspaceTabs("#tab-configuration", (tabName, button, pane) => {
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
      refreshTaskTab();
    }
  });

  // activate default configuration tab
  const defaultTabBtn = container?.querySelector('.workspace-tab-button');
  if (defaultTabBtn) defaultTabBtn.click();
}
