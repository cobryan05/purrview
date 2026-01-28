import { setupWorkspaceTabs } from '/app-static/js/app/utils/tabs.js';
import { renderClassList } from '/app-static/js/ui/classes.js';
import { renderModelClassAssignments } from '/app-static/js/ui/models.js';
import { renderSourceManager } from '/app-static/js/ui/sources.js';
import { renderTagList } from '/app-static/js/ui/tags.js';
import { renderActiveTasks, renderTaskConfigs } from '/app-static/js/ui/tasks.js';

export async function init() {
  const container = setupWorkspaceTabs("#tab-configuration", tabName => {
    if (tabName === 'tab-tag-management') {
      renderClassList({ target: "classes-list" });
      renderTagList({ target: "tags-list" });
      renderModelClassAssignments({ target: "model-classes-box" });
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
