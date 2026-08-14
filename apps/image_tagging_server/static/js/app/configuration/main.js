import { setupWorkspaceTabs } from '/app-static/js/app/utils/tabs.js';
import { renderConfigLanding } from '/app-static/js/app/configuration/landing.js';
import { renderLabelList } from '/app-static/js/ui/labels.js';
import { renderModelLabelAssignments } from '/app-static/js/ui/models.js';
import { renderModelConfigManager } from '/app-static/js/ui/modelConfigs.js';
import { renderModelStackManager } from '/app-static/js/ui/modelStacks.js';
import { renderTagList } from '/app-static/js/ui/tags.js';
import { renderDatasetManager } from '/app-static/js/ui/datasets.js';
import { renderTrainingConfigManager } from '/app-static/js/ui/trainingConfigs.js';
import { renderFilterChainManager } from '/app-static/js/ui/filterChains.js';
import { renderWatchProfileManager } from '/app-static/js/ui/watchers.js';
import { renderSourceManager } from '/app-static/js/ui/sources.js';
import { renderTaskConfigs } from '/app-static/js/ui/tasks.js';
import { refreshListEditorHeights } from '/app-static/js/app/utils/listEditorHeight.js';

export async function init() {
  // Declared first so the onTabChange callback can reach activateTab: setupTabs
  // invokes the callback for the initial tab *before* it returns, so the
  // landing gets a lambda that resolves at click time rather than the value.
  let tabs;

  tabs = setupWorkspaceTabs("#tab-configuration", async (tabName, button, pane) => {
    if (tabName === 'tab-config-overview') {
      await renderConfigLanding({
        target: 'config-overview-box',
        activateTab: name => tabs.activateTab(name),
      });
    } else if (tabName === 'tab-tag-management') {
      renderLabelList({ target: "labels-list" });
      renderTagList({ target: "tags-list" });
      renderModelLabelAssignments({ target: "model-labels-box" });
    } else if (tabName === 'tab-dataset-management') {
      await renderDatasetManager({ target: 'dataset-management-box' });
    } else if (tabName === 'tab-model-management') {
      await renderModelConfigManager({ target: 'model-config-box' });
    } else if (tabName === 'tab-modelstack-management') {
      await renderModelStackManager({ target: 'model-stack-management-box' });
    } else if (tabName === 'tab-training-management') {
      await renderTrainingConfigManager({ target: 'training-config-box' });
    } else if (tabName === 'tab-filterchain-management') {
      await renderFilterChainManager({ target: 'filter-chain-management-box' });
    } else if (tabName === 'tab-watcher-management') {
      await renderWatchProfileManager({ target: 'watcher-management-box' });
    } else if (tabName === 'tab-source-management') {
      // Moved in from the dissolved Operations tab — both of these are
      // definition editors of the same list-beside-editor shape as their
      // neighbours here. Running what they define is the Runs tab's job.
      await renderSourceManager({ target: 'source-management-box' });
    } else if (tabName === 'tab-task-management') {
      const refresh = () => renderTaskConfigs({ target: 'task-config-list', refresh });
      await renderTaskConfigs({ target: 'task-config-list', refresh });
    }

    // These panes mount a .list-editor-split whose height is measured, not
    // hard-coded; it can only be measured once the pane is laid out.
    refreshListEditorHeights();
  });

  // activate default configuration tab
  const defaultTabBtn = tabs.container?.querySelector('.workspace-tab-button');
  if (defaultTabBtn) defaultTabBtn.click();
}
