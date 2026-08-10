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
    }
  });

  // activate default configuration tab
  const defaultTabBtn = tabs.container?.querySelector('.workspace-tab-button');
  if (defaultTabBtn) defaultTabBtn.click();
}
