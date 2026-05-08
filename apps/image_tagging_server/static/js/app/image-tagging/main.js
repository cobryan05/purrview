import { FileNavigation } from '../../ui/utils/fileNavigation.js';
import { init as initCanvas } from './canvas/main.js';
import { init as initFiles } from './files.js';
import { init as initSidebar } from './sidebar.js';
import { init as initToolbar } from './toolbar.js';
import { setupWorkspaceTabs } from '/app-static/js/app/utils/tabs.js';


/**
 * @param {import('@image_tagging_types').ImageTaggingRuntime} runtime
 * @param {HTMLElement} tab_pane
 */
export async function init(runtime, tab_pane) {
  runtime.setImageTaggingPane(tab_pane);
  runtime.setFileNavigation(new FileNavigation());

  const { container, activateTab } = setupWorkspaceTabs("#tab-image-tagging", (tabName, button, pane)  => {
    if (tabName === 'tab-files') {
      initFiles(runtime);
    }
  });

  await Promise.all([initCanvas(runtime), initSidebar(runtime), initToolbar(runtime)]);

  // activate default workspace tab
  const defaultTabBtn = tab_pane?.querySelector('.workspace-tab-button');
  if (defaultTabBtn) defaultTabBtn.click();
}