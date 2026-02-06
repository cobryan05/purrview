import { FileNavigation } from '../../ui/utils/fileNavigation.js';
import { init as initCanvas } from './canvas/main.js';
import { init as initFiles } from './files.js';
import { init as initSidebar } from './sidebar.js';
import { init as initToolbar } from './toolbar.js';
import { setupWorkspaceTabs } from '/app-static/js/app/utils/tabs.js';


/**
 * @param {import('@image_tagging_types').ImageTaggingState} state
 */
export async function init(state) {
  if( !state.fileNavigation ) {
    state.fileNavigation = new FileNavigation();
  }
  const { container, activateTab } = setupWorkspaceTabs("#tab-image-tagging", tabName => {
    if (tabName === 'tab-files') {
      initFiles(state);
    }
  });

  await Promise.all([initCanvas(state), initSidebar(state), initToolbar(state)]);

  // activate default workspace tab
  const defaultTabBtn = container?.querySelector('.workspace-tab-button');
  if (defaultTabBtn) defaultTabBtn.click();
}