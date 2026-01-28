import { init as initCanvas } from './canvas/main.js';
import { init as initFiles } from './files.js';
import { init as initSidebar } from './sidebar.js';
import { setupWorkspaceTabs } from '/app-static/js/app/utils/tabs.js';

export async function init() {
  const container = setupWorkspaceTabs("#tab-image-tagging", tabName => {
    if (tabName === 'tab-files' && !initFiles._initialized) {
      initFiles();
      initFiles._initialized = true;
    }
  });

  await Promise.all([initCanvas(), initSidebar()]);

  // activate default workspace tab
  const defaultTabBtn = container?.querySelector('.workspace-tab-button');
  if (defaultTabBtn) defaultTabBtn.click();
}