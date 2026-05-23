import { FileNavigation } from '../../ui/utils/fileNavigation.js';
import { initCanvasStage } from './canvas/main.js';
import { addInferenceResults, loadImageOntoCanvas } from './canvas/image.js';
import { init as initFiles } from './files.js';
import { init as initSidebar } from './sidebar.js';
import { init as initToolbar } from './toolbar.js';
import { setupWorkspaceTabs } from '/app-static/js/app/utils/tabs.js';
import { runInference } from '/app-static/js/shared/api/inference.js';
import { events, EventTypes } from '/app-static/js/shared/events/index.js';
import { Logger } from '/app-static/js/ui/utils/index.js';


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

  await initCanvasStage(runtime);

  // App-specific event subscriptions that only apply to the main edit tab.
  events.subscribe(EventTypes.RUN_INFERENCE_ON_CANVAS, async ({ modelName }) => {
    try {
      const img = runtime.state?.image?.img;
      if (!img) throw new Error('No image loaded');
      const result = await runInference(modelName, img);
      await addInferenceResults(runtime, result);
    } catch (err) {
      Logger.error('Failed to run inference:', err);
    }
  });

  events.subscribe(EventTypes.LOAD_IMAGE_ONTO_CANVAS, ({ path, showCanvas }) => {
    loadImageOntoCanvas(runtime, path);
    if (showCanvas) {
      events.publish(EventTypes.SHOW_CANVAS);
    }
  });

  await Promise.all([initSidebar(runtime), initToolbar(runtime)]);

  // activate default workspace tab
  const defaultTabBtn = tab_pane?.querySelector('.workspace-tab-button');
  if (defaultTabBtn) defaultTabBtn.click();
}
