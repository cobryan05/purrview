import { FileNavigation } from '../../ui/utils/fileNavigation.js';
import { createCanvasInstance } from './canvas/instance.js';
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
  runtime.setPaneRef(tab_pane);
  runtime.setFileNavigation(new FileNavigation());

  const { container, activateTab } = setupWorkspaceTabs("#tab-image-tagging", (tabName, button, pane)  => {
    if (tabName === 'tab-files') {
      initFiles(runtime);
    }
  });

  const canvasContainer = tab_pane.querySelector('.canvas-container');
  await createCanvasInstance(canvasContainer, { runtime });

  // App-specific event subscriptions scoped to this canvas's runtime.
  events.subscribeToCanvas(runtime.id, EventTypes.RUN_INFERENCE_ON_CANVAS, async ({ modelName }) => {
    try {
      const img = runtime.state?.image?.img;
      if (!img) throw new Error('No image loaded');
      const result = await runInference(modelName, img);
      await addInferenceResults(runtime, result);
    } catch (err) {
      Logger.error('Failed to run inference:', err);
    }
  });

  events.subscribeToCanvas(runtime.id, EventTypes.LOAD_IMAGE_ONTO_CANVAS, ({ path, showCanvas }) => {
    loadImageOntoCanvas(runtime, path);
    if (showCanvas) {
      events.publish(EventTypes.SHOW_MAIN_CANVAS, { runtimeId: runtime.id });
    }
  });

  const toolbarEl = tab_pane.querySelector('.canvas-toolbar');
  // Store handle on runtime so destroy() is reachable if a cleanup path is added later.
  runtime._toolbarHandle = initToolbar(runtime, toolbarEl);
  await initSidebar(runtime);

  // activate default workspace tab
  const defaultTabBtn = tab_pane?.querySelector('.workspace-tab-button');
  if (defaultTabBtn) defaultTabBtn.click();
}
