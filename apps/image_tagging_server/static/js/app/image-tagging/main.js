import { FileNavigation } from '../../ui/utils/fileNavigation.js';
import { createCanvasInstance } from './canvas/instance.js';
import { loadImageOntoCanvas } from './canvas/image.js';
import { wireInferenceRunner } from './canvasWiring.js';
import { init as initFiles } from './files.js';
import { init as initSidebar } from './sidebar.js';
import { init as initToolbar } from './toolbar.js';
import { initHotkeyBindings } from './hotkeyBindings.js';
import { setupWorkspaceTabs } from '/app-static/js/app/utils/tabs.js';
import { events, EventTypes } from '/app-static/js/shared/events/index.js';
import { initSplitResize } from '/app-static/js/ui/utils/splitResize.js';


let _resolveCanvasReady;
/**
 * Resolves once this tab's canvas is mounted and its LOAD_IMAGE_ONTO_CANVAS
 * subscription is live. A caller elsewhere in the app (see OPEN_IMAGE_ON_MAIN_CANVAS
 * in app/main.js) can focus Annotate and then await this rather than guessing a delay
 * long enough for the lazy init — publish too early and the image simply never loads.
 * @type {Promise<void>}
 */
export const canvasReady = new Promise(resolve => { _resolveCanvasReady = resolve; });

/**
 * @param {import('@image_tagging_types').ImageTaggingRuntime} runtime
 * @param {HTMLElement} tab_pane
 */
export async function init(runtime, tab_pane) {
  // Load hotkey bindings from DB early so keydown handlers have the right map.
  initHotkeyBindings().catch(err => console.warn('Hotkey init failed:', err));

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
  wireInferenceRunner(runtime);

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

  // Sidebar drag-resize — store handle alongside _toolbarHandle so destroy() is
  // reachable if a cleanup path is added later (avoids permanently leaking document listeners).
  const sidebarEl = document.getElementById('sidebar');
  const resizeHandleEl = document.getElementById('sidebar-resize-handle');
  if (sidebarEl && resizeHandleEl) {
    runtime._sidebarResizeHandle = initSplitResize(resizeHandleEl, sidebarEl, {
      storageKey: 'wtch_sidebar_width',
      defaultWidth: 240,
      min: 120,
      max: 500,
    });
  }

  // activate default workspace tab
  const defaultTabBtn = tab_pane?.querySelector('.workspace-tab-button');
  if (defaultTabBtn) defaultTabBtn.click();

  _resolveCanvasReady();
}
