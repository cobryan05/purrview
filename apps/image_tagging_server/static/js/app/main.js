import { registerKeyboardShortcuts } from './hotkeys.js';
import { init as initImageTagging } from './image-tagging/main.js';
import { init as initConfiguration } from './configuration/main.js';
import { init as initReview } from './review/main.js';
import { setupMainTabs } from '/app-static/js/app/utils/tabs.js';
import { events, EventTypes } from '/app-static/js/shared/events/index.js';
import { createImageTaggingRuntime } from './image-tagging/runtime.js';

const initializedFeatures = new Set();

/**
 * @typedef {Object} AppState
 * @property {import('@image_tagging_types').ImageTaggingRuntime} imgRuntime
 */

/** @type {AppState} */
const _appState = {
  imgRuntime: createImageTaggingRuntime()
}

/**
 * @param {AppState} appState
 */
export async function init(appState) {
  // Use the new helper to manage top-level tabs

  const { container, activateTab } = setupMainTabs('#main-tabs', async (tabName, button, pane) => {
    if (!initializedFeatures.has(tabName)) {
      if (tabName === 'tab-image-tagging') {
        await initImageTagging(appState.imgRuntime, pane);
      } else if (tabName === 'tab-configuration') {
        await initConfiguration?.();
      } else if (tabName === 'tab-review') {
        await initReview?.();
      }
      initializedFeatures.add(tabName);
    }
  });


  events.subscribe(EventTypes.SHOW_CANVAS, () => {
    activateMainTab(container, 'tab-image-tagging');
    activateWorkspaceTab('#tab-image-tagging', 'tab-canvas');
  });


  // Activate default top-level tab
  const defaultTabBtn = container?.querySelector('.main-tab-button.active');
  if (defaultTabBtn) defaultTabBtn.click();

  registerKeyboardShortcuts(appState);
}

function activateMainTab(container, tabId) {
  container.querySelector(`[data-tab="${tabId}"]`)?.click();
}

function activateWorkspaceTab(featureId, tabId) {
  document
    .querySelector(`${featureId} [data-tab="${tabId}"]`)
    ?.click();
}

init(_appState);
