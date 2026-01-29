import { registerKeyboardShortcuts } from './shortcuts.js';
import { init as initImageTagging } from './image-tagging/main.js';
import { init as initConfiguration } from './configuration/main.js';
import { setupMainTabs } from '/app-static/js/app/utils/tabs.js';
import { events, EventTypes } from '/app-static/js/shared/events/index.js';

import { appState } from './state.js';

const initializedFeatures = new Set();

/**
 * @param {import('@app_types').AppState} state
 */
export async function init(state) {
  // Use the new helper to manage top-level tabs

  const { container, activateTab } = setupMainTabs('#main-tabs', async (tabName) => {
    if (!initializedFeatures.has(tabName)) {
      if (tabName === 'tab-image-tagging') {
        await initImageTagging(state.imageTagging);
      } else if (tabName === 'tab-configuration') {
        await initConfiguration?.();
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

  registerKeyboardShortcuts();
}

function activateMainTab(container, tabId) {
  container.querySelector(`[data-tab="${tabId}"]`)?.click();
}

function activateWorkspaceTab(featureId, tabId) {
  document
    .querySelector(`${featureId} [data-tab="${tabId}"]`)
    ?.click();
}

init(appState);
