import { init as initImageTagging } from './image-tagging/main.js';
import { init as initConfiguration } from './configuration/main.js';
import { init as initReview, applyFilterWhenReady } from './review/main.js';
import { init as initSettings } from './settings/main.js';
import { setupMainTabs } from '/app-static/js/app/utils/tabs.js';
import { events, EventTypes } from '/app-static/js/shared/events/index.js';
import { createImageTaggingRuntime } from './image-tagging/runtime.js';
import { activeModel } from '/app-static/js/shared/activeModel.js';
import { mountActiveModelBar } from '/app-static/js/ui/activeModelBar.js';
import { setPendingDatasetPrefill } from '/app-static/js/ui/datasets.js';

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
  // Initialise session-scoped active model state (reads sessionStorage + fetches config if needed)
  await activeModel.init();
  mountActiveModelBar('active-model-bar');

  // Use the new helper to manage top-level tabs

  const { container, activateTab } = setupMainTabs('#main-tabs', async (tabName, button, pane) => {
    if (!initializedFeatures.has(tabName)) {
      if (tabName === 'tab-image-tagging') {
        await initImageTagging(appState.imgRuntime, pane);
      } else if (tabName === 'tab-configuration') {
        await initConfiguration?.();
      } else if (tabName === 'tab-review') {
        await initReview?.();
      } else if (tabName === 'tab-settings') {
        await initSettings?.();
      }
      initializedFeatures.add(tabName);
    }
  });


  events.subscribeToCanvas(_appState.imgRuntime.id, EventTypes.SHOW_MAIN_CANVAS, () => {
    activateMainTab(container, 'tab-image-tagging');
    activateWorkspaceTab('#tab-image-tagging', 'tab-canvas');
  });

  events.subscribe(EventTypes.OPEN_REVIEW_WITH_FILTERS, ({ tagUuids }) => {
    activateMainTab(container, 'tab-review');
    // applyFilterWhenReady applies immediately if the panel is already up, or
    // queues the state so init() applies it once createReviewPanel resolves.
    applyFilterWhenReady({ tagUuids: tagUuids ?? [] });
  });

  events.subscribe(EventTypes.EXPORT_FILTERS_TO_DATASET, ({ conditions, labelUuids }) => {
    setPendingDatasetPrefill({
      conditions,
      labelUuids,
      name: 'New Imported Dataset',
      description: 'New dataset imported from search filters',
    });
    activateMainTab(container, 'tab-configuration');
    // Delay workspace-tab click until after the configuration pane mounts/settles.
    setTimeout(() => {
      activateWorkspaceTab('#tab-configuration', 'tab-dataset-management');
    }, 50);
  });


  // Activate default top-level tab
  const defaultTabBtn = container?.querySelector('.main-tab-button.active');
  if (defaultTabBtn) defaultTabBtn.click();

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
