import { init as initImageTagging } from './image-tagging/main.js';
import { init as initConfiguration } from './configuration/main.js';
import { init as initReview, applyFilterWhenReady } from './review/main.js';
import { init as initSettings } from './settings/main.js';
import { setupMainTabs, navigateTabs } from '/app-static/js/app/utils/tabs.js';
import { events, EventTypes } from '/app-static/js/shared/events/index.js';
import { createImageTaggingRuntime, setPrimaryRuntimeId } from './image-tagging/runtime.js';
import { initContextMenus } from '/app-static/js/ui/contextMenu/index.js';
import { activeModel } from '/app-static/js/shared/activeModel.js';
import { activeTrainingConfig } from '/app-static/js/shared/activeTrainingConfig.js';
import { mountActiveModelBar } from '/app-static/js/ui/activeModelBar.js';
import { mountNoticesBar } from '/app-static/js/ui/noticesBar.js';
import { setPendingDatasetPrefill } from '/app-static/js/ui/datasets.js';

const initializedFeatures = new Set();

/**
 * @typedef {Object} AppState
 * @property {import('@image_tagging_types').ImageTaggingRuntime} imgRuntime
 */

/** @type {AppState} */
const _appState = {
  imgRuntime: createImageTaggingRuntime()
};
setPrimaryRuntimeId(_appState.imgRuntime.id);

/**
 * @param {AppState} appState
 */
export async function init(appState) {
  // Initialise session-scoped visibility prefs and active training config state
  activeModel.init();
  await activeTrainingConfig.init();
  mountActiveModelBar('active-model-bar');
  mountNoticesBar('notices-bar');

  // Initialise the right-click context menu system
  initContextMenus();

  // Use the new helper to manage top-level tabs

  const { container, activateTab } = setupMainTabs('#main-tabs', async (tabName, button, pane) => {
    if (!initializedFeatures.has(tabName)) {
      // Mark initialized BEFORE awaiting: a fast A→B→A tab bounce re-enters
      // this callback while the first init() is still in flight, and a
      // post-await add() would let it start a second concurrent init
      // (duplicate Konva stage / review panel, leaked subscriptions).
      initializedFeatures.add(tabName);
      if (tabName === 'tab-image-tagging') {
        await initImageTagging(appState.imgRuntime, pane);
      } else if (tabName === 'tab-configuration') {
        await initConfiguration?.();
      } else if (tabName === 'tab-review') {
        await initReview?.();
      } else if (tabName === 'tab-settings') {
        await initSettings?.();
      }
    }
    // Let features react to main-tab focus changes (e.g. the Configuration
    // tab closes its task SSE stream when it loses focus).
    events.publish(EventTypes.MAIN_TAB_CHANGED, { tabName });
  });


  events.subscribeToCanvas(_appState.imgRuntime.id, EventTypes.SHOW_MAIN_CANVAS, () => {
    activateMainTab(container, 'tab-image-tagging');
    activateWorkspaceTab('#tab-image-tagging', 'tab-canvas');
  });

  events.subscribe(EventTypes.OPEN_REVIEW_WITH_FILTERS, ({ tagUuids, labelUuids, verificationCheck }) => {
    activateMainTab(container, 'tab-review');
    // applyFilterWhenReady applies immediately if the panel is already up, or
    // queues the state so init() applies it once createReviewPanel resolves.
    const state = { tagUuids: tagUuids ?? [] };
    if (labelUuids !== undefined) state.labelUuids = labelUuids;
    if (verificationCheck !== undefined) state.verificationCheck = verificationCheck;
    applyFilterWhenReady(state);
  });

  events.subscribe(EventTypes.EXPORT_FILTERS_TO_DATASET, ({ filterJson, labelUuids }) => {
    setPendingDatasetPrefill({
      filterJson,
      labelUuids,
      name: 'New Imported Dataset',
      description: 'New dataset imported from search filters',
    });
    navigateTabs('tab-configuration', 'tab-dataset-management');
  });


  // Activate default top-level tab
  const defaultTabBtn = container?.querySelector('.main-tab-button.active');
  if (defaultTabBtn) defaultTabBtn.click();

}

function activateMainTab(container, tabId) {
  container.querySelector(`[data-tab="${tabId}"]`)?.click();
}

/**
 * Navigate to a main tab and then (after a short delay) to a workspace sub-tab.
 * The delay gives the main-tab's lazy-init time to mount its pane before the
 * workspace-tab click fires.  Both arguments are tab ids (the data-tab value).
 *
 * @param {string} mainTabId       e.g. 'tab-configuration'
 * @param {string} workspaceTabId  e.g. 'tab-dataset-management'
 */


function activateWorkspaceTab(featureId, tabId) {
  document
    .querySelector(`${featureId} [data-tab="${tabId}"]`)
    ?.click();
}

init(_appState);
