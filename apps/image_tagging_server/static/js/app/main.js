import { init as initImageTagging, canvasReady } from './image-tagging/main.js';
import { init as initConfiguration } from './configuration/main.js';
import { init as initReview, applyFilterWhenReady } from './review/main.js';
import { init as initRuns } from './runs/main.js';
import { init as initNvr } from './nvr/main.js';
import { init as initSettings } from './settings/main.js';
import { setupMainTabs, setupSectionTabs, navigateTabs } from '/app-static/js/app/utils/tabs.js';
import { initScrollChrome, showChrome } from '/app-static/js/app/utils/scrollChrome.js';
import { initListEditorHeights, refreshListEditorHeights } from '/app-static/js/app/utils/listEditorHeight.js';
import { events, EventTypes } from '/app-static/js/shared/events/index.js';
import { createImageTaggingRuntime, setPrimaryRuntimeId } from './image-tagging/runtime.js';
import { initContextMenus } from '/app-static/js/ui/contextMenu/index.js';
import { activeModel } from '/app-static/js/shared/activeModel.js';
import { activeTrainingConfig } from '/app-static/js/shared/activeTrainingConfig.js';
import { mountActiveModelBar } from '/app-static/js/ui/activeModelBar.js';
import { mountNoticesBar } from '/app-static/js/ui/noticesBar.js';
import { setPendingDatasetPrefill } from '/app-static/js/ui/datasets.js';
import { seedRunsBadge } from '/app-static/js/ui/runsBadge.js';
import { initTours } from '/app-static/js/ui/tour/index.js';
import { Logger, toast } from '/app-static/js/ui/utils/logging.js';

const initializedFeatures = new Set();

/** Section tabs inside Annotate; kept so activateReview can reach activateTab. */
let _annotateTabs = null;

/**
 * Wire Annotate's two section tabs (Canvas / Review) and lazy-init whichever is
 * focused. Both features are heavy — Canvas mounts a Konva stage, Review builds
 * a full filter panel and card grid — so neither is built until it is looked at.
 *
 * The same "mark before await" guard the main-tab callback uses applies here for
 * the same reason: a fast Canvas→Review→Canvas bounce re-enters this callback
 * while the first init is still in flight, and a post-await add() would start a
 * second concurrent init (duplicate Konva stage, leaked subscriptions).
 *
 * @param {AppState} appState
 */
async function initAnnotate(appState) {
  const initializedSections = new Set();

  _annotateTabs = setupSectionTabs('#tab-image-tagging', async (tabName, button, pane) => {
    if (initializedSections.has(tabName)) return;
    initializedSections.add(tabName);

    if (tabName === 'tab-annotate-workspace') {
      await initImageTagging(appState.imgRuntime, pane);
    } else if (tabName === 'tab-review') {
      await initReview?.();
    }
  });

  // Activate the default section tab. setupSectionTabs already activates the
  // first one on setup, so this only matters if none carried `active` markup.
  const defaultSectionBtn = document.querySelector('#tab-image-tagging .section-tab-button');
  if (defaultSectionBtn && !defaultSectionBtn.classList.contains('active')) defaultSectionBtn.click();
}

/**
 * Focus Annotate → Review from anywhere, initialising both tiers on the way.
 *
 * Review is no longer a main tab, so reaching it means activating Annotate
 * first and its Review section tab second — and the second click has to happen
 * after Annotate's lazy init has built the section tabs, hence the await.
 *
 * @param {Element|null} container - the main-tabs container
 */
async function activateReview(container) {
  activateMainTab(container, 'tab-image-tagging');
  // Annotate's init is kicked off synchronously by the click above but resolves
  // later; yield until the section tabs exist rather than racing them.
  for (let i = 0; i < 50 && !_annotateTabs; i++) {
    await new Promise(resolve => setTimeout(resolve, 10));
  }
  _annotateTabs?.activateTab('tab-review');
}

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
        await initAnnotate(appState);
      } else if (tabName === 'tab-configuration') {
        await initConfiguration?.();
      } else if (tabName === 'tab-nvr') {
        await initNvr?.();
      } else if (tabName === 'tab-runs') {
        await initRuns?.();
      } else if (tabName === 'tab-settings') {
        await initSettings?.();
      }
    }
    // A tab switch can arrive programmatically (OPEN_REVIEW_WITH_FILTERS, the
    // canvas "back to review" path) while the chrome is scrolled out of view —
    // land on the new tab with it showing rather than seemingly headerless.
    showChrome();
    // Let features react to main-tab focus changes (e.g. the Jobs tab closes
    // its task SSE stream when it loses focus).
    events.publish(EventTypes.MAIN_TAB_CHANGED, { tabName });
    // The pane that just became visible may hold a list-editor split, which can
    // only be measured once it is actually laid out.
    refreshListEditorHeights();
  });


  events.subscribeToCanvas(_appState.imgRuntime.id, EventTypes.SHOW_MAIN_CANVAS, () => {
    activateMainTab(container, 'tab-image-tagging');
    // Canvas now sits one tier deeper: Annotate's section tab, then the
    // Canvas/Files workspace tab inside it.
    _annotateTabs?.activateTab('tab-annotate-workspace');
    activateWorkspaceTab('#tab-image-tagging', 'tab-canvas');
  });

  // "Open this image on the canvas", published from tabs that have no canvas of their
  // own (the NVR live view's saved frame, today). Annotate is focused first and the
  // load is forwarded only once its lazy init has actually mounted the canvas — see
  // OPEN_IMAGE_ON_MAIN_CANVAS in shared/events/types.js.
  events.subscribe(EventTypes.OPEN_IMAGE_ON_MAIN_CANVAS, async ({ path }) => {
    if (!path) return;
    activateMainTab(container, 'tab-image-tagging');
    _annotateTabs?.activateTab('tab-annotate-workspace');
    await canvasReady;
    activateWorkspaceTab('#tab-image-tagging', 'tab-canvas');
    events.publish(EventTypes.LOAD_IMAGE_ONTO_CANVAS, {
      path,
      showCanvas: true,
      runtimeId: appState.imgRuntime.id,
    });
  });

  // "Show me the clip this harvested frame came from." Imported on demand rather than
  // at module load: the event player and its overlay are only ever reached from a
  // tracking-blip run, and everyone else should not pay for them.
  events.subscribe(EventTypes.OPEN_EVENT_CLIP, async ({ eventUuid, atS = null }) => {
    if (!eventUuid) return;
    try {
      const { openEventClipAt } = await import('/app-static/js/app/nvr/events/main.js');
      await openEventClipAt(eventUuid, { atS });
    } catch (err) {
      Logger.error(err);
      toast('Could not open the clip for this frame', 5000, 'error');
    }
  });

  // Carried in the tab's pending-selection drop box rather than handed to the Events
  // page directly: the navigation re-renders that page, so a filter pushed into the
  // render that is on screen now would be thrown away a moment later. `selectUuid` also
  // forces the re-render when Events is already the active sub-tab, and drops the
  // request if the pane never wires up.
  events.subscribe(EventTypes.OPEN_NVR_EVENTS_FOR_SOURCE, ({ sourceUuid }) => {
    if (!sourceUuid) return;
    navigateTabs('tab-nvr', 'tab-nvr-events', { selectUuid: sourceUuid });
  });

  events.subscribe(EventTypes.OPEN_REVIEW_WITH_FILTERS, ({
    tagUuids, imageTagUuids, labelUuids, verificationCheck, bboxVerification,
  }) => {
    activateReview(container);
    // applyFilterWhenReady applies immediately if the panel is already up, or
    // queues the state so init() applies it once createReviewPanel resolves.
    const state = { tagUuids: tagUuids ?? [] };
    // Whole-image tags (image_tags), as opposed to the bbox tags above — the
    // only filter that can match a frame collected without any bboxes.
    // Always written, so a caller that passes none clears a previous selection
    // rather than silently leaving it applied (same reason bboxVerification is
    // always present in datasetFormToReviewState).
    state.imageTagConditions = (imageTagUuids ?? []).map(uuid => ({ tagUuid: uuid, state: 'present' }));
    if (labelUuids !== undefined) state.labelUuids = labelUuids;
    if (verificationCheck !== undefined) state.verificationCheck = verificationCheck;
    // 'any' | 'verified' | 'unverified' — the bbox verification *tag*, not the
    // whole-image label-completeness check above. Passing it through lets a
    // caller open Review already narrowed to what still needs annotating.
    if (bboxVerification !== undefined) state.bboxVerification = bboxVerification;
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


  trackMainTabsHeight(container);
  initScrollChrome();
  initListEditorHeights();

  // Activate default top-level tab
  const defaultTabBtn = container?.querySelector('.main-tab-button.active');
  if (defaultTabBtn) defaultTabBtn.click();

  // The Jobs badge is the one piece of a tab that must be right before the tab
  // is ever opened — it exists to tell you something is running from wherever
  // you are. Jobs' own init is lazy, so seed the count here with a single fetch
  // (no SSE stream: the stream is Jobs' to open when it has focus).
  seedRunsBadge();

  // Last, and not awaited: a tour is a guest on a booted page, never something
  // boot waits on, and its workspace check is a couple of cached fetches.
  void initTours();
}

/**
 * Publish the pinned main-tab header's height as --main-tabs-h so the other
 * sticky bars (the Review toolbar) can park below it. Measured rather than
 * hard-coded in CSS: the header's height moves with font size and with the
 * Activity badge, and an offset that drifts either overlaps the header or
 * leaves a strip of scrolling content showing between the two bars.
 *
 * @param {Element|null} container
 */
function trackMainTabsHeight(container) {
  const header = container?.querySelector('.main-tab-header');
  if (!header) return;
  const publish = () =>
    document.documentElement.style.setProperty('--main-tabs-h', `${header.offsetHeight}px`);
  publish();
  if (typeof ResizeObserver !== 'undefined') new ResizeObserver(publish).observe(header);
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
