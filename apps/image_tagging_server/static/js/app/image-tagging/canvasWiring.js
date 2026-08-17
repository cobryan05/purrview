/**
 * Wiring shared by the two places a canvas + sidebar is mounted: the Image
 * Tagging tab (`image-tagging/main.js` + `image-tagging/sidebar.js`) and the
 * Review tab's canvas modal (`review/main.js`'s `showCanvasModal`).
 *
 * ⚠️ **This is wiring, not DOM construction, and that is deliberate.** The tab's
 * sidebar is declared in `index.html` with global ids (`#sidebar`,
 * `#labels-tool-list`, `#tab-inspector`, `#inference-panel-container`) that CSS
 * and many Playwright tests target; the modal hand-builds its own with
 * `modal-tab-*` ids precisely because ids must be unique per document. Sharing
 * the *markup* would mean either rewriting `index.html` or cloning a template
 * and rewriting ids (fragile — `sidebar.js` resolves its panes by id). So each
 * side keeps building its own elements and passes the references in here.
 *
 * What genuinely was duplicated, and now lives here:
 *   1. the RUN_INFERENCE_ON_CANVAS subscription loop (byte-identical but for a
 *      log prefix),
 *   2. the label-list + tag-panel construction pair,
 *   3. the sidebar tab-switching rules (bbox tool → Labels, tag tool → Tags,
 *      bbox double-click → Tags), which differ only in tab ids.
 *
 * The inference *panel* is deliberately NOT created here: the tab creates it
 * lazily on first activation of its Inference tab, the modal eagerly. That
 * difference is real and stays with the callers.
 */

import { parseToolUuid } from './canvas/tools.js';
import { addInferenceResults } from './canvas/image.js';
import { runDetectionBatch } from '/app-static/js/shared/api/inference.js';
import { events, EventTypes } from '/app-static/js/shared/events/index.js';
import { Logger } from '/app-static/js/ui/utils/index.js';
import { createLabelList } from '/app-static/js/ui/labelList.js';
import { createTagPanel } from '/app-static/js/ui/inspector.js';

/**
 * Subscribe this runtime's canvas to RUN_INFERENCE_ON_CANVAS and apply the
 * results to it.
 *
 * The models are inferred in ONE request (see `runDetectionBatch`) rather than
 * one request each: they all run over the same canvas image, and sending it
 * per model made the server base64-decode and letterbox the identical image
 * once per model. Selecting four models used to mean four uploads and four
 * decodes of the same picture, run back-to-back.
 *
 * Results are still APPLIED sequentially, in the order the models were
 * selected — each one applies `replaceExisting` independently against whatever
 * the canvas looks like when its turn comes, so the combined result depends on
 * that order. That is the pre-existing (accepted) tradeoff for multi-select,
 * and it is deliberately unchanged: batching alters when the inference happens,
 * never the order its results land in.
 *
 * @param {import('@image_tagging_types').ImageTaggingRuntime} runtime
 * @param {{ logPrefix?: string }} [opts]  prefix for error logs, e.g. 'Canvas modal: '
 * @returns {() => void} unsubscribe
 */
export function wireInferenceRunner(runtime, { logPrefix = '' } = {}) {
  return events.subscribeToCanvas(
    runtime.id,
    EventTypes.RUN_INFERENCE_ON_CANVAS,
    async ({ modelNames, confThresh, replaceExisting }) => {
      const img = runtime.state?.image?.img;
      if (!img) {
        Logger.error(`${logPrefix}Failed to run inference: no image loaded`);
        return;
      }
      let outcomes;
      try {
        outcomes = await runDetectionBatch(modelNames, img, confThresh);
      } catch (err) {
        // The whole request failed (inference server down, bad image). Per-model
        // failures do not come through here — they arrive as `error` below, so
        // one broken model still lets the others apply.
        Logger.error(`${logPrefix}Failed to run inference:`, err);
        return;
      }
      for (const { modelName, result, error } of outcomes) {
        if (error || !result) {
          Logger.error(`${logPrefix}Failed to run inference (${modelName}):`, error);
          continue;
        }
        try {
          await addInferenceResults(runtime, result, { replaceExisting });
        } catch (err) {
          Logger.error(`${logPrefix}Failed to apply inference (${modelName}):`, err);
        }
      }
    },
  );
}

/**
 * Create the two eagerly-built sidebar panels for a canvas runtime.
 *
 * Both must be alive before the first CANVAS_TOOL_CHANGED / CANVAS_BBOX_CLICKED
 * event: the label panel's "auto-select first label on bare bbox tool" logic and
 * the tag panel's single-click updates both depend on already being subscribed.
 *
 * Either element may be absent (the tab resolves its own by id, and those can
 * legitimately be missing) — that panel is simply skipped.
 *
 * @param {import('@image_tagging_types').ImageTaggingRuntime} runtime
 * @param {{ labelsListEl?: HTMLElement|null, tagsPaneEl?: HTMLElement|null }} els
 * @returns {Promise<{ destroy(): void }>} destroys both panels
 */
export async function createSidebarPanels(runtime, { labelsListEl = null, tagsPaneEl = null } = {}) {
  const labelHandle = labelsListEl
    ? await createLabelList(labelsListEl, { mode: 'select', runtime })
    : null;
  const tagPanelHandle = tagsPaneEl ? createTagPanel(tagsPaneEl, runtime) : null;

  return {
    destroy() {
      labelHandle?.destroy?.();
      tagPanelHandle?.destroy?.();
    },
  };
}

/**
 * Subscribe the sidebar tab-switching rules: picking the bbox tool shows the
 * Labels tab and picking the tag tool shows the Tags tab (that's where their
 * pickers live), and double-clicking a bbox shows Tags.
 *
 * @param {import('@image_tagging_types').ImageTaggingRuntime} runtime
 * @param {{
 *   activateTab: (tabId: string) => void,
 *   labelsTabId: string,
 *   tagsTabId: string,
 * }} opts
 * @returns {() => void} unsubscribe both
 */
export function wireSidebarTabRules(runtime, { activateTab, labelsTabId, tagsTabId }) {
  const unsubToolChanged = events.subscribeToCanvas(
    runtime.id,
    EventTypes.CANVAS_TOOL_CHANGED,
    ({ tool }) => {
      const { tool: kind } = parseToolUuid(tool);
      if (kind === 'bbox') activateTab(labelsTabId);
      else if (kind === 'tag') activateTab(tagsTabId);
    },
  );

  const unsubDblClick = events.subscribeToCanvas(
    runtime.id,
    EventTypes.CANVAS_BBOX_DOUBLE_CLICKED,
    () => activateTab(tagsTabId),
  );

  return () => {
    unsubToolChanged?.();
    unsubDblClick?.();
  };
}
