import { createImageTaggingRuntime } from '../runtime.js';
import { initCanvasStage } from './main.js';
import { loadImageOntoCanvas } from './image.js';
import { attachHotkeys } from '../../hotkeys.js';
import { events, EventTypes } from '/app-static/js/shared/events/index.js';
import { fetchTags } from '/app-static/js/shared/api/tags.js';
import { defaultEnhanceMode } from './enhance.js';

/**
 * Shared canvas setup: wires stage events, hotkeys, and live tag-icon refresh.
 * Call this after the runtime has a canvas (setCanvasContainer or setImageTaggingPane).
 *
 * @param {import('@image_tagging_types').ImageTaggingRuntime} runtime
 * @param {{ hotkeys?: boolean }} [opts] - hotkeys: false for canvases the user
 *   didn't open and isn't driving (see attachHotkeys' warning).
 * @returns {Promise<{ destroy: () => void }>}
 */
export async function initCanvasCore(runtime, { hotkeys = true } = {}) {
  await initCanvasStage(runtime);
  const hotkeyHandle = hotkeys ? attachHotkeys(runtime) : null;

  const unsubTagsUpdated = events.subscribe(EventTypes.TAGS_UPDATED, async () => {
    const tagsMap = await fetchTags();
    runtime.canvas?.layer?.find('.bbox')?.forEach(bboxView => {
      const updatedTags = (bboxView.uiBBox?.tags ?? []).map(t => ({
        uuid: t.uuid,
        icon: tagsMap.get(t.uuid)?.icon ?? null,
      }));
      bboxView.uiBBox.tags = updatedTags;
      bboxView.refreshIcons(updatedTags);
    });
  });

  return {
    destroy() {
      hotkeyHandle?.destroy();
      unsubTagsUpdated();
    },
  };
}

/**
 * Creates a self-contained, destroyable canvas instance.
 *
 * The container must have explicit CSS dimensions before this is called;
 * the Konva stage will size itself to the container's current clientWidth/clientHeight.
 *
 * @param {HTMLElement} container
 * @param {{ tool?: string, enhance?: string, lockEnhance?: boolean, hotkeys?: boolean }} [options]
 *   enhance: initial display-enhancement mode id. Defaults to whatever mode the
 *   user last selected this session, so a modal opened while working through
 *   dark frames comes up already enhanced. Pass 'none' for canvases with no
 *   enhance control (read-only previews), which also skips the CPU cost.
 *   lockEnhance: pin that mode — the enhance hotkey becomes a no-op on this
 *   canvas, so a focused read-only preview can't change the session default.
 *   hotkeys: false to skip attachHotkeys entirely. Required for any canvas that
 *   appears on its own (embedded previews in polled status cards): attaching
 *   puts it on top of the hotkey stack, stealing every key from the canvas the
 *   user actually opened.
 * @returns {Promise<{ runtime: import('@image_tagging_types').ImageTaggingRuntime, loadImage: (path: string) => Promise<void>, destroy: () => void }>}
 */
export async function createCanvasInstance(container, options = {}) {
  const runtime = options.runtime ?? createImageTaggingRuntime();
  runtime.setCanvasContainer(container);
  if (runtime.canvas) {
    runtime.canvas.enhanceMode = options.enhance ?? defaultEnhanceMode();
    runtime.canvas.enhanceLocked = !!options.lockEnhance;
  }
  const coreHandle = await initCanvasCore(runtime, { hotkeys: options.hotkeys !== false });
  runtime.setTool(options.tool ?? 'select');

  return {
    runtime,
    loadImage: (path, opts) => loadImageOntoCanvas(runtime, path, opts),
    destroy() {
      coreHandle.destroy();
      runtime.destroyCanvas();
    },
  };
}
