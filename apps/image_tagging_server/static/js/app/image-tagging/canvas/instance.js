import { createImageTaggingRuntime } from '../runtime.js';
import { initCanvasStage } from './main.js';
import { loadImageOntoCanvas } from './image.js';
import { attachHotkeys } from '../../hotkeys.js';
import { events, EventTypes } from '/app-static/js/shared/events/index.js';
import { fetchTags } from '/app-static/js/shared/api/tags.js';

/**
 * Shared canvas setup: wires stage events, hotkeys, and live tag-icon refresh.
 * Call this after the runtime has a canvas (setCanvasContainer or setImageTaggingPane).
 *
 * @param {import('@image_tagging_types').ImageTaggingRuntime} runtime
 * @returns {Promise<{ destroy: () => void }>}
 */
export async function initCanvasCore(runtime) {
  await initCanvasStage(runtime);
  const hotkeyHandle = attachHotkeys(runtime);

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
      hotkeyHandle.destroy();
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
 * @param {{ tool?: string }} [options]
 * @returns {Promise<{ runtime: import('@image_tagging_types').ImageTaggingRuntime, loadImage: (path: string) => Promise<void>, destroy: () => void }>}
 */
export async function createCanvasInstance(container, options = {}) {
  const runtime = options.runtime ?? createImageTaggingRuntime();
  runtime.setCanvasContainer(container);
  const coreHandle = await initCanvasCore(runtime);
  runtime.setTool(options.tool ?? 'select');

  return {
    runtime,
    loadImage: (path) => loadImageOntoCanvas(runtime, path),
    destroy() {
      coreHandle.destroy();
      runtime.destroyCanvas();
    },
  };
}
