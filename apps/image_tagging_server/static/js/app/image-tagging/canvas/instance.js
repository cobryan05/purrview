import { createImageTaggingRuntime } from '../runtime.js';
import { initCanvasStage } from './main.js';
import { loadImageOntoCanvas } from './image.js';

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
  const runtime = createImageTaggingRuntime();
  runtime.setCanvasContainer(container);
  await initCanvasStage(runtime);
  runtime.setTool(options.tool ?? 'select');

  return {
    runtime,
    loadImage: (path) => loadImageOntoCanvas(runtime, path),
    destroy() {
      runtime.destroyCanvas();
    },
  };
}
