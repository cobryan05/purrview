import {
  addInferenceResults,
  clearAnnotations,
  deleteSelected,
  loadImageOntoCanvas,
  reloadImage,
  saveAnnotations
} from './image.js';
import {
  handleClick,
  handleContextMenu,
  handleMouseDown,
  handleMouseMove,
  handleMouseUp,
  handleResize,
  handleWheel
} from './interaction.js';
import { setTool } from './tools.js';
import { runInference } from '/app-static/js/shared/api/inference.js';
import { events, EventTypes } from '/app-static/js/shared/events/index.js';
import { Logger } from '/app-static/js/ui/utils/index.js';

/**
 * @param {import('@image_tagging_types').ImageTaggingRuntime} runtime
 */
export async function init(runtime) {
  // Initialize stage and set default tool
  const container = runtime.imageTaggingPane;
  if (!container) {
    Logger.error("Failed to find canvas container");
    return;
  }
  runtime.setTool('select');
  // Expose functions globally for HTML onclick handlers
  Object.assign(window, {
    setTool,
    reloadImage,
    clearAnnotations,
    saveAnnotations,
    deleteSelected,
    loadImageAndMetadata: loadImageOntoCanvas,
  });

  const stage = runtime.canvas.stage;
  if (!stage) {
    Logger.error('Failed to initialize canvas stage');
    return;
  }

  stage.on('mousedown', e => handleMouseDown(e, runtime));
  stage.on('mousemove', e => handleMouseMove(e, runtime));
  stage.on('mouseup', e => handleMouseUp(e, runtime));
  stage.on('wheel', e => handleWheel(e, runtime));
  stage.on('click', e => handleClick(e, runtime));
  stage.on('contextmenu', e => handleContextMenu(e, runtime));

  // Resize stage on window resize
  window.addEventListener('resize', e => { handleResize(e, runtime); });
  // if (container) {
  //   stage.width(container.clientWidth);
  //   stage.height(container.clientHeight);
  //   stage.draw();
  // } else {
  //   Logger.warn('Failed to resize canvas stage');
  // }

  //});


  events.subscribe(EventTypes.CANVAS_BBOX_CLICKED, ({ bboxId }) => {
    const { transformer, layer } = state.canvas;
    const group = state.getBboxGroup(bboxId);
    if (!transformer || !layer || !group) return;

    transformer.nodes([group.metadata.rect]);
    transformer.moveToTop();
    layer.batchDraw();
  });

  events.subscribe(EventTypes.RUN_INFERENCE_ON_CANVAS, async ({ modelName }) => {
    try {
      const img = state.image?.img;
      if (!img) throw new Error('No image loaded');

      /** @type {import('@web_api').InferenceResultModel} */
      const result = await runInference(modelName, img);
      await addInferenceResults(state, result);
    } catch (err) {
      Logger.error('Failed to run inference:', err);
    }
  });

  events.subscribe(EventTypes.LOAD_IMAGE_ONTO_CANVAS, ({ path, showCanvas }) => {
    loadImageOntoCanvas(runtime, path);
    if (showCanvas) {
      events.publish(EventTypes.SHOW_CANVAS);
    }
  });
}
