import {
  addInferenceResults,
  clearAnnotations,
  deleteSelected,
  exportAnnotations,
  loadImageAndMetadata,
  reloadImage,
  saveAnnotations,
} from './image.js';
import {
  handleClick,
  handleContextMenu,
  handleMouseDown,
  handleMouseMove,
  handleMouseUp,
  handleWheel,
} from './interaction.js';
import { runInference } from '/app-static/js/shared/api/inference.js';
import { setTool } from './tools.js';
import { events, EventTypes } from '/app-static/js/shared/events/index.js';
import { Logger } from '/app-static/js/ui/utils/index.js';

/**
 * @param {import('@image_tagging_types').ImageTaggingState} state
 */
export async function init(state) {
  // Initialize stage and set default tool
  const container = document.getElementById('draw-container');
  state.initCanvas(container);
  state.setTool('select');
  // Expose functions globally for HTML onclick handlers
  Object.assign(window, {
    setTool,
    reloadImage,
    clearAnnotations,
    saveAnnotations,
    deleteSelected,
    loadImageAndMetadata,
  });

  const stage = state.canvas.stage;
  stage.on('mousedown', e => handleMouseDown(e, state));
  stage.on('mousemove', e => handleMouseMove(e, state));
  stage.on('mouseup', e => handleMouseUp(e, state));
  stage.on('wheel', e => handleWheel(e, state));
  stage.on('click', e => handleClick(e, state));
  stage.on('contextmenu', e => handleContextMenu(e, state));

  // Resize stage on window resize
  window.addEventListener('resize', () => {
    const container = document.getElementById('draw-container');
    stage.width(container.clientWidth);
    stage.height(container.clientHeight);
    stage.draw();
  });


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

      const result = await runInference(modelName, img);
      await addInferenceResults(state, result);
    } catch (err) {
      Logger.error('Failed to run inference:', err);
    }
  });

  events.subscribe(EventTypes.LOAD_IMAGE_ONTO_CANVAS, ({ path, showCanvas }) => {
    loadImageAndMetadata(state, path);
    if (showCanvas) {
       events.publish(EventTypes.SHOW_CANVAS);
    }
  });
}
