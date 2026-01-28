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
import { state } from './state.js';
import { selectBboxTool, setTool, updateToolbarButtons } from './tools.js';
import { events, EventTypes } from '/app-static/js/shared/events/index.js';
import { Logger } from '/app-static/js/ui/utils/index.js';

export async function init() {
  // Initialize stage and set default tool
  const container = document.getElementById('draw-container');
  state.initStage(container);
  state.setTool('select');

  // Expose functions globally for HTML onclick handlers
  window.setTool = setTool;
  window.reloadImage = reloadImage;
  window.clearAnnotations = clearAnnotations;
  window.selectBboxTool = selectBboxTool;
  window.saveAnnotations = saveAnnotations;
  window.deleteSelected = deleteSelected;
  window.loadImageAndMetadata = loadImageAndMetadata;
  window.refreshAnnotations = exportAnnotations;

  let stage = state.canvas.stage;
  // Attach stage event listeners
  stage.on('mousedown', handleMouseDown);
  stage.on('mousemove', handleMouseMove);
  stage.on('mouseup', handleMouseUp);
  stage.on('wheel', handleWheel);
  stage.on('click', handleClick);
  stage.on('contextmenu', handleContextMenu);

  // Resize stage on window resize
  window.addEventListener('resize', () => {
    const container = document.getElementById('draw-container');
    stage.width(container.clientWidth);
    stage.height(container.clientHeight);
    stage.draw();
  });

  // Update toolbar buttons on startup
  updateToolbarButtons();



  events.subscribe(EventTypes.CANVAS_BBOX_CLICKED, ({ bboxId }) => {
    const transformer = state.canvas.transformer;
    const layer = state.canvas.layer;
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
      await addInferenceResults(result);
    } catch (err) {
      Logger.error('Failed to run inference:', err);
    }
  });
}


