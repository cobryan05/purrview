import {
  handleClick,
  handleContextMenu,
  handleMouseDown,
  handleMouseMove,
  handleMouseUp,
  handleWheel,
} from './interaction.js';
import {
  clearAnnotations,
  deleteSelected,
  exportAnnotations,
  loadImageAndMetadata,
  reloadImage,
  saveAnnotations,
} from './image.js';
import { state } from './state.js';
import { selectBboxTool, setTool, updateToolbarButtons } from './tools.js';

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
