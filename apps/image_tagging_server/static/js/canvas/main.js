import { initStage, getStage } from './state.js';
import { setTool, updateToolbarButtons } from './tools.js';
import {
  handleMouseDown,
  handleMouseMove,
  handleMouseUp,
  handleWheel,
  handleClick,
  handleContextMenu,
} from './interaction.js';
import {
  exportAnnotations,
  loadImageAndMetadata,
  clearAnnotations,
  saveAnnotations,
  deleteSelected,
} from './io.js';

// Initialize stage and set default tool
initStage();
setTool('select');

// Expose functions globally for HTML onclick handlers
window.setTool = setTool;
window.clearAnnotations = clearAnnotations;
window.saveAnnotations = saveAnnotations;
window.deleteSelected = deleteSelected;
window.loadImageAndMetadata = loadImageAndMetadata;
window.refreshAnnotations = exportAnnotations;

let stage = getStage()
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
