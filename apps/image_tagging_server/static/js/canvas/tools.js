import { setCurrentTool, getCurrentTool, getStage } from './state.js';

// Clear selection implementation (adapt or import your existing logic)
export function clearSelection() {
  if (window.selectedShape) {
    window.selectedShape = null;
  }
  if (window.transformer) {
    window.transformer.nodes([]);
    window.transformer.getLayer()?.draw();
  }
  getStage().batchDraw();
}

// Set the current tool, update UI and cursor
export function setTool(tool) {
  console.log('[DEBUG] setTool called with', tool);
  setCurrentTool(tool);
  clearSelection();
  updateToolbarButtons();
  getStage().container().style.cursor = tool === 'select' ? 'default' : 'crosshair';
}

// Get the current tool
export function getTool() {
  return getCurrentTool();
}

// Highlight toolbar buttons based on current tool
export function updateToolbarButtons() {
  const current = getCurrentTool();
  document.querySelectorAll('#toolbar button[data-tool]').forEach(btn => {
    btn.classList.toggle('selected', btn.dataset.tool === current);
  });
}
