import { appState } from '../../state.js';
import { events, EventTypes } from '/app-static/js/shared/events/index.js';

// Clear selection implementation
export function clearSelection() {
  const transformer = appState.imageTagging.canvas.transformer;
  if (transformer) {
    transformer.nodes([]);
    transformer.getLayer()?.draw();
  }
  appState.imageTagging.setSelectedBboxUuid(null);
  appState.imageTagging?.canvas?.stage?.batchDraw();
}

// Set the current tool, update UI and cursor
/**
 * @param {import('@image_tagging_types').ImageTaggingState} state
 * @param {string} tool
 */
export function setTool(state, tool) {
  state.setTool(tool);
  clearSelection();
  state.canvas.stage.container().style.cursor = tool === 'select' ? 'default' : 'crosshair';
  events.publish(EventTypes.CANVAS_TOOL_CHANGED, { tool });
}

/**
 * @param {import('@image_tagging_types').ImageTaggingState} state
 */
export function getTool(state) {
  return state.currentTool;
}

export function parseToolUuid(str) {
  const [tool, uuid] = str.split(':', 2);
  return { tool, uuid: uuid === '' ? null : uuid };
}
