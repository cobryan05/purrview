import { events, EventTypes } from '/app-static/js/shared/events/index.js';

// Clear selection implementation
export function clearSelection() {
  // const transformer = state.imageTagging.canvas.transformer;
  // if (transformer) {
  //   transformer.nodes([]);
  //   transformer.getLayer()?.draw();
  // }
  // state.imageTagging.setSelectedBboxUuid(null);
  // state.imageTagging?.canvas?.stage?.batchDraw();
}

// Set the current tool, update UI and cursor
/**
 * @param {import('@image_tagging_types').ImageTaggingRuntime} runtime
 * @param {string} tool
 */
export function setTool(runtime, tool) {
  runtime.setTool(tool);
  clearSelection();
  runtime.canvas.stage.container().style.cursor = tool === 'select' ? 'default' : 'crosshair';
  events.publish(EventTypes.CANVAS_TOOL_CHANGED, { tool });
}

/**
 * @param {import('@image_tagging_types').ImageTaggingRuntime} runtime
 */
export function getTool(runtime) {
  return runtime.tool;
}

export function parseToolUuid(str) {
  const [tool, uuid] = str.split(':', 2);
  return { tool, uuid: uuid === '' ? null : uuid };
}
