import { events, EventTypes } from '/app-static/js/shared/events/index.js';

/** True for tools that draw new bboxes ('bbox' and 'bbox:<labelUuid>'). */
export function isDrawTool(tool) {
  return tool === 'bbox' || tool.startsWith('bbox:');
}

/** True for the tag-application tool ('tag:<tagUuid>'). */
export function isTagTool(tool) {
  return tool.startsWith('tag:');
}

/**
 * Existing bboxes are only draggable in the 'select' tool. While a draw or tag
 * tool is active a pointer-down over a bbox must start drawing / apply, not drag.
 * @param {import('@image_tagging_types').ImageTaggingRuntime} runtime
 */
export function syncBboxDraggable(runtime) {
  const layer = runtime?.canvas?.layer;
  if (!layer) return;
  const draggable = runtime.tool === 'select';
  layer.find('.bbox').forEach(n => n.draggable(draggable));
}

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
  syncBboxDraggable(runtime);
  runtime.canvas.stage.container().style.cursor =
    isDrawTool(tool) ? 'crosshair' : isTagTool(tool) ? 'pointer' : 'default';
  events.publish(EventTypes.CANVAS_TOOL_CHANGED, { tool, runtimeId: runtime.id });
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
