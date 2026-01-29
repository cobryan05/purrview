// @ts-check
import { canvasState } from './canvas/state.js';
import { Logger } from '/app-static/js/ui/utils/index.js';

/**
 * @typedef {import('@image_tagging_types').ImageTaggingState} ImageTaggingState
 */

/** @type {ImageTaggingState} */
export const imageTaggingState = {
  canvas: canvasState,
  image: null,
  currentTool: 'select',
  currentClassUuid: null,
  selectedBoxUuid: null,

  get selectedTool() {
    return this.currentTool;
  },
  get selectedBboxUuid() {
    return this.selectedBoxUuid;
  },
  get classUuid() {
    return this.currentClassUuid;
  },

  getBboxGroup(uuid) {
    return this.canvas.bboxes.get(uuid);
  },

  // --- Setters ---
  setTool(tool) {
    Logger.debug('setTool:', tool);
    const parts = tool.split(':');
    this.currentTool = parts[0];
    if (parts[1]) this.setClassUuid(parts[1]);
  },

  setClassUuid(uuid) {
    this.currentClassUuid = uuid;
  },

  setImage(image) {
    this.image = image;
  },

  setSelectedBboxUuid(uuid) {
    this.selectedBoxUuid = uuid;
  },

  setTransformer(t) {
    this.canvas.transformer = t;
  },

  removeBboxGroup(bboxGroup) {
    this.canvas.bboxes.delete(bboxGroup.metadata.runtimeBboxInfo.uuid);
  },

  updateCanvasBbox(bboxGroup) {
    this.canvas.bboxes.set(bboxGroup.metadata.runtimeBboxInfo.uuid, bboxGroup);
  },

  clearBboxes() {
    this.canvas.bboxes.clear();
    this.image?.bboxes?.clear();
  },

  clearSelection() {
    this.canvas.transformer?.setNodes([]);
  },

  // --- Initialization ---
  initCanvas(container) {
    this.canvas.init(container);
  },
};
