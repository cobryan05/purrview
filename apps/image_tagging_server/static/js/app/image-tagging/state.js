// @ts-check

/**
  @param {HTMLElement} container
  @returns {import('@canvas_types').ImageTaggingState}
 */
export function createImageTaggingState(container) {
  const stage = new Konva.Stage({
    container: container,
    width: container.clientWidth,
    height: container.clientHeight,
  });


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
  currentLabelUuid: null,
  selectedBoxUuid: null,

  get selectedTool() {
    return this.currentTool;
  },
  get selectedBboxUuid() {
    return this.selectedBoxUuid;
  },
  get labelUuid() {
    return this.currentLabelUuid;
  },

  getBboxGroup(uuid) {
    return this.canvas.bboxes.get(uuid);
  },

  // --- Setters ---
  setTool(tool) {
    Logger.debug('setTool:', tool);
    const parts = tool.split(':');
    this.currentTool = parts[0];
    if (parts[1]) this.setLabelUuid(parts[1]);
  },

  setLabelUuid(uuid) {
    this.currentLabelUuid = uuid;
  },

  /** @type {import('@domain_image').UIImage} */
  setImage(image) {
    this.image = image;
    // if( image.bboxes ) {
    //   for( const [uuid, box] of image.bboxes.entries() ) {
    //     const canvasBbox = new CanvasBboxGroup(box, this);
    //     this.updateCanvasBbox(canvasBbox);
    //   }
    // }
  },

  setSelectedBboxUuid(uuid) {
    this.selectedBoxUuid = uuid;
  },

  setTransformer(t) {
    this.canvas.transformer = t;
  },

  removeBboxGroup(bboxGroup) {
    this.canvas.bboxes.delete(bboxGroup.metadata.bboxInfo.uuid);
  },

  updateCanvasBboxView(bboxGroup) {
    this.canvas.bboxes.set(bboxGroup.metadata.bboxInfo.uuid, bboxGroup);
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
