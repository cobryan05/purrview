// @ts-check
import { Logger } from '/app-static/js/ui/utils/index.js';

/**
 * @typedef {import('@app_types').State} State
 */


/** @type {State} */
export const state = {
  // --- Canvas context ---
  canvas: {
    stage: null,
    layer: null,
    transformer: null,

    bboxes: new Map(),
  },

  // Current tool + selection
  currentTool: 'select',
  currentClassUuid: null,
  selectedBoxUuid: null,

  // Current image context
  image: null,

  // --- Getters ---
  get selectedTool() { return this.currentTool; },
  get selectedBboxUuid() { return this.selectedBoxUuid; },
  get classUuid() { return this.currentClassUuid; },

  getBboxGroup(uuid) {
    return this.canvas.bboxes?.get(uuid);
  },
  // --- Setters ---
  setTool(tool) {
    Logger.debug("setCurrentTool:", tool);
    const parts = tool.split(':');
    this.currentTool = parts[0];
    if (parts.length === 2) this.currentClassUuid = parts[1];
  },

  setClassUuid(uuid) { this.currentClassUuid = uuid; },
  setImage(image) {
    this.image = image
  },
  /** @param {import('@konva_groups').CanvasBboxGroup} bboxGroup */
  removeBboxGroup(bboxGroup) {
    this.canvas.bboxes?.delete(bboxGroup.metadata.uuid);
  },
  /** @param {import('@konva_groups').CanvasBboxGroup} bboxGroup */
  updateCanvasBbox(bboxGroup) {
    this.canvas.bboxes?.set(bboxGroup.metadata.uuid, bboxGroup);
  },
  setSelectedBboxUuid(bboxUuid) { this.selectedBoxUuid = bboxUuid; },
  setTransformer(t) { this.canvas.transformer = t; },
  clearBboxes() { this.canvas.bboxes?.clear(); },
  clearSelection() { state.canvas.transformer?.setNodes([]); },

  // --- Initialization ---
  initStage(container) {
    this.canvas.stage = new Konva.Stage({
      container: 'draw-container',
      width: container.clientWidth,
      height: container.clientHeight,
    });

    this.canvas.transformer = new Konva.Transformer({
      rotateEnabled: false,
      borderStroke: 'yellow',
      borderDash: [4, 4],
      anchorStroke: 'red',
      anchorFill: 'white',
      anchorSize: 10,
      anchorCornerRadius: 5,
      enabledAnchors: ['top-left', 'top-right', 'bottom-left', 'bottom-right'],
      ignoreStroke: true,
      keepRatio: false,
      /**
       * @param {KonvaBox} oldBox
       * @param {KonvaBox} newBox
       * @returns {KonvaBox}
       */
      boundBoxFunc: (oldBox, newBox) => {
        const minSize = 10;
        if (Math.abs(newBox.width) < minSize || Math.abs(newBox.height) < minSize) {
          return oldBox;
        }
        if (newBox.width < 0 || newBox.height < 0) {
          return oldBox;
        }
        return newBox;
      },
    });
    this.canvas.layer = new Konva.Layer();
    if (!(this.canvas.transformer && this.canvas.layer && this.canvas.stage)) {
      Logger.error("Failed to initialize canvas components");
      return;
    }

    this.canvas.layer.add(this.canvas.transformer);
    this.canvas.stage.add(this.canvas.layer);

    this.canvas.bboxes = new Map();
  },
};
