import { Logger } from '/app-static/js/ui/utils/index.js';

export const state = {
  // --- Canvas context ---
  canvas: {
    stage: null,
    layer: null,
    transformer: null,

    bboxes: null,
  },

  // Current tool + selection
  currentTool: 'select',
  currentClassUuid: null,
  currentTagUuid: null,
  selectedBoxUuid: null,

  // Current image context
  image: {
    name: null,
    data: null,
    bboxes: null
  },

  // --- Getters ---
  get selectedTool() { return this.currentTool; },
  get selectedBboxUuid() { return this.selectedBoxUuid; },
  get classUuid() { return this.currentClassUuid; },
  get tagUuid() { return this.currentTagUuid; },
  get bboxes() { return this.image.bboxes; },

  getBbox(uuid) {
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
  setTagUuid(uuid) { this.currentTagUuid = uuid; },
  setImage(name, img, bboxes) {
    this.image.name = name;
    this.image.data = img;
    this.image.bboxes = bboxes;
  },
  setBbox(uuid, bboxInfo) {
    if (bboxInfo == null) {
      this.canvas.bboxes?.delete(uuid);
    } else {
      this.canvas.bboxes?.set(uuid, bboxInfo);
    }
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
      keepRatio: false,
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
