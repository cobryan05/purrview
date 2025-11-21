import { Logger } from '/app-static/js/ui/utils/index.js';

export const state = {
  // Kanvas context
  stage: null,
  layer: null,
  transformer: null,

  // Current tool + selection
  currentTool: 'select',
  currentClassUuid: null,
  currentTagUuid: null,
  currentBbox: null,

  // Current image context
  currentImage: null,
  currentImageName: null,
  currentBboxes: null,

  // --- Getters ---
  get tool() { return this.currentTool; },
  get classUuid() { return this.currentClassUuid; },
  get tagUuid() { return this.currentTagUuid; },
  get image() { return this.currentImage; },
  get imageName() { return this.currentImageName; },
  get bboxes() { return this.currentBboxes; },
  get bbox() { return this.currentBbox; },
  get stageRef() { return this.stage; },
  get layerRef() { return this.layer; },
  get transformerRef() { return this.transformer; },

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
    this.currentImageName = name;
    this.currentImage = img;
    this.currentBboxes = bboxes;
  },
  setBbox(bbox) { this.currentBbox = bbox; },
  setTransformer(t) { this.transformer = t; },

  // --- Initialization ---
  initStage(container) {
    this.stage = new Konva.Stage({
      container: 'draw-container',
      width: container.clientWidth,
      height: container.clientHeight,
    });

    this.transformer = new Konva.Transformer({
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

    this.layer = new Konva.Layer();
    this.layer.add(this.transformer);
    this.stage.add(this.layer);
  },
};
