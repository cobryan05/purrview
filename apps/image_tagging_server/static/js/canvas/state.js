// @ts-check
import { Logger } from '/app-static/js/ui/utils/index.js';

/**
 * @typedef {import('@konva').default.Stage} Stage
 * @typedef {import('@konva').default.Layer} Layer
 * @typedef {import('@konva').default.Transformer} Transformer
 * @typedef {import('@konva').default.Group} Group
 */


/**
 * @typedef {Object} BBoxInfo
 * @property {number} x
 * @property {number} y
 * @property {number} width
 * @property {number} height
 * @property {string} [classUuid]
 * @property {string[]} [tagUuids]
 */

/**
 * @typedef {Object} ImageState
 * @property {string|null} name
 * @property {HTMLImageElement|ImageBitmap|null} data
 * @property {Map<string,BBoxInfo>|null} bboxes
 */

/**
 * @typedef {Object} CanvasState
 * @property {Stage|null} stage
 * @property {Layer|null} layer
 * @property {Transformer|null} transformer
 * @property {Map<string,Group>|null} bboxes
 */

/**
 * @typedef {Object} State
 * @property {CanvasState} canvas
 * @property {string} currentTool
 * @property {string|null} currentClassUuid
 * @property {string|null} selectedBoxUuid
 * @property {ImageState} image
 * @property {string} selectedTool
 * @property {string|null} selectedBboxUuid
 * @property {string|null} classUuid
 * @property {(uuid:string) => Group|undefined} getBboxGroup
 * @property {(tool:string) => void} setTool
 * @property {(uuid:string) => void} setClassUuid
 * @property {(name:string, img:any, bboxes:Map<string,BBoxInfo>) => void} setImage
 * @property {(uuid:string, group:Group|null) => void} setBboxGroup
 * @property {(bboxUuid:string) => void} setSelectedBboxUuid
 * @property {(t:Transformer) => void} setTransformer
 * @property {() => void} clearBboxes
 * @property {() => void} clearSelection
 * @property {(container:HTMLElement) => void} initStage
 */

/** @type {State} */
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
  setImage(name, img, bboxes) {
    this.image.name = name;
    this.image.data = img;
    this.image.bboxes = bboxes;
  },
  setBboxGroup(uuid, group) {
    if (group == null) {
      this.canvas.bboxes?.delete(uuid);
    } else {
      this.canvas.bboxes?.set(uuid, group);
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
