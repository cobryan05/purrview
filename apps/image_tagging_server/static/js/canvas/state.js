import { notify } from './utils.js';

let stage = null;
let layer = null;
let transformer = null;

let currentTool = 'select';
let currentImageName = null;
let currentLabelUuid = null;
let currentLabelText = null

let labelList = [];
let labelUuidMap = new Map();

export function getCurrentTool() {
  return currentTool;
}

export function setCurrentTool(tool) {
  notify("setCurrentTool called with", tool);
  const parts = tool.split(':');
  currentTool = parts[0];
  if (parts.length === 2) {
    setCurrentLabelUuid(parts[1]);
  }
}

export function setLabelList(labels) {
  labelList = labels;
  labelUuidMap.clear();
  function indexLabels(labels) {
    for (const label of labels) {
      if (label.metadata?.uuid) {
        labelUuidMap.set(label.metadata.uuid, label);
      }
      if (label.children) indexLabels(label.children);
    }
  }
  indexLabels(labels);
}

export function getLabelList() {
  return labelList;
}

export function getLabelByUuid(uuid) {
  return labelUuidMap.get(uuid) || null;
}

export function getCurrentLabelUuid() {
  return currentLabelUuid;
}

export function setCurrentLabelUuid(uuid) {
  currentLabelUuid = uuid
}

export function getCurrentImageName() {
  return currentImageName;
}

export function setCurrentImageName(name) {
  currentImageName = name;
}

export function getTransformer() {
  return transformer;
}

export function setTransformer(newTransformer) {
  transformer = newTransformer;
}

export function getStage() {
  return stage;
}

export function getLayer() {
  return layer;
}

export function initStage() {
  const container = document.getElementById('draw-container');

  stage = new Konva.Stage({
    container: 'draw-container',
    width: container.clientWidth,
    height: container.clientHeight,
  });

  transformer = new Konva.Transformer({
    rotateEnabled: false,
    borderStroke: 'yellow',
    borderDash: [4, 4],
    anchorStroke: 'red',
    anchorFill: 'white',
    anchorSize: 10,
    anchorCornerRadius: 5,
    enabledAnchors: ['top-left', 'top-right', 'bottom-left', 'bottom-right'],
    keepRatio: false,

    // Prevent flip and tiny shapes
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

  setTransformer(transformer);

  layer = new Konva.Layer();
  layer.add(transformer);
  stage.add(layer);
}
