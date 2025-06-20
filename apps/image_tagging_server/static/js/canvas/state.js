let stage = null;
let layer = null;
let transformer = null;

let currentTool = 'select';
let currentImageName = null;

// === Getters and setters for state variables ===

export function getCurrentTool() {
  return currentTool;
}

export function setCurrentTool(tool) {
  console.debug("[DEBUG] setTool called with", tool);
  currentTool = tool;
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

// === Initialization functions ===
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
  });
  setTransformer(transformer);

  layer = new Konva.Layer();
  layer.add(transformer)
  stage.add(layer);
}