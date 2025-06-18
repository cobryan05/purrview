// interaction.js

import { getCurrentTool, setCurrentTool, getStage, getLayer } from './state.js';
import { clearSelection, setTool } from './tools.js';
import { createShape } from './drawing.js';
import { selectShape } from './selection.js';

// State variables for interaction
let tempShape = null;
let startPos = null;
let isPanning = false;
let lastPanPos = null;

// Helper to get pointer position relative to stage with transforms
function getPointerPosition() {
  let stage = getStage();
  const pos = stage.getPointerPosition();
  if (!pos) return null;
  return {
    x: (pos.x - stage.x()) / stage.scaleX(),
    y: (pos.y - stage.y()) / stage.scaleY(),
  };
}

// Mouse down event handler
export function handleMouseDown(e) {
  let stage = getStage();
  if (e.evt.button === 1) { // Middle click = start panning
    isPanning = true;
    lastPanPos = { x: e.evt.clientX, y: e.evt.clientY };
    stage.container().style.cursor = 'move';
    e.evt.preventDefault();
    return;
  }

  if (e.evt.button !== 0) return; // Only left click beyond this
  if (isPanning) return;

  const currentTool = getCurrentTool();
  if (currentTool === 'select') return;

  const pos = getPointerPosition();
  if (!pos) return;

  startPos = pos;
  tempShape = createShape(currentTool, pos.x, pos.y);
  if (tempShape) layer.add(tempShape);
}

// Mouse move event handler
export function handleMouseMove(e) {
  let stage = getStage();
  if (isPanning) {
    const dx = e.evt.clientX - lastPanPos.x;
    const dy = e.evt.clientY - lastPanPos.y;
    lastPanPos = { x: e.evt.clientX, y: e.evt.clientY };

    stage.x(stage.x() + dx);
    stage.y(stage.y() + dy);
    stage.batchDraw();
    return;
  }

  if (!tempShape) return;

  const pos = getPointerPosition();
  if (!pos) return;

  const dx = pos.x - startPos.x;
  const dy = pos.y - startPos.y;

  if (tempShape instanceof Konva.Rect) {
    tempShape.width(dx);
    tempShape.height(dy);
  } else if (tempShape instanceof Konva.Circle) {
    tempShape.radius(Math.sqrt(dx * dx + dy * dy));
  } else if (tempShape instanceof Konva.Line) {
    tempShape.points([startPos.x, startPos.y, pos.x, pos.y]);
  }

  layer.batchDraw();
}

// Mouse up event handler
export function handleMouseUp(e) {
  const currentTool = getCurrentTool();
  getStage().container().style.cursor = currentTool === 'select' ? 'default' : 'crosshair';

  if (e.evt.button === 1) { // middle button up ends panning
    isPanning = false;
    return;
  }
  tempShape = null;
}

// Mouse wheel zoom handler
export function handleWheel(e) {
  let stage = getStage();
  e.evt.preventDefault();

  const oldScale = stage.scaleX();
  const pointer = stage.getPointerPosition();
  if (!pointer) return;

  const scaleBy = 1.05;
  const direction = e.evt.deltaY > 0 ? -1 : 1;
  const newScale = direction > 0 ? oldScale * scaleBy : oldScale / scaleBy;

  stage.scale({ x: newScale, y: newScale });

  const mousePointTo = {
    x: (pointer.x - stage.x()) / oldScale,
    y: (pointer.y - stage.y()) / oldScale,
  };

  const newPos = {
    x: pointer.x - mousePointTo.x * newScale,
    y: pointer.y - mousePointTo.y * newScale,
  };

  stage.position(newPos);
  stage.batchDraw();
}

// Click event to clear selection if clicking empty area
export function handleClick(e) {
  let stage = getStage();
  if (e.target === stage) {
    clearSelection();
    layer.draw();
  }
}

// Right-click event handler to clear selection and switch tool
export function handleContextMenu(e) {
  e.evt.preventDefault();
  clearSelection();
  setTool('select');
  layer.draw();
}
