import { createBboxGroup, updateBoundingBox } from './drawing.js';
import { state } from './state.js'
import { clearSelection, selectBboxTool, setTool, selectBbox } from './tools.js';

let pendingDraggedBbox = null;
let startPos = null;
let isPanning = false;
let lastPanPos = null;
let _lastClickTime = 0;
let _lastClickPos = null;
const DOUBLE_CLICK_TIME_MS = 400;
const DOUBLE_CLICK_DISTANCE_PX = 10;
let crosshairV = null;
let crosshairH = null;

function createCrosshairLines() {
  const layer = state.canvas.layer;
  if (crosshairV && crosshairH) return; // already created

  const stage = state.canvas.stage;
  const width = stage.width();
  const height = stage.height();

  crosshairV = new Konva.Line({
    points: [0, 0, 0, height],
    stroke: 'rgba(127,127,127,0.5)',
    strokeWidth: 1,
    dash: [4, 4],
    listening: false,  // don't interfere with events
  });

  crosshairH = new Konva.Line({
    points: [0, 0, width, 0],
    stroke: 'rgba(0,0,0,0.3)',
    strokeWidth: 1,
    dash: [4, 4],
    listening: false,
  });

  layer.add(crosshairV);
  layer.add(crosshairH);
}

function getPointerPosition() {
  const stage = state.canvas.stage;
  const pos = stage.getPointerPosition();
  if (!pos) return null;
  return {
    x: (pos.x - stage.x()) / stage.scaleX(),
    y: (pos.y - stage.y()) / stage.scaleY(),
  };
}

export async function handleMouseDown(e) {
  const stage = state.canvas.stage;
  const layer = state.canvas.layer;
  if (e.evt.button === 1) {
    isPanning = true;
    lastPanPos = { x: e.evt.clientX, y: e.evt.clientY };
    stage.container().style.cursor = 'move';
    e.evt.preventDefault();
    return;
  }

  if (e.evt.button !== 0) return;
  if (isPanning) return;

  const currentTool = state.currentTool;
  if (currentTool === 'select') return;

  const pos = getPointerPosition();
  if (!pos) return;

  startPos = pos;
  const currentClassUuid = state.currentClassUuid;
  pendingDraggedBbox = createBboxGroup(
    pos.x, pos.y, { width: 1, height: 1, metadata: { classUuid: currentClassUuid } });
  if (pendingDraggedBbox) {
    layer.add(pendingDraggedBbox);
    pendingDraggedBbox.name('annotation');
    state.setBboxGroup(pendingDraggedBbox.metadata.uuid, pendingDraggedBbox);
  }
}

export async function handleMouseMove(e) {
  const stage = state.canvas.stage;
  const layer = state.canvas.layer;
  if (isPanning) {
    const dx = e.evt.clientX - lastPanPos.x;
    const dy = e.evt.clientY - lastPanPos.y;
    lastPanPos = { x: e.evt.clientX, y: e.evt.clientY };
    stage.x(stage.x() + dx);
    stage.y(stage.y() + dy);
    stage.batchDraw();
    return;
  }

  const pos = getPointerPosition();
  if (!pos) return;

  const currentTool = state.currentTool;
  if (currentTool != 'select') {
    createCrosshairLines();
    // Position lines at mouse X,Y spanning full height,width
    crosshairV.points([pos.x, 0, pos.x, stage.height()]);
    crosshairH.points([0, pos.y, stage.width(), pos.y]);
    crosshairV.show();
    crosshairH.show();
    layer.batchDraw();
  } else {
    if (crosshairV) crosshairV.hide();
    if (crosshairH) crosshairH.hide();
    layer.batchDraw();
  }

  if (!pendingDraggedBbox) return;

  const dx = pos.x - startPos.x;
  const dy = pos.y - startPos.y;

  const box = pendingDraggedBbox.findOne('.box');
  const cls = pendingDraggedBbox.findOne('.class');

  if (!box || !cls) return;

  const newX = dx < 0 ? pos.x : startPos.x;
  const newY = dy < 0 ? pos.y : startPos.y;
  const newWidth = Math.abs(dx);
  const newHeight = Math.abs(dy);

  pendingDraggedBbox.position({ x: newX, y: newY });
  box.size({ width: newWidth, height: newHeight });
  cls.y(-18);

  layer.batchDraw();
}

export async function handleMouseUp(e) {
  const currentTool = state.currentTool;
  const isSelectTool = currentTool === 'select';
  state.canvas.stage.container().style.cursor =
   isSelectTool ? 'default' : 'crosshair';

  if (e.evt.button === 1) {
    isPanning = false;
    return;
  }

  if (pendingDraggedBbox) {
    const box = pendingDraggedBbox.findOne('.box');
    const stage = state.canvas.stage;

    if (box.width() < DOUBLE_CLICK_DISTANCE_PX || box.height() < DOUBLE_CLICK_DISTANCE_PX) {
      pendingDraggedBbox.destroy();
    } else {
      selectShape(pendingDraggedBbox, box);
    }

    state.canvas.layer.draw();
    pendingDraggedBbox = null;
  }

  const pos = getPointerPosition();
  const hitGroup = findGroupAtPoint(pos);

  const now = Date.now();
  const timeDelta = now - _lastClickTime;
  const clickDist = _lastClickPos ? Math.hypot(pos.x - _lastClickPos.x, pos.y - _lastClickPos.y) : null;
  const isDoubleClick = (clickDist != null && timeDelta < DOUBLE_CLICK_TIME_MS && clickDist < DOUBLE_CLICK_DISTANCE_PX);
  _lastClickTime = now;
  _lastClickPos = pos;

  if (hitGroup) {
    if (isDoubleClick) {
      if (!isSelectTool) {
        const currentClassUuid = state.currentClassUuid;
        updateBoundingBox(hitGroup, { metadata: { classUuid: currentClassUuid } });
      } else {
        selectBbox(hitGroup.metadata.uuid);
      }
    } else {
      const box = hitGroup.findOne('.box');
      selectShape(hitGroup, box);
    }
  }


}

export async function handleWheel(e) {
  const stage = state.canvas.stage;
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

export function handleClick(e) {
  const stage = state.canvas.stage;
  const layer = state.canvas.layer;
  if (e.target === stage) {
    clearSelection();
    layer.draw();
  }
}

export function handleContextMenu(e) {
  const layer = state.canvas.layer;
  e.evt.preventDefault();
  clearSelection();
  if (state.currentTool === 'select') {
    selectBboxTool();
  } else {
    setTool('select');
  }
  layer.draw();
}

export function selectShape(group, highlight_shape = null) {
  const layer = state.canvas.layer;
  const transformer = state.canvas.transformer;
  const shape = highlight_shape ?? group;
  if (!shape || !layer) return;

  transformer.nodes([shape]);
  transformer.moveToTop();
}

export function findGroupAtPoint(pos) {
  const layer = state.canvas.layer;
  const children = layer.getChildren(node => node.name() === 'annotation');

  for (const group of children) {
    const rect = group.getClientRect({ relativeTo: layer });
    if (
      pos.x >= rect.x &&
      pos.x <= rect.x + rect.width &&
      pos.y >= rect.y &&
      pos.y <= rect.y + rect.height
    ) {
      return group;
    }
  }
  return null;
}