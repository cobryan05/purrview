import { createBoundingBox, updateBoundingBox } from './drawing.js';
import { selectShape, findGroupAtPoint } from './selection.js';
import { getCurrentTool, getLayer, getStage, getCurrentLabelUuid } from './state.js';
import { clearSelection, selectBboxTool, setTool } from './tools.js';

let pendingGroup = null;
let startPos = null;
let isPanning = false;
let lastPanPos = null;
let lastDiscardTime = 0;
let lastDiscardPos = null;
const DOUBLE_CLICK_TIME_MS = 400;
const DOUBLE_CLICK_DISTANCE_PX = 10;

function getPointerPosition() {
  const stage = getStage();
  const pos = stage.getPointerPosition();
  if (!pos) return null;
  return {
    x: (pos.x - stage.x()) / stage.scaleX(),
    y: (pos.y - stage.y()) / stage.scaleY(),
  };
}

export function handleMouseDown(e) {
  const stage = getStage();
  const layer = getLayer();
  if (e.evt.button === 1) {
    isPanning = true;
    lastPanPos = { x: e.evt.clientX, y: e.evt.clientY };
    stage.container().style.cursor = 'move';
    e.evt.preventDefault();
    return;
  }

  if (e.evt.button !== 0) return;
  if (isPanning) return;

  const currentTool = getCurrentTool();
  if (currentTool === 'select') return;

  const pos = getPointerPosition();
  if (!pos) return;

  startPos = pos;
  const currentLabelUuid = getCurrentLabelUuid();
  pendingGroup = createBoundingBox(
    pos.x, pos.y, { width: 1, height: 1, metadata: { labelUuid: currentLabelUuid } });
  if (pendingGroup) {
    layer.add(pendingGroup);
  }
}

export function handleMouseMove(e) {
  const stage = getStage();
  const layer = getLayer();
  if (isPanning) {
    const dx = e.evt.clientX - lastPanPos.x;
    const dy = e.evt.clientY - lastPanPos.y;
    lastPanPos = { x: e.evt.clientX, y: e.evt.clientY };
    stage.x(stage.x() + dx);
    stage.y(stage.y() + dy);
    stage.batchDraw();
    return;
  }

  if (!pendingGroup) return;

  const pos = getPointerPosition();
  if (!pos) return;

  const dx = pos.x - startPos.x;
  const dy = pos.y - startPos.y;

  const box = pendingGroup.findOne('.box');
  const label = pendingGroup.findOne('.label');

  if (!box || !label) return;

  const newX = dx < 0 ? pos.x : startPos.x;
  const newY = dy < 0 ? pos.y : startPos.y;
  const newWidth = Math.abs(dx);
  const newHeight = Math.abs(dy);

  pendingGroup.position({ x: newX, y: newY });
  box.size({ width: newWidth, height: newHeight });
  label.y(-18);

  layer.batchDraw();
}

export function handleMouseUp(e) {
  const currentTool = getCurrentTool();
  getStage().container().style.cursor =
    currentTool === 'select' ? 'default' : 'crosshair';

  if (e.evt.button === 1) {
    isPanning = false;
    return;
  }

  if (pendingGroup) {
    const box = pendingGroup.findOne('.box');
    const stage = getStage();
    const now = Date.now();
    const pos = getPointerPosition();

    if (box.width() < DOUBLE_CLICK_DISTANCE_PX || box.height() < DOUBLE_CLICK_DISTANCE_PX) {
      pendingGroup.destroy();

      // Check for double-click-like behavior
      if (
        lastDiscardPos &&
        now - lastDiscardTime < DOUBLE_CLICK_TIME_MS &&
        Math.hypot(pos.x - lastDiscardPos.x, pos.y - lastDiscardPos.y) < DOUBLE_CLICK_DISTANCE_PX
      ) {
        const hitGroup = findGroupAtPoint(pos);
        if( hitGroup ) {
            const currentLabelUuid = getCurrentLabelUuid();
            updateBoundingBox(hitGroup, { metadata: { labelUuid: currentLabelUuid } });
            selectShape(hitGroup);
        }
      }

      lastDiscardTime = now;
      lastDiscardPos = pos;
    } else {
      selectShape(pendingGroup);
    }

    getLayer().draw();
    pendingGroup = null;
  }
}

export function handleWheel(e) {
  const stage = getStage();
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
  const stage = getStage();
  const layer = getLayer();
  if (e.target === stage) {
    clearSelection();
    layer.draw();
  }
}

export function handleContextMenu(e) {
  const layer = getLayer();
  e.evt.preventDefault();
  clearSelection();
  if (getCurrentTool() === 'select') {
    selectBboxTool();
  } else {
    setTool('select');
  }
  layer.draw();
}
