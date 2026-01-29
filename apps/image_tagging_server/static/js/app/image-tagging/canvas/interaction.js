import { Logger } from '../../../ui/utils/logging.js';
import { generateUUID } from '../../../ui/utils/utils.js';
import { CanvasBboxGroup } from './groups/CanvasBboxGroup.js';
import { clearSelection, setTool } from './tools.js';
import { events, EventTypes } from '/app-static/js/shared/events/index.js';

/** @type {import('@app_types').CanvasBboxGroup | null} */
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

function createCrosshairLines(state) {
  if (crosshairV && crosshairH) return;

  const layer = state.canvas.layer;
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

  layer.add(crosshairV, crosshairH);
}

/**
 * @param {import('@image_tagging_types').ImageTaggingState} state - The image tagging state for this canvas.
 * @returns {x, y}
 */
function getPointerPosition(state) {
  const stage = state.canvas.stage;
  const pos = stage.getPointerPosition();
  if (!pos) return null;
  return {
    x: (pos.x - stage.x()) / stage.scaleX(),
    y: (pos.y - stage.y()) / stage.scaleY(),
  };
}


/** Return bbox group under pointer, if any
 *
 * @param {import('@image_tagging_types').ImageTaggingState} state - The image tagging state for this canvas.
 * @param {{x: number, y: number}} pos - The pointer position in canvas coordinates.
 * @returns {import('./groups/CanvasBboxGroup.js').CanvasBboxGroup|null} The bbox group under the pointer, or null if none.
 */
function findGroupAtPoint(state, pos) {
  const layer = state.canvas.layer;
  const children = layer.getChildren(node => node.name() === 'annotation');
  for (const group of children) {
    const rect = group.getClientRect({ relativeTo: layer });
    if (pos.x >= rect.x && pos.x <= rect.x + rect.width &&
      pos.y >= rect.y && pos.y <= rect.y + rect.height) {
      return group;
    }
  }
  return null;
}

/**
 * @param {Konva.KonvaEventObject<MouseEvent>} e - The mouse event object from Konva.
 * @param {import('@image_tagging_types').ImageTaggingState} state - The image tagging state for this canvas.
 * @returns {Promise<void>}
 */
export async function handleMouseDown(e, state) {
  const stage = state.canvas.stage;
  const layer = state.canvas.layer;
  if (!stage || !layer) {
    Logger.error("No canvas!");
    return;
  }
  // Middle mouse: start panning
  if (e.evt.button === 1) {
    isPanning = true;
    lastPanPos = { x: e.evt.clientX, y: e.evt.clientY };
    stage.container().style.cursor = 'move';
    e.evt.preventDefault();
    return;
  }

  if (e.evt.button !== 0 || isPanning) return;

  const currentTool = state.currentTool;
  if (currentTool === 'select') return;

  const pos = getPointerPosition(state);
  if (!pos) return;

  startPos = pos;

  /** @type {import('@app_types').RuntimeBboxInfo} */
  const emptyBbox = { x: pos.x, y: pos.y, width: 1, height: 1, classUuid: state.currentClassUuid, uuid: generateUUID() };
  pendingDraggedBbox = new CanvasBboxGroup(emptyBbox, state);
  if (pendingDraggedBbox) {
    state.updateCanvasBbox(pendingDraggedBbox);
    layer.add(pendingDraggedBbox);
  }
}

/**
 * @param {Konva.KonvaEventObject<MouseEvent>} e - The mouse event object from Konva.
 * @param {import('@image_tagging_types').ImageTaggingState} state - The image tagging state for this canvas.
 * @returns {Promise<void>}
 */
export async function handleMouseMove(e, state) {
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

  const pos = getPointerPosition(state);
  if (!pos) return;

  const currentTool = state.currentTool;
  if (currentTool != 'select') {
    createCrosshairLines(state);
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
  const newX = dx < 0 ? pos.x : startPos.x;
  const newY = dy < 0 ? pos.y : startPos.y;
  const newWidth = Math.abs(dx);
  const newHeight = Math.abs(dy);
  pendingDraggedBbox.updatePosition({ x: newX, y: newY, width: newWidth, height: newHeight });
  //layer.batchDraw();
}

/**
 * @param {Konva.KonvaEventObject<MouseEvent>} e - The mouse event object from Konva.
 * @param {import('@image_tagging_types').ImageTaggingState} state - The image tagging state for this canvas.
 * @returns {Promise<void>}
 */
export async function handleMouseUp(e, state) {
  const currentTool = state.currentTool;
  const isSelectTool = currentTool === 'select';
  state.canvas.stage.container().style.cursor =
    isSelectTool ? 'default' : 'crosshair';

  if (e.evt.button === 1) {
    isPanning = false;
    return;
  }

  if (pendingDraggedBbox) {
    const box = pendingDraggedBbox.metadata.rect;

    if (box.width() < DOUBLE_CLICK_DISTANCE_PX || box.height() < DOUBLE_CLICK_DISTANCE_PX) {
      state.removeBboxGroup(pendingDraggedBbox);
      pendingDraggedBbox.destroy();
    } else {
      events.publish(EventTypes.CANVAS_BBOX_CLICKED, {
        bboxId: pendingDraggedBbox.metadata.runtimeBboxInfo.uuid
      });
    }

    state.canvas.layer.draw();
    pendingDraggedBbox = null;
  }

  const pos = getPointerPosition(state);
  const hitGroup = findGroupAtPoint(state, pos);

  const now = Date.now();
  const timeDelta = now - _lastClickTime;
  const clickDist = _lastClickPos ? Math.hypot(pos.x - _lastClickPos.x, pos.y - _lastClickPos.y) : null;
  const isDoubleClick = (clickDist != null && timeDelta < DOUBLE_CLICK_TIME_MS && clickDist < DOUBLE_CLICK_DISTANCE_PX);
  _lastClickTime = now;
  _lastClickPos = pos;

  if (hitGroup && hitGroup instanceof CanvasBboxGroup) {
    const selectedUuid = hitGroup.metadata.runtimeBboxInfo.uuid;
    if (isDoubleClick) {
      if (!isSelectTool) {
        hitGroup.updateMetadata({ classUuid: state.currentClassUuid });
      } else {
        state.setSelectedBboxUuid(selectedUuid);
        events.publish(EventTypes.CANVAS_BBOX_DOUBLE_CLICKED, {
          bboxId: selectedUuid
        });
      }
    } else {
      state.setSelectedBboxUuid(selectedUuid);
      events.publish(EventTypes.CANVAS_BBOX_CLICKED, { bboxId: selectedUuid });
    }
  }


}

/**
 * @param {Konva.KonvaEventObject<MouseEvent>} e - The mouse event object from Konva.
 * @param {import('@image_tagging_types').ImageTaggingState} state - The image tagging state for this canvas.
 * @returns {Promise<void>}
 */
export async function handleWheel(e, state) {
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

/**
 * @param {Konva.KonvaEventObject<MouseEvent>} e - The mouse event object from Konva.
 * @param {import('@image_tagging_types').ImageTaggingState} state - The image tagging state for this canvas.
 * @returns {Promise<void>}
 */
export function handleClick(e, state) {
  const stage = state.canvas.stage;
  const layer = state.canvas.layer;
  if (e.target === stage) {
    clearSelection();
    layer.draw();
  }
}

/**
 * @param {Konva.KonvaEventObject<MouseEvent>} e - The mouse event object from Konva.
 * @param {import('@image_tagging_types').ImageTaggingState} state - The image tagging state for this canvas.
 * @returns {Promise<void>}
 */
export function handleContextMenu(e, state) {
  const layer = state.canvas.layer;
  e.evt.preventDefault();
  clearSelection();
  if (state.currentTool === 'select') {
    setTool(state, `bbox:`)
  } else {
    setTool(state, 'select');
  }
  layer.draw();
}
