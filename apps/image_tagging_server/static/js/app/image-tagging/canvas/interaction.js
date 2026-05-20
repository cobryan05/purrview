import { Logger } from '../../../ui/utils/logging.js';
import { generateUUID } from '../../../ui/utils/utils.js';
import { BboxView } from '/app-static/js/app/image-tagging/canvas/views/bboxView.js';
import { createUIBBbox } from '/app-static/js/shared/domain/image/mapper.js';
import { clearSelection, setTool } from './tools.js';
import { events, EventTypes } from '/app-static/js/shared/events/index.js';
import { setCanvasViewport } from './manager.js';
import { screenToImage } from './viewport.js';

/** @type {import('@canvas_types').BboxView | null} */
let _pendingDraggedBbox = null;
/** @type {import('@canvas_types').BboxView | null} */
let _selectedBbox = null;

const DOUBLE_CLICK_TIME_MS = 400;
const DOUBLE_CLICK_DISTANCE_PX = 10;
const BBOX_MIN_PX = 10;
let _startPos = null;
let _isPanning = false;
let _lastPanPos = null;
let _lastClickTime = 0;
let _lastClickPos = null;
let _crosshairV = null;
let _crosshairH = null;

/**
 * Creates crosshair lines on the canvas.
 * @param {import('@image_tagging_types').CanvasState} canvasState - The canvas state.
 */
function createCrosshairLines(canvasState) {
  if (_crosshairV && _crosshairH) return;

  const layer = canvasState.layer;
  const stage = canvasState.stage;
  const width = stage.width();
  const height = stage.height();

  _crosshairV = new Konva.Line({
    points: [0, 0, 0, height],
    stroke: 'rgba(127,127,127,0.5)',
    strokeWidth: 1,
    dash: [4, 4],
    listening: false,  // don't interfere with events
  });

  _crosshairH = new Konva.Line({
    points: [0, 0, width, 0],
    stroke: 'rgba(0,0,0,0.3)',
    strokeWidth: 1,
    dash: [4, 4],
    listening: false,
  });

  layer.add(_crosshairV, _crosshairH);
}

/**
 * @param {import('@image_tagging_types').CanvasState} canvasState - The canvas state
 * @returns {x, y}
 */
function getPointerPosition(canvasState) {
  const stage = canvasState.stage;
  const pos = stage.getPointerPosition();
  if (!pos) return null;
  return {
    x: (pos.x - stage.x()) / stage.scaleX(),
    y: (pos.y - stage.y()) / stage.scaleY(),
  };
}


/** Return bbox group under pointer, if any
 *
 * @param {import('@image_tagging_types').ImageTaggingRuntime} runtime - The image tagging runtime for this canvas.
 * @param {{x: number, y: number}} pos - The pointer position in canvas coordinates.
 * @returns {import('@canvas_types').BboxView|null} The bbox view under the pointer, or null if none.
 */
function findGroupAtPoint(runtime, pos) {
  const layer = runtime.canvas.layer;
  const children = layer.getChildren(node => node.name() === 'bbox');
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
 * @param {import('@image_tagging_types').ImageTaggingRuntime} runtime - The image tagging runtime for this canvas.
 * @returns {Promise<void>}
 */
export async function handleMouseDown(e, runtime) {
  const stage = runtime.canvas.stage;
  const background = runtime.canvas?.backgroundImage;
  const layer = runtime.canvas.layer;

  if (!stage || !layer) {
    Logger.error("No canvas!");
    return;
  }

  // Middle mouse: start panning
  if (e.evt.button === 1) {
    _isPanning = true;
    _lastPanPos = { x: e.evt.clientX, y: e.evt.clientY };
    stage.container().style.cursor = 'move';
    e.evt.preventDefault();
    return;
  }

  if (e.evt.button !== 0 || _isPanning || runtime.tool === 'select') return;

  const pos = getPointerPosition(runtime.canvas);
  if (!pos) return;

  _startPos = pos;
}

/**
 * @param {Konva.KonvaEventObject<MouseEvent>} e - The mouse event object from Konva.
 * @param {import('@image_tagging_types').ImageTaggingRuntime} runtime - The image tagging runtime for this canvas.
 * @returns {Promise<void>}
 */
export async function handleMouseMove(e, runtime) {
  const stage = runtime.canvas.stage;
  const layer = runtime.canvas.layer;
  if (_isPanning) {
    const dx = e.evt.clientX - _lastPanPos.x;
    const dy = e.evt.clientY - _lastPanPos.y;
    _lastPanPos = { x: e.evt.clientX, y: e.evt.clientY };
    const viewport = runtime.canvas?.viewport;
    if (!viewport) return;

    const pannedViewport = { ...viewport, offsetX: viewport.offsetX + dx, offsetY: viewport.offsetY + dy };
    await setCanvasViewport(runtime.canvas, pannedViewport);
  }

  const pos = getPointerPosition(runtime.canvas);
  if (!pos) return;

  const currentTool = runtime.tool;
  if (currentTool != 'select') {
    createCrosshairLines(runtime.canvas);
    // Position lines at mouse X,Y spanning full height,width
    _crosshairV.points([pos.x, 0, pos.x, stage.height()]);
    _crosshairH.points([0, pos.y, stage.width(), pos.y]);
    _crosshairV.show();
    _crosshairH.show();
    layer.batchDraw();
  } else {
    if (_crosshairV) _crosshairV.hide();
    if (_crosshairH) _crosshairH.hide();
    layer.batchDraw();
  }

  if (!_startPos) return;

  const dx = pos.x - _startPos.x;
  const dy = pos.y - _startPos.y;
  if (!_pendingDraggedBbox && (dx * dx + dy * dy) > BBOX_MIN_PX ** 2) {
    /** @type {import('@web_api').BoundingBoxMetadataModel} */
    const emptyBbox = { x: pos.x, y: pos.y, width: 1, height: 1, labelUuid: runtime.state?.currentLabelUuid, uuid: generateUUID() };
    _pendingDraggedBbox = createUIBBbox({
      x: _startPos.x,
      y: _startPos.y,
      width: dx,
      height: dy,
      labelUuid: runtime.state?.currentLabelUuid,
      uuid: generateUUID()
    });
    if (_pendingDraggedBbox) {
      runtime.state.updateCanvasBboxView(_pendingDraggedBbox);
      layer.add(_pendingDraggedBbox);
    }
  }

  if (_pendingDraggedBbox) {
    const newX = dx < 0 ? pos.x : _startPos.x;
    const newY = dy < 0 ? pos.y : _startPos.y;
    const newWidth = Math.abs(dx);
    const newHeight = Math.abs(dy);
    _pendingDraggedBbox.updatePosition({ x: newX, y: newY, width: newWidth, height: newHeight });
    //layer.batchDraw();
  }
}

/**
 * @param {Konva.KonvaEventObject<MouseEvent>} e - The mouse event object from Konva.
 * @param {import('@image_tagging_types').ImageTaggingRuntime} runtime - The image tagging runtime for this canvas.
 * @returns {Promise<void>}
 */
export async function handleMouseUp(e, runtime) {
  const isSelectTool = runtime.tool === 'select';
  runtime.canvas.stage.container().style.cursor =
    isSelectTool ? 'default' : 'crosshair';

  if (e.evt.button === 1) {
    _isPanning = false;
    return;
  }
  _startPos = null;
  if (_pendingDraggedBbox) {
    const box = _pendingDraggedBbox.metadata.rect;

    if (box.width() < BBOX_MIN_PX || box.height() < BBOX_MIN_PX) {
      state.removeBboxGroup(_pendingDraggedBbox);
      _pendingDraggedBbox.destroy();
    } else {
      events.publish(EventTypes.CANVAS_BBOX_CLICKED, {
        bboxId: _pendingDraggedBbox.metadata.bboxInfo.uuid
      });
    }

    state.canvas.layer.draw();
    _pendingDraggedBbox = null;
  }

  const pos = getPointerPosition(runtime.canvas);
  const hitGroup = findGroupAtPoint(runtime, pos);

  const now = Date.now();
  const timeDelta = now - _lastClickTime;
  const clickDist = _lastClickPos ? Math.hypot(pos.x - _lastClickPos.x, pos.y - _lastClickPos.y) : null;
  const isDoubleClick = (clickDist != null && timeDelta < DOUBLE_CLICK_TIME_MS && clickDist < DOUBLE_CLICK_DISTANCE_PX);
  _lastClickTime = now;
  _lastClickPos = pos;

  if (hitGroup instanceof BboxView && isDoubleClick) {
    events.publish(EventTypes.CANVAS_BBOX_DOUBLE_CLICKED, { bboxId: hitGroup.id() });
  }
}

/**
 * @param {Konva.KonvaEventObject<MouseEvent>} e - The mouse event object from Konva.
 * @param {import('@image_tagging_types').ImageTaggingRuntime} runtime - The image tagging runtime for this canvas.
 * @returns {Promise<void>}
 */
export async function handleWheel(e, runtime) {
  e.evt.preventDefault();
  const { stage, viewport } = runtime.canvas;
  const pointer = stage.getPointerPosition();
  if (!pointer) return;

  const zoomFactor = 1.1;
  const direction = e.evt.deltaY > 0 ? 1 : -1;
  const newScale = direction > 0 ? viewport.scale / zoomFactor : viewport.scale * zoomFactor;

  // mouse-centered zoom
  const imagePoint = screenToImage(pointer.x, pointer.y, viewport);
  const zoomedViewport = {
    ...viewport,
    scale: newScale,
    offsetX: pointer.x - imagePoint.x * newScale,
    offsetY: pointer.y - imagePoint.y * newScale,
  };
  await setCanvasViewport(runtime.canvas, zoomedViewport);
}

/**
 * @param {Konva.KonvaEventObject<MouseEvent>} e - The mouse event object from Konva.
 * @param {import('@image_tagging_types').ImageTaggingRuntime} runtime - The image tagging runtime for this canvas.
 * @returns {Promise<void>}
 */
export function handleClick(e, runtime) {
  if (!runtime.canvas) return;
  const { stage, layer, transformer } = runtime.canvas;

  // Walk up from the click target to find a BboxView group
  let node = e.target;
  let bboxGroup = null;
  while (node && node !== stage) {
    if (node instanceof BboxView) {
      bboxGroup = node;
      break;
    }
    node = node.getParent();
  }

  if (bboxGroup instanceof BboxView) {
    if (_selectedBbox && _selectedBbox !== bboxGroup) {
      _selectedBbox.off('transformend.transformer');
      _selectedBbox = null;
    }
    _selectedBbox = bboxGroup;
    transformer.nodes([bboxGroup]);
    transformer.moveToTop();
    bboxGroup.on('transformend.transformer', () => transformer.forceUpdate());
    events.publish(EventTypes.CANVAS_BBOX_CLICKED, { bboxId: bboxGroup.id() });
  } else if (e.target === stage) {
    _selectedBbox?.off('transformend.transformer');
    _selectedBbox = null;
    transformer.nodes([]);
  }
  layer.batchDraw();
}

/**
 * @param {Konva.KonvaEventObject<MouseEvent>} e - The mouse event object from Konva.
 * @param {import('@image_tagging_types').ImageTaggingRuntime} runtime - The image tagging runtime for this canvas.
 * @returns {Promise<void>}
 */
export function handleContextMenu(e, runtime) {
  const layer = runtime.canvas.layer;
  e.evt.preventDefault();
  clearSelection();
  if (state.currentTool === 'select') {
    setTool(state, `bbox:`)
  } else {
    setTool(state, 'select');
  }
  layer.draw();
}


/**
 * @param {UIEvent} e
 * @param {import('@image_tagging_types').ImageTaggingRuntime} runtime
 * @returns {Promise<void>}
 */
export async function handleResize(e, runtime) {
  const canvas = runtime.canvas;

  if (!canvas) {
    return;
  }

  const { container } = canvas;

  const width = container.clientWidth;
  const height = container.clientHeight;

  if (width === 0 || height === 0) {
    return;
  }

  await setCanvasViewport(canvas, canvas.viewport, { width, height, });
}