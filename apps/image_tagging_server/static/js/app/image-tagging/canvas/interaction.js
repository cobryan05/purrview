import { Logger } from '../../../ui/utils/logging.js';
import { generateUUID } from '../../../ui/utils/utils.js';
import { BboxView } from '/app-static/js/app/image-tagging/canvas/views/bboxView.js';
import { createUIBBbox } from '/app-static/js/shared/domain/image/mapper.js';
import { clearSelection, setTool, parseToolUuid } from './tools.js';
import { fetchLabelByUuid } from '/app-static/js/shared/api/labels.js';
import { events, EventTypes } from '/app-static/js/shared/events/index.js';
import { UNVERIFIED_TAG_UUID, VERIFICATION_TAG_UUIDS } from '/app-static/js/shared/api/review.js';
import { setCanvasViewport } from './manager.js';
import { screenToImage } from './viewport.js';

/**
 * Mutates uiBBox.tags in-place: if no verification tag is present, adds UNVERIFIED.
 * Bboxes that already carry any verification tag (UNVERIFIED, VERIFIED, FALSE_POS) are left untouched.
 */
function applyDefaultVerificationTag(uiBBox) {
  const tags = uiBBox.tags ?? [];
  // Skip if already carrying any verification tag.
  if (tags.some(t => VERIFICATION_TAG_UUIDS.has(t.uuid))) return;
  uiBBox.tags = [...tags, { uuid: UNVERIFIED_TAG_UUID, icon: null }];
}

const DOUBLE_CLICK_TIME_MS = 400;
const DOUBLE_CLICK_DISTANCE_PX = 10;
const BBOX_MIN_PX = 10;

/**
 * Creates crosshair lines on the canvas.
 * @param {import('@image_tagging_types').CanvasState} canvasState - The canvas state.
 */
function createCrosshairLines(canvasState) {
  const ist = canvasState.interactionState;
  if (ist.crosshairV && ist.crosshairH) return;

  const layer = canvasState.layer;
  const stage = canvasState.stage;
  const width = stage.width();
  const height = stage.height();

  ist.crosshairV = new Konva.Line({
    points: [0, 0, 0, height],
    stroke: 'rgba(127,127,127,0.5)',
    strokeWidth: 1,
    dash: [4, 4],
    listening: false,
  });

  ist.crosshairH = new Konva.Line({
    points: [0, 0, width, 0],
    stroke: 'rgba(0,0,0,0.3)',
    strokeWidth: 1,
    dash: [4, 4],
    listening: false,
  });

  layer.add(ist.crosshairV, ist.crosshairH);
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
  const layer = runtime.canvas.layer;
  const ist = runtime.canvas.interactionState;

  if (!stage || !layer) {
    Logger.error("No canvas!");
    return;
  }

  // Middle mouse: start panning
  if (e.evt.button === 1) {
    ist.isPanning = true;
    ist.lastPanPos = { x: e.evt.clientX, y: e.evt.clientY };
    stage.container().style.cursor = 'move';
    e.evt.preventDefault();
    return;
  }

  // Left click on empty canvas in select mode: pan
  if (e.evt.button === 0 && runtime.tool === 'select') {
    const pos = getPointerPosition(runtime.canvas);
    if (pos && !findGroupAtPoint(runtime, pos)) {
      ist.isPanning = true;
      ist.lastPanPos = { x: e.evt.clientX, y: e.evt.clientY };
      stage.container().style.cursor = 'grabbing';
      return;
    }
  }

  if (e.evt.button !== 0 || ist.isPanning || runtime.tool === 'select') return;

  const pos = getPointerPosition(runtime.canvas);
  if (!pos) return;

  ist.startPos = pos;
}

/**
 * @param {Konva.KonvaEventObject<MouseEvent>} e - The mouse event object from Konva.
 * @param {import('@image_tagging_types').ImageTaggingRuntime} runtime - The image tagging runtime for this canvas.
 * @returns {Promise<void>}
 */
export async function handleMouseMove(e, runtime) {
  const stage = runtime.canvas.stage;
  const layer = runtime.canvas.layer;
  const ist = runtime.canvas.interactionState;

  if (ist.isPanning) {
    const dx = e.evt.clientX - ist.lastPanPos.x;
    const dy = e.evt.clientY - ist.lastPanPos.y;
    ist.lastPanPos = { x: e.evt.clientX, y: e.evt.clientY };
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
    ist.crosshairV.points([pos.x, 0, pos.x, stage.height()]);
    ist.crosshairH.points([0, pos.y, stage.width(), pos.y]);
    ist.crosshairV.show();
    ist.crosshairH.show();
    layer.batchDraw();
  } else {
    if (ist.crosshairV) ist.crosshairV.hide();
    if (ist.crosshairH) ist.crosshairH.hide();
    layer.batchDraw();
  }

  if (!ist.startPos) return;

  const dx = pos.x - ist.startPos.x;
  const dy = pos.y - ist.startPos.y;
  if (!ist.pendingDraggedBbox && (dx * dx + dy * dy) > BBOX_MIN_PX ** 2) {
    const viewport = runtime.canvas?.viewport;
    if (!viewport) return;
    const { uuid: labelUuid } = parseToolUuid(runtime.tool);
    let label = undefined;
    if (labelUuid) {
      const labelData = await fetchLabelByUuid(labelUuid);
      if (labelData) {
        label = { uuid: labelUuid, text: labelData.name, color: labelData.color };
      }
    }
    const uiBBox = createUIBBbox({ uuid: generateUUID(), label });
    ist.pendingDraggedBbox = new BboxView(uiBBox, viewport);
    layer.add(ist.pendingDraggedBbox);
  }

  if (ist.pendingDraggedBbox) {
    ist.pendingDraggedBbox.updatePosition({
      x: dx < 0 ? pos.x : ist.startPos.x,
      y: dy < 0 ? pos.y : ist.startPos.y,
      width: Math.abs(dx),
      height: Math.abs(dy),
    });
  }
}

/**
 * @param {Konva.KonvaEventObject<MouseEvent>} e - The mouse event object from Konva.
 * @param {import('@image_tagging_types').ImageTaggingRuntime} runtime - The image tagging runtime for this canvas.
 * @returns {Promise<void>}
 */
export async function handleMouseUp(e, runtime) {
  if (!runtime.canvas) return;
  const ist = runtime.canvas.interactionState;
  const isSelectTool = runtime.tool === 'select';
  runtime.canvas.stage.container().style.cursor =
    isSelectTool ? 'default' : 'crosshair';

  if (e.evt.button === 1 || ist.isPanning) {
    ist.isPanning = false;
    return;
  }
  ist.startPos = null;
  if (ist.pendingDraggedBbox) {
    const w = ist.pendingDraggedBbox.width();
    const h = ist.pendingDraggedBbox.height();

    if (w < BBOX_MIN_PX || h < BBOX_MIN_PX) {
      ist.pendingDraggedBbox.destroy();
    } else {
      // normalize back to 0-1 image space before registering
      const vp = runtime.canvas.viewport;
      const sw = vp.imageWidth * vp.scale;
      const sh = vp.imageHeight * vp.scale;
      const uiBBox = ist.pendingDraggedBbox.uiBBox;
      uiBBox.x = (ist.pendingDraggedBbox.x() - vp.offsetX) / sw;
      uiBBox.y = (ist.pendingDraggedBbox.y() - vp.offsetY) / sh;
      uiBBox.width = w / sw;
      uiBBox.height = h / sh;
      runtime.state.image.bboxes.set(uiBBox.uuid, uiBBox);
      applyDefaultVerificationTag(uiBBox);
      ist.pendingDraggedBbox.refreshIcons(uiBBox.tags);
      events.publish(EventTypes.CANVAS_BBOX_CLICKED, { bboxId: uiBBox.uuid, runtimeId: runtime.id });
      events.publish(EventTypes.CANVAS_BBOXES_CHANGED, { runtimeId: runtime.id });
    }

    runtime.canvas.layer.draw();
    ist.pendingDraggedBbox = null;
  }

  const pos = getPointerPosition(runtime.canvas);
  const hitGroup = findGroupAtPoint(runtime, pos);

  const now = Date.now();
  const timeDelta = now - ist.lastClickTime;
  const clickDist = ist.lastClickPos ? Math.hypot(pos.x - ist.lastClickPos.x, pos.y - ist.lastClickPos.y) : null;
  const isDoubleClick = (clickDist != null && timeDelta < DOUBLE_CLICK_TIME_MS && clickDist < DOUBLE_CLICK_DISTANCE_PX);
  ist.lastClickTime = now;
  ist.lastClickPos = pos;

  if (hitGroup instanceof BboxView && isDoubleClick) {
    const { uuid: labelUuid } = parseToolUuid(runtime.tool);
    if (runtime.tool === 'select') {
      events.publish(EventTypes.CANVAS_BBOX_DOUBLE_CLICKED, { bboxId: hitGroup.id(), bboxView: hitGroup, runtimeId: runtime.id });
    } else if (labelUuid) {
      const labelData = await fetchLabelByUuid(labelUuid);
      if (labelData) {
        hitGroup.uiBBox.label = { uuid: labelUuid, text: labelData.name, color: labelData.color };
        hitGroup.updateAppearance(labelData.name, labelData.color);
        applyDefaultVerificationTag(hitGroup.uiBBox);
        hitGroup.refreshIcons(hitGroup.uiBBox.tags);
        events.publish(EventTypes.CANVAS_BBOXES_CHANGED, { runtimeId: runtime.id });
      }
    }
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
  const ist = runtime.canvas.interactionState;

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
    if (ist.selectedBbox && ist.selectedBbox !== bboxGroup) {
      ist.selectedBbox.off('transformend.transformer');
      ist.selectedBbox = null;
    }
    ist.selectedBbox = bboxGroup;
    transformer.nodes([bboxGroup]);
    transformer.moveToTop();
    bboxGroup.on('transformend.transformer', () => transformer.forceUpdate());
    events.publish(EventTypes.CANVAS_BBOX_CLICKED, { bboxId: bboxGroup.id(), runtimeId: runtime.id });
  } else if (e.target === stage) {
    ist.selectedBbox?.off('transformend.transformer');
    ist.selectedBbox = null;
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
  const { stage, layer } = runtime.canvas;
  e.evt.preventDefault();

  let node = e.target;
  while (node && node !== stage) {
    if (node instanceof BboxView) return;
    node = node.getParent();
  }

  clearSelection();
  if (runtime.tool === 'select') {
    setTool(runtime, 'bbox:');
  } else {
    setTool(runtime, 'select');
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