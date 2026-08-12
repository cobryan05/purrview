import { Logger } from '../../../ui/utils/logging.js';
import { generateUUID } from '../../../ui/utils/utils.js';
import { BboxView } from '/app-static/js/app/image-tagging/canvas/views/bboxView.js';
import { createUIBBbox } from '/app-static/js/shared/domain/image/mapper.js';
import { clearSelection, setTool, parseToolUuid, isDrawTool, isTagTool, syncBboxDraggable } from './tools.js';
import { fetchLabelByUuid } from '/app-static/js/shared/api/labels.js';
import { fetchTagByUuid } from '/app-static/js/shared/api/tags.js';
import { events, EventTypes } from '/app-static/js/shared/events/index.js';
import { UNVERIFIED_TAG_UUID, VERIFIED_TAG_UUID, VERIFICATION_TAG_UUIDS } from '/app-static/js/shared/api/review.js';
import { setCanvasViewport, visibleContainerHeight } from './manager.js';
import { screenToImage } from './viewport.js';
import { openContextMenuForObjects } from '/app-static/js/ui/contextMenu/registry.js';
import { makeCanvasBboxAdapter } from './contextTargets.js';

/**
 * Mutates uiBBox.tags in-place: if no verification tag is present, assigns one based on
 * whether the bbox has a label (VERIFIED) or not (UNVERIFIED).
 */
function applyDefaultVerificationTag(uiBBox) {
  const tags = uiBBox.tags ?? [];
  // Skip if already carrying any verification tag.
  if (tags.some(t => VERIFICATION_TAG_UUIDS.has(t.uuid))) return;
  if (uiBBox.label?.uuid) {
    uiBBox.tags = [...tags, { uuid: VERIFIED_TAG_UUID, icon: null }];
  } else {
    uiBBox.tags = [...tags, { uuid: UNVERIFIED_TAG_UUID, icon: null }];
  }
}

/**
 * Strips all verification tags from a bbox in-place and adds VERIFIED.
 * Call this whenever a human assigns a label to a bbox.
 */
export function applyVerifiedTag(uiBBox) {
  uiBBox.tags = [
    ...(uiBBox.tags ?? []).filter(t => !VERIFICATION_TAG_UUIDS.has(t.uuid)),
    { uuid: VERIFIED_TAG_UUID, icon: null },
  ];
}

/**
 * Toggle a tag on a bbox (used by the canvas "tag tool" on double-click).
 * Adding a verification-group tag removes any other verification tag first.
 * @param {import('@canvas_types').BboxView} bboxView
 * @param {string} tagUuid
 * @param {import('@image_tagging_types').ImageTaggingRuntime} runtime
 */
export async function toggleBboxTag(bboxView, tagUuid, runtime) {
  const tag = await fetchTagByUuid(tagUuid);
  if (!tag) return;
  const tags = bboxView.uiBBox.tags ?? [];
  const has = tags.some(t => t.uuid === tagUuid);
  let next;
  if (has) {
    next = tags.filter(t => t.uuid !== tagUuid);
  } else {
    const base = VERIFICATION_TAG_UUIDS.has(tagUuid)
      ? tags.filter(t => !VERIFICATION_TAG_UUIDS.has(t.uuid))
      : tags;
    next = [...base, { uuid: tagUuid, icon: tag.icon ?? null }];
  }
  bboxView.uiBBox.tags = next;
  bboxView.refreshIcons(next);
  runtime.pushState();
  events.publish(EventTypes.CANVAS_BBOXES_CHANGED, { runtimeId: runtime.id });
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
/**
 * Return every bbox group under the pointer, smallest-area first so inner/stacked
 * boxes stay reachable.
 * @param {import('@image_tagging_types').ImageTaggingRuntime} runtime
 * @param {{x: number, y: number}} pos
 * @returns {import('@canvas_types').BboxView[]}
 */
function findGroupsAtPoint(runtime, pos) {
  const layer = runtime.canvas.layer;
  const children = layer.getChildren(node => node.name() === 'bbox');
  const hits = [];
  for (const group of children) {
    const rect = group.getClientRect({ relativeTo: layer });
    if (pos.x >= rect.x && pos.x <= rect.x + rect.width &&
      pos.y >= rect.y && pos.y <= rect.y + rect.height) {
      hits.push({ group, area: rect.width * rect.height });
    }
  }
  hits.sort((a, b) => a.area - b.area);
  return hits.map(h => h.group);
}

/** Return the smallest bbox group under the pointer, if any. */
function findGroupAtPoint(runtime, pos) {
  return findGroupsAtPoint(runtime, pos)[0] ?? null;
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

  // Left click on empty canvas with a non-draw tool: pan
  if (e.evt.button === 0 && !isDrawTool(runtime.tool)) {
    const pos = getPointerPosition(runtime.canvas);
    if (pos && !findGroupAtPoint(runtime, pos)) {
      ist.isPanning = true;
      ist.lastPanPos = { x: e.evt.clientX, y: e.evt.clientY };
      stage.container().style.cursor = 'grabbing';
      return;
    }
  }

  // Draw tools always start a new bbox, even when the pointer is over an existing one.
  if (e.evt.button !== 0 || ist.isPanning || !isDrawTool(runtime.tool)) return;

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
  if (isDrawTool(currentTool)) {
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
  runtime.canvas.stage.container().style.cursor =
    isDrawTool(runtime.tool) ? 'crosshair' : isTagTool(runtime.tool) ? 'pointer' : 'default';

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
      syncBboxDraggable(runtime);
      // Drawing uses manual mouse positioning, not Konva drag — 'dragend' does not fire here.
      runtime.pushState();
      events.publish(EventTypes.CANVAS_BBOX_CLICKED, { bboxId: uiBBox.uuid, bboxView: ist.pendingDraggedBbox, runtimeId: runtime.id });
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
  ist.lastWasDoubleClick = isDoubleClick;

  // Select-tool selection (including stacked-bbox cycling) is handled in
  // handleClick, which fires right after this on the same gesture.
  if (hitGroup instanceof BboxView && isDoubleClick) {
    const { tool: toolKind, uuid: toolUuid } = parseToolUuid(runtime.tool);
    if (toolKind === 'bbox' && toolUuid) {
      const labelData = await fetchLabelByUuid(toolUuid);
      if (labelData) {
        hitGroup.uiBBox.label = { uuid: toolUuid, text: labelData.name, color: labelData.color };
        hitGroup.updateAppearance(labelData.name, labelData.color);
        applyVerifiedTag(hitGroup.uiBBox);
        hitGroup.refreshIcons(hitGroup.uiBBox.tags);
        // Labeling is a pure data mutation — no Konva drag/transform event fires for this.
        runtime.pushState();
        events.publish(EventTypes.CANVAS_BBOXES_CHANGED, { runtimeId: runtime.id });
      }
    } else if (toolKind === 'tag' && toolUuid) {
      await toggleBboxTag(hitGroup, toolUuid, runtime);
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
 * Make `bboxGroup` the canvas's selected bbox: attach the transformer, record
 * it as the selection, and wire the resize handler that keeps a transform from
 * being lost.
 *
 * ⚠️ The ONLY supported way to put the transformer on a bbox. Calling
 * `transformer.nodes([view])` directly renders live resize anchors while
 * skipping this bookkeeping, so dragging them mutates the box without ever
 * marking the canvas dirty — the edit looks applied, the Save button stays
 * clean, and it is discarded on close. That is exactly the bug the review
 * modal's open-on-a-bbox highlight shipped with.
 *
 * @param {import('@image_tagging_types').ImageTaggingRuntime} runtime
 * @param {import('./views/bboxView.js').BboxView} bboxGroup
 */
export function selectBboxView(runtime, bboxGroup) {
  if (!runtime.canvas || !bboxGroup) return;
  const { transformer } = runtime.canvas;
  const ist = runtime.canvas.interactionState;

  if (ist.selectedBbox && ist.selectedBbox !== bboxGroup) {
    ist.selectedBbox.off('transformend.transformer');
  }
  ist.selectedBbox = bboxGroup;
  transformer.nodes([bboxGroup]);
  transformer.moveToTop();

  // Konva .on() with a namespace appends rather than replaces, so re-selecting
  // the same bbox would stack a duplicate handler (N undo pushes per resize).
  // Clear unconditionally before wiring.
  bboxGroup.off('transformend.transformer');

  // Resize handler: Konva fires transformend from the Transformer node (not the
  // BboxView group), so the stage-level bubbling guard won't catch it. Wire it
  // directly on the group so we reliably get pushState + dirty notification.
  bboxGroup.on('transformend.transformer', () => {
    transformer.forceUpdate();
    runtime.pushState();
    events.publish(EventTypes.CANVAS_BBOXES_CHANGED, { runtimeId: runtime.id });
  });
}

/**
 * @param {Konva.KonvaEventObject<MouseEvent>} e - The mouse event object from Konva.
 * @param {import('@image_tagging_types').ImageTaggingRuntime} runtime - The image tagging runtime for this canvas.
 * @returns {Promise<void>}
 */
export function handleClick(e, runtime) {
  if (!runtime.canvas) return;
  const { layer, transformer } = runtime.canvas;
  const ist = runtime.canvas.interactionState;

  // Selection (and stacked-bbox cycling) only happens with the select tool.
  if (runtime.tool !== 'select') return;

  const pos = getPointerPosition(runtime.canvas);
  const groups = pos ? findGroupsAtPoint(runtime, pos) : [];

  // Clicked empty space → clear selection.
  if (!groups.length) {
    ist.selectedBbox?.off('transformend.transformer');
    ist.selectedBbox = null;
    ist.cycleAnchor = null;
    transformer.nodes([]);
    layer.batchDraw();
    return;
  }

  // A single click selects the smallest box under the pointer. Double-clicking
  // at ~the same point rotates to the next box in the overlapping stack;
  // clicking elsewhere starts over from the smallest. The index persists across
  // the two clicks of a double-click so each double-click advances by exactly one.
  const sameSpot = ist.cycleAnchor &&
    Math.hypot(pos.x - ist.cycleAnchor.x, pos.y - ist.cycleAnchor.y) < DOUBLE_CLICK_DISTANCE_PX;
  if (!sameSpot) {
    ist.cycleIndex = 0;
  } else if (ist.lastWasDoubleClick) {
    ist.cycleIndex = (ist.cycleIndex + 1) % groups.length;
  } else {
    ist.cycleIndex = ist.cycleIndex % groups.length;
  }
  ist.cycleAnchor = pos;

  const bboxGroup = groups[ist.cycleIndex];

  // Capture whether this exact box was already selected BEFORE this click.
  // Used below to decide between tool-switch and inspector-open on double-click.
  const wasAlreadySelected = (ist.selectedBbox === bboxGroup);

  selectBboxView(runtime, bboxGroup);

  events.publish(EventTypes.CANVAS_BBOX_CLICKED, { bboxId: bboxGroup.id(), bboxView: bboxGroup, runtimeId: runtime.id });

  // Double-click action:
  //   • Already-selected box in select mode → switch to that box's label tool
  //     so the user can immediately relabel / verify with another double-click.
  //   • All other cases → open the inspector.
  if (ist.lastWasDoubleClick) {
    const labelUuid = bboxGroup.uiBBox?.label?.uuid;
    if (wasAlreadySelected && labelUuid) {
      setTool(runtime, `bbox:${labelUuid}`);
    } else {
      events.publish(EventTypes.CANVAS_BBOX_DOUBLE_CLICKED, { bboxId: bboxGroup.id(), bboxView: bboxGroup, runtimeId: runtime.id });
    }
  }

  layer.batchDraw();
}

/**
 * @param {Konva.KonvaEventObject<MouseEvent>} e - The mouse event object from Konva.
 * @param {import('@image_tagging_types').ImageTaggingRuntime} runtime - The image tagging runtime for this canvas.
 * @returns {Promise<void>}
 */
export async function handleContextMenu(e, runtime) {
  const { stage } = runtime.canvas;
  e.evt.preventDefault();

  // Find bbox under the right-click, if any
  let node = e.target;
  let bboxGroup = null;
  while (node && node !== stage) {
    if (node instanceof BboxView) { bboxGroup = node; break; }
    node = node.getParent();
  }

  // Only right-clicking a bbox opens a context menu here. Right-clicking empty canvas
  // space intentionally falls straight through to the tool-toggle behavior below; the
  // canvas surface menu (similar/dissimilar search) is reached via the toolbar button
  // instead (see toolbar.js), not via right-click.
  if (bboxGroup) {
    const obj = { adapters: { bbox: makeCanvasBboxAdapter(bboxGroup, runtime) } };
    const opened = await openContextMenuForObjects([obj], { x: e.evt.clientX, y: e.evt.clientY });
    if (opened) return;
  }

  // FALLBACK: original tool-toggle behavior (unchanged)
  clearSelection();
  const bboxLabelUuid = bboxGroup?.uiBBox?.label?.uuid;
  if (bboxLabelUuid) {
    setTool(runtime, runtime.tool === `bbox:${bboxLabelUuid}` ? 'select' : `bbox:${bboxLabelUuid}`);
  } else {
    setTool(runtime, runtime.tool === 'select' ? 'bbox' : 'select');
  }
  runtime.canvas.layer.draw();
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

  await setCanvasViewport(canvas, canvas.viewport, { width, height: visibleContainerHeight(container) });
}