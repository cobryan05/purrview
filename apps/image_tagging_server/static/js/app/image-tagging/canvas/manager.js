
import { computeViewport } from './viewport.js';
import { BboxView } from './views/bboxView.js';
import { clearSelectionState } from './selection.js';
import { renderEnhanced, DEFAULT_ENHANCE_MODE } from './enhance.js';

/**
 * @param {import('@image_tagging_types').CanvasState} canvas
 * @param {import('@image_tagging_types').UIImage} uiImage
 */
export async function setCanvasImage(canvas, uiImage) {
  const { stage, layer } = canvas;

  // Clean up existing background image; bbox cleanup handled by rerenderBboxViews
  layer.find('.background-image').forEach(n => n.destroy());

  // The enhancement mode is a property of the canvas, not of the image, so it
  // carries across file navigation without the user re-picking it each frame.
  canvas.sourceImage = uiImage.img;

  const konvaImg = new Konva.Image({
    image: renderEnhanced(uiImage.img, canvas.enhanceMode ?? DEFAULT_ENHANCE_MODE),
    name: 'background-image',
    width: stage.width(),
    height: stage.height(),
    listening: false,
  });

  layer.add(konvaImg);
  konvaImg.moveToBottom();

  const visibleHeight = visibleContainerHeight(canvas.container);
  const viewport = computeViewport(
    canvas.container.clientWidth,
    visibleHeight,
    uiImage.img.width,
    uiImage.img.height
  );
  canvas.backgroundImage = konvaImg;

  rerenderBboxViews(canvas, uiImage.bboxes, viewport);

  setCanvasViewport(canvas, viewport, { width: canvas.container.clientWidth, height: visibleHeight });
}

/**
* @param {import('@image_tagging_types').CanvasState} canvas
* @param {import('@image_tagging_types').ViewportTransform} viewport
* @param { { width: number; height: number } } size
*/
export async function setCanvasViewport(canvas, viewport, size) {
  const { stage, layer, transformer } = canvas;

  if (!stage || !viewport) {
    return;
  }

  if (size) {
    stage.width(size.width);
    stage.height(size.height);
  }

  // Place background image
  if (canvas.backgroundImage) {
    const imgWidth = viewport.imageWidth * viewport.scale;
    const imgHeight = viewport.imageHeight * viewport.scale;

    canvas.backgroundImage.setAttrs({
      x: viewport.offsetX,
      y: viewport.offsetY,
      width: imgWidth,
      height: imgHeight,
    });

    // Dashed border tracing the image bounds, so it's always obvious where
    // the image edges are even when zoomed/panned away from fit-to-view.
    let border = layer.findOne('.image-border');
    if (!border) {
      border = new Konva.Rect({
        name: 'image-border',
        stroke: '#4a90d9',
        strokeWidth: 1,
        dash: [6, 4],
        listening: false,
      });
      layer.add(border);
    }
    border.setAttrs({
      x: viewport.offsetX,
      y: viewport.offsetY,
      width: imgWidth,
      height: imgHeight,
    });
    // Keep it just above the background image, below bboxes/handles.
    border.moveToBottom();
    border.moveUp();
  }

  // Place bounding boxes
  const annotations = layer.find('.bbox');

  annotations.forEach((bboxView) => {
    if (typeof bboxView.updateViewport === 'function') {
      bboxView.updateViewport(viewport);
    }
  });

  // Sync Transformer
  // This ensures that if a box is selected, the selection handles
  // move with the box during zoom/pan.
  if (transformer && transformer.nodes().length > 0) {
    transformer.forceUpdate();
  }

  layer.batchDraw();

  // clone to avoid reference leaks (?)
  canvas.viewport = { ...viewport };
}

/**
 * Destroys all existing BboxView nodes and rebuilds them from the given bbox map.
 * Used for undo/redo restore and initial image render. Does not touch the background image or viewport.
 * @param {import('@image_tagging_types').CanvasState} canvas
 * @param {Map<string, import('@image_tagging_types').UIBbox>} bboxes
 * @param {import('@image_tagging_types').ViewportTransform} [viewport] - Defaults to canvas.viewport if omitted.
 */
export function rerenderBboxViews(canvas, bboxes, viewport) {
  const { layer, transformer } = canvas;
  const vp = viewport ?? canvas.viewport;
  transformer.nodes([]);
  // Every view below is about to be destroyed, so the selection set must be
  // emptied too — otherwise interactionState holds references to dead nodes
  // and the next delete or copy operates on them.
  clearSelectionState(canvas);
  layer.find('.bbox').forEach(n => n.destroy());
  bboxes.forEach(bbox => layer.add(new BboxView(bbox, vp)));
  layer.batchDraw();
}

/**
 * Returns the visible height of the canvas container after subtracting any
 * overlap with the fixed #bottom-bars element.
 * @param {HTMLElement} container
 * @returns {number}
 */
export function visibleContainerHeight(container) {
  const containerRect = container.getBoundingClientRect();
  const bottomBars = document.getElementById('bottom-bars');
  if (bottomBars) {
    const barsTop = bottomBars.getBoundingClientRect().top;
    const overlap = containerRect.bottom - barsTop;
    if (overlap > 0) return container.clientHeight - overlap;
  }
  return container.clientHeight;
}

/**
 * Reset the canvas viewport to the default fit-to-container zoom.
 * @param {import('@image_tagging_types').CanvasState} canvas
 */
export function resetCanvasZoom(canvas) {
  if (!canvas?.viewport?.imageWidth) return;
  const visibleHeight = visibleContainerHeight(canvas.container);
  const viewport = computeViewport(
    canvas.container.clientWidth,
    visibleHeight,
    canvas.viewport.imageWidth,
    canvas.viewport.imageHeight,
  );
  setCanvasViewport(canvas, viewport, { width: canvas.container.clientWidth, height: visibleHeight });
}
