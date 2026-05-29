
import { computeViewport } from './viewport.js';
import { BboxView } from './views/bboxView.js';

/**
 * @param {import('@image_tagging_types').CanvasState} canvas
 * @param {import('@image_tagging_types').UIImage} uiImage
 */
export async function setCanvasImage(canvas, uiImage) {
  const { stage, layer } = canvas;

  // Clean up existing background image; bbox cleanup handled by rerenderBboxViews
  layer.find('.background-image').forEach(n => n.destroy());

  const konvaImg = new Konva.Image({
    image: uiImage.img,
    name: 'background-image',
    width: stage.width(),
    height: stage.height(),
    listening: false,
  });

  layer.add(konvaImg);
  konvaImg.moveToBottom();

  const viewport = computeViewport(
    canvas.container.clientWidth,
    canvas.container.clientHeight,
    uiImage.img.width,
    uiImage.img.height
  );
  canvas.backgroundImage = konvaImg;

  rerenderBboxViews(canvas, uiImage.bboxes, viewport);

  setCanvasViewport(canvas, viewport, { width: canvas.container.clientWidth, height: canvas.container.clientHeight });
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
    canvas.backgroundImage.setAttrs({
      x: viewport.offsetX,
      y: viewport.offsetY,
      width: viewport.imageWidth * viewport.scale,
      height: viewport.imageHeight * viewport.scale,
    });
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
  layer.find('.bbox').forEach(n => n.destroy());
  bboxes.forEach(bbox => layer.add(new BboxView(bbox, vp)));
  layer.batchDraw();
}
