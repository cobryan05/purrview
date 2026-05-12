
import { computeViewport } from './viewport.js';
import { BboxView } from './views/bboxView.js';

/**
 * @param {import('@image_tagging_types').CanvasState} canvas
 * @param {import('@image_tagging_types').UIImage} uiImage
 */
export async function setCanvasImage(canvas, uiImage) {
  const { stage, layer, transformer } = canvas;

  // Clean up existing
  transformer.nodes([]);
  layer.find('.annotation').forEach(n => n.destroy());
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

  uiImage.bboxes.forEach(bbox => {
    const bboxView = new BboxView(bbox, viewport);
    layer.add(bboxView);
  });

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
  const annotations = layer.find('.annotation');
  const fullScaledWidth = viewport.imageWidth * viewport.scale;
  const fullScaledHeight = viewport.imageHeight * viewport.scale;
  annotations.forEach((node) => {
    const uiBBox = node.uiBBox;
    if (!uiBBox) return;

    node.setAttrs({
      x: (uiBBox.x * fullScaledWidth) + viewport.offsetX,
      y: (uiBBox.y * fullScaledHeight) + viewport.offsetY,
      width: uiBBox.width * fullScaledWidth,
      height: uiBBox.height * fullScaledHeight
    });
    if (typeof node.updateSize === 'function') {
      node.updateSize(node.width(), node.height());
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