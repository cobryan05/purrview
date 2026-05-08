import { computeViewport } from './viewport.js';

/**
 * @param {import('@image_tagging_types').CanvasState} canvas
 * @param {import('@image_tagging_types').UIImage} uiImage
 */
export async function setCanvasImage(canvas, uiImage) {
  const { stage, layer, transformer } = canvas;

  // 1. Internal Cleanup
  transformer.nodes([]);
  layer.find('.annotation').forEach(n => n.destroy());
  layer.find('.background-image').forEach(n => n.destroy());

  const konvaImg = new Konva.Image({
    image: uiImage.img,
    name: 'background-image',
    width: stage.width(),
    height: stage.height()
  });

  layer.add(konvaImg);
  konvaImg.moveToBottom();

  // // 3. Add Bboxes
  // uiImage.bboxes.forEach(bbox => {
  //     if (bbox.view) layer.add(bbox.view);
  // });


  const viewport = computeViewport(
    canvas.container.clientWidth,
    canvas.container.clientHeight,
    uiImage.img.width,
    uiImage.img.height
  );
  canvas.backgroundImage = konvaImg;
  setCanvasViewport(canvas, viewport, { width: canvas.container.clientWidth, height: canvas.container.clientHeight });
  // layer.batchDraw();
}

/**
* @param {import('@image_tagging_types').CanvasState} canvas
* @param {import('@image_tagging_types').ViewportTransform} viewport
* @param { { width: number; height: number } } size
*/
export async function setCanvasViewport(canvas, viewport, size) {
  const { stage, layer } = canvas;

  if (!stage || !viewport) {
    return;
  }

  if (size) {
    stage.width(size.width);
    stage.height(size.height);
  }

  if (canvas.backgroundImage) {
    canvas.backgroundImage.position({
      x: viewport.offsetX,
      y: viewport.offsetY,
    });

    canvas.backgroundImage.size({
      width: viewport.imageWidth * viewport.scale,
      height: viewport.imageHeight * viewport.scale,
    });
  }

  layer.batchDraw();

  // IMPORTANT: clone to avoid reference leaks
  canvas.viewport = { ...viewport };
}