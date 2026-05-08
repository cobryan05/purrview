
/**
  @param {HTMLElement} container
  @returns {import('@image_tagging_types').CanvasState}
 */
export function createCanvasState(container) {
  const stage = new Konva.Stage({
    container: container.id,
    width: container.clientWidth,
    height: container.clientHeight,
  });

  const transformer = new Konva.Transformer({
    rotateEnabled: false,
    borderStroke: 'yellow',
    borderDash: [4, 4],
    anchorStroke: 'red',
    anchorFill: 'white',
    anchorSize: 10,
    anchorCornerRadius: 5,
    enabledAnchors: ['top-left', 'top-right', 'bottom-left', 'bottom-right'],
    ignoreStroke: true,
    keepRatio: false,
    boundBoxFunc: (oldBox, newBox) => {
      const minSize = 10;
      if (Math.abs(newBox.width) < minSize || Math.abs(newBox.height) < minSize) return oldBox;
      if (newBox.width < 0 || newBox.height < 0) return oldBox;
      return newBox;
    },
  });

  const layer = new Konva.Layer();
  layer.add(transformer);
  stage.add(layer);

  /** @type {import('@image_tagging_types').ViewportTransform} */
  const viewport = { offsetX: 0, offsetY: 0, scale: 1.0, imageHeight: 0, imageWidth: 0 };

  /** @type {import('@image_tagging_types').CanvasState} */
  const canvasState = {
    container,
    stage, layer, transformer, viewport
  };

  return canvasState;
};
