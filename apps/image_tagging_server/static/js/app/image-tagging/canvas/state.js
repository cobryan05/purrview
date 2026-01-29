/**
 * @typedef {import('@canvas_types').CanvasState} CanvasState
 */


/** @type {CanvasState} */
export const canvasState = {
  stage: null,
  layer: null,
  transformer: null,
  bboxes: new Map(),

  init(container) {
    this.stage = new Konva.Stage({
      container: container.id,
      width: container.clientWidth,
      height: container.clientHeight,
    });

    this.transformer = new Konva.Transformer({
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

    this.layer = new Konva.Layer();
    this.layer.add(this.transformer);
    this.stage.add(this.layer);

    this.bboxes = new Map();
  },
};
