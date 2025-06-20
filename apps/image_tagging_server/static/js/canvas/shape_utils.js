import { getLayer } from './state.js';
import { selectShape } from './selection.js';

export function setupShape(shape) {
  shape.name('annotation');
  shape.metadata = shape.metadata || {};

  // Prevent dragging with middle-click
  shape.on('mousedown', e => {
    if (e.evt.button === 1) shape.draggable(false);
    else shape.draggable(true);
  });

  shape.on('mouseup dragend', () => shape.draggable(true));

  // Handle selection logic
  shape.on('click', e => {
    if (e.evt.button !== 0) return;
    e.cancelBubble = true;
    selectShape(shape);
  });

  // Apply transform logic with scale reset
  shape.on('transform', () => {
    const scaleX = shape.scaleX();
    const scaleY = shape.scaleY();
    shape.scaleX(1);
    shape.scaleY(1);

    if (shape instanceof Konva.Rect) {
      shape.width(shape.width() * scaleX);
      shape.height(shape.height() * scaleY);
    }

    getLayer().batchDraw();
  });

  return shape;
}