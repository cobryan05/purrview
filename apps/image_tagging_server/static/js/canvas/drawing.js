import { getLayer } from './state.js';
import { selectShape } from './selection.js';

// Create a new shape depending on the tool
export function createShape(tool, x, y) {
  const common = {
    stroke: 'red',
    strokeWidth: 2,
    draggable: true,
  };

  let shape;
  if (tool === 'rect') {
    shape = new Konva.Rect({ ...common, x, y, width: 1, height: 1 });
  } else {
    return null;
  }

  // Prevent dragging with middle-click
  shape.on('mousedown', e => {
    if (e.evt.button === 1) shape.draggable(false);
    else shape.draggable(true);
  });

  shape.on('mouseup dragend', () => shape.draggable(true));

  shape.on('click', e => {
    if (e.evt.button !== 0) return;
    e.cancelBubble = true;
    selectShape(shape);
  });

  shape.on('transform', () => {
    const scaleX = shape.scaleX();
    const scaleY = shape.scaleY();
    shape.scaleX(1);
    shape.scaleY(1);
    shape.width(shape.width() * scaleX);
    shape.height(shape.height() * scaleY);
    getLayer().batchDraw();
  });

  return shape;
}
