
import { setupShape } from './shape_utils.js';

export function createShape(tool, x, y, props = {}) {
  const common = {
    stroke: 'red',
    strokeWidth: 2,
    draggable: true,
    ...props,  // allow override
  };

  let shape;
  if (tool === 'rect') {
    shape = new Konva.Rect({ ...common, x, y, width: common.width ?? 1, height: common.height ?? 1 });
  } else {
    return null;
  }

  return setupShape(shape);
}
