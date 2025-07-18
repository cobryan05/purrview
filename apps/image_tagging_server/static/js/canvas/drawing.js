import { getLayer, getTransformer, getCurrentTool, getCurrentLabelUuid, getLabelByUuid } from './state.js';
import { selectShape } from './selection.js';
import { generateUUID } from './utils.js';
import { openInspectorTab } from '../sidebar/main.js';

/**
 * Create a bounding box group with a rectangle, label, confidence,
 * and set up all relevant event handlers here.
 */
export function createBoundingBox(x, y, props = {}) {
  const width = props.width ?? 50;
  const height = props.height ?? 50;
  const labelUuid = props.metadata?.labelUuid ?? ''
  const confidence = props.metadata?.confidence;
  const uuid = generateUUID();
  const label = getLabelByUuid(labelUuid) ;
  const labelText = (label?.metadata?.name ?? props.metadata?.label) ?? 'Unknown';
  const group = new Konva.Group({
    x,
    y,
    draggable: true,
    name: 'annotation',
  });

  const color = label?.metadata?.color ?? 'red';

  const rect = new Konva.Rect({
    name: 'box',
    width,
    height,
    stroke: color,
    strokeWidth: 2,
  });

  const bboxText = `${labelText}${confidence != null ? ` (${(confidence * 100).toFixed(1)}%)` : ''}`;

  const text = new Konva.Text({
    name: 'label',
    text: bboxText,
    fontSize: 14,
    fill: color,
    y: -18,
    x: 0,
  });

  group.metadata = {
    label,
    confidence,
    ...props.metadata,
    uuid,
  };

  group.add(rect);
  group.add(text);

  // --- Event handlers setup ---

  // Disable dragging with middle mouse button down
  group.on('mousedown', e => {
    if (e.evt.button === 1) {
      group.draggable(false);
    } else {
      group.draggable(getCurrentTool() === 'select');
    }
  });

  // Restore draggable on mouseup or dragend
  group.on('mouseup dragend', () => group.draggable(true));

  // On click, select the shape and attach transformer only to the rectangle
  group.on('click', e => {
    if (e.evt.button !== 0) {
      return;
    }
    e.cancelBubble = true;

    const transformer = getTransformer();
    transformer.nodes([rect]);
    transformer.moveToTop();

    selectShape(group);
    getLayer().draw();
  });

  group.on('dblclick', () => {
    selectShape(group);
    openInspectorTab();
  });

  function updateBoundingBoxLayout() {
    // Always keep rect at (0,0) in group
    rect.x(0);
    rect.y(0);
    // Keep label at top-left, above the box
    text.x(0);
    text.y(-18);
  }

  rect.on('transform', () => {
    const layer = getLayer();

    const scaleX = rect.scaleX();
    const scaleY = rect.scaleY();

    let newWidth = rect.width() * scaleX;
    let newHeight = rect.height() * scaleY;

    const MIN_SIZE = 10;

    // Clamp sizes to MIN_SIZE, avoid negative or too small values
    newWidth = Math.max(newWidth, MIN_SIZE);
    newHeight = Math.max(newHeight, MIN_SIZE);

    // Compute left and top edge relative to group + rect position + scale
    // rect.x() and rect.y() are relative to group coords
    const rectLeft = rect.x();
    const rectTop = rect.y();

    // Calculate group's new position to always be at the top-left corner of the bounding box
    let newGroupX = group.x() + rectLeft;
    let newGroupY = group.y() + rectTop;

    // Apply new size to rect
    rect.width(newWidth);
    rect.height(newHeight);

    // Update group's position
    group.position({
      x: newGroupX,
      y: newGroupY,
    });

    // Reset scale
    rect.scaleX(1);
    rect.scaleY(1);

    updateBoundingBoxLayout();
    layer.batchDraw();
  });
  updateBoundingBoxLayout();
  return group;
}
