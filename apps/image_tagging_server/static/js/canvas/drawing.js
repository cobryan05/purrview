import { selectShape } from './selection.js';
import { getCurrentTool, getLayer, getTransformer } from './state.js';
import { generateUUID } from './utils.js';
import { fetchLabelByUuid } from '/app-static/js/ui/utils/index.js';


/**
 * Create a bounding box group with a rectangle, label, confidence,
 * and set up all relevant event handlers here.
 */

export function createBoundingBox(x, y, props = {}) {
  const width = props.width ?? 50;
  const height = props.height ?? 50;
  const labelUuid = props.metadata?.labelUuid ?? '';
  const confidence = props.metadata?.confidence;
  const uuid = generateUUID();

  // Initial placeholder values
  let labelText = props.metadata?.label ?? 'Unknown';
  let color = 'grey';

  const group = new Konva.Group({
    x,
    y,
    draggable: true,
    name: 'annotation',
  });

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
    label: null, // will be filled in once async fetch completes
    confidence,
    ...props.metadata,
    uuid,
  };

  group.add(rect);
  group.add(text);

  // Kick off async label resolution
  if (labelUuid) {
    fetchLabelByUuid(labelUuid).then(label => {
      if (!label) return;

      group.metadata.label = label;
      const newColor = label?.metadata?.color ?? 'red';
      const newLabelText = label?.metadata?.name ?? 'Unknown';

      rect.stroke(newColor);

      text.text(
        `${newLabelText}${confidence != null ? ` (${(confidence * 100).toFixed(1)}%)` : ''}`
      );
      text.fill(newColor);

      applyBoundingBoxLayout(group);
      getLayer().batchDraw();
    });
  }

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
  });

  rect.on('transform', () => {
    const layer = getLayer();

    const scaleX = rect.scaleX();
    const scaleY = rect.scaleY();

    let newWidth = rect.width() * scaleX;
    let newHeight = rect.height() * scaleY;

    const MIN_SIZE = 10;
    newWidth = Math.max(newWidth, MIN_SIZE);
    newHeight = Math.max(newHeight, MIN_SIZE);

    const rectLeft = rect.x();
    const rectTop = rect.y();

    let newGroupX = group.x() + rectLeft;
    let newGroupY = group.y() + rectTop;

    rect.width(newWidth);
    rect.height(newHeight);

    group.position({
      x: newGroupX,
      y: newGroupY,
    });

    rect.scaleX(1);
    rect.scaleY(1);

    applyBoundingBoxLayout(group);
    layer.batchDraw();
  });

  applyBoundingBoxLayout(group);
  return group;
}

/**
 * Update an existing bounding box with new metadata, such as label or color.
 */
export function updateBoundingBox(group, props = {}) {
  const rect = group.findOne('.box');
  const text = group.findOne('.label');
  if (!rect || !text) return;

  const oldMetadata = group.metadata ?? {};
  const newMetadata = { ...oldMetadata, ...props.metadata };

  const labelUuid = newMetadata.labelUuid ?? '';
  const confidence = newMetadata.confidence;

  fetchLabelByUuid(labelUuid)
    .then(label => {
      const color = label?.metadata?.color ?? 'red';
      const labelText = (label?.metadata?.name ?? props.metadata?.label) ?? 'Unknown';

      // Update metadata
      group.metadata = {
        ...newMetadata,
        label,
        confidence,
      };

      // Update label text
      const bboxText = `${labelText}${confidence != null ? ` (${(confidence * 100).toFixed(1)}%)` : ''}`;
      text.text(bboxText);
      text.fill(color);

      // Update stroke color
      rect.stroke(color);

      applyBoundingBoxLayout(group);
      getLayer().batchDraw();
    })
    .catch(err => {
      console.error('Failed to fetch label:', err);
      // optionally fallback
    });
}


/**
 * Keep rect at (0,0) and label positioned just above.
 */
function applyBoundingBoxLayout(group) {
  const rect = group.findOne('.box');
  const text = group.findOne('.label');
  if (!rect || !text) return;

  rect.x(0);
  rect.y(0);
  text.x(0);
  text.y(-18);
}
