import { state } from './state.js';
import { generateUUID } from './utils.js';
import { fetchClassByUuid } from '/app-static/js/ui/utils/index.js';


/**
 * Create a bounding box group with a rectangle, class, confidence,
 * and set up all relevant event handlers here.
 */

export function createBoundingBox(x, y, props = {}) {
  const width = props.width ?? 50;
  const height = props.height ?? 50;
  const classUuid = props.metadata?.classUuid ?? '';
  const confidence = props.metadata?.confidence ?? null;
  const uuid = props.metadata?.uuid ?? generateUUID();

  // Initial placeholder values
  let classText = props.metadata?.class ?? 'Unknown';
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

  const bboxText = `${classText}${confidence != null ? ` (${(confidence * 100).toFixed(1)}%)` : ''}`;

  const text = new Konva.Text({
    name: 'class',
    text: bboxText,
    fontSize: 14,
    fill: color,
    y: -18,
    x: 0,
  });

  state.setBbox(uuid, group);

  group.metadata = {
    class: null, // will be filled in once async fetch completes
    rect,
    text: text,
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
      group.draggable(state.currentTool === 'select');
    }
  });

  // Restore draggable on mouseup or dragend
  group.on('mouseup dragend', () => group.draggable(true));

  rect.on('transform', () => {
    const layer = state.canvas.layer;

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

  updateBoundingBox(group);
  return group;
}

/**
 * Update an existing bounding box with new metadata, such as class or color.
 */
export function updateBoundingBox(group, props = {}) {
  const rect = group.findOne('.box');
  const text = group.findOne('.class');
  if (!rect || !text) return;

  const oldMetadata = group.metadata ?? {};
  const newMetadata = { ...oldMetadata, ...props.metadata };

  const classUuid = newMetadata.classUuid ?? '';
  const confidence = newMetadata.confidence;

  // Kick off async class resolution
  fetchClassByUuid(classUuid)
    .then(cls => {
      const color = cls?.metadata?.color ?? 'red';
      const classText = (cls?.metadata?.name ?? props.metadata?.class) ?? 'Unknown';

      // Update metadata
      group.metadata = {
        ...newMetadata,
        class: cls,
        confidence,
      };

      // Update class text
      const bboxText = `${classText}${confidence != null ? ` (${(confidence * 100).toFixed(1)}%)` : ''}`;
      text.text(bboxText);
      text.fill(color);

      // Update stroke color
      rect.stroke(color);

      applyBoundingBoxLayout(group);
      state.canvas.layer.batchDraw();
    })
    .catch(err => {
      console.error('Failed to fetch class:', err);
      // optionally fallback
    });
}


/**
 * Keep rect at (0,0) and class positioned just above.
 */
function applyBoundingBoxLayout(group) {
  const rect = group.findOne('.box');
  const text = group.findOne('.class');
  if (!rect || !text) return;

  rect.x(0);
  rect.y(0);
  text.x(0);
  text.y(-18);
}
