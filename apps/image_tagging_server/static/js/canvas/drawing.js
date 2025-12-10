import { state } from './state.js';
import { generateUUID, fetchClassByUuid } from '/app-static/js/ui/utils/index.js';


/**
 * Create a bounding box group with a rectangle, class, confidence,
 * and set up all relevant event handlers here.
 *
 * @param {import('@app_types').RuntimeBbox} bbox
 */

export function createGroupFromBbox(bbox) {
  // TODO: Fix this up, why are two places setting the text
  // Initial placeholder values
  let labelText = bbox.text ?? 'Unknown';
  let color = 'grey';

  const group = new Konva.Group({
    x: bbox.x,
    y: bbox.y,
    draggable: true,
    name: 'annotation',
  });

  const rect = new Konva.Rect({
    name: 'box',
    width: bbox.width,
    height: bbox.height,
    stroke: color,
    strokeWidth: 2,
  });

  const bboxText = `${labelText}${bbox.confidence != null ? ` (${(bbox.confidence * 100).toFixed(1)}%)` : ''}`;

  const text = new Konva.Text({
    name: 'class',
    text: bboxText,
    fontSize: 14,
    fill: color,
    y: -18,
    x: 0,
  });

  group.setAttr('metadata', {
    uuid: bbox.uuid,
    classUuid: bbox.classUuid,
    class: null, // will be filled in once async fetch completes
    rect,
    text: text,
    confidence: bbox.confidence
  });

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
    layer?.batchDraw();
  });

  updateGroupMetadata(group);
  return group;
}

/**
 * Update an existing bounding box with new metadata, such as class or color.
 */
export function updateGroupMetadata(group, metadata = {}) {
  const rectRef = group.findOne('.box');
  const textRef = group.findOne('.class');
  if (!rectRef || !textRef) return;

  const oldMetadata = group.getAttr('metadata') ?? {};
  const newMetadata = { ...oldMetadata, ...metadata };

  const classUuid = newMetadata.classUuid ?? null;
  const confidence = newMetadata.confidence;

  // Kick off async class resolution
  (classUuid
    ? fetchClassByUuid(classUuid)
    : Promise.resolve(null)
  ).then(cls => {
    let bboxText;
    const classText = (cls?.metadata?.name ?? newMetadata?.class) ?? newMetadata?.text ?? 'Unknown';
    const color = cls?.metadata?.color ?? 'grey';
    bboxText = `${classText}${confidence != null ? ` (${(confidence * 100).toFixed(1)}%)` : ''}`;

    // Update metadata
    group.setAttr('metadata', {
      ...newMetadata,
      class: cls,
      confidence,
    });

    textRef.text(bboxText);
    textRef.fill(color);
    rectRef.stroke(color);

    applyBoundingBoxLayout(group);
    state.canvas.layer?.batchDraw();
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
