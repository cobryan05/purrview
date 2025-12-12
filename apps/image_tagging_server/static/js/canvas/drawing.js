import { state } from './state.js';
import { generateUUID, fetchClassByUuid } from '/app-static/js/ui/utils/index.js';
import { BboxGroup } from './groups/BboxGroup.js';

/**
 * Create a bounding box group from a bbox object.
 * @param {import('@app_types').RuntimeBbox} bbox
 * @returns {BboxGroup}
 */
export function createGroupFromBbox(bbox) {
  const bboxGroup = new BboxGroup(bbox);

  // --- Event handlers setup ---
  // Disable dragging with middle mouse button down
  bboxGroup.on('mousedown', e => {
    if (e.evt.button === 1) {
      bboxGroup.draggable(false);
    } else {
      bboxGroup.draggable(state.currentTool === 'select');
    }
  });

  // Restore draggable on mouseup or dragend
  bboxGroup.on('mouseup dragend', () => bboxGroup.draggable(true));

  // The following fixes the layout of the bbox when rescaling it.
  const rect = bboxGroup.metadata.rect;
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

    let newGroupX = bboxGroup.x() + rectLeft;
    let newGroupY = bboxGroup.y() + rectTop;

    rect.width(newWidth);
    rect.height(newHeight);

    bboxGroup.position({
      x: newGroupX,
      y: newGroupY,
    });

    rect.scaleX(1);
    rect.scaleY(1);

    applyBoundingBoxLayout(bboxGroup);
    layer?.batchDraw();
  });

  bboxGroup.name('annotation');
  bboxGroup.updateMetadata();
  return bboxGroup;
}

// /**
//  * Update an existing bounding box with new metadata, such as class or color.
//  * @param {import('@konva_groups').BboxGroup} bboxGroup
//  * @param {Partial<import('@konva_groups').BboxGroupMetadata>} metadata
//   */
// export function updateGroupMetadata(bboxGroup, metadata = {}) {
//   const rectRef = bboxGroup.metadata.rect;
//   const textRef = bboxGroup.metadata.text;
//   if (!rectRef || !textRef) return;

//   // Merge the partial metadata with the current metadata
//   const mergedMetadata = { ...bboxGroup.metadata };
//   for (const key in metadata) {
//     if (Object.prototype.hasOwnProperty.call(metadata, key)) {
//       const value = metadata[key];
//       if (value !== undefined) {
//         mergedMetadata[key] = value;
//       }
//     }
//   }

//   // Kick off async class resolution
//   (mergedMetadata.classUuid
//     ? fetchClassByUuid(mergedMetadata.classUuid)
//     : Promise.resolve(null)
//   ).then(resolvedClass => {
//     let bboxText;
//     const classText = resolvedClass?.metadata?.name ?? mergedMetadata?.text ?? 'Unknown';
//     const color = resolvedClass?.metadata?.color ?? 'grey';
//     bboxText = `${classText}${mergedMetadata.confidence != null ? ` (${(mergedMetadata.confidence * 100).toFixed(1)}%)` : ''}`;

//     bboxGroup.metadata = mergedMetadata;

//     textRef.text(bboxText);
//     textRef.fill(color);
//     rectRef.stroke(color);

//     applyBoundingBoxLayout(bboxGroup);
//     state.canvas.layer?.batchDraw();
//   })
//     .catch(err => {
//       console.error('Failed to fetch class:', err);
//       // optionally fallback
//     });
// }


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
