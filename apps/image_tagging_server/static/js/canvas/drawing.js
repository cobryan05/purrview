import { state } from './state.js';
import { generateUUID, fetchClassByUuid } from '/app-static/js/ui/utils/index.js';
import { CanvasBboxGroup } from './groups/CanvasBboxGroup.js';

/**
 * Create a bounding box group from a bbox object.
 * @param {import('@app_types').RuntimeBbox} bbox
 * @returns {CanvasBboxGroup}
 */
export function createCanvasBboxFromRuntimeBbox(bbox) {
  const canvasBbox = new CanvasBboxGroup(bbox);
  return canvasBbox;
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

