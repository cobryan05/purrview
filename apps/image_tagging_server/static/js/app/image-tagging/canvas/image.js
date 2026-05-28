import { BboxView } from '/app-static/js/app/image-tagging/canvas/views/bboxView.js';
import { fetchImage, updateImage } from '/app-static/js/shared/api/images.js';
import { fetchLabels } from '/app-static/js/shared/api/labels.js';
import { createUIBBbox } from '/app-static/js/shared/domain/image/mapper.js';
import { Logger, toast } from '/app-static/js/ui/utils/index.js';
import { events, EventTypes } from '/app-static/js/shared/events/index.js';
import { UNVERIFIED_TAG_UUID } from '/app-static/js/shared/api/review.js';
import { fetchTags } from '/app-static/js/shared/api/tags.js';

/**
 * @param {import('@image_tagging_types').ImageTaggingRuntime} runtime
 */
/**
 * Returns true if the current canvas image has any unsaved (dirty) bboxes.
 * @param {import('@image_tagging_types').ImageTaggingRuntime} runtime
 */
function hasDirtyChanges(runtime) {
  const bboxes = runtime.state?.image?.bboxes;
  if (!bboxes) return false;
  for (const bbox of bboxes.values()) {
    if (bbox.dirty) return true;
  }
  return false;
}

/**
 * Marks all bboxes on the current image as clean (not dirty).
 * @param {import('@image_tagging_types').ImageTaggingRuntime} runtime
 */
function markSaved(runtime) {
  const bboxes = runtime.state?.image?.bboxes;
  if (!bboxes) return;
  for (const bbox of bboxes.values()) {
    bbox.dirty = false;
  }
}

export async function reloadImage(runtime) {
  const imageName = runtime.state.image?.image_path;
  if (!imageName) {
    Logger.notify('No image loaded to reload');
    return;
  }
  await loadImageOntoCanvas(runtime, imageName);
}

/**
 * @param {import('@image_tagging_types').ImageTaggingRuntime} runtime
 * @param {string} imageName
 */
export async function loadImageOntoCanvas(runtime, imageName) {
  Logger.debug('loadImageOntoCanvas called with imageName:', imageName);

  if (!runtime.canvas) return;

  // Warn if switching to a different image with unsaved changes
  const currentPath = runtime.state?.image?.image_path;
  if (currentPath && currentPath !== imageName && hasDirtyChanges(runtime)) {
    const proceed = confirm('You have unsaved annotation changes. Discard them and load the new image?');
    if (!proceed) return;
  }

  // Create a new load id and invalidate all previous loads for this canvas instance
  const loadId = ++runtime.canvas.activeImageLoadId;
  /** @type {import('@domain_image').UIImage} */
  const imageInfo = await fetchImage(imageName);

  // Abort if canvas was destroyed or a newer load started while we waited
  if (!runtime.canvas || loadId !== runtime.canvas.activeImageLoadId) {
    Logger.debug('Discarding stale image load for', imageName);
    return;
  }

  runtime.setImage(imageInfo);
  events.publish(EventTypes.IMAGE_LOADED_ONTO_CANVAS, { path: imageName, runtimeId: runtime.id });
}


/**
 * @param {import('@image_tagging_types').ImageTaggingRuntime} runtime
 */
export function clearAnnotations(runtime) {
  if (!runtime.canvas) return;
  const { layer, transformer } = runtime.canvas;
  transformer?.nodes([]);
  runtime.clearBboxes();

  const children = [...layer.getChildren()];
  children.forEach(child => {
    if (child.name() === 'bbox') {
      child.destroy();
    }
  });
  layer.draw();
}

/**
 * @param {import('@image_tagging_types').ImageTaggingRuntime} runtime
 */
export async function saveAnnotations(runtime) {
  if (!runtime.state?.image) {
    toast('No image loaded to save annotations!');
    return false;
  }

  try {
    await updateImage(runtime.state.image);
    markSaved(runtime);
    toast('Annotations saved successfully!');
    return true;
  }
  catch (err) {
    Logger.error('Failed to save annotations:', err);
    return false;
  }
}

/**
 * @param {import('@image_tagging_types').ImageTaggingRuntime} runtime
 */
export function deleteSelected(runtime) {
  const transformer = runtime.canvas?.transformer;
  const layer = runtime.canvas?.layer;
  if (!transformer || !layer) return;

  const selectedNodes = transformer.nodes();
  if (!selectedNodes.length) return;

  // Walk up from the selected node to find the BboxView group
  let node = selectedNodes[0];
  let bboxView = null;
  while (node && node !== layer) {
    if (node instanceof BboxView) {
      bboxView = node;
      break;
    }
    node = node.getParent();
  }

  if (bboxView) {
    runtime.state.image?.bboxes.delete(bboxView.uiBBox.uuid);
    bboxView.destroy();
    events.publish(EventTypes.CANVAS_BBOXES_CHANGED, { runtimeId: runtime.id });
  }

  transformer.nodes([]);
  layer.draw();
}

/**
 * @param {import('@image_tagging_types').ImageTaggingState} state
 */
export function exportAnnotations(state) {
  const layer = state.canvas.layer;
  const shapes = layer.getChildren().filter(s => s.name() === 'annotation');

  const annotations = shapes
    .map(group => {
      const rect = group.findOne('.box');
      if (!rect) return null;

      return {
        type: 'rect',
        x: group.x(),
        y: group.y(),
        width: rect.width(),
        height: rect.height(),
        metadata: group.metadata || {},
      };
    })
    .filter(Boolean);

  const pre = document.getElementById('annotation-json');
  if (pre) {
    pre.textContent = JSON.stringify(annotations, null, 2);
  }
}

/**
 * Adds recognition results to the current canvas
 * @param {import('@image_tagging_types').ImageTaggingRuntime} runtime
 * @param {import('@web_api').InferenceResultModel} results - The inference results to add to the current canvas
 */
export async function addInferenceResults(runtime, results) {
  if (!runtime.canvas || !runtime.state?.image) {
    Logger.warn('addInferenceResults: no canvas or image loaded');
    return;
  }

  const { layer, viewport } = runtime.canvas;
  const [labelsMap, tagsMap] = await Promise.all([fetchLabels(), fetchTags()]);
  const unverifiedIcon = tagsMap.get(UNVERIFIED_TAG_UUID)?.icon ?? null;

  for (const detection of results.detections) {
    const { bbox } = detection;

    const labelData = bbox.labelUuid ? labelsMap.get(bbox.labelUuid) : null;
    const label = labelData
      ? { uuid: labelData.uuid, text: labelData.name, color: labelData.color }
      : bbox.labelUuid
        ? { uuid: bbox.labelUuid, text: bbox.classStr ?? bbox.labelUuid }
        : bbox.classStr
          ? { text: bbox.classStr }
          : undefined;

    // Always include the UNVERIFIED tag on inference results
    const baseTags = (bbox.tagUuids ?? []).map(id => ({ uuid: id, icon: tagsMap.get(id)?.icon ?? null }));
    const tags = baseTags.some(t => t.uuid === UNVERIFIED_TAG_UUID)
      ? baseTags
      : [{ uuid: UNVERIFIED_TAG_UUID, icon: unverifiedIcon }, ...baseTags];

    const uiBbox = createUIBBbox({
      uuid: bbox.uuid,
      label,
      tags,
      x: bbox.x,
      y: bbox.y,
      width: bbox.width,
      height: bbox.height,
      selected: false,
      dirty: true,
    });

    runtime.state.image.bboxes.set(uiBbox.uuid, uiBbox);
    const bboxView = new BboxView(uiBbox, viewport);
    layer.add(bboxView);
  }

  layer.draw();
}
