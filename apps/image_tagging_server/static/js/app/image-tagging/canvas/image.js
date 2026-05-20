import { BboxView } from '/app-static/js/app/image-tagging/canvas/views/bboxView.js';
import { fetchImage, updateImage } from '/app-static/js/shared/api/images.js';
import { fetchLabels } from '/app-static/js/shared/api/labels.js';
import { createUIBBbox } from '/app-static/js/shared/domain/image/mapper.js';
import { Logger, toast } from '/app-static/js/ui/utils/index.js';

let _activeImageLoadId = 0;

/**
 * @param {import('@image_tagging_types').ImageTaggingRuntime} runtime
 */
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

  // Create a new load id and invalidate all previous loads
  const loadId = ++_activeImageLoadId;
  /** @type {import('@domain_image').UIImage} */
  const imageInfo = await fetchImage(imageName);

  // If another load started while we waited abort
  if (loadId !== _activeImageLoadId) {
    Logger.debug('Discarding stale image load for', imageName);
    return;
  }

  runtime.setImage(imageInfo);
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
    return;
  }

  try {
    await updateImage(runtime.state.image);
    toast('Annotations saved successfully!');
  }
  catch (err) {
    Logger.error('Failed to save annotations:', err);
  }
}

/**
 * @param {import('@image_tagging_types').ImageTaggingState} state
 */
export function deleteSelected(state) {
  const transformer = state.canvas.transformer;
  const layer = state.canvas.layer;
  if (!transformer) return;

  const selectedNodes = transformer.nodes();
  if (!selectedNodes.length) return;

  let node = selectedNodes[0];
  let group = node.getParent();

  // Walk up until we find the group named "annotation"
  while (group && group.name() !== 'annotation') {
    group = group.getParent();
  }

  if (group && group.name() === 'annotation') {
    group.destroy();
    state.removeBboxGroup(group);
  } else {
    node.destroy(); // fallback
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
  const labelsMap = await fetchLabels();

  for (const detection of results.detections) {
    const { bbox } = detection;

    const labelData = bbox.labelUuid ? labelsMap.get(bbox.labelUuid) : null;
    const label = labelData
      ? { uuid: labelData.uuid, text: labelData.name, color: labelData.color }
      : bbox.labelUuid
        ? { uuid: bbox.labelUuid, text: bbox.classStr ?? bbox.labelUuid }
        : undefined;

    const uiBbox = createUIBBbox({
      uuid: bbox.uuid,
      label,
      tagUuids: bbox.tagUuids ?? [],
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
