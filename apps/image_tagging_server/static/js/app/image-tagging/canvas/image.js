import { CanvasBboxGroup } from './groups/CanvasBboxGroup.js';
import { fetchImage, updateImage } from '/app-static/js/shared/api/images.js';
import { Logger, toast } from '/app-static/js/ui/utils/index.js';

let _activeImageLoadId = 0;

/**
 * @param {import('@image_tagging_types').ImageTaggingState} state
 */
export async function reloadImage(state) {
  const imageName = state.image.name;
  if (!imageName) {
    Logger.notify('No image loaded to reload');
    return;
  }
  await loadImageAndMetadata(state, imageName);
}

/**
 * @param {import('@image_tagging_types').ImageTaggingState} state
 * @param {string} imageName
 */
export async function loadImageAndMetadata(state, imageName) {
  Logger.debug('loadImageAndMetadata called with imageName:', imageName);

  // Create a new load id and invalidate all previous loads
  const loadId = ++_activeImageLoadId;
  /** @type {import('@canvas_types').RuntimeImage} */
  const imageInfo = await fetchImage(imageName);

  // If another load started while we waited → abort
  if (loadId !== _activeImageLoadId) {
    Logger.debug('Discarding stale image load for', imageName);
    return;
  }

  clearAnnotations(state);
  state.setImage(imageInfo);

  if (imageInfo.bboxes && imageInfo.img) {
    for (const [uuid, box] of imageInfo.bboxes.entries()) {
      const canvasBbox = new CanvasBboxGroup(box, state);
      state.updateCanvasBbox(canvasBbox);
    }
  }
  await refreshCanvas(state);
}

/**
 * @param {import('@image_tagging_types').ImageTaggingState} state
 */
export async function refreshCanvas(state) {
  const layer = state.canvas.layer;
  const img = state.image?.img;
  if (!layer || !img) {
    Logger.warn("Can't refresh without a canvas and image");
    return;
  }

  const oldBg = layer.findOne('.background');
  if (oldBg) {
    oldBg.destroy();
  }

  const bg = new Konva.Image({
    image: img,
    x: 0,
    y: 0,
    width: img.width,
    height: img.height,
    listening: false,
    name: 'background',
  });
  layer.add(bg);
  bg.moveToBottom();

  for (const [key, group] of (state.canvas.bboxes ?? [])) {
    layer.add(group);
    group.moveToTop();
  }

  // === Zoom to fit the image with padding ===
  const stage = state.canvas.stage;
  const container = stage?.container();
  if (stage && container) {
    const padding = 20;

    const scaleX = (container.clientWidth - padding * 2) / img.width;
    const scaleY = (container.clientHeight - padding * 2) / img.height;
    const scale = Math.min(scaleX, scaleY);

    // Center the image
    const newWidth = img.width * scale;
    const newHeight = img.height * scale;

    const offsetX = (container.clientWidth - newWidth) / 2;
    const offsetY = (container.clientHeight - newHeight) / 2;

    stage.scale({ x: scale, y: scale });
    stage.position({ x: offsetX, y: offsetY });
    stage.batchDraw();
  }

  layer.draw();
  Logger.notify(`Loaded image and metadata for ${state.image.name}`);
}


/**
 * @param {import('@image_tagging_types').ImageTaggingState} state
 */
export function clearAnnotations(state) {
  const layer = state.canvas.layer;
  state.clearSelection();
  state.clearBboxes();

  const children = [...layer.getChildren()];
  children.forEach(child => {
    if (child.name() === 'annotation') {
      child.destroy();
    }
  });
  layer.draw();
}

/**
 * @param {import('@image_tagging_types').ImageTaggingState} state
 */
export async function saveAnnotations(state) {
  if (!state.image) {
    toast('No image loaded to save annotations!');
    return;
  }

  try {
    // Sync the canvas positions to the image
    state.image.bboxes?.clear();
    for (const [key, canvasGroup] of state.canvas.bboxes) {
      canvasGroup.updatePosition({});
      state.image.bboxes?.set(key, canvasGroup.metadata.bboxInfo);
    }
    await updateImage(state.image);
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
 * @param {import('@image_tagging_types').ImageTaggingState} state
 * @param {import('@web_api').InferenceResultModel} results - The inference results to add to the current canvas
 */
export async function addInferenceResults(state, results) {
  const layer = state.canvas.layer;
  if (!layer) {
    Logger.error("Couldn't find image and canvas to add results");
    return;
  }
  results.detections.forEach(box => {

    const bboxGroup = new CanvasBboxGroup(box, state);
    state.updateCanvasBbox(bboxGroup);
  });

  await refreshCanvas(state);
}
