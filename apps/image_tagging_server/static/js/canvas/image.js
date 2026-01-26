import { CanvasBboxGroup } from './groups/CanvasBboxGroup.js';
import { state } from './state.js'
import { Logger, fetchImage, generateUUID, toast, updateImage } from '/app-static/js/ui/utils/index.js';

/**
 * @typedef {import('@app_types').RuntimeBboxInfo} RuntimeBboxInfo
 * @typedef {import('@app_types').InferenceResult} InferenceResult

 */

export async function reloadImage() {
  const imageName = state.image.name;
  if (!imageName) {
    Logger.notify('No image loaded to reload');
    return;
  }
  await loadImageAndMetadata(imageName);
}

export async function loadImageAndMetadata(imageName) {
  Logger.debug('loadImageAndMetadata called with imageName:', imageName);

  clearAnnotations();
  const imageInfo = await fetchImage(imageName);
  state.setImage(imageInfo);

  if (imageInfo.bboxes && imageInfo.img) {
    const { width, height } = imageInfo.img;

    for (const [uuid, box] of imageInfo.bboxes.entries()) {
      const canvasBbox = new CanvasBboxGroup(box);
      state.updateCanvasBbox(canvasBbox);
    }
  }
  await refreshCanvas();
}


export async function refreshCanvas() {
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


export function clearAnnotations() {
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

export async function saveAnnotations() {
  if (!state.image) {
    toast('No image loaded to save annotations!');
    return;
  }

  try {
    // Sync the canvas positions to the image
    state.image.bboxes?.clear();
    for (const [key, canvasGroup] of state.canvas.bboxes) {
      canvasGroup.updatePosition({});
      state.image.bboxes?.set(key, canvasGroup.metadata.runtimeBboxInfo);
    }
    await updateImage(state.image);
    toast('Annotations saved successfully!');
  }
  catch (err) {
    Logger.error('Failed to save annotations:', err);
  }
}

export function deleteSelected() {
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

export function exportAnnotations() {
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
 * @param {InferenceResult} results - The inference results to add to the current canvas
 */
export async function addInferenceResults(results) {
  const layer = state.canvas.layer;
  if (!layer) {
    Logger.error("Couldn't find image and canvas to add results");
    return;
  }
  results.detections.forEach(box => {
    const bboxGroup = new CanvasBboxGroup(box);
    state.updateCanvasBbox(bboxGroup);
  });

  await refreshCanvas();
}
