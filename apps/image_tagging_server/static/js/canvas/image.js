import { createCanvasBboxFromRuntimeBbox } from './drawing.js';
import { CanvasBboxGroup } from './groups/CanvasBboxGroup.js';
import { state } from './state.js'
import { Logger, fetchImage, generateUUID } from '/app-static/js/ui/utils/index.js';

/**
 * @typedef {import('@app_types').RuntimeBbox} BBoxInfo
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
      const canvasBbox = createCanvasBboxFromRuntimeBbox(box);
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
  if (!state.image?.name || !state.image?.img) {
    toast('No image loaded to save annotations!');
    return;
  }

  try {
    const boxes = Array.from(state.canvas.bboxes.values())
      .filter(canvasBbox => canvasBbox.metadata.classUuid != null)
      .map(canvasBbox => {
        const rect = canvasBbox.metadata.rect;
        const uuid = canvasBbox.metadata.uuid ?? generateUUID();
        const bbox_uuid = canvasBbox.metadata.uuid ?? generateUUID();
        const bbox_class_uuid = canvasBbox.metadata.classUuid ?? null;

        const { width, height } = state.image?.img;

        return {
          uuid: bbox_uuid,
          class_uuid: bbox_class_uuid,
          x: canvasBbox.x() / width,
          y: canvasBbox.y() / height,
          width: rect.width() / width,
          height: rect.height() / height,
          extra: canvasBbox.metadata.extra || {}  // Arbitrary key-value pairs
        };
      });
    const missingCnt = state.canvas.bboxes.size - boxes.length
    if( missingCnt > 0 ) {
      Logger.error(`Discarding ${missingCnt} boxes with missing data`);
    }

    const payload = {
      image_path: state.image.name,
      boxes: boxes,
      extra: {}  // optional image-level metadata (e.g., tags, reviewer, etc.)
    };

    const res = await fetch('/api/images/metadata/update', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });

    if (!res.ok) throw new Error(`Save failed with status ${res.status}`);
    Logger.notify(`Annotations saved successfully for image ${state.imageName}`);
  } catch (err) {
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
    const bboxGroup = createCanvasBboxFromRuntimeBbox(box);
    state.updateCanvasBbox(bboxGroup);
  });

  await refreshCanvas();
}
