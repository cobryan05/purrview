import { createBboxGroup, updateBoundingBox } from './drawing.js';
import { state } from './state.js'
import { Logger, fetchImage, generateUUID } from '/app-static/js/ui/utils/index.js';

/**
 * @typedef {import('../types.js').BBoxInfo} BBoxInfo
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
  const { image: img, bboxes } = await fetchImage(imageName);
  state.setImage(imageName, img, bboxes);

  // Initialize bounding boxes from the image
  if (state.image.bboxes) {
    for (const [uuid, box] of state.image.bboxes.entries()) {
      const absX = box.x * img.width;
      const absY = box.y * img.height;
      const absWidth = box.width * img.width;
      const absHeight = box.height * img.height;

      const group = createBboxGroup(absX, absY, {
        width: absWidth,
        height: absHeight,
        metadata: {
          uuid: box.uuid,
          classUuid: box.class_uuid,
          tagUuids: box.tagUuids?.map(l => l.uuid) ?? [],
          extra: box.extra ?? {}
        }
      });
      group.name('annotation');
      state.updateBboxGroup(group);
    }
  }
  await refreshCanvas();
  _recenterImage();
  Logger.notify(`Loaded image and metadata for ${state.image.name}`);
}

function _recenterImage() {
  // === Zoom to fit the image with padding ===
  const stage = state.canvas.stage;
  const container = stage?.container();
  const img = state.image.data;
  const padding = 20;

  if (!img || !stage || !container) {
    return;
  }

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

export async function refreshCanvas() {
  const layer = state.canvas.layer;
  layer?.clear();

  const bg = new Konva.Image({
    image: state.image.data,
    x: 0,
    y: 0,
    width: state.image.data.width,
    height: state.image.data.height,
    listening: false,
    name: 'background',
  });
  layer.add(bg);
  bg.moveToBottom();

  // Create canvas.bboxes from the image.bboxes
  const img = state.image.data;
  for (const [key, group] of state.canvas.bboxes) {
    updateBoundingBox(group);
    layer.add(group);
  }

  layer.draw();
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
  if (!state.image.name || !state.image.data) {
    toast('No image loaded to save annotations!');
    return;
  }

  try {

    let emptyClassCnt = 0;
    const boxes = Array.from(state.canvas.bboxes.values()).map(group => {
      const bbox_class_uuid = group.metadata.classUuid ?? null;
      if (!bbox_class_uuid) {
        return null;
      }
      const rect = group.metadata.rect;
      const uuid = group.metadata.uuid ?? generateUUID();
      const bbox_uuid = group.metadata.uuid ?? generateUUID();

      return {
        uuid: bbox_uuid,
        class_uuid: bbox_class_uuid,
        x: group.x() / state.image.data.width,
        y: group.y() / state.image.data.height,
        width: rect.width() / state.image.data.width,
        height: rect.height() / state.image.data.height,
        extra: group.metadata.extra || {}  // Arbitrary key-value pairs
      };
    }).filter(box => {
      if (box === null) {
        emptyClassCnt++;
        return false;
      }
      return true;
    });

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
    Logger.notify(`Annotations saved successfully for image ${state.image.name}`);
    if (emptyClassCnt > 0) {
      Logger.error(`${emptyClassCnt} bounding boxes have no class assigned and were not saved.`);
    }
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
    state.removeBboxGroup(group);
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
 * @param {BBoxInfo[]} results
 * @param {HTMLImageElement} image - base64 image element to run inference on.
 */
export async function addRecognizedBoxes(results, image) {
  const imageWidth = image.width;
  const imageHeight = image.height;

  results.forEach(obj => {
    const x = obj.x * imageWidth;
    const y = obj.y * imageHeight;
    const width = obj.width * imageWidth;
    const height = obj.height * imageHeight;

    const group = createBboxGroup(x, y, {
      width,
      height,
      metadata: {
        text: obj.text,
        confidence: obj.confidence,
        classUuid: obj.classUuid,
      },
    });

    group.name('annotation');
    state.updateBboxGroup(group);
  })
  await refreshCanvas();
}
