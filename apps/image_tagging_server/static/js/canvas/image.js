import { createGroupFromBbox } from './drawing.js';
import { BboxGroup } from './groups/BboxGroup.js';
import { state } from './state.js'
import { Logger, fetchImage, generateUUID } from '/app-static/js/ui/utils/index.js';

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
      const group = createGroupFromBbox(box);
      state.updateBboxGroup(group);
      // const bboxGroup = new BboxGroup(box);

      // const absX = box.x * width;
      // const absY = box.y * height;
      // const absWidth = box.width * width;
      // const absHeight = box.height * height;

      // /** @type {import('@app_types').RuntimeBbox} */
      // const bbox = {
      //   x: absX,
      //   y: absY,
      //   width: absWidth,
      //   height: absHeight,
      //   uuid,
      //   classUuid: box.classUuid,
      //   tagUuids: box.tagUuids ?? [],
      // };
      // const group = createGroupFromBbox(bbox);
      // state.updateBboxGroup(group);
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
    const boxes = Array.from(state.canvas.bboxes.values()).map(bboxGroup => {
      const rect = bboxGroup.metadata.rect;
      const uuid = bboxGroup.metadata.uuid ?? generateUUID();
      const bbox_uuid = bboxGroup.metadata.uuid ?? generateUUID();
      const bbox_class_uuid = bboxGroup.metadata.classUuid ?? null;

      const { width, height } = state.image?.img;

      return {
        uuid: bbox_uuid,
        class_uuid: bbox_class_uuid,
        x: bboxGroup.x() / width,
        y: bboxGroup.y() / height,
        width: rect.width() / width,
        height: rect.height() / height,
        extra: bboxGroup.metadata.extra || {}  // Arbitrary key-value pairs
      };
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

export function addRecognizedBoxes(results) {
  const layer = state.canvas.layer;
  if (!layer) {
    Logger.error("Couldn't find layer to add recognition results to");
    return;
  }

  const bg = layer.findOne(
    node => node.name() === 'background' && node instanceof Konva.Image);
  if (!bg) {
    Logger.error('No background image found!');
    return;
  }

  const imageWidth = bg.width();
  const imageHeight = bg.height();

  results.forEach(obj => {
    const [x_norm, y_norm, w_norm, h_norm] = obj.bounding_box;

    const x = x_norm * imageWidth;
    const y = y_norm * imageHeight;
    const width = w_norm * imageWidth;
    const height = h_norm * imageHeight;
    const uuid = generateUUID();

    /** @type {import('@app_types').RuntimeBbox} */
    const bbox = {
      uuid,
      x,
      y,
      width,
      height,
      classUuid: obj.class_uuid,
      confidence: obj.confidence,
      text: obj.class.name
    };
    const shape = createGroupFromBbox(bbox);


    layer.add(shape);
  });

  layer.draw();
}
