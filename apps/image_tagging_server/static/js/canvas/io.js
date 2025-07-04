import { refreshLabelList } from '../sidebar/main.js';
import { createBoundingBox } from './drawing.js';
import { getCurrentImageName, getLayer, getStage, getTransformer, setCurrentImageName } from './state.js';
import { debug, error, notify } from './utils.js';

export async function reloadImage() {
  const imageName = getCurrentImageName();
  if (!imageName) {
    notify('No image loaded to reload');
    return;
  }
  await loadImageAndMetadata(imageName);
}

export async function loadImageAndMetadata(imageName) {
  debug('loadImageAndMetadata called with imageName:', imageName);
  if (!imageName) {
    toast('Please enter an image name!');
    return;
  }

  try {
    const transformer = getTransformer();
    const layer = getLayer();
    clearAnnotations();
    setCurrentImageName(imageName);
    refreshLabelList({})

    // === Load image blob + metadata via unified API ===
    const imageRes = await fetch(`/api/images/get?path=${encodeURIComponent(imageName)}`);
    if (!imageRes.ok) throw new Error(`Failed to load image via API for ${imageName}`);

    const imageJson = await imageRes.json();
    if (imageJson.status !== 'success' || !imageJson.content) {
      throw new Error(`Invalid image API response for ${imageName}`);
    }

    // === Decode image ===
    const img = new Image();
    img.src = `data:${imageJson.mime_type};base64,${imageJson.content}`;
    await new Promise((resolve, reject) => {
      img.onload = resolve;
      img.onerror = () =>
        reject(new Error(`Failed to decode base64 image for ${imageName}`));
    });

    transformer.nodes([]);
    layer.getChildren().forEach(child => {
      if (child !== transformer) child.destroy();
    });

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
    layer.moveToBottom();

    // === Load metadata from imageJson.boxes ===
    for (const box of imageJson.boxes || []) {
      const absX = box.x * img.width;
      const absY = box.y * img.height;
      const absWidth = box.width * img.width;
      const absHeight = box.height * img.height;

      const shape = createBoundingBox(absX, absY, {
        width: absWidth,
        height: absHeight,
        metadata: {
          id: box.id,
          labelUuid: box.label_uuid,
          tags: box.tags?.map(l => l.id) ?? [],
          extra: box.extra ?? {}
        }
      });

      shape.name('annotation');
      layer.add(shape);
    }
    // === Zoom to fit the image with padding ===
    const stage = getStage();
    const container = stage.container();
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

    layer.draw();
    notify(`Loaded image and metadata for ${imageName}`);
  } catch (err) {
    error('Failed to load image or annotations:', err);
  }
}

export function clearAnnotations() {
  const layer = getStage().findOne('Layer');
  const transformer = getTransformer();
  transformer.nodes([]);
  const children = [...layer.getChildren()];
  children.forEach(child => {
    if (child.name() === 'annotation') {
      child.destroy();
    }
  });
  layer.draw();
}

export async function saveAnnotations() {
  const layer = getLayer();
  const imagePath = getCurrentImageName();
  const image = layer.findOne('.background')?.image();
  if (!imagePath || !image) {
    toast('No image loaded to save annotations!');
    return;
  }

  const imgWidth = image.width;
  const imgHeight = image.height;

  try {
    const boxes = layer.getChildren()
      .filter(shape => shape.name() === 'annotation')
      .map(group => {
        const rect = group.findOne('.box');
        if (!rect) return null;

        const bbox_meta = group.metadata || {};
        const bbox_id = bbox_meta.id ?? null;
        const bbox_label_uuid = bbox_meta.label.metadata.uuid;

        return {
          id: bbox_id,
          label_uuid: bbox_label_uuid,
          x: group.x() / imgWidth,
          y: group.y() / imgHeight,
          width: rect.width() / imgWidth,
          height: rect.height() / imgHeight,
          extra: bbox_meta.extra || {}  // Arbitrary key-value pairs
        };
      })
      .filter(Boolean);

    const payload = {
      image_path: imagePath,
      boxes: boxes,
      extra: {}  // optional image-level metadata (e.g., tags, reviewer, etc.)
    };

    const res = await fetch('/api/images/metadata/update', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });

    if (!res.ok) throw new Error(`Save failed with status ${res.status}`);
    notify(`Annotations saved successfully for image ${imagePath}`);
  } catch (err) {
    error('Failed to save annotations:', err);
  }
}

export function deleteSelected() {
  const transformer = getTransformer();
  const layer = getLayer();
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
  const layer = getLayer();
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
  const layer = getLayer();
  const stage = getStage();

  const bg = layer.findOne(
    node => node.name() === 'background' && node instanceof Konva.Image);
  if (!bg) {
    error('No background image found!');
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

    const shape = createBoundingBox(x, y, {
      width,
      height,
      metadata: {
        label: obj.class_name,
        confidence: obj.confidence,
      },
    });

    shape.name('annotation');
    layer.add(shape);
  });

  layer.draw();
}
