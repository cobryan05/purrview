import { createBoundingBox } from './drawing.js';
import { getCurrentImageName, getLayer, getStage, getTransformer, setCurrentImageName } from './state.js';
import { debug, warn, error, notify } from './utils.js';

export async function loadImageAndMetadata(imageName) {
  debug('loadImageAndMetadata called with imageName:', imageName);
  if (!imageName) {
    toast('Please enter an image name!');
    return;
  }

  try {
    const transformer = getTransformer();
    const layer = getLayer();
    setCurrentImageName(imageName);

    const imageUrl = `/images/${imageName}.jpg`;
    const img = new Image();
    img.src = imageUrl;

    await new Promise((resolve, reject) => {
      img.onload = resolve;
      img.onerror = () =>
        reject(new Error(`Failed to load image: ${imageUrl}`));
    });

    transformer.nodes([]);

    layer.getChildren().forEach(child => {
      if (child !== transformer) {
        child.destroy();
      }
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

    const metadataUrl = `/metadata/${imageName}.json?t=${Date.now()}`;
    let metadata = { annotations: [] };
    const res = await fetch(metadataUrl);
    if (res.ok) {
      metadata = await res.json();
    } else {
      debug(`No metadata found for ${imageName}, starting with empty annotations.`);
    }

    for (const ann of metadata.annotations) {
      let shape = null;

      switch (ann.type) {
        case 'rect':
          shape = createBoundingBox(ann.x, ann.y, {
            width: ann.width,
            height: ann.height,
            metadata: ann.metadata,
          });
          break;
        default:
          warn('Unknown annotation type:', ann.type);
          continue;
      }

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

export function loadImageFromInput() {
  const input = document.getElementById('imageNameInput');
  if (!input) {
    toast('Image name input not found!');
    return;
  }
  const imageName = input.value.trim();
  if (!imageName) {
    toast('Please enter an image name');
    return;
  }
  loadImageAndMetadata(imageName);
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
  let layer = getLayer();
  const imageName = getCurrentImageName();
  if (!imageName) {
    toast('No image loaded to save annotations!');
    return;
  }

  try {
    const shapes = layer.getChildren();
    const data = shapes.filter(shape => shape.name() === 'annotation')
      .map(group => {
        const rect = group.findOne('.box');
        if (!rect) return null;
        const { uuid, ...metadataWithoutUuid } =
          group.metadata || {};
        return {
          type: 'rect',
          x: group.x(),
          y: group.y(),
          width: rect.width(),
          height: rect.height(),
          metadata: metadataWithoutUuid
        };
      })
      .filter(Boolean);

    const res = await fetch('/save-annotations', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image: imageName, annotations: data }),
    });

    if (!res.ok) throw new Error(`Save failed with status ${res.status}`);

    notify(`Annotations saved successfully for image ${imageName}`);
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
