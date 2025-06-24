import { getLayer, getTransformer, getStage } from './state.js';
import { createBoundingBox } from './drawing.js';
import { debug } from './utils.js';
import { setCurrentImageName, getCurrentImageName } from './state.js';

const undoStack = [];
const redoStack = [];

export async function loadImageAndMetadata(imageName) {
  debug('loadImageAndMetadata called with imageName:', imageName);
  if (!imageName) {
    alert('Please enter an image name!');
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
      img.onerror = () => reject(new Error(`Failed to load image: ${imageUrl}`));
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
            label: ann.metadata?.label ?? '',
          });
          break;
        default:
          debug('Unknown annotation type:', ann.type);
          continue;
      }

      shape.name('annotation');
      layer.add(shape);
    }

    layer.draw();
    debug(`Loaded image and metadata for ${imageName}`);
  } catch (err) {
    debug('Error loading image and metadata:', err);
    alert('Failed to load image or annotations.');
  }
}

export function loadImageFromInput() {
  const input = document.getElementById('imageNameInput');
  if (!input) {
    alert('Image name input not found!');
    return;
  }
  const imageName = input.value.trim();
  if (!imageName) {
    alert('Please enter an image name');
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
    alert('No image loaded to save annotations!');
    return;
  }

  try {
    const shapes = layer.getChildren();
    const data = shapes.filter(shape => shape.name() === 'annotation').map(group => {
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
    }).filter(Boolean);

    const res = await fetch('/save-annotations', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image: imageName, annotations: data }),
    });

    if (!res.ok) throw new Error(`Save failed with status ${res.status}`);

    debug(`Annotations saved successfully for image ${imageName}`);
  } catch (err) {
    debug('Failed to save annotations:', err);
    alert('Failed to save annotations.');
  }
}

export function deleteSelected() {
  let transformer = getTransformer();
  let layer = getLayer();
  if (!transformer) {
    debug('No transformer available for deleteSelected');
    return;
  }

  const selectedNodes = transformer.nodes();
  if (!selectedNodes.length) {
    debug('No shape selected to delete');
    return;
  }

  const shape = selectedNodes[0];
  undoStack.push({ action: 'delete', shape });
  redoStack.length = 0;

  shape.destroy();
  transformer.nodes([]);
  layer.draw();
  debug('Deleted selected shape');
}

export function undo() {
  let layer = getLayer();
  if (!undoStack.length) {
    debug('Nothing to undo');
    return;
  }

  const lastAction = undoStack.pop();

  switch (lastAction.action) {
    case 'delete':
      layer.add(lastAction.shape);
      lastAction.shape.show();
      redoStack.push(lastAction);
      layer.draw();
      debug('Undo: restored deleted shape');
      break;
    default:
      debug('Undo: Unknown action', lastAction.action);
  }
}

export function redo() {
  let layer = getLayer();
  if (!redoStack.length) {
    debug('Nothing to redo');
    return;
  }

  const lastUndone = redoStack.pop();

  switch (lastUndone.action) {
    case 'delete':
      lastUndone.shape.destroy();
      undoStack.push(lastUndone);
      layer.draw();
      debug('Redo: deleted shape again');
      break;
    default:
      debug('Redo: Unknown action', lastUndone.action);
  }
}

export function exportAnnotations() {
  const layer = getLayer();
  const shapes = layer.getChildren().filter(s => s.name() === 'annotation');

  const annotations = shapes.map(group => {
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
  }).filter(Boolean);

  const pre = document.getElementById('annotation-json');
  if (pre) pre.textContent = JSON.stringify(annotations, null, 2);
}

export function addRecognizedBoxes(results) {
  const layer = getLayer();
  const stage = getStage();

  const bg = layer.findOne(node => node.name() === 'background' && node instanceof Konva.Image);
  if (!bg) {
    alert('No background image found!');
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
