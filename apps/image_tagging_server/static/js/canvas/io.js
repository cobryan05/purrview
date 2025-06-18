// io.js
import { getLayer, getTransformer } from './state.js';
import { createShape } from './drawing.js';
import { debug } from './utils.js';
import { setCurrentImageName, getCurrentImageName } from './state.js';

const undoStack = [];
const redoStack = [];

/**
 * Load image and metadata for given image name.
 * Clears current canvas and redraws background + shapes.
 * @param {string} imageName - Base name without extension
 */
export async function loadImageAndMetadata(imageName) {
  let transformer = getTransformer();
  let layer = getLayer();
  debug('loadImageAndMetadata called with imageName:', imageName);
  if (!imageName) {
    alert('Please enter an image name!');
    return;
  }

  try {
    setCurrentImageName(imageName);

    // Load image
    const imageUrl = `/images/${imageName}.jpg`;
    const img = new Image();
    img.src = imageUrl;

    await new Promise((resolve, reject) => {
      img.onload = () => resolve();
      img.onerror = () => reject(new Error(`Failed to load image: ${imageUrl}`));
    });

    // Clear previous shapes (including background image)
    layer.destroyChildren();

    // Re-add transformer to layer if it's defined
    let transformer = getTransformer()
    if (transformer && !transformer.getLayer()) {
      layer.add(transformer);
    }

    // Add background image to bottom
    const bg = new Konva.Image({
      image: img,
      x: 0,
      y: 0,
      width: img.width,
      height: img.height,
      listening: false,
    });
    layer.add(bg);
    layer.moveToBottom();

    // Fetch metadata JSON
    const metadataUrl = `/metadata/${imageName}.json`;
    let metadata = { annotations: [] };
    const res = await fetch(metadataUrl);
    if (res.ok) {
      metadata = await res.json();
    } else {
      debug(`No metadata found for ${imageName}, starting with empty annotations.`);
    }

    // Recreate shapes from metadata
    for (const ann of metadata.annotations) {
      let shape = null;

      switch (ann.type) {
        case 'rect':
          shape = createShape('rect', ann.x, ann.y);
          shape.width(ann.width);
          shape.height(ann.height);
          break;

        case 'circle':
          shape = createShape('circle', ann.x, ann.y);
          shape.radius(ann.radius);
          break;

        case 'line':
          shape = createShape('line', 0, 0);
          shape.points(ann.points);
          break;

        default:
          debug('Unknown annotation type:', ann.type);
          continue;
      }

      if (shape) {
        shape.metadata = ann.metadata || {};
        layer.add(shape);
      }
    }

    layer.draw();
    debug(`Loaded image and metadata for ${imageName}`);
  } catch (err) {
    debug('Error loading image and metadata:', err);
    alert('Failed to load image or annotations.');
  }
}

/**
 * Load the image from the textbox
 */
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

/**
 * Save current annotations to server for currently loaded image.
 */
export async function saveAnnotations() {
  let layer = getLayer();
  const imageName = getCurrentImageName();
  if (!imageName) {
    alert('No image loaded to save annotations!');
    return;
  }

  try {
    // Convert shapes to annotation data
    const shapes = layer.getChildren();

    // Skip background image (first child)
    const data = shapes.slice(1).map(shape => {
      switch (shape.className) {
        case 'Rect':
          return {
            type: 'rect',
            x: shape.x(),
            y: shape.y(),
            width: shape.width(),
            height: shape.height(),
            metadata: shape.metadata || {},
          };
        case 'Circle':
          return {
            type: 'circle',
            x: shape.x(),
            y: shape.y(),
            radius: shape.radius(),
            metadata: shape.metadata || {},
          };
        case 'Line':
          return {
            type: 'line',
            points: shape.points(),
            metadata: shape.metadata || {},
          };
        default:
          return null;
      }
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

/**
 * Delete the currently selected shape.
 * Also pushes the deleted shape onto undo stack.
 */
export function deleteSelected() {
  let transformer = getTransformer()
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
  redoStack.length = 0; // clear redo stack on new action

  shape.destroy();
  transformer.nodes([]);
  layer.draw();
  debug('Deleted selected shape');
}

/**
 * Undo the last action (only supports delete for now).
 */
export function undo() {
  if (!undoStack.length) {
    debug('Nothing to undo');
    return;
  }

  const lastAction = undoStack.pop();

  switch (lastAction.action) {
    case 'delete':
      // Restore the deleted shape
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

/**
 * Redo the last undone action.
 */
export function redo() {
  if (!redoStack.length) {
    debug('Nothing to redo');
    return;
  }

  const lastUndone = redoStack.pop();

  switch (lastUndone.action) {
    case 'delete':
      // Remove the shape again
      lastUndone.shape.destroy();
      undoStack.push(lastUndone);
      layer.draw();
      debug('Redo: deleted shape again');
      break;

    default:
      debug('Redo: Unknown action', lastUndone.action);
  }
}
