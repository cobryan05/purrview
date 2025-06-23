import { getLayer, getTransformer } from './state.js';

export function selectShape(shape) {
  const layer = getLayer();
  let transformer = getTransformer();

  if (!shape || !layer) return;
  transformer.nodes([shape]);
  layer.draw();

  const labelInput = document.getElementById('labelInput');
  const tagsInput = document.getElementById('tagsInput');

  labelInput.value = shape.metadata?.label || '';
  tagsInput.value = (shape.metadata?.tags || []).join(', ');

  labelInput.oninput = () => {
    shape.metadata = shape.metadata || {};
    shape.metadata.label = labelInput.value;
  };

  tagsInput.oninput = () => {
    shape.metadata = shape.metadata || {};
    shape.metadata.tags = tagsInput.value.split(',').map(t => t.trim()).filter(Boolean);
  };
}