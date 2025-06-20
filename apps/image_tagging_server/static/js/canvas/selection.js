import { getLayer, getTransformer } from './state.js';

export function selectShape(shape) {
  const layer = getLayer();
  let transformer = getTransformer();

  if (!shape || !layer) return;
  transformer.nodes([shape]);
  layer.draw();

  // const label = prompt('Label:', shape._metadata?.label || '');
  // const tags = prompt('Tags (comma-separated):', shape._metadata?.tags?.join(', ') || '');
  // shape._metadata = {
  //   label: label || '',
  //   tags: tags.split(',').map(t => t.trim()).filter(Boolean),
  // };
}
