import { getLayer, getTransformer, setTransformer } from './state.js';

export function selectShape(shape) {
  const layer = getLayer();
  let transformer = getTransformer();

  if (!shape || !layer) return;

  if (!transformer) {
    transformer = new Konva.Transformer({
      rotateEnabled: false,
      borderStroke: 'red',
      borderDash: [4, 4],
      anchorStroke: 'red',
      anchorFill: 'white',
      anchorSize: 10,
      anchorCornerRadius: 5,
      enabledAnchors: ['top-left', 'top-right', 'bottom-left', 'bottom-right'],
      keepRatio: false,
    });
    setTransformer(transformer);
    layer.add(transformer);
  } else if (!layer.findOne(node => node === transformer)) {
    // Transformer got wiped out by destroyChildren — re-add it
    layer.add(transformer);
  }

  transformer.nodes([shape]);
  layer.draw();

  // const label = prompt('Label:', shape._metadata?.label || '');
  // const tags = prompt('Tags (comma-separated):', shape._metadata?.tags?.join(', ') || '');
  // shape._metadata = {
  //   label: label || '',
  //   tags: tags.split(',').map(t => t.trim()).filter(Boolean),
  // };
}
