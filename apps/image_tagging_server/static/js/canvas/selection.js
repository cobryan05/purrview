import { getLayer, getTransformer } from './state.js';

export function selectShape(group) {
  const layer = getLayer();
  const transformer = getTransformer();

  if (!group || !layer) return;

  // Update metadata UI
  const labelInput = document.getElementById('labelInput');
  const tagsInput = document.getElementById('tagsInput');

  const metadata = group.metadata ?? {};
  labelInput.value = metadata.label ?? '';
  tagsInput.value = (metadata.tags ?? []).join(', ');

  labelInput.oninput = () => {
    metadata.label = labelInput.value;
    group.metadata = metadata;
    const labelNode = group.findOne('.label');
    if (labelNode) {
      const conf = metadata.confidence;
      labelNode.text(`${metadata.label}${conf != null ? ` (${(conf * 100).toFixed(1)}%)` : ''}`);
      layer.batchDraw();
    }
  };

  tagsInput.oninput = () => {
    metadata.tags = tagsInput.value.split(',').map(s => s.trim()).filter(Boolean);
    group.metadata = metadata;
  };
}
