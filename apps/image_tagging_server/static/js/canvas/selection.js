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
    metadata.classUuid = null;
    group.metadata = metadata;
    const classNode = group.findOne('.class');
    if (classNode) {
      const conf = metadata.confidence;
      classNode.text(`${metadata.label}${conf != null ? ` (${(conf * 100).toFixed(1)}%)` : ''}`);
      layer.batchDraw();
    }
  };

  tagsInput.oninput = () => {
    metadata.tags = tagsInput.value.split(',').map(s => s.trim()).filter(Boolean);
    group.metadata = metadata;
  };
}

export function findGroupAtPoint(pos) {
  const layer = getLayer();
  const children = layer.getChildren(node => node.name() === 'annotation');

  for (const group of children) {
    const rect = group.getClientRect({ relativeTo: layer });
    if (
      pos.x >= rect.x &&
      pos.x <= rect.x + rect.width &&
      pos.y >= rect.y &&
      pos.y <= rect.y + rect.height
    ) {
      return group;
    }
  }

  return null;
}