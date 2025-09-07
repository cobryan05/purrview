let _modelCache = null;

export function clearModelsCache() {
  _modelCache = null;
}

export async function fetchModelsList() {
  if (_modelCache) {
    return [..._modelCache.keys()];
  }

  const res = await fetch('/api/models/list');
  const { models } = await res.json();
  _modelCache = new Map(models.map(model => [model, null]));
  return [..._modelCache.keys()];
}

export async function fetchModelLabelMappings(modelName) {
  const models = await fetchModelsList();
  if (!models.includes(modelName)) {
    throw new Error(`Model ${modelName} not found`);
  }

  let labelMapping = _modelCache.get(modelName);
  if (labelMapping == null) {
    const res = await fetch('/api/models/labels/get', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model_name: modelName }),
    });
    const { labels } = await res.json();
    labelMapping = new Map(Object.entries(labels));
    _modelCache.set(modelName, labelMapping);
  }
  return new Map(labelMapping);
}

/**
 * Associate a label with a model’s class.
 * Updates both server and local cache.
 */
export async function associateLabel(modelName, modelClass, labelUuid) {
  const models = await fetchModelsList();
  if (!models.includes(modelName)) {
    throw new Error(`Model ${modelName} not found`);
  }

  await fetch('/api/models/labels/associate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      model_name: modelName,
      model_class: modelClass,
      label_uuid: labelUuid,
    }),
  });

  // Invalidate cache
  _modelCache.set(modelName, null);
}
