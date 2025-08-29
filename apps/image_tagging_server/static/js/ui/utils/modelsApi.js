let _models = new Set();                     // just the names
let _modelLabelMappings = new Map();         // modelName → { class → label_uuid }

/**
 * Refresh the model list cache from /api/models/list.
 */
export async function refreshModelCache() {
  const res = await fetch('/api/models/list');
  const { models } = await res.json();
  _models = new Set(models);
  // Don't clear mappings; they stay cached until refreshed explicitly
}

/**
 * Return all known model names as an array.
 */
export function getAllModelNames() {
  return Array.from(_models);
}

/**
 * Ensure class→label mappings for a model are loaded into cache.
 */
async function ensureModelMappings(modelName) {
  if (!_modelLabelMappings.has(modelName)) {
    const res = await fetch('/api/models/labels/get', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model_name: modelName }),
    });
    const { labels: classMap } = await res.json(); // { className: label_uuid }
    _modelLabelMappings.set(modelName, classMap);
  }
}

/**
 * Get class→label mappings for a given model.
 * Fetches from server if not already cached.
 */
export async function getModelMappings(modelName) {
  await ensureModelMappings(modelName);
  return _modelLabelMappings.get(modelName);
}

/**
 * Associate a label with a model’s class.
 * Updates both server and local cache.
 */
export async function associateLabel(modelName, modelClass, labelUuid) {
  await fetch('/api/models/labels/associate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      model_name: modelName,
      model_class: modelClass,
      label_uuid: labelUuid,
    }),
  });

  // Update cache
  await ensureModelMappings(modelName);
  const classMap = _modelLabelMappings.get(modelName);
  classMap[modelClass] = labelUuid;
}
