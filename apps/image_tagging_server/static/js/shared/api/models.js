// modelsApi.js
import { createCachedFetcher } from '/app-static/js/ui/utils/data/createCachedFetcher.js';
import { wrapSingleKey, ignoreKeyReturn } from './apiUtils.js';

export const modelsFetcher = createCachedFetcher(async (keys) => {
  const res = await fetch('/api/models');
  const data = await res.json();

  if (!res.ok || data.status === 'failure') {
    throw new Error(data.message || 'Failed to fetch models list');
  }

  const { models } = data;
  // Store as Map: modelName -> labelMapping (null initially)
  const map = new Map(models.map(model => [model, null]));

  return ignoreKeyReturn(keys, map);
});

export const modelLabelsFetcher = createCachedFetcher(async (modelNames) => {
  const res = await fetch('/api/models/classes/bulk', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model_names: modelNames }),
  });

  const data = await res.json();
  if (!res.ok || data.status === 'failure') {
    throw new Error(data.message || 'Failed to fetch bulk model classes');
  }

  const modelMap = new Map();
  for (const [name, classes] of Object.entries(data.models)) {
    modelMap.set(name, new Map(Object.entries(classes)));
  }
  return modelMap;
});

/** -----------------------------
 * CACHE CLEAR
 * -----------------------------
 */
export function clearModelsCache() {
  modelsFetcher.clear();
  modelLabelsFetcher.clear();
}

/** -----------------------------
 * FETCH FUNCTIONS
 * -----------------------------
 */
export async function fetchModelsList() {
  const map = await modelsFetcher.fetch('modelsMap');
  return [...map.keys()];
}

export async function _fetchModelsLabelMappings({ modelNames }) {
  const models = await fetchModelsList();
  for (const modelName of modelNames) {
    if (!models.includes(modelName)) throw new Error(`Model ${modelName} not found`);
  }

  const modelLabelMap = await modelLabelsFetcher.fetch(modelNames);

  const modelsMap = await modelsFetcher.fetch('modelsMap');
  for (const modelName of modelNames) {
    const labels = modelLabelMap.get(modelName);
    modelsMap.set(modelName, labels);
  }

  return modelsMap;
}
export const fetchModelsLabelMappings = wrapSingleKey(_fetchModelsLabelMappings);

/** -----------------------------
 * UPDATE FUNCTION
 * -----------------------------
 */
export async function associateLabel(modelName, modelClass, labelUuid) {
  const models = await fetchModelsList();
  if (!models.includes(modelName)) throw new Error(`Model ${modelName} not found`);

  // Path updated to /api/models/{model_name}/classes/{class_name}/label
  // Method updated to PUT as suggested for RESTful updates
  const res = await fetch(`/api/models/${modelName}/classes/${modelClass}/label`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      label_uuid: labelUuid,
    }),
  });

  const data = await res.json();

  if (!res.ok || data.status === 'failure') {
    throw new Error(data.message || 'Failed to associate label');
  }

  // Invalidate label mapping cache for this model
  modelLabelsFetcher.delete(modelName);

  const modelsMap = await modelsFetcher.fetch('modelsMap');
  modelsMap.set(modelName, null);
}