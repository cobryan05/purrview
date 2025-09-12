// modelsApi.js
import { createCachedFetcher } from './createCachedFetcher.js';

/** -----------------------------
 * MODELS CACHE
 * -----------------------------
 */
export const modelsFetcher = createCachedFetcher(async () => {
  const res = await fetch('/api/models/list');
  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.message || 'Failed to fetch models list');
  }
  const { models } = await res.json();
  // Store as Map: modelName -> labelMapping (null initially)
  const map = new Map(models.map(model => [model, null]));
  return map;
});

export const modelLabelsFetcher = createCachedFetcher(async (modelName) => {
  const res = await fetch('/api/models/labels/get', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model_name: modelName }),
  });

  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.message || `Failed to fetch labels for model ${modelName}`);
  }

  const { labels } = await res.json();
  return new Map(Object.entries(labels));
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

export async function fetchModelLabelMappings(modelName) {
  const models = await fetchModelsList();
  if (!models.includes(modelName)) throw new Error(`Model ${modelName} not found`);

  const labels = await modelLabelsFetcher.fetch(modelName);

  // Ensure we also update the models map with the label mapping
  const modelsMap = await modelsFetcher.fetch('modelsMap');
  modelsMap.set(modelName, labels);

  return new Map(labels);
}


/** -----------------------------
 * UPDATE FUNCTION
 * -----------------------------
 */
export async function associateLabel(modelName, modelClass, labelUuid) {
  const models = await fetchModelsList();
  if (!models.includes(modelName)) throw new Error(`Model ${modelName} not found`);

  const res = await fetch('/api/models/labels/associate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      model_name: modelName,
      model_class: modelClass,
      label_uuid: labelUuid,
    }),
  });

  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.message || 'Failed to associate label');
  }

  // Invalidate label mapping cache for this model
  modelLabelsFetcher.delete(modelName);
  // Optionally also reset the entry in models map
  const modelsMap = await modelsFetcher.fetch('modelsMap');
  modelsMap.set(modelName, null);
}
