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
  // Store as Map: modelName -> classMapping (null initially)
  const map = new Map(models.map(model => [model, null]));
  return map;
});

export const modelClassesFetcher = createCachedFetcher(async (modelName) => {
  const res = await fetch('/api/models/classes/get', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model_name: modelName }),
  });

  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.message || `Failed to fetch classes for model ${modelName}`);
  }

  const { classes } = await res.json();
  return new Map(Object.entries(classes));
});


/** -----------------------------
 * CACHE CLEAR
 * -----------------------------
 */
export function clearModelsCache() {
  modelsFetcher.clear();
  modelClassesFetcher.clear();
}


/** -----------------------------
 * FETCH FUNCTIONS
 * -----------------------------
 */
export async function fetchModelsList() {
  const map = await modelsFetcher.fetch('modelsMap');
  return [...map.keys()];
}

export async function fetchModelClassMappings(modelName) {
  const models = await fetchModelsList();
  if (!models.includes(modelName)) throw new Error(`Model ${modelName} not found`);

  const classes = await modelClassesFetcher.fetch(modelName);
  // Ensure we also update the models map with the class mapping
  const modelsMap = await modelsFetcher.fetch('modelsMap');
  modelsMap.set(modelName, classes);

  return new Map(classes);
}


/** -----------------------------
 * UPDATE FUNCTION
 * -----------------------------
 */
export async function associateClass(modelName, modelClass, classUuid) {
  const models = await fetchModelsList();
  if (!models.includes(modelName)) throw new Error(`Model ${modelName} not found`);

  const res = await fetch('/api/models/classes/associate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      model_name: modelName,
      model_class: modelClass,
      class_uuid: classUuid,
    }),
  });

  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.message || 'Failed to associate class');
  }

  // Invalidate class mapping cache for this model
  modelClassesFetcher.delete(modelName);
  // Optionally also reset the entry in models map
  const modelsMap = await modelsFetcher.fetch('modelsMap');
  modelsMap.set(modelName, null);
}
