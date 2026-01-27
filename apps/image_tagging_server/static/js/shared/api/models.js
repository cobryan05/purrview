// modelsApi.js
import { createCachedFetcher } from '/app-static/js/ui/utils/data/createCachedFetcher.js';
import { wrapSingleKey, ignoreKeyReturn } from './apiUtils.js';
/** -----------------------------
 * MODELS CACHE
 * -----------------------------
 */
export const modelsFetcher = createCachedFetcher(async (keys) => {
  const res = await fetch('/api/models/list');
  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.message || 'Failed to fetch models list');
  }
  const { models } = await res.json();
  // Store as Map: modelName -> classMapping (null initially)
  const map = new Map(models.map(model => [model, null]));

  return ignoreKeyReturn(keys, map);
});

export const modelClassesFetcher = createCachedFetcher(async (modelNames) => {
  const res = await fetch('/api/models/classes/get', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model_names: modelNames }),
  });

  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.message || `Failed to fetch classes for model ${modelName}`);
  }

  const { models } = await res.json();

  const modelMap = new Map();
  for (const [modelName, classObj] of Object.entries(models)) {
    // Convert inner object → Map
    const innerMap = new Map(Object.entries(classObj));
    modelMap.set(modelName, innerMap);
  }

  return modelMap;
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

export async function _fetchModelsClassMappings({modelNames}) {
  const models = await fetchModelsList();
  for (const modelName of modelNames) {
    if (!models.includes(modelName)) throw new Error(`Model ${modelName} not found`);
  }

  const modelClassMap = await modelClassesFetcher.fetch(modelNames);
  // Ensure we also update the models map with the class mapping
  const modelsMap = await modelsFetcher.fetch('modelsMap');
  for (const modelName of modelNames) {
    const classes = modelClassMap.get(modelName);
    modelsMap.set(modelName, classes);
  }

  return modelsMap;
}
export const fetchModelsClassMappings = wrapSingleKey(_fetchModelsClassMappings);


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
