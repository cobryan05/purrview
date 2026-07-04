// modelsApi.js
import { createCachedFetcher } from '/app-static/js/ui/utils/data/createCachedFetcher.js';
import { apiFetch, wrapSingleKey, ignoreKeyReturn } from './apiUtils.js';

export const modelsFetcher = createCachedFetcher(async (keys) => {
  const res = await apiFetch('/api/models');
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
  const res = await apiFetch('/api/models/classes/bulk', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ modelNames }),
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
  _modelKindCache.clear();
}

/** @type {Map<string, string>} model name → modelKind */
const _modelKindCache = new Map();

/**
 * Fetch the model_kind for each name in *modelNames*.
 * Results are cached for the session (kinds are stable until models dir changes).
 * @param {string[]} modelNames
 * @returns {Promise<Map<string, string>>} name → modelKind ("image_classifier" | "multi_label_classifier" | "detector")
 */
export async function fetchModelsKinds(modelNames) {
  const result = new Map();
  const missing = modelNames.filter(n => !_modelKindCache.has(n));

  await Promise.all(missing.map(async name => {
    try {
      const res = await apiFetch(`/api/models/${name}/meta`);
      const data = await res.json().catch(() => ({}));
      const kind = (res.ok && data.status === 'success') ? (data.modelKind ?? 'detector') : 'detector';
      _modelKindCache.set(name, kind);
    } catch {
      _modelKindCache.set(name, 'detector');
    }
  }));

  for (const name of modelNames) {
    result.set(name, _modelKindCache.get(name) ?? 'detector');
  }
  return result;
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

/**
 * Fetch the training and validation image UUIDs stored in a model's sidecar JSON.
 * @param {string} modelName
 * @returns {Promise<{train: string[], val: string[]}>}
 */
export async function fetchModelTrainingImages(modelName) {
  const res = await apiFetch(`/api/models/${encodeURIComponent(modelName)}/training-images`);
  const data = await res.json();
  if (!res.ok || data.status === 'failure') {
    throw new Error(data.message || 'Failed to fetch training images');
  }
  return { train: data.train_image_uuids ?? [], val: data.val_image_uuids ?? [] };
}

/**
 * Fetch the device preference ("auto" | "gpu" | "cpu") stored in a model's sidecar JSON.
 * @param {string} modelName
 * @returns {Promise<string>}
 */
export async function fetchModelDevice(modelName) {
  const res = await apiFetch(`/api/models/${encodeURIComponent(modelName)}/meta`);
  const data = await res.json().catch(() => ({}));
  return (res.ok && data.status === 'success') ? (data.device ?? 'auto') : 'auto';
}

/**
 * Set the device preference ("auto" | "gpu" | "cpu") for a model, persisted to its sidecar JSON.
 * @param {string} modelName
 * @param {string} device
 * @returns {Promise<boolean>}
 */
export async function setModelDevice(modelName, device) {
  const res = await apiFetch(`/api/models/${encodeURIComponent(modelName)}/device`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ device }),
  });

  const data = await res.json();
  if (!res.ok || data.status === 'failure') {
    throw new Error(data.message || 'Failed to set model device');
  }
  return data.deviceSet;
}

/** -----------------------------
 * UPDATE FUNCTION
 * -----------------------------
 */
export async function associateLabel(modelName, modelClass, labelUuid) {
  const models = await fetchModelsList();
  if (!models.includes(modelName)) throw new Error(`Model ${modelName} not found`);

  // Path updated to /api/models/{model_name}/classes/{class_name}/label
  // Method updated to PUT as suggested for RESTful updates
  const res = await apiFetch(`/api/models/${modelName}/classes/${modelClass}/label`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      labelUuid: labelUuid,
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