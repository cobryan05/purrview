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

/**
 * Drop the cached model list *and* tell every open picker to rebuild from it.
 *
 * Call this from anywhere the set of models changes — an upload, a delete, a
 * finished import/convert/train run. `clearModelsCache()` on its own only helps
 * the caller: a panel that was built earlier keeps the list it already rendered,
 * which is why a freshly trained model needed a page reload before it appeared
 * anywhere. Ordering matters — the cache is cleared first so subscribers that
 * refetch synchronously do not re-read what was just invalidated.
 *
 * Imported lazily so `shared/api/*` keeps its one-way dependency on the event bus
 * out of the module graph's hot path; the publish is fire-and-forget either way.
 */
export function notifyModelsChanged() {
  clearModelsCache();
  import('/app-static/js/shared/events/index.js')
    .then(({ events, EventTypes }) => events.publish(EventTypes.MODELS_UPDATED, {}))
    .catch(() => { /* the cache is already clear; the next fetch is correct regardless */ });
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
 * Fetch the training and validation image UUIDs stored in a model's sidecar JSON,
 * plus the run provenance (`trainedAt`, `modelConfigNodeUuid`) a caller needs to
 * confirm the model really is the one a given training config last produced.
 * @param {string} modelName
 * @returns {Promise<{train: string[], val: string[], trainedAt: string|null, modelConfigNodeUuid: string|null}>}
 */
export async function fetchModelTrainingImages(modelName) {
  const res = await apiFetch(`/api/models/${encodeURIComponent(modelName)}/training-images`);
  const data = await res.json();
  if (!res.ok || data.status === 'failure') {
    throw new Error(data.message || 'Failed to fetch training images');
  }
  // ModelTrainingImagesResponse inherits StatusResponse's to_camel alias generator,
  // so the wire fields are camelCase. This used to read only the snake_case names
  // and therefore always resolved to empty arrays — the Training Data modal
  // reported "no training image UUIDs recorded" for every model, including ones
  // whose sidecar had a full manifest. Snake_case is kept as a fallback so an
  // older/by-name-serialised response still works.
  return {
    train: data.trainImageUuids ?? data.train_image_uuids ?? [],
    val: data.valImageUuids ?? data.val_image_uuids ?? [],
    trainedAt: data.trainedAt ?? data.trained_at ?? null,
    modelConfigNodeUuid: data.modelConfigNodeUuid ?? data.model_config_node_uuid ?? null,
  };
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

/**
 * Fetch the curated base-model list (key + display name + whether it needs
 * training-server conversion) offered for cold-start import. The full list
 * (including onnx_url/pt_url/model_family) lives server-side in
 * apps/helpers/curatedModels.py.
 * @returns {Promise<Array<{key: string, displayName: string, requiresConversion: boolean}>>}
 */
export async function fetchCuratedModels() {
  const res = await apiFetch('/api/models/curated');
  const data = await res.json();
  if (!res.ok || data.status === 'failure') {
    throw new Error(data.message || 'Failed to fetch curated models list');
  }
  return data.models ?? [];
}

/**
 * Lightweight reachability check for the training server, used to enable/
 * disable "needs conversion" curated downloads in the UI. Reuses the existing
 * `/api/training/status` health endpoint (proxied straight through to the
 * training server) — no new backend endpoint needed.
 * @returns {Promise<boolean>}
 */
export async function isTrainingServerReachable() {
  try {
    const res = await apiFetch('/api/training/status');
    if (!res.ok) return false;
    const data = await res.json();
    return data.status === 'success';
  } catch {
    return false;
  }
}

/**
 * Upload a local .onnx file to the models directory; the server parses its
 * classes from the ONNX's own `names` metadata and writes a sidecar.
 * @param {{modelName: string, onnxBase64: string, overwrite?: boolean}} args
 * @returns {Promise<{modelName: string, classes: string[]}>}
 */
export async function importModelUpload({ modelName, onnxBase64, overwrite = false }) {
  const res = await apiFetch('/api/models/import/upload', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ modelName, onnxBase64, overwrite }),
  });

  const data = await res.json();
  if (!res.ok || data.status === 'failure') {
    throw new Error(data.message || 'Failed to import model');
  }
  return { modelName: data.modelName, classes: data.classes ?? [] };
}

/**
 * Delete a model's ONNX weights and sidecar JSON from the models directory.
 * @param {string} modelName
 * @returns {Promise<void>}
 */
export async function deleteModel(modelName) {
  const res = await apiFetch(`/api/models/${encodeURIComponent(modelName)}`, {
    method: 'DELETE',
  });

  const data = await res.json();
  if (!res.ok || data.status === 'failure') {
    throw new Error(data.message || 'Failed to delete model');
  }
}

/**
 * Detect the format of a to-be-imported model from a small header slice, and
 * report whether server-side conversion to ONNX is supported for it.
 * @param {{filename: string, headerBase64: string}} args
 * @returns {Promise<{format: string, isOnnx: boolean, conversionSupported: boolean}>}
 */
export async function detectModel({ filename, headerBase64 }) {
  const res = await apiFetch('/api/models/detect', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ filename, headerBase64 }),
  });

  const data = await res.json();
  if (!res.ok || data.status === 'failure') {
    throw new Error(data.message || 'Failed to detect model format');
  }
  return { format: data.format, isOnnx: data.isOnnx, conversionSupported: data.conversionSupported };
}

/**
 * Stage a non-ONNX source model file (uploaded in full) on the training
 * server ahead of a background conversion task.
 * @param {{sourceBase64: string}} args
 * @returns {Promise<string>} stagingId
 */
export async function stageSourceModel({ sourceBase64 }) {
  const res = await apiFetch('/api/models/import/stage', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ sourceBase64 }),
  });

  const data = await res.json();
  if (!res.ok || data.status === 'failure') {
    throw new Error(data.message || 'Failed to stage model for conversion');
  }
  return data.stagingId;
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