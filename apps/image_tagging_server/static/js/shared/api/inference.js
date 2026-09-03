// inferenceApi.js
import { apiFetch } from './apiUtils.js';

import { generateUUID, Logger, toast } from '/app-static/js/ui/utils/index.js';

/**
 * Runs inference on the given image using the specified model.
 * @param {string} model_name - The name of the model to use for inference.
 * @param {HTMLImageElement} image - base64 image element to run inference on.
 * @returns {Promise<import('@web_api').InferenceResultModel>}
 */
export async function runDetection(model_name, image, confThresh = 0.25) {
  if (!image.src.startsWith("data:")) {
    throw new Error("Image is not a base64 data URI");
  }
  const base64String = image.src.split(',')[1]; // strip "data:image/png;base64,"

  // Prepare JSON payload
  /** @type {import('@web_api').InferencePayload} */
  const payload = {
    confThresh,
    returnAnnotatedImg: false,
    pinId: null,
    imageBase64: base64String,
  };

  toast(`Sending recognition request for model ${model_name}...`);
  // Send recognition request
  const res = await apiFetch(`/api/models/${model_name}/detect`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(payload),
  });

  if (!res.ok) {
    throw new Error(`Recognition request failed: ${res.statusText}`);
  }

  /** @type {import('@web_api').InferenceResponse} */
  const recRes = await res.json();
  const { result } = recRes;

  Logger.debug('Recognize response:', recRes);

  if (recRes.status != 'success' || !result) {
    throw new Error(`Recognition failed: ${recRes.message}`);
  }
  toast(`${result?.detections?.length} objects detected`);

  return result;
}

/**
 * Runs several models over ONE image in a single request.
 *
 * The per-model {@link runDetection} uploads its own copy of the image, and the
 * server then base64-decodes and letterboxes that identical image once per
 * model. Running a four-model stack that way costs four uploads and ~13.7 ms of
 * duplicated CPU per frame; this sends the image once and the server shares the
 * decode and (where the models agree on input size and letterboxing) the
 * preprocessing.
 *
 * Results are returned **in the order the models were requested**, not in the
 * order they finished. Callers merge overlapping detections by position —
 * first-model-wins — so a result order that depended on which model happened to
 * finish first would silently change which model's label survives.
 *
 * @param {string[]} modelNames
 * @param {HTMLImageElement} image - base64 image element to run inference on.
 * @param {number} [confThresh]
 * @returns {Promise<Array<{modelName: string, result: ?import('@web_api').InferenceResultModel, error: ?string}>>}
 */
export async function runDetectionBatch(modelNames, image, confThresh = 0.25) {
  if (!image.src.startsWith('data:')) {
    throw new Error('Image is not a base64 data URI');
  }
  const base64String = image.src.split(',')[1];

  toast(`Sending recognition request for ${modelNames.length} model(s)...`);
  const res = await apiFetch('/api/models/detect-batch', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ modelNames, confThresh, imageBase64: base64String }),
  });

  if (!res.ok) {
    throw new Error(`Recognition request failed: ${res.statusText}`);
  }

  const recRes = await res.json();
  Logger.debug('Batch recognize response:', recRes);
  if (recRes.status !== 'success') {
    throw new Error(`Recognition failed: ${recRes.message}`);
  }

  const results = recRes.results || {};
  const errors = recRes.errors || {};
  // Rebuilt from the caller's own list rather than by iterating the response:
  // that makes the order a property of the request, not of JSON object key
  // ordering across a proxy and two serializers.
  const ordered = modelNames.map(modelName => ({
    modelName,
    result: results[modelName] ?? null,
    error: errors[modelName] ?? (results[modelName] ? null : 'no result returned'),
  }));

  const total = ordered.reduce((n, r) => n + (r.result?.detections?.length ?? 0), 0);
  toast(`${total} objects detected`);
  return ordered;
}

/**
 * Run a whole hierarchical model config over one image, as a single model.
 *
 * One request runs the top model AND every sub-model beneath it, and the detections
 * that come back are already refined ("Willow", not "Cat"). No cascade is orchestrated
 * here on purpose: the second pass has to behave identically for the canvas, the
 * watcher and the review preview, so it lives in exactly one place — inference_server's
 * `detect-unit` — and this is a thin call into it.
 *
 * Returns the same `{ modelName, result, error }` shape as {@link runDetectionBatch} so
 * callers apply unit results and plain model results through the same code path, plus
 * `notes` for anything the server found but could not run (an untrained or broken
 * sub-model). Ignoring `notes` is what turns a half-trained hierarchy into detections
 * that silently stop at a parent label.
 *
 * @param {string} modelConfigUuid
 * @param {HTMLImageElement} image
 * @param {number} confThresh
 * @returns {Promise<{outcomes: Array<{modelName: string, result: object|null, error: string|null}>, notes: string[], subModelNames: string[]}>}
 */
export async function runDetectionUnit(modelConfigUuid, image, confThresh = 0.25) {
  if (!image.src.startsWith('data:')) {
    throw new Error('Image is not a base64 data URI');
  }
  const base64String = image.src.split(',')[1];

  const res = await apiFetch('/api/models/detect-unit', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ modelConfigUuid, confThresh, imageBase64: base64String }),
  });

  if (!res.ok) {
    throw new Error(`Recognition request failed: ${res.statusText}`);
  }

  const recRes = await res.json();
  Logger.debug('Unit recognize response:', recRes);
  if (recRes.status !== 'success') {
    throw new Error(`Recognition failed: ${recRes.message}`);
  }

  const results = recRes.results || {};
  const errors = recRes.errors || {};
  // Ordered by the server's detector list, which is the unit's own stack order — the
  // caller merges first-wins, so this order decides which label survives an overlap.
  const outcomes = (recRes.detectorNames || Object.keys(results)).map(modelName => ({
    modelName,
    result: results[modelName] ?? null,
    error: errors[modelName] ?? (results[modelName] ? null : 'no result returned'),
  }));

  const total = outcomes.reduce((n, r) => n + (r.result?.detections?.length ?? 0), 0);
  const subs = recRes.subModelNames || [];
  toast(`${total} objects detected${subs.length ? ` (refined by ${subs.length} sub-model(s))` : ''}`);
  return { outcomes, notes: recRes.notes || [], subModelNames: subs };
}

/**
 * List the model configs this inference server holds trained artifacts for.
 *
 * Names are NOT included: inference_server has no database, so the caller joins these
 * uuids against the model-configs API rather than trusting a second source for a name.
 *
 * @returns {Promise<Array<{modelConfigUuid: string, detectorNames: string[], subModelNames: string[], parts: object[], notes: string[]}>>}
 */
export async function fetchModelUnits() {
  const res = await apiFetch('/api/models/units');
  const body = await res.json().catch(() => ({}));
  // Throws rather than returning [] on failure, and the difference matters: callers use
  // an empty list to mean "this config has never been trained" and draw a warning from
  // it. An unreachable inference server returning [] would render that warning on every
  // healthy hierarchy — crying wolf until the real one is ignored.
  if (!res.ok || body.status !== 'success') {
    throw new Error(body.message || `Failed to list model units: ${res.statusText}`);
  }
  return body.units || [];
}

/**
 * Run an ONNX classifier on an image and return per-label predictions.
 * @param {string} model_name
 * @param {HTMLImageElement} image
 * @returns {Promise<Array<{labelUuid: string, confidence: number, isPresent: boolean}>>}
 */
export async function runClassify(model_name, image) {
  if (!image.src.startsWith('data:')) throw new Error('Image is not a base64 data URI');
  const base64String = image.src.split(',')[1];

  const res = await apiFetch(`/api/models/${model_name}/classify`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ imageBase64: base64String }),
  });

  const data = await res.json().catch(() => ({}));
  if (!res.ok || data.status !== 'success') throw new Error(data.message || 'Classify request failed');

  return data.result?.predictions ?? [];
}
