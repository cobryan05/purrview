// inferenceApi.js

import { fetchLabelByUuid } from '/app-static/js/shared/api/labels.js';
import { generateUUID, Logger, toast } from '/app-static/js/ui/utils/index.js';

/**
 * Runs inference on the given image using the specified model.
 * @param {string} model_name - The name of the model to use for inference.
 * @param {HTMLImageElement} image - base64 image element to run inference on.
 * @returns {Promise<import('@web_api').InferenceResultModel>}
 */
export async function runInference(model_name, image) {
  if (!image.src.startsWith("data:")) {
    throw new Error("Image is not a base64 data URI");
  }
  const base64String = image.src.split(',')[1]; // strip "data:image/png;base64,"

  // Prepare JSON payload
  /** @type {import('@web_api').RecognizePayload} */
  const payload = {
    model_name: model_name,
    conf_thresh: 0.25,
    return_annotated_img: false,
    pin_id: null, // optional
    image_base64: base64String,
  };

  toast(`Sending recognition request for model ${model_name}...`);
  // Send recognition request
  const res = await fetch('/api/recognize', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(payload),
  });

  if (!res.ok) {
    throw new Error(`Recognition request failed: ${res.statusText}`);
  }

  /** @type {import('@web_api').RecognizeResponse} */
  const recRes = await res.json();
  const { result } = recRes;

  Logger.debug('Recognize response:', recRes);

  if (recRes.status != 'success' || !result) {
    throw new Error(`Recognition failed: ${recRes.message}`);
  }
  toast(`${result?.detections?.length} objects detected`);

  return result;
}
