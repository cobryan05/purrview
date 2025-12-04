// inferenceApi.js

import { Logger, toast, fetchClassByUuid } from '/app-static/js/ui/utils/index.js';

/**
 * @typedef {import('../../types').BBoxInfo} BBoxInfo
 * @typedef {import('../../types').InferenceResult} InferenceResult
 */

/**
 * Runs inference on the given image using the specified model.
 * @param {string} model_name - The name of the model to use for inference.
 * @param {HTMLImageElement} image - base64 image element to run inference on.
 * @returns {Promise<InferenceResult>}
 */
export async function runInference(model_name, image) {
  if (!image.src.startsWith("data:")) {
    throw new Error("Image is not a base64 data URI");
  }
  const base64String = image.src.split(',')[1]; // strip "data:image/png;base64,"

  // Prepare JSON payload
  const payload = {
    model_name: model_name,
    conf_thresh: 0.25,
    return_annotated: false,
    pin_id: null, // optional
    image_base64: base64String,
  };

  toast(`Sending recognition request for model ${model_name}...`);
  // Send recognition request
  const response = await fetch('/api/recognize', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    throw new Error(`Recognition request failed: ${response.statusText}`);
  }

  const result = await response.json();
  Logger.debug('Recognize response:', result);

  if (result.status != 'success') {
    throw new Error(`Recognition failed: ${result.message}`);
  }
  toast(`${result.detections.length} objects detected`);

  // Convert from JSON result to BBoxInfo type
  /** @type {BBoxInfo[]} */
  const bboxList = await Promise.all(
    result.detections.map(async (det) => {
      const [x, y, width, height] = det.bounding_box;
      return {
        x,
        y,
        width,
        height,
        confidence: det.confidence,
        text: (await fetchClassByUuid(det.class_uuid))?.metadata.name || det.class_name,
        classUuid: det.class_uuid || undefined,
        tagUuids: [] // TODO: set 'unverified' tag
      };
    })
  );

  /** @type {InferenceResult} */
  const ret = {
    detections: bboxList
  };

  return ret;
}
