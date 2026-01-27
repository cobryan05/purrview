// imagesApi.js
import { Logger, generateUUID } from '/app-static/js/ui/utils/index.js';

/**
 * Fetch a list of files from the API.
 *
 * @param {string} path - The folder path to list files from.
 * @returns {Promise<import('@web_api').FileEntry[]>} Array of file entries from the server.
 * @throws {Error} If the fetch fails or the API returns an error status.
 */
export async function fetchImageList(path) {
  /** @type {import('@web_api').ListFilesPayload} */
  const payload = { path };

  const res = await fetch('/api/images/list', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });

  /** @type {import('@web_api').ListFilesResponse} */
  const data = await res.json();

  if (!res.ok || data.status !== 'success') {
    throw new Error(data.message || 'Failed to fetch image list');
  }

  return data.files || [];
}

/**
 * Fetch an image and its metadata from the server.
 *
 * @param {string} path - Path of the image to fetch.
 * @returns {Promise<import('@app_types').RuntimeImage>}
 *          Object containing the loaded Image element and the bounding boxes.
 * @throws {Error} If the fetch fails or the API response is invalid.
 */
export async function fetchImage(path) {
  /** @type {import('@web_api').GetFilePayload} */
  const payload = { path };

  const res = await fetch('/api/images/get', {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  if (!res.ok) {
    const errData = await res.json().catch(() => ({}));
    throw new Error(errData.message || `Failed to load image via API for ${path}`);
  }

  /** @type {import('@web_api').GetFileResponse} */
  const imageRes = await res.json();

  if (imageRes.status !== 'success' || !imageRes.image_base64) {
    throw new Error(`Invalid image API response for ${path}`);
  }

  const img = new Image();
  img.src = `data:${imageRes.mime_type};base64,${imageRes.image_base64}`;

  // Block until the image is fully loaded to resolve dimensions
  await new Promise((resolve, reject) => {
    img.onload = resolve;
    img.onerror = () => reject(new Error(`Failed to decode base64 image for ${path}`));
  });

  // Convert web_api format to app_types Map
  /** @type {Map<string,import('@app_types').RuntimeBboxInfo>} */
  const bboxMap = new Map();
  (imageRes.metadata?.boxes || []).forEach(box => {

    const tagUuids = (box.tag_uuids ?? []).map(tag => tag);
    const absX = box.x * img.width;
    const absY = box.y * img.height;
    const absHeight = box.height * img.height;
    const absWidth = box.width * img.width;

    /** @type {import('@app_types').RuntimeBboxInfo} */
    const bboxInfo = {
      uuid: box.uuid ?? generateUUID(),
      x: absX,
      y: absY,
      width: absWidth,
      height: absHeight,
      confidence: undefined,
      text: undefined,
      classUuid: box.class_uuid,
      tagUuids
    };
    bboxMap.set(bboxInfo.uuid, bboxInfo);
  });

  /** @type {import('@app_types').RuntimeImage} */
  const ret = { name: path, img, bboxes: bboxMap };
  return ret;
}


/* ---- Metadata Updating --- */
/**

 * @param {import('@app_types').RuntimeImage} runtimeImage
 * @throws {Error} If the update fails or the API response is invalid.
 */
export async function updateImage(runtimeImage) {
  const img = runtimeImage.img;
  if (!runtimeImage.name || !img || !runtimeImage.bboxes) {
    throw new Error("No image name provided");
  }
  // Convert from runtime type to web api type

  /** @type {import('@web_api').BoundingBoxInput[]} */
  const boxes = Array.from(runtimeImage.bboxes.values())
    .filter(bbox => bbox.classUuid != null)
    .map(bbox => {
      if (bbox.classUuid == null) {
        throw new Error("Invalid bbox classUuid");
      }
      return {
        uuid: bbox.uuid,
        class_uuid: bbox.classUuid,
        x: bbox.x / img.width,
        y: bbox.y / img.height,
        width: bbox.width / img.width,
        height: bbox.height / img.height,
        tag_uuids: bbox.tagUuids || [],
        extra: null
      };
    });

  /** @type {import('@web_api').UpdateMetadataPayload} */
  const payload = {
    image_path: runtimeImage.name,
    boxes: boxes,
    extra: null
  };

  const res = await fetch('/api/images/metadata/update', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload)
  });


  if (!res.ok) {
    const errData = await res.json().catch(() => ({}));
    throw new Error(errData.message || `Failed to update image via API for ${runtimeImage.name}`);
  }

  /** @type {import('@web_api').UpdateMetadataResponse} */
  const updatRes = await res.json();

  if (updatRes.status !== 'success') {
    throw new Error(`${updatRes.message}`);
  }
}
