// imagesApi.js


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

  // Convert web_api format to app_types Map
  /** @type {Map<string,import('@app_types').RuntimeBbox>} */
  const bboxMap = new Map();
  (imageRes.metadata?.boxes || []).forEach(box => {

    const tagUuids = (box.tags ?? []).map(tag => tag.uuid);

    /** @type {import('@app_types').RuntimeBbox} */
    const bboxInfo = {
      uuid: box.uuid,
      x: box.x,
      y: box.y,
      width: box.width,
      height: box.height,
      confidence: undefined,
      text: undefined,
      classUuid: box.class_uuid,
      tagUuids
    };
    bboxMap.set(bboxInfo.uuid, bboxInfo);
  });

  // Block until the image is fully loaded
  await new Promise((resolve, reject) => {
    img.onload = resolve;
    img.onerror = () => reject(new Error(`Failed to decode base64 image for ${path}`));
  });

  /** @type {import('@app_types').RuntimeImage} */
  const ret = { name: path, img, bboxes: bboxMap };
  return ret;
}


/* ---- Metadata Updating --- */
/**
 * @param {UpdateMetadataPayload} metadata
 */
export async function updateImage(path, metadata) {
  const payload = {
    image_path: state.image.name,
    boxes: boxes,
    extra: {}  // optional image-level metadata (e.g., tags, reviewer, etc.)
  };

  const res = await fetch('/api/images/metadata/update', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload)
  });




  const imageRes = await fetch('/api/images/get', {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ path: path }),
  });
  if (!imageRes.ok) throw new Error(`Failed to load image via API for ${path}`);

  const imageJson = await imageRes.json();
  if (imageJson.status !== 'success' || !imageJson.content) {
    throw new Error(`Invalid image API response for ${path}`);
  }

  const img = new Image();
  img.src = `data:${imageJson.mime_type};base64,${imageJson.content}`;
  await new Promise((resolve, reject) => {
    img.onload = resolve;
    img.onerror = () =>
      reject(new Error(`Failed to decode base64 image for ${path}`));
  });
  return { image: img, bboxes: imageJson.boxes || [] };
}