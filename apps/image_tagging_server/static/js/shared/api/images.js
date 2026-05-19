// imagesApi.js
import { imageRecordToUI } from '/app-static/js/shared/domain/image/mapper.js';

/**
 * Fetch a list of files from the API.
 *
 * @param {string} path - The folder path to list files from.
 * @param {string} [pattern="*"] - Glob pattern to filter files.
 * @param {boolean} [recursive=false] - Whether to search subdirectories.
 * @returns {Promise<import('@web_api').FileEntry[]>} Array of file entries from the server.
 * @throws {Error} If the fetch fails or the API returns an error status.
 */
export async function fetchImageList(path, pattern = "*", recursive = false) {
  // Construct the URL with search parameters
  const params = new URLSearchParams({
    path: path,
    pattern: pattern,
    recursive: recursive.toString(),
  });

  const res = await fetch(`/api/images/list?${params.toString()}`, {
    method: 'GET',
    headers: { 'Accept': 'application/json' },
  });

  /** @type {import('@web_api').ListFilesResponse} */
  const data = await res.json();

  // Check both the HTTP status and our custom status field
  if (!res.ok || data.status !== 'success') {
    throw new Error(data.message || 'Failed to fetch image list');
  }

  return data.files || [];
}

/**
 * Fetch an image and its metadata from the server using the new GET endpoint.
 *
 * @param {string} path - Relative path of the image to fetch.
 * @returns {Promise<import('@domain_image').UIImage>}
 * @throws {Error} If the fetch fails or the API response is invalid.
 */
export async function fetchImage(path) {
  const url = `/api/images/${encodeURIComponent(path)}/file`;

  const res = await fetch(url, {
    method: "GET",
    headers: { "Accept": "application/json" },
  });

  if (!res.ok) {
    const errData = await res.json().catch(() => ({}));
    throw new Error(errData.message || `Failed to load image via API for ${path}`);
  }

  /** @type {import('@web_api').GetFileResponse} */
  const imageRes = await res.json();

  if (imageRes.status !== 'success' || !imageRes.imageBase64 || !imageRes.image) {
    throw new Error(`Invalid image API response for ${path}`);
  }

  const img = new Image();
  img.src = `data:${imageRes.mimeType ?? 'image/jpeg'};base64,${imageRes.imageBase64}`;

  // Block until the image is fully decoded so we have access to width/height
  await new Promise((resolve, reject) => {
    img.onload = resolve;
    img.onerror = () => reject(new Error(`Failed to decode base64 image for ${path}`));
  });

  const ret = imageRecordToUI(imageRes.image, img, path);
  return ret;
}

/* ---- Metadata Updating --- */
/**
 * @param {import('@domain_image').UIImage} uiImage
 * @throws {Error} If the update fails or the API response is invalid.
 */
export async function updateImage(uiImage) {
  if (!uiImage.image_path || !uiImage.bboxes) {
    throw new Error("No image path provided");
  }

  /** @type {import('@web_api').BoundingBoxMetadataModel[]} */
  const boxes = Array.from(uiImage.bboxes.values())
    .filter(bbox => bbox.label?.uuid != null)
    .map(bbox => ({
      uuid: bbox.uuid,
      labelUuid: /** @type {string} */ (bbox.label?.uuid),
      x: bbox.x,
      y: bbox.y,
      width: bbox.width,
      height: bbox.height,
      tagUuids: [],
    }));

  /** @type {import('@web_api').UpdateMetadataPayload} */
  const payload = {
    imagePath: uiImage.image_path,
    boxes,
  };

  const res = await fetch('/api/images/metadata/update', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload)
  });

  if (!res.ok) {
    const errData = await res.json().catch(() => ({}));
    throw new Error(errData.message || `Failed to update image via API for ${uiImage.image_path}`);
  }

  /** @type {import('@web_api').UpdateMetadataResponse} */
  const updateRes = await res.json();

  if (updateRes.status !== 'success') {
    throw new Error(`${updateRes.message}`);
  }
}
