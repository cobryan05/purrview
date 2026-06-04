// imagesApi.js
import { apiFetch } from './apiUtils.js';
import { imageRecordToUI } from '/app-static/js/shared/domain/image/mapper.js';

/**
 * Fetch a list of files from the API.
 *
 * @param {string} path - The folder path to list files from.
 * @param {string} [pattern="*"] - Glob pattern to filter files.
 * @param {boolean} [recursive=false] - Whether to search subdirectories.
 * @param {boolean} [includeDbStatus=false] - When true, each file entry's `uuid` field is
 *   populated if the file is registered in the image database.
 * @returns {Promise<import('@web_api').FileEntryRead[]>} Array of file entries from the server.
 * @throws {Error} If the fetch fails or the API returns an error status.
 */
export async function fetchImageList(path, pattern = "*", recursive = false, includeDbStatus = false) {
  const params = new URLSearchParams({
    path,
    pattern,
    recursive: recursive.toString(),
    include_db_status: includeDbStatus.toString(),
  });

  const res = await apiFetch(`/api/images/list?${params.toString()}`, {
    method: 'GET',
    headers: { 'Accept': 'application/json' },
  });

  /** @type {import('@web_api').ListFilesResponse} */
  const data = await res.json();

  if (!res.ok || data.status !== 'success') {
    throw new Error(data.message || 'Failed to fetch image list');
  }

  return data.files || [];
}

/**
 * Create a directory under FILES_ROOT.
 *
 * @param {string} path - Relative path of the directory to create.
 * @throws {Error} If the request fails or the server returns an error.
 */
export async function createDirectory(path) {
  const res = await apiFetch('/api/images/mkdir', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ path }),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok || data.status !== 'success') {
    throw new Error(data.message || 'Failed to create directory');
  }
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

  const res = await apiFetch(url, {
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
      tagUuids: (bbox.tags ?? []).map(t => t.uuid),
    }));

  /** @type {import('@web_api').UpdateMetadataPayload} */
  const payload = {
    imagePath: uiImage.image_path,
    boxes,
  };

  const res = await apiFetch('/api/images/metadata/update', {
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
