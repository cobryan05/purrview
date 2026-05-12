// imagesApi.js
import { Logger, generateUUID } from '/app-static/js/ui/utils/index.js';
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

  if (imageRes.status !== 'success' || !imageRes.image_base64) {
    throw new Error(`Invalid image API response for ${path}`);
  }

  const img = new Image();
  img.src = `data:${imageRes.mime_type};base64,${imageRes.image_base64}`;

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

 * @param {import('@canvas_types').RuntimeImage} runtimeImage
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
    .filter(bbox => bbox.labelUuid != null)
    .map(bbox => {
      if (bbox.labelUuid == null) {
        throw new Error("Invalid bbox labelUuid");
      }
      return {
        uuid: bbox.uuid,
        labelUuid: bbox.labelUuid,
        x: bbox.x / img.width,
        y: bbox.y / img.height,
        width: bbox.width / img.width,
        height: bbox.height / img.height,
        tagUuids: bbox.tagUuids || [],
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
  // TODO:???
}
