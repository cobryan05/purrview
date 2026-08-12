// imagesApi.js
import { apiFetch } from './apiUtils.js';
import { imageRecordToUI } from '/app-static/js/shared/domain/image/mapper.js';
import { events, EventTypes } from '/app-static/js/shared/events/index.js';

/**
 * Fetch a page of files from the API.
 *
 * @param {string} path - The folder path to list files from.
 * @param {string} [pattern="*"] - Glob pattern to filter files.
 * @param {boolean} [recursive=false] - Whether to search subdirectories.
 * @param {boolean} [includeDbStatus=false] - When true, each file entry's `uuid` field is
 *   populated if the file is registered in the image database.
 * @param {{ offset?: number, limit?: number }} [pagination] - Pagination options.
 * @returns {Promise<{ files: import('@web_api').FileEntryRead[], hasMore: boolean }>}
 * @throws {Error} If the fetch fails or the API returns an error status.
 */
export async function fetchImageList(path, pattern = "*", recursive = false, includeDbStatus = false, { offset = 0, limit = 200 } = {}) {
  const params = new URLSearchParams({
    path,
    pattern,
    recursive: recursive.toString(),
    include_db_status: includeDbStatus.toString(),
    offset: offset.toString(),
    limit: limit.toString(),
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

  return { files: data.files || [], hasMore: data.hasMore ?? false };
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
  if (!res.ok) {
    const errData = await res.json().catch(() => ({}));
    throw new Error(errData.message || `Failed to create directory (HTTP ${res.status})`);
  }
  const data = await res.json();
  if (data.status !== 'success') {
    throw new Error(data.message || 'Failed to create directory');
  }
}

/**
 * Fetch + decode one image, with no caching. The expensive half of `fetchImage`.
 *
 * `/file` is a single request that reads the whole file, base64s it into the
 * JSON body AND does the image-metadata DB read, so its latency is the sum of
 * both — which is what makes it worth prefetching over a slow DB.
 *
 * @param {string} path
 * @returns {Promise<{imageRes: import('@web_api').GetFileResponse, img: HTMLImageElement}>}
 */
async function _fetchImageRaw(path) {
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

  return { imageRes, img };
}

/**
 * Bounded LRU of in-flight-or-settled `_fetchImageRaw` promises, keyed by path.
 *
 * Deliberately holds the RAW response + decoded element rather than the mapped
 * `UIImage`: a canvas mutates the UIBbox objects it is handed, so serving two
 * loads the same mapped object would let an edit in one canvas appear in the
 * other. `fetchImage` re-maps per call, giving every caller its own boxes while
 * still sharing the one expensive thing — the decoded bitmap.
 *
 * Small on purpose: entries pin full-resolution decoded images, not thumbnails.
 * That bound is also why this is not `createCachedFetcher` — that helper never
 * evicts, which is fine for labels and tags and not for decoded bitmaps.
 */
const IMAGE_CACHE_MAX = 5;
/** @type {Map<string, Promise<{imageRes: object, img: HTMLImageElement}>>} */
const _imageCache = new Map();

function _cachedFetchImage(path) {
  const hit = _imageCache.get(path);
  if (hit) {
    // Re-insert so Map iteration order stays least-recent-first for eviction.
    _imageCache.delete(path);
    _imageCache.set(path, hit);
    return hit;
  }

  const pending = _fetchImageRaw(path);
  _imageCache.set(path, pending);
  // A failure must not be cached, or one transient error makes the path
  // permanently unloadable for as long as the entry survives.
  pending.catch(() => {
    if (_imageCache.get(path) === pending) _imageCache.delete(path);
  });

  for (const oldest of _imageCache.keys()) {
    if (_imageCache.size <= IMAGE_CACHE_MAX) break;
    _imageCache.delete(oldest);
  }
  return pending;
}

/**
 * Drop a path's cached image (or the whole cache when called with no path).
 *
 * Anything that changes what `/file` would return for an image MUST call this,
 * or navigation can land back on a pre-edit copy. The mutation events below
 * cover the app's own writes; callers doing an explicit "reload from server"
 * should pass `allowCached: false` instead of relying on invalidation ordering.
 *
 * @param {string} [path]
 */
export function invalidateCachedImage(path) {
  if (path === undefined) _imageCache.clear();
  else _imageCache.delete(path);
}

/**
 * Warm the cache for a path without blocking or throwing.
 *
 * Errors are swallowed on purpose: a prefetch is speculative, and a failure
 * here must not surface a toast for an image the user may never open. The real
 * load re-fetches and reports the error itself.
 *
 * @param {string} path
 */
export function prefetchImage(path) {
  if (!path) return;
  _cachedFetchImage(path).catch(() => {});
}

// Invalidate on every committed change to an image's boxes, and on deletion.
// IMAGE_ANNOTATIONS_SAVED / IMAGE_BBOXES_CHANGED both carry `imageFilename`,
// which is the same key `loadImage` uses.
events.subscribe(EventTypes.IMAGE_ANNOTATIONS_SAVED, ({ imageFilename } = {}) => invalidateCachedImage(imageFilename));
events.subscribe(EventTypes.IMAGE_BBOXES_CHANGED, ({ imageFilename } = {}) => invalidateCachedImage(imageFilename));
events.subscribe(EventTypes.IMAGE_DELETED, ({ imagePath } = {}) => invalidateCachedImage(imagePath));

/**
 * Fetch an image and its metadata from the server using the new GET endpoint.
 *
 * @param {string} path - Relative path of the image to fetch.
 * @param {{allowCached?: boolean}} [opts] - `allowCached` lets the call be
 *   served from (and populate) the prefetch cache. Off by default: an explicit
 *   reload — after a save, or on an external-change refresh — must see the
 *   server's current state, not a copy taken before the write.
 * @returns {Promise<import('@domain_image').UIImage>}
 * @throws {Error} If the fetch fails or the API response is invalid.
 */
export async function fetchImage(path, { allowCached = false } = {}) {
  const { imageRes, img } = allowCached ? await _cachedFetchImage(path) : await _fetchImageRaw(path);
  return imageRecordToUI(imageRes.image, img, path);
}

/**
 * Fetch lightweight metadata (including bboxes) for an image, without binary data.
 * Returns null if the image isn't registered in the database.
 *
 * @param {string} path - Relative path of the image.
 * @returns {Promise<import('@web_api').ImageRecordRead|null>}
 * @throws {Error} If the request fails.
 */
export async function fetchImageMetadata(path) {
  const res = await apiFetch('/api/images/metadata/get', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ imagePath: path }),
  });

  /** @type {import('@web_api').GetImageMetadataResponse} */
  const data = await res.json();
  if (!res.ok || data.status !== 'success') return null;
  return data.image ?? null;
}

/* ---- Metadata Updating --- */
/**
 * @param {import('@domain_image').UIImage} uiImage
 * @throws {Error} If the update fails or the API response is invalid.
 */
/**
 * Find images similar to the given image UUID.
 * @param {string} imageUuid
 * @param {number} threshold
 * @returns {Promise<Array<{imageUuid: string, distance: number}>>}
 */
export async function findSimilarImagesByUuid(imageUuid, threshold = 5) {
  const res = await apiFetch('/api/images/find-similar-by-image-uuid', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ imageUuid, threshold }),
  });
  const data = await res.json();
  if (!res.ok || data.status !== 'success') throw new Error(data.message || 'Failed to find similar images');
  return (data.matches ?? []).map(m => ({ imageUuid: m.imageUuid, distance: m.distance }));
}

/**
 * Find bboxes similar to the given bbox UUID.
 * @param {string} bboxUuid
 * @param {number} threshold
 * @param {string|null} [labelUuid]
 * @returns {Promise<Array<{bboxUuid: string, imageUuid: string|null, labelUuid: string, distance: number}>>}
 */
export async function findSimilarBboxesByUuid(bboxUuid, threshold = 8, labelUuids = [], tagUuids = [], excludeTagUuids = [], tagMatchMode = 'any') {
  const res = await apiFetch('/api/images/bbox/find-similar-by-bbox-uuid', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ bboxUuid, threshold, labelUuids, tagUuids, excludeTagUuids, tagMatchMode }),
  });
  const data = await res.json();
  if (!res.ok || data.status !== 'success') throw new Error(data.message || 'Failed to find similar bboxes');
  return (data.matches ?? []).map(m => ({
    bboxUuid: m.bboxUuid,
    imageUuid: m.imageUuid ?? null,
    labelUuid: m.labelUuid,
    distance: m.distance,
  }));
}

export async function findDissimilarImagesByUuid(imageUuid, count = 100, candidateUuids = null) {
  const res = await apiFetch('/api/images/find-dissimilar-by-image-uuid', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ imageUuid, count, candidateUuids }),
  });
  const data = await res.json();
  if (!res.ok || data.status !== 'success') throw new Error(data.message || 'Failed to find dissimilar images');
  return (data.matches ?? []).map(m => ({ imageUuid: m.imageUuid, rank: m.rank }));
}

export async function findDissimilarBboxesByUuid(bboxUuid, count = 100, labelUuids = []) {
  const res = await apiFetch('/api/images/bbox/find-dissimilar-by-bbox-uuid', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ bboxUuid, count, labelUuids }),
  });
  const data = await res.json();
  if (!res.ok || data.status !== 'success') throw new Error(data.message || 'Failed to find dissimilar bboxes');
  return (data.matches ?? []).map(m => ({
    bboxUuid: m.bboxUuid,
    imageUuid: m.imageUuid ?? null,
    labelUuid: m.labelUuid,
    rank: m.rank,
  }));
}

/**
 * Most visually distinct images. With no `candidateUuids` the server scopes the
 * walk to unverified images; pass a filtered UUID set to scope it to that instead.
 * An empty array means "the filter matched nothing" and returns no matches.
 */
export async function findDissimilarGlobal(count = 100, seedImageUuid = null, candidateUuids = null) {
  const res = await apiFetch('/api/images/find-dissimilar-global', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ count, seedImageUuid, candidateUuids }),
  });
  const data = await res.json();
  if (!res.ok || data.status !== 'success') throw new Error(data.message || 'Failed to find most unique images');
  return (data.matches ?? []).map(m => ({ imageUuid: m.imageUuid, rank: m.rank }));
}

export async function findSimilarBboxesByCoords(imageUuid, x, y, w, h, threshold = 8, labelUuids = [], tagUuids = [], excludeTagUuids = [], tagMatchMode = 'any') {
  const res = await apiFetch('/api/images/bbox/find-similar-by-coords', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ imageUuid, x, y, w, h, threshold, labelUuids, tagUuids, excludeTagUuids, tagMatchMode }),
  });
  const data = await res.json();
  if (!res.ok || data.status !== 'success') throw new Error(data.message || 'Failed to find similar bboxes');
  return (data.matches ?? []).map(m => ({
    bboxUuid: m.bboxUuid,
    imageUuid: m.imageUuid ?? null,
    labelUuid: m.labelUuid,
    distance: m.distance,
  }));
}

export async function fetchImagesByUuids(imageUuids) {
  if (!imageUuids || imageUuids.length === 0) return [];
  const res = await apiFetch('/api/images/batch-by-uuids', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ imageUuids }),
  });
  const data = await res.json();
  if (!res.ok || data.status !== 'success') throw new Error(data.message || 'Failed to fetch images by UUIDs');
  return data.images ?? [];
}

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
      // Omitted when empty rather than sent as {}: this endpoint replaces the
      // whole bbox list, and the server reads a missing `extra` as "keep the
      // stored metadata" but an explicit {} as "clear it".
      ...(bbox.metadataJson && Object.keys(bbox.metadataJson).length > 0
        ? { extra: bbox.metadataJson }
        : {}),
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

/**
 * Permanently delete an image from the DB and disk.
 * @param {string} imageUuid
 * @returns {Promise<{ imageUuid: string, filename: string }>}
 */
export async function deleteImage(imageUuid) {
  const res = await apiFetch(`/api/images/${encodeURIComponent(imageUuid)}`, {
    method: 'DELETE',
  });

  const data = await res.json().catch(() => ({}));
  if (!res.ok || data.status !== 'success') {
    throw new Error(data.message || `Failed to delete image ${imageUuid}`);
  }
  return { imageUuid: data.imageUuid, filename: data.filename };
}

export async function deleteImageDbOnly(imageUuid) {
  const res = await apiFetch(`/api/images/${encodeURIComponent(imageUuid)}/db-only`, {
    method: 'DELETE',
  });

  const data = await res.json().catch(() => ({}));
  if (!res.ok || data.status !== 'success') {
    throw new Error(data.message || `Failed to remove image ${imageUuid} from DB`);
  }
  return { imageUuid: data.imageUuid, filename: data.filename };
}
