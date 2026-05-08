// @ts-check

/**
 * @typedef {import('@domain_image').UIImage} UIImage
 * @typedef {import('@domain_image').UIBbox} UIBbox
 * @typedef {import('@image_tagging_types').ViewportTransform} ViewportTransform
 */


/**
 * Compute an initial viewport transform that fits the image
 * into the container while preserving aspect ratio.
 *
 * The viewport transform maps image-space coordinates to screen-space:
 *
 *   screenX = imageX * scale + offsetX
 *   screenY = imageY * scale + offsetY
 *
 * @param {number} containerWidth
 * @param {number} containerHeight
 * @param {number} imageWidth
 * @param {number} imageHeight
 * @returns {ViewportTransform}
 */
export function computeViewport(containerWidth, containerHeight, imageWidth, imageHeight) {
  const scale = Math.min(
    containerWidth / imageWidth,
    containerHeight / imageHeight
  );

  return {
    scale,

    imageWidth,
    imageHeight,

    offsetX: (containerWidth - imageWidth * scale) / 2,
    offsetY: (containerHeight - imageHeight * scale) / 2,
  };
}

/**
 * Convert normalized bbox coords to screen coords.
 *
 * @param {UIBbox} bbox
 * @param {ViewportTransform} viewport
 */
function bboxToScreen(bbox, viewport) {
  return {
    x: viewport.x + bbox.x * viewport.width,
    y: viewport.y + bbox.y * viewport.height,
    width: bbox.width * viewport.width,
    height: bbox.height * viewport.height,
  };
}


/**
 * Convert screen coords to image coords.
 * @param {number} x
 * @param {number} y
 * @param {ViewportTransform} viewport
 */
export function screenToImage(x, y, viewport) {
  return {
    x: (x - viewport.offsetX) / viewport.scale,
    y: (y - viewport.offsetY) / viewport.scale,
  };
}
