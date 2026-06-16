/**
 * Shared pagination utilities used by similarSearch.js and dissimilarSearch.js.
 */

/**
 * Return the start index for a keyset-paginated slice of an ordered array.
 * Each element must have an `imageUuid` property.
 *
 * @param {Array<{imageUuid: string}>} order
 * @param {string|null} afterUuid
 * @returns {number}
 */
export function pageStartIdx(order, afterUuid) {
  return afterUuid !== null ? order.findIndex(e => e.imageUuid === afterUuid) + 1 : 0;
}
