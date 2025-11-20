import { createCachedFetcher } from './createCachedFetcher.js';
import { wrapSingleKey } from './apiUtils.js';

// This cache is meant for runtime caching of bounding box information. It does not
// itself commit changes to the server. Updates are local only!


/**
 * Caches
 */
export const bboxesInfoFetcher = createCachedFetcher(async (bboxUuids) => {
  const res = await fetch('/api/bboxes/get', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ bbox_uuids: bboxUuids }),
  });

  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.message || 'Failed to fetch bbox info');
  }

  const { info } = await res.json();
  return info;
});


/** -----------------------------
 * CACHE CLEAR FUNCTIONS
 * -----------------------------
 */
export function clearBboxInfoCache() {
  bboxesInfoFetcher.clear();
}


/** -----------------------------
 * FETCH FUNCTIONS
 * -----------------------------
 */

export async function _fetchBboxesInfo(bboxUuids) {
  const result = new Map();
  for (const uuid of bboxUuids) {
    const info = await bboxesInfoFetcher.fetch(uuid);
    result.set(uuid, info);
  }
  return result;
}
export const fetchBboxesInfo = wrapSingleKey(_fetchBboxesInfo);


async function _updateBboxInfo({ bboxUuids }) {
  const res = await fetch('/api/bboxes/update', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ bboxes: bboxUuids }),
  });
  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.message || 'Failed to update bounding boxes');
  }
}
export const updateBboxesInfo = wrapSingleKey(_updateBboxInfo);

/** -----------------------------
 * SOURCE FUNCTIONS
 * -----------------------------
 */
// async function _deleteSources({ uuids }) {
//   const res = await fetch('/api/sources/delete', {
//     method: 'POST',
//     headers: { 'Content-Type': 'application/json' },
//     body: JSON.stringify({ source_uuids: uuids }),
//   });

//   if (!res.ok) {
//     const error = await res.json();
//     throw new Error(error.message || 'Failed to delete source');
//   }

//   const map = await bboxesFetcher.fetch('sourcesMap');
//   uuids.forEach(uuid => map.delete(uuid));
// }
// export const deleteSources = wrapSingleKey(_deleteSources);
