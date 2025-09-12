import { createCachedFetcher } from './createCachedFetcher.js';
import { wrapSingleKey } from './apiUtils.js';

/**
 * Caches
 */
export const sourcesFetcher = createCachedFetcher(async () => {
  const res = await fetch('/api/sources/get');
  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.message || 'Failed to fetch sources');
  }
  const { sources } = await res.json();
  // Store as Map for internal cache
  const map = new Map(Object.entries(sources));
  return map;
});

export const imageProviderFetcher = createCachedFetcher(async () => {
  const res = await fetch('/api/sources/image-providers/list');
  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.message || 'Failed to fetch image provider list');
  }
  const { providers } = await res.json();
  return Array.from(providers);
});

export const providerSchemaFetcher = createCachedFetcher(async (providerName) => {
  const res = await fetch('/api/sources/image-providers/schema', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ image_provider: providerName }),
  });

  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.message || 'Failed to fetch image provider schema');
  }

  const { schema } = await res.json();
  return schema;
});


/** -----------------------------
 * CACHE CLEAR FUNCTIONS
 * -----------------------------
 */
export function clearSourceCache() {
  sourcesFetcher.clear();
}

export function clearImageProviderCache() {
  imageProviderFetcher.clear();
  providerSchemaFetcher.clear();
}


/** -----------------------------
 * FETCH FUNCTIONS
 * -----------------------------
 */
export async function fetchSourceList() {
  const map = await sourcesFetcher.fetch('sourcesMap');
  // Return a copy of the Map to prevent external mutation
  return new Map(map);
}

export async function fetchImageProviderList() {
  return await imageProviderFetcher.fetch('providerList');
}

export async function fetchImageProviderSchema(providerName) {
  return await providerSchemaFetcher.fetch(providerName);
}


/** -----------------------------
 * SOURCE CRUD FUNCTIONS
 * -----------------------------
 */
export async function createSource({ name, providerName = null, params = {} }) {
  const res = await fetch('/api/sources/create', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ source_name: name, image_provider: providerName, params }),
  });

  if (!res.ok) {
    const error = await res.json();
    throw new Error(error.message || 'Failed to create source');
  }

  const { source } = await res.json();
  // Update the cached Map
  const map = await sourcesFetcher.fetch('sourcesMap');
  map.set(source.uuid, source);
  return source;
}

export async function updateSource({ uuid, name, providerName, params }) {
  const res = await fetch('/api/sources/update', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ source_uuid: uuid, source_name: name, image_provider: providerName, params }),
  });

  if (!res.ok) {
    const error = await res.json();
    throw new Error(error.message || 'Failed to update source');
  }

  const { source } = await res.json();
  const map = await sourcesFetcher.fetch('sourcesMap');
  map.set(source.uuid, source);
  return source;
}

async function _deleteSources({ uuids }) {
  const res = await fetch('/api/sources/delete', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ source_uuids: uuids }),
  });

  if (!res.ok) {
    const error = await res.json();
    throw new Error(error.message || 'Failed to delete source');
  }

  const map = await sourcesFetcher.fetch('sourcesMap');
  uuids.forEach(uuid => map.delete(uuid));
}
export const deleteSources = wrapSingleKey(_deleteSources);
