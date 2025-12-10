import { ignoreKeyReturn, wrapSingleKey } from './apiUtils.js';
import { createCachedFetcher } from './createCachedFetcher.js';

/**
 * Caches
 */
export const sourcesFetcher = createCachedFetcher(async (keys) => {
  const res = await fetch('/api/sources/get');
  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.message || 'Failed to fetch sources');
  }
  const { sources } = await res.json();
  // Store as Map for internal cache
  const map = new Map(Object.entries(sources));
  return ignoreKeyReturn(keys, map);
});

export const imageProviderFetcher = createCachedFetcher(async (keys) => {
  const res = await fetch('/api/sources/image-providers/list');
  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.message || 'Failed to fetch image provider list');
  }
  const { image_providers } = await res.json();
  return ignoreKeyReturn(keys, Array.from(image_providers));
});

export const providerSchemaFetcher = createCachedFetcher(async (providerNames) => {
  const results = new Map();
  for (const name of providerNames) {
    const res = await fetch('/api/sources/image-providers/schema', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image_provider: name }),
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.message || `Failed to fetch image provider schema for ${name}`);
    }

    const { image_provider_schema } = await res.json();
    results.set(name, image_provider_schema);
  }

  return results
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
 * SOURCE FUNCTIONS
 * -----------------------------
 */
export async function createSource({ name, provider = null, provider_params = {} }) {
  const res = await fetch('/api/sources/create', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ source_name: name, image_provider: provider, provider_params }),
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

export async function updateSource({ uuid, name, provider, provider_params }) {
  const res = await fetch('/api/sources/update', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ source_uuid: uuid, source_name: name, image_provider: provider, provider_params }),
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
