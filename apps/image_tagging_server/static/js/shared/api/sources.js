import { ignoreKeyReturn, wrapSingleKey } from './apiUtils.js';
import { createCachedFetcher } from '/app-static/js/ui/utils/data/createCachedFetcher.js';

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
  const { imageProviders } = await res.json();
  return ignoreKeyReturn(keys, Array.from(imageProviders));
});

export const providerSchemaFetcher = createCachedFetcher(async (providerNames) => {
  const results = new Map();
  for (const name of providerNames) {
    const res = await fetch('/api/sources/image-providers/schema', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ imageProvider: name }),
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.message || `Failed to fetch image provider schema for ${name}`);
    }

    const { imageProviderSchema } = await res.json();
    results.set(name, imageProviderSchema);
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
export async function createSource({ name, provider = null, providerParams = {} }) {
  const res = await fetch('/api/sources/create', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ sourceName: name, imageProvider: provider, providerParams }),
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

export async function updateSource({ uuid, name, provider, providerParams }) {
  const res = await fetch('/api/sources/update', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ sourceUuid: uuid, sourceName: name, imageProvider: provider, providerParams }),
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
    body: JSON.stringify({ sourceUuids: uuids }),
  });

  if (!res.ok) {
    const error = await res.json();
    throw new Error(error.message || 'Failed to delete source');
  }

  const map = await sourcesFetcher.fetch('sourcesMap');
  uuids.forEach(uuid => map.delete(uuid));
}
export const deleteSources = wrapSingleKey(_deleteSources);
