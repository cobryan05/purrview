let _sourcesMap = null; // uuid -> full source data
let _imageProviders = null; // Cache for image providers
let _providerSchemas = new Map(); // Cache for provider schemas

export function clearSourceCache() {
  _sourcesMap = null;
}

export function clearImageProviderCache() {
  _imageProviders = null;
  _providerSchemas = null;
}

/**
 * Refresh the source list cache
 */
export async function fetchSourceList() {
  if (!_sourcesMap) {
    const res = await fetch('/api/sources/get');
    const { sources } = await res.json();

    // Store everything as a Map (uuid -> source object)
    _sourcesMap = new Map(Object.entries(sources));
  }
  return new Map(_sourcesMap);
}

/**
 * Get the list of available image providers from the cache.
 * Throws an error if the cache is not yet populated.
 */
export async function fetchImageProviderList() {
  if (!_imageProviders) {
    const res = await fetch('/api/sources/image-providers/list');
    if (res.ok) {
      const { providers } = await res.json();
      _imageProviders = providers;
    } else {
      const error = await res.json();
      throw new Error(error.message || 'Failed to fetch image provider list');
    }
  }
  return Array.from(_imageProviders);
}

/**
 * Fetch the schema for a specific image provider, with caching.
 */
export async function fetchImageProviderSchema(providerName) {
  if (_providerSchemas.has(providerName)) {
    return _providerSchemas.get(providerName);
  }

  const res = await fetch('/api/sources/image-providers/schema', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ image_provider: providerName }),
  });

  if (res.ok) {
    const { schema } = await res.json();
    _providerSchemas.set(providerName, schema);
    return schema;
  } else {
    const error = await res.json();
    throw new Error(error.message || 'Failed to fetch image provider schema');
  }
}

/**
 * Create a new source.
 */
export async function createSource({ name, providerName = null, params = {} }) {
  const res = await fetch('/api/sources/create', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ source_name: name, image_provider: providerName, params }),
  });

  if (res.ok) {
    const { source } = await res.json();
    _sourcesMap.set(source.uuid, source);
  } else {
    const error = await res.json();
    throw new Error(error.message || 'Failed to create source');
  }
}

/**
 * Update an existing source.
 */
export async function updateSource({ uuid, name, providerName, params }) {
  const res = await fetch('/api/sources/update', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ source_uuid: uuid, source_name: name, image_provider: providerName, params }),
  });

  if (res.ok) {
    const { source } = await res.json();
    _sourcesMap.set(source.uuid, source);
  } else {
    const error = await res.json();
    throw new Error(error.message || 'Failed to update source');
  }
}

/**
 * Delete a source (or multiple sources).
 */
export async function deleteSources({ uuids }) {
  if (!Array.isArray(uuids)) {
    uuids = [uuids];
  }

  const res = await fetch('/api/sources/delete', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ source_uuids: uuids }),
  });

  if (res.ok) {
    uuids.forEach(sourceUuid => _sourcesMap.delete(sourceUuid));
  } else {
    const error = await res.json();
    throw new Error(error.message || 'Failed to delete source');
  }
}
