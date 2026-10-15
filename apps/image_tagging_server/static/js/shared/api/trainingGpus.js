/**
 * The GPUs the training host can see, for the per-run GPU override.
 *
 * Served by training_server, not by the frontend: training may be a remote GPU node
 * with entirely different cards from the inference host, so there is no single
 * machine-wide GPU list to ask for.
 *
 * Cached for the life of the page — cards do not come and go, and the training config
 * form asks for them on every open.
 */

import { createCachedFetcher } from '/app-static/js/ui/utils/data/createCachedFetcher.js';
import { apiFetch, ignoreKeyReturn } from './apiUtils.js';

const gpusFetcher = createCachedFetcher(async (keys) => {
  const res = await apiFetch('/api/training/gpus');
  const data = await res.json();
  if (!res.ok) throw new Error(data.detail || data.message || 'Failed to fetch training GPUs');
  return ignoreKeyReturn(keys, {
    // Only cards actually exposed to the container can be selected: one left out of
    // GPU_DEVICES has no CUDA ordinal, so naming it would pick a different card.
    gpus: (data.gpus || []).filter(g => g.cuda_ordinal !== null && g.cuda_ordinal !== undefined),
    defaultCudaOrdinal: data.default_cuda_ordinal ?? null,
  });
});

/**
 * @returns {Promise<{ gpus: Array<{index:number,name:string,bus_id:string,memory_mb:?number,cuda_ordinal:number,label:string}>, defaultCudaOrdinal: ?number }>}
 *   Empty `gpus` on a CPU-only or GPU-less host — the caller hides its control rather
 *   than offering a choice of nothing.
 */
export async function fetchTrainingGpus() {
  try {
    return await gpusFetcher.fetch('all');
  } catch {
    // A training node that is down or too old to serve this endpoint must not break
    // the training form — the run simply uses the deployment default.
    return { gpus: [], defaultCudaOrdinal: null };
  }
}
