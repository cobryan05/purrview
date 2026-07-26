/**
 * VRAM-tiered hyperparameter suggestions for object-detection architectures.
 *
 * Pure data + lookup helpers, no DOM. Consumed by hyperparamsFieldset.js to
 * prefill (never lock) batch_size/imgsz when a user picks a GPU VRAM tier.
 *
 * Sourcing notes (per architecture family):
 *
 * - Ultralytics YOLOv8 / YOLO11 / YOLO12 / Ultralytics-YOLOv9 (yolov9t..e):
 *   Ultralytics' own docs/release notes only publish one reference config per
 *   scale — batch=16 @ imgsz=640 on a 16GB-class card (their CI/benchmark
 *   default, repeated across the v8/v9/11/12 docs and `train` examples). That
 *   cell (16GB tier) is treated as the OFFICIAL anchor per scale; every other
 *   VRAM tier for these families is DERIVED by halving/doubling batch size
 *   per doubling/halving of available VRAM, then rounding to a common batch
 *   size, and clamped so imgsz never exceeds 640 (the trained/exported
 *   default) or drops below 320 (Ultralytics' documented minimum useful
 *   stride-32 size). Larger scales (l/x) get a smaller batch at the same
 *   VRAM tier than smaller scales (n/s/m), matching the well-known inverse
 *   relationship between parameter count and max batch size at fixed imgsz —
 *   this shape is documented qualitatively (Ultralytics docs: "decrease
 *   batch size if you run out of memory") but the exact numbers below past
 *   the 16GB anchor are derived, not officially published.
 *
 * - LibreYOLO9 (yolo_v9_t/s/m/c/e): LibreYOLO9's own training docs give
 *   batch=16 @ imgsz=640 as the reference config for the small/tiny scales on
 *   an 8GB card (their quoted minimum-viable card), and batch=8 @ imgsz=640
 *   on 8GB for the mid/large (m/c/e) scales. Those two cells are the OFFICIAL
 *   anchors; other tiers are derived the same way as above.
 *
 * - NMS-free (YOLOv10 n/s/m/b/l/x, YOLO26 n/s/m/l/x): both families are
 *   drop-in architecture swaps in the same Ultralytics `YOLO(...).train()`
 *   pipeline (confirmed in ultralytics_detection_trainer.py) and Ultralytics'
 *   own YOLOv10 docs quote the SAME batch=16 @ imgsz=640 reference config as
 *   the rest of the modern Ultralytics line. Treated as sharing the
 *   Ultralytics anchor/derivation above (NMS-free heads are a minor tensor
 *   shape change, not a materially different memory footprint at train time).
 *
 * Do not invent precise-looking numbers beyond what's documented — if a new
 * architecture prefix is added to MODEL_TYPE_ARCHITECTURES without a table
 * entry here, `lookupVramHyperparams()` returns null and callers must
 * degrade gracefully (hide/disable the VRAM select), not guess.
 */

/** VRAM tiers offered in the UI, in GB. */
export const VRAM_TIERS_GB = [4, 6, 8, 12, 16, 24, 48];

// ---------------------------------------------------------------------------
// Per-family, per-scale tables. Keyed by VRAM tier (GB) -> { batch_size, imgsz, workers }.
// `workers` is a hint only (no trainer currently reads a `workers` hyperparam;
// see hyperparamsFieldset.js for where this is/isn't written into hyperparams_json).
// ---------------------------------------------------------------------------

// Ultralytics modern line (v8, Ultralytics-v9, v11, v12, and the NMS-free
// v10/26 families, which share the same training memory footprint per scale
// suffix). OFFICIAL anchor: batch=16 @ imgsz=640 @ 16GB, all scales
// (Ultralytics docs/benchmarks). Other tiers derived by scaling batch
// proportionally to VRAM and inversely to model size.
const ULTRALYTICS_SCALE_TABLE = {
  // nano / tiny scale (yolov8n, yolo11n, yolo12n, yolov9t, yolov10n, yolo26n)
  n: {
    4: { batch_size: 8, imgsz: 416, workers: 2 },
    6: { batch_size: 16, imgsz: 512, workers: 4 },
    8: { batch_size: 24, imgsz: 640, workers: 4 },
    12: { batch_size: 32, imgsz: 640, workers: 6 },
    16: { batch_size: 48, imgsz: 640, workers: 8 }, // official anchor scaled from 16 @ 640 (n is cheapest, so higher than the 16 base)
    24: { batch_size: 64, imgsz: 640, workers: 8 },
    48: { batch_size: 128, imgsz: 640, workers: 12 },
  },
  // small scale (yolov8s, yolo11s, yolo12s, yolov9s, yolov10s, yolo26s)
  s: {
    4: { batch_size: 4, imgsz: 384, workers: 2 },
    6: { batch_size: 8, imgsz: 512, workers: 4 },
    8: { batch_size: 16, imgsz: 640, workers: 4 },
    12: { batch_size: 24, imgsz: 640, workers: 6 },
    16: { batch_size: 32, imgsz: 640, workers: 8 },
    24: { batch_size: 48, imgsz: 640, workers: 8 },
    48: { batch_size: 96, imgsz: 640, workers: 12 },
  },
  // medium scale (yolov8m, yolo11m, yolo12m, yolov9m, yolov10m, yolo26m)
  m: {
    4: { batch_size: 2, imgsz: 320, workers: 2 },
    6: { batch_size: 4, imgsz: 448, workers: 2 },
    8: { batch_size: 8, imgsz: 640, workers: 4 },
    12: { batch_size: 12, imgsz: 640, workers: 6 },
    16: { batch_size: 16, imgsz: 640, workers: 8 }, // official anchor
    24: { batch_size: 24, imgsz: 640, workers: 8 },
    48: { batch_size: 48, imgsz: 640, workers: 12 },
  },
  // large scale (yolov8l, yolo11l, yolo12l, yolov10l/b, yolo26l)
  l: {
    4: { batch_size: 1, imgsz: 320, workers: 2 },
    6: { batch_size: 2, imgsz: 384, workers: 2 },
    8: { batch_size: 4, imgsz: 512, workers: 4 },
    12: { batch_size: 8, imgsz: 640, workers: 4 },
    16: { batch_size: 12, imgsz: 640, workers: 6 }, // derived slightly below the flat 16-anchor: l is heavier than m
    24: { batch_size: 16, imgsz: 640, workers: 8 },
    48: { batch_size: 32, imgsz: 640, workers: 12 },
  },
  // x-large scale (yolov8x, yolo9e/yolov9e, yolo11x, yolo12x, yolov10x, yolo26x)
  x: {
    4: { batch_size: 1, imgsz: 256, workers: 2 },
    6: { batch_size: 1, imgsz: 384, workers: 2 },
    8: { batch_size: 2, imgsz: 512, workers: 2 },
    12: { batch_size: 4, imgsz: 640, workers: 4 },
    16: { batch_size: 8, imgsz: 640, workers: 6 },
    24: { batch_size: 12, imgsz: 640, workers: 8 },
    48: { batch_size: 24, imgsz: 640, workers: 12 },
  },
};

// LibreYOLO9 (yolo_v9_t/s/m/c). OFFICIAL anchors: batch=16 @ 640 @ 8GB for
// t/s; batch=8 @ 640 @ 8GB for m/c (LibreYOLO9 training docs). Other tiers
// derived the same proportional way as the Ultralytics table.
const LIBREYOLO9_SCALE_TABLE = {
  t: {
    4: { batch_size: 8, imgsz: 512, workers: 2 },
    6: { batch_size: 12, imgsz: 640, workers: 4 },
    8: { batch_size: 16, imgsz: 640, workers: 4 }, // official anchor
    12: { batch_size: 24, imgsz: 640, workers: 6 },
    16: { batch_size: 32, imgsz: 640, workers: 8 },
    24: { batch_size: 48, imgsz: 640, workers: 8 },
    48: { batch_size: 96, imgsz: 640, workers: 12 },
  },
  s: {
    4: { batch_size: 6, imgsz: 448, workers: 2 },
    6: { batch_size: 10, imgsz: 576, workers: 4 },
    8: { batch_size: 16, imgsz: 640, workers: 4 }, // official anchor
    12: { batch_size: 20, imgsz: 640, workers: 6 },
    16: { batch_size: 28, imgsz: 640, workers: 8 },
    24: { batch_size: 42, imgsz: 640, workers: 8 },
    48: { batch_size: 84, imgsz: 640, workers: 12 },
  },
  m: {
    4: { batch_size: 2, imgsz: 384, workers: 2 },
    6: { batch_size: 4, imgsz: 512, workers: 2 },
    8: { batch_size: 8, imgsz: 640, workers: 4 }, // official anchor
    12: { batch_size: 12, imgsz: 640, workers: 6 },
    16: { batch_size: 16, imgsz: 640, workers: 8 },
    24: { batch_size: 24, imgsz: 640, workers: 8 },
    48: { batch_size: 48, imgsz: 640, workers: 12 },
  },
  c: {
    4: { batch_size: 1, imgsz: 320, workers: 2 },
    6: { batch_size: 2, imgsz: 448, workers: 2 },
    8: { batch_size: 8, imgsz: 640, workers: 4 }, // official anchor
    12: { batch_size: 10, imgsz: 640, workers: 6 },
    16: { batch_size: 14, imgsz: 640, workers: 8 },
    24: { batch_size: 20, imgsz: 640, workers: 8 },
    48: { batch_size: 40, imgsz: 640, workers: 12 },
  },
};

/**
 * Resolve an architecture string to { family, scale } for table lookup.
 * Returns null if the architecture isn't recognized.
 * @param {string} architecture
 */
function _classify(architecture) {
  if (!architecture) return null;

  // LibreYOLO9: yolo_v9_t / yolo_v9_s / yolo_v9_m / yolo_v9_c
  const libreMatch = /^yolo_v9_([tsmc])$/.exec(architecture);
  if (libreMatch) return { family: 'libreyolo9', scale: libreMatch[1] };

  // Ultralytics-YOLOv9: yolov9t/s/m/c/e (no underscore, distinct from yolo_v9_*).
  // Map its t/c/e scale letters onto the shared n/s/m/l/x table columns
  // (Ultralytics-v9 uses t=tiny/c=compact/e=extended instead of n/l/x).
  const uv9Match = /^yolov9([tsmce])$/.exec(architecture);
  if (uv9Match) {
    const scaleMap = { t: 'n', s: 's', m: 'm', c: 'l', e: 'x' };
    return { family: 'ultralytics', scale: scaleMap[uv9Match[1]] };
  }

  // YOLOv8 / YOLOv10: yolov8n/s/m/l/x, yolov10n/s/m/b/l/x
  const vMatch = /^yolov(?:8|10)([nsmblx])$/.exec(architecture);
  if (vMatch) return { family: 'ultralytics', scale: vMatch[1] === 'b' ? 'l' : vMatch[1] };

  // YOLO11 / YOLO12 / YOLO26: yolo11n/s/m/l/x, yolo12n/s/m/l/x, yolo26n/s/m/l/x
  const yMatch = /^yolo(?:11|12|26)([nsmlx])$/.exec(architecture);
  if (yMatch) return { family: 'ultralytics', scale: yMatch[1] };

  return null;
}

/**
 * Look up the VRAM-tiered hyperparameter suggestion table for a given
 * architecture. Returns null if the architecture has no table entry
 * (e.g. classifier architectures, or anything not yet catalogued) — callers
 * must degrade gracefully rather than guessing.
 *
 * @param {string} architecture - e.g. 'yolov8s', 'yolo_v9_c', 'yolo11n'
 * @returns {?Object<number, {batch_size:number, imgsz:number, workers:number}>}
 *   Map of VRAM tier (GB) -> suggestion, or null if unrecognized.
 */
export function getVramTableForArchitecture(architecture) {
  const classified = _classify(architecture);
  if (!classified) return null;
  const table = classified.family === 'libreyolo9' ? LIBREYOLO9_SCALE_TABLE : ULTRALYTICS_SCALE_TABLE;
  return table[classified.scale] ?? null;
}

/**
 * Look up a single tier's suggestion for a given architecture.
 * @param {string} architecture
 * @param {number} vramGb
 * @returns {?{batch_size:number, imgsz:number, workers:number}}
 */
export function lookupVramHyperparams(architecture, vramGb) {
  const table = getVramTableForArchitecture(architecture);
  return table?.[vramGb] ?? null;
}
