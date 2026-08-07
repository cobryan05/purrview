// Display-only image enhancement for the annotation canvases.
//
// Grainy / dark camera frames (IR night shots, a dark tupperware in a dark
// room) are hard to draw boxes on. Each mode re-renders the *source* pixels
// into an offscreen canvas which is then swapped into the existing
// Konva.Image background — nothing here touches the stored image, the bboxes,
// or what gets sent to inference or training. It is purely a viewing aid.
//
// Everything is plain ImageData math (no dependencies). Results are cached per
// (source image, mode) so cycling back to a mode you already viewed is free.

import { events, EventTypes } from '/app-static/js/shared/events/index.js';

/**
 * Cycle order. `id` is what gets stored on the canvas state; `short` is what
 * the toolbar button shows.
 * @type {Array<{id: string, short: string, title: string}>}
 */
export const ENHANCE_MODES = [
  { id: 'none', short: 'Off', title: 'Enhancement off — original pixels' },
  { id: 'levels', short: 'Levels', title: 'Auto levels — stretch each channel to full range (blown-out contrast)' },
  { id: 'clahe', short: 'Local', title: 'CLAHE — local contrast equalization; best for a dark object in a dark scene' },
  { id: 'shadows', short: 'Shadows', title: 'Shadow boost — gamma lift that opens up near-black detail' },
  { id: 'edges', short: 'Edges', title: 'Sobel edges drawn in cyan over a dimmed, contrast-stretched frame' },
  { id: 'canny', short: 'Canny', title: 'Canny edge detection — binary outlines only' },
];

const MODE_BY_ID = new Map(ENHANCE_MODES.map(m => [m.id, m]));

export const DEFAULT_ENHANCE_MODE = 'none';

// Source frames are processed at native resolution so zooming in stays sharp,
// but a very large frame would block the main thread for a noticeable beat.
// Above this budget the working buffer is downscaled (aspect preserved — the
// Konva node's own width/height is untouched, so it just stretches back).
const MAX_PIXELS = 4_000_000;

// Per source image, keep at most this many rendered modes alive. The outer map
// is weak, so everything for an image is released when the image itself is.
const CACHE_PER_IMAGE = 3;

/** @type {WeakMap<HTMLImageElement, Map<string, HTMLCanvasElement>>} */
const _cache = new WeakMap();

// Last mode the user picked, reused as the default for canvases created later
// in the session (open a modal while annotating dark frames and it comes up in
// the same mode). Deliberately in-memory only — a fresh page load starts off.
let _lastMode = DEFAULT_ENHANCE_MODE;

// ── Public API ───────────────────────────────────────────────────────────────

/** @param {string} id @returns {{id:string, short:string, title:string}} */
export function getEnhanceMode(id) {
  return MODE_BY_ID.get(id) ?? ENHANCE_MODES[0];
}

/** The mode a newly created canvas should start in. */
export function defaultEnhanceMode() {
  return _lastMode;
}

/** @param {string} id @returns {string} the next mode id in cycle order */
export function nextEnhanceModeId(id) {
  const idx = ENHANCE_MODES.findIndex(m => m.id === id);
  return ENHANCE_MODES[(idx + 1) % ENHANCE_MODES.length].id;
}

/**
 * Point the canvas background at the given mode's rendering of its source
 * image. Safe to call with no image loaded (records the mode for the next
 * load) and safe to call repeatedly with the same mode.
 *
 * @param {import('@image_tagging_types').CanvasState} canvas
 * @param {string} modeId
 * @param {string} [runtimeId] - published with CANVAS_ENHANCE_CHANGED when given
 */
export function applyCanvasEnhancement(canvas, modeId, runtimeId) {
  if (!canvas) return;
  const mode = getEnhanceMode(modeId);
  canvas.enhanceMode = mode.id;

  const source = canvas.sourceImage;
  if (source && canvas.backgroundImage) {
    canvas.backgroundImage.image(renderEnhanced(source, mode.id));
    canvas.layer?.batchDraw();
  }

  if (runtimeId) {
    events.publish(EventTypes.CANVAS_ENHANCE_CHANGED, { runtimeId, mode: mode.id });
  }
}

/**
 * Advance the canvas to the next enhancement mode.
 * @param {import('@image_tagging_types').ImageTaggingRuntime} runtime
 * @returns {string|null} the new mode id, or null if there is no canvas
 */
export function cycleCanvasEnhancement(runtime) {
  if (!runtime?.canvas) return null;
  // Read-only previews opt out entirely (see createCanvasInstance's lockEnhance):
  // they have no enhance control, and the hotkey reaching one would both enhance a
  // widget nobody asked to enhance and poison the default for every later canvas.
  if (runtime.canvas.enhanceLocked) return null;
  const next = nextEnhanceModeId(runtime.canvas.enhanceMode ?? DEFAULT_ENHANCE_MODE);
  _lastMode = next;
  applyCanvasEnhancement(runtime.canvas, next, runtime.id);
  return next;
}

/**
 * The image source Konva should draw for this mode — the original element for
 * 'none', otherwise a cached offscreen canvas.
 * @param {HTMLImageElement} img
 * @param {string} modeId
 * @returns {HTMLImageElement|HTMLCanvasElement}
 */
export function renderEnhanced(img, modeId) {
  if (!img || modeId === 'none' || !MODE_BY_ID.has(modeId)) return img;

  let perImage = _cache.get(img);
  if (!perImage) {
    perImage = new Map();
    _cache.set(img, perImage);
  }
  const hit = perImage.get(modeId);
  if (hit) return hit;

  const rendered = _render(img, modeId);
  perImage.set(modeId, rendered);
  while (perImage.size > CACHE_PER_IMAGE) {
    perImage.delete(perImage.keys().next().value);   // Maps iterate in insertion order
  }
  return rendered;
}

// ── Rendering ────────────────────────────────────────────────────────────────

/**
 * @param {HTMLImageElement} img
 * @param {string} modeId
 * @returns {HTMLCanvasElement}
 */
function _render(img, modeId) {
  const scale = Math.min(1, Math.sqrt(MAX_PIXELS / Math.max(1, img.width * img.height)));
  const w = Math.max(1, Math.round(img.width * scale));
  const h = Math.max(1, Math.round(img.height * scale));

  const canvas = document.createElement('canvas');
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  ctx.drawImage(img, 0, 0, w, h);

  const imageData = ctx.getImageData(0, 0, w, h);
  switch (modeId) {
    case 'levels': autoLevels(imageData.data); break;
    case 'clahe': clahe(imageData, 8, 3.0); break;
    case 'shadows': autoLevels(imageData.data); applyGamma(imageData.data, 0.45); break;
    case 'edges': edgeOverlay(imageData); break;
    case 'canny': canny(imageData); break;
  }
  ctx.putImageData(imageData, 0, 0);
  return canvas;
}

// ── Point operations ─────────────────────────────────────────────────────────

const LOW_CLIP = 0.005;    // fraction of pixels sacrificed at each end of the
const HIGH_CLIP = 0.005;   // histogram, so a few hot/dead pixels can't pin the
                           // stretch to 0..255 and do nothing.

/**
 * Per-channel percentile contrast stretch, in place. On a near-monochrome IR
 * frame this also neutralises the colour cast, which is usually what you want.
 * @param {Uint8ClampedArray} d
 */
export function autoLevels(d) {
  const total = d.length / 4;
  for (let c = 0; c < 3; c++) {
    const hist = new Uint32Array(256);
    for (let i = c; i < d.length; i += 4) hist[d[i]]++;

    const lo = percentile(hist, total, LOW_CLIP);
    const hi = percentile(hist, total, 1 - HIGH_CLIP);
    if (hi <= lo) continue;

    const lut = new Uint8ClampedArray(256);
    const span = hi - lo;
    for (let v = 0; v < 256; v++) lut[v] = ((v - lo) * 255) / span;
    for (let i = c; i < d.length; i += 4) d[i] = lut[d[i]];
  }
}

/**
 * Gamma < 1 lifts shadows (opens up near-black detail), > 1 crushes them.
 * @param {Uint8ClampedArray} d
 * @param {number} gamma
 */
export function applyGamma(d, gamma) {
  const lut = new Uint8ClampedArray(256);
  for (let v = 0; v < 256; v++) lut[v] = 255 * Math.pow(v / 255, gamma);
  for (let i = 0; i < d.length; i += 4) {
    d[i] = lut[d[i]];
    d[i + 1] = lut[d[i + 1]];
    d[i + 2] = lut[d[i + 2]];
  }
}

/** Value below which `frac` of the histogram's mass lies. */
function percentile(hist, total, frac) {
  const target = total * frac;
  let acc = 0;
  for (let v = 0; v < 256; v++) {
    acc += hist[v];
    if (acc >= target) return v;
  }
  return 255;
}

// ── CLAHE ────────────────────────────────────────────────────────────────────

/**
 * Contrast Limited Adaptive Histogram Equalization on luminance, with the
 * original hue carried over by scaling RGB by the luminance ratio.
 *
 * This is the mode that actually finds a dark object against a dark
 * background: it equalizes each tile independently, so a locally flat region
 * gets stretched even when the frame as a whole already spans 0..255. The clip
 * limit is what stops it from amplifying sensor grain into confetti.
 *
 * @param {ImageData} imageData
 * @param {number} tiles - grid is tiles x tiles
 * @param {number} clipLimit - multiples of the tile's average bin height
 */
export function clahe(imageData, tiles = 8, clipLimit = 3.0) {
  const { data: d, width: w, height: h } = imageData;
  const gray = toGray(d, w, h);

  const tw = Math.ceil(w / tiles);
  const th = Math.ceil(h / tiles);

  /** @type {Uint8ClampedArray[]} one 256-entry LUT per tile */
  const luts = new Array(tiles * tiles);

  for (let ty = 0; ty < tiles; ty++) {
    for (let tx = 0; tx < tiles; tx++) {
      const x0 = tx * tw, x1 = Math.min(x0 + tw, w);
      const y0 = ty * th, y1 = Math.min(y0 + th, h);
      const hist = new Uint32Array(256);
      let count = 0;
      for (let y = y0; y < y1; y++) {
        const row = y * w;
        for (let x = x0; x < x1; x++) {
          hist[gray[row + x]]++;
          count++;
        }
      }

      const lut = new Uint8ClampedArray(256);
      if (count === 0) {
        for (let v = 0; v < 256; v++) lut[v] = v;
        luts[ty * tiles + tx] = lut;
        continue;
      }

      // Clip tall bins and hand the excess back out evenly.
      const limit = Math.max(1, (clipLimit * count) / 256);
      let excess = 0;
      for (let v = 0; v < 256; v++) {
        if (hist[v] > limit) {
          excess += hist[v] - limit;
          hist[v] = limit;
        }
      }
      const bonus = excess / 256;

      let cdf = 0;
      for (let v = 0; v < 256; v++) {
        cdf += hist[v] + bonus;
        lut[v] = (255 * cdf) / count;
      }
      luts[ty * tiles + tx] = lut;
    }
  }

  // Bilinear blend between the four nearest tile LUTs, otherwise tile seams
  // show up as hard rectangles across the frame.
  for (let y = 0; y < h; y++) {
    const fy = y / th - 0.5;
    const ty0 = clampInt(Math.floor(fy), 0, tiles - 1);
    const ty1 = clampInt(ty0 + 1, 0, tiles - 1);
    const wy = clamp01(fy - ty0);

    for (let x = 0; x < w; x++) {
      const fx = x / tw - 0.5;
      const tx0 = clampInt(Math.floor(fx), 0, tiles - 1);
      const tx1 = clampInt(tx0 + 1, 0, tiles - 1);
      const wx = clamp01(fx - tx0);

      const i = y * w + x;
      const v = gray[i];
      const top = luts[ty0 * tiles + tx0][v] * (1 - wx) + luts[ty0 * tiles + tx1][v] * wx;
      const bot = luts[ty1 * tiles + tx0][v] * (1 - wx) + luts[ty1 * tiles + tx1][v] * wx;
      const out = top * (1 - wy) + bot * wy;

      const p = i * 4;
      if (v < 1) {
        // No hue to preserve at true black — a ratio here would be a divide by
        // (almost) zero and blow one channel out arbitrarily.
        d[p] = d[p + 1] = d[p + 2] = out;
      } else {
        const ratio = out / v;
        d[p] = d[p] * ratio;
        d[p + 1] = d[p + 1] * ratio;
        d[p + 2] = d[p + 2] * ratio;
      }
    }
  }
}

// ── Edge detection ───────────────────────────────────────────────────────────

/**
 * Sobel magnitude drawn in cyan on top of a dimmed, contrast-stretched copy of
 * the frame. Keeping the scene visible underneath is the point — a bare edge
 * map loses the context you need to place a box correctly.
 * @param {ImageData} imageData
 */
export function edgeOverlay(imageData) {
  const { data: d, width: w, height: h } = imageData;
  const { mag, max } = sobelMagnitude(prepGray(d, w, h), w, h);

  autoLevels(d);

  // Normalising against the frame's own strongest edge keeps a low-contrast
  // night shot from producing an almost-black overlay.
  const norm = max > 0 ? 255 / max : 0;
  for (let i = 0; i < mag.length; i++) {
    const e = Math.min(255, mag[i] * norm);
    const p = i * 4;
    d[p] = d[p] * 0.45;
    d[p + 1] = d[p + 1] * 0.45 + e;
    d[p + 2] = d[p + 2] * 0.45 + e;
  }
}

const CANNY_HIGH_PERCENTILE = 0.92;   // of edge magnitudes; the rest is grain
const CANNY_LOW_RATIO = 0.4;          // low threshold as a fraction of high

/**
 * Canny: denoise → blur → Sobel → non-maximum suppression → hysteresis.
 * Output is binary white-on-black. Thresholds are derived from the frame's own
 * magnitude distribution, so it needs no per-image tuning.
 * @param {ImageData} imageData
 */
export function canny(imageData) {
  const { data: d, width: w, height: h } = imageData;
  const { mag, dir, max } = sobelMagnitude(prepGray(d, w, h), w, h);

  // Non-maximum suppression: keep a pixel only if it is a local peak along the
  // gradient direction, quantized to the 4 nearest neighbour axes.
  const thin = new Float32Array(mag.length);
  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      const i = y * w + x;
      const m = mag[i];
      if (m === 0) continue;

      let a = (dir[i] * 180) / Math.PI;
      if (a < 0) a += 180;

      let n1, n2;
      if (a < 22.5 || a >= 157.5) { n1 = mag[i - 1]; n2 = mag[i + 1]; }
      else if (a < 67.5) { n1 = mag[i - w + 1]; n2 = mag[i + w - 1]; }
      else if (a < 112.5) { n1 = mag[i - w]; n2 = mag[i + w]; }
      else { n1 = mag[i - w - 1]; n2 = mag[i + w + 1]; }

      if (m >= n1 && m >= n2) thin[i] = m;
    }
  }

  const high = magnitudePercentile(thin, max, CANNY_HIGH_PERCENTILE);
  const low = high * CANNY_LOW_RATIO;

  // Hysteresis: strong pixels are edges; weak pixels are edges only if they are
  // reachable from a strong one. Explicit stack, not recursion — a long edge in
  // a 4MP frame would otherwise blow the JS stack.
  const edge = new Uint8Array(thin.length);
  const stack = new Int32Array(thin.length);
  let sp = 0;
  for (let i = 0; i < thin.length; i++) {
    if (thin[i] >= high) { edge[i] = 1; stack[sp++] = i; }
  }
  while (sp > 0) {
    const i = stack[--sp];
    const x = i % w, y = (i / w) | 0;
    for (let dy = -1; dy <= 1; dy++) {
      const ny = y + dy;
      if (ny < 0 || ny >= h) continue;
      for (let dx = -1; dx <= 1; dx++) {
        const nx = x + dx;
        if (nx < 0 || nx >= w) continue;
        const j = ny * w + nx;
        if (!edge[j] && thin[j] >= low) { edge[j] = 1; stack[sp++] = j; }
      }
    }
  }

  for (let i = 0; i < edge.length; i++) {
    const v = edge[i] ? 255 : 0;
    const p = i * 4;
    d[p] = d[p + 1] = d[p + 2] = v;
  }
}

/** Value at the given percentile of the non-zero magnitudes. */
function magnitudePercentile(mag, max, frac) {
  if (max <= 0) return Infinity;
  const hist = new Uint32Array(257);
  let count = 0;
  for (let i = 0; i < mag.length; i++) {
    if (mag[i] <= 0) continue;
    hist[Math.min(256, ((mag[i] / max) * 256) | 0)]++;
    count++;
  }
  if (count === 0) return Infinity;

  const target = count * frac;
  let acc = 0;
  for (let b = 0; b < 257; b++) {
    acc += hist[b];
    if (acc >= target) return (b / 256) * max;
  }
  return max;
}

// ── Grayscale helpers ────────────────────────────────────────────────────────

/** @returns {Uint8ClampedArray} one luminance byte per pixel */
function toGray(d, w, h) {
  const gray = new Uint8ClampedArray(w * h);
  for (let i = 0, p = 0; i < gray.length; i++, p += 4) {
    gray[i] = 0.299 * d[p] + 0.587 * d[p + 1] + 0.114 * d[p + 2];
  }
  return gray;
}

/**
 * Grayscale prepared for edge detection: a 3x3 median kills the isolated
 * speckle that low-light sensors produce (which a blur alone only smears into
 * blobs the Sobel then happily outlines), then a small Gaussian smooths what's
 * left so gradients are stable.
 */
function prepGray(d, w, h) {
  return gaussian5(median3(toGray(d, w, h), w, h), w, h);
}

/** 3x3 median filter. Edge pixels are passed through unchanged. */
function median3(src, w, h) {
  const out = new Uint8ClampedArray(src.length);
  out.set(src);
  const win = new Uint8Array(9);
  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      let k = 0;
      for (let dy = -1; dy <= 1; dy++) {
        const row = (y + dy) * w + x;
        win[k++] = src[row - 1];
        win[k++] = src[row];
        win[k++] = src[row + 1];
      }
      // Insertion sort of 9 elements beats Array#sort here by a wide margin.
      for (let i = 1; i < 9; i++) {
        const v = win[i];
        let j = i - 1;
        while (j >= 0 && win[j] > v) { win[j + 1] = win[j]; j--; }
        win[j + 1] = v;
      }
      out[y * w + x] = win[4];
    }
  }
  return out;
}

/** Separable [1 4 6 4 1]/16 Gaussian (sigma ~1). Borders clamp. */
function gaussian5(src, w, h) {
  const K = [1, 4, 6, 4, 1];
  const tmp = new Float32Array(src.length);
  const out = new Uint8ClampedArray(src.length);

  for (let y = 0; y < h; y++) {
    const row = y * w;
    for (let x = 0; x < w; x++) {
      let acc = 0;
      for (let k = -2; k <= 2; k++) acc += src[row + clampInt(x + k, 0, w - 1)] * K[k + 2];
      tmp[row + x] = acc / 16;
    }
  }
  for (let x = 0; x < w; x++) {
    for (let y = 0; y < h; y++) {
      let acc = 0;
      for (let k = -2; k <= 2; k++) acc += tmp[clampInt(y + k, 0, h - 1) * w + x] * K[k + 2];
      out[y * w + x] = acc / 16;
    }
  }
  return out;
}

/**
 * @param {Uint8ClampedArray} gray
 * @returns {{mag: Float32Array, dir: Float32Array, max: number}}
 */
function sobelMagnitude(gray, w, h) {
  const mag = new Float32Array(w * h);
  const dir = new Float32Array(w * h);
  let max = 0;
  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      const i = y * w + x;
      const tl = gray[i - w - 1], t = gray[i - w], tr = gray[i - w + 1];
      const l = gray[i - 1], r = gray[i + 1];
      const bl = gray[i + w - 1], b = gray[i + w], br = gray[i + w + 1];

      const gx = tl + 2 * l + bl - tr - 2 * r - br;
      const gy = tl + 2 * t + tr - bl - 2 * b - br;
      const m = Math.hypot(gx, gy);
      mag[i] = m;
      dir[i] = Math.atan2(gy, gx);
      if (m > max) max = m;
    }
  }
  return { mag, dir, max };
}

function clampInt(v, lo, hi) { return v < lo ? lo : v > hi ? hi : v; }
function clamp01(v) { return v < 0 ? 0 : v > 1 ? 1 : v; }
