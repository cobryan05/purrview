import { fetchImageMetadata } from '/app-static/js/shared/api/images.js';
import { getImageThumbnailUrl } from '/app-static/js/shared/api/review.js';
import { getSetting } from '/app-static/js/shared/api/settings.js';

/**
 * Hover-triggered popover that shows an annotated thumbnail preview for an
 * image entry. Single module covering both call styles that previously lived
 * in `app/review/hoverPreview.js` (rich cards, known bboxes, optional
 * highlighted bbox, shared `imageCache`) and `ui/fileEntryHoverPreview.js`
 * (file-browser entries, bboxes fetched lazily, no cache).
 */

const DEFAULT_SHOW_DELAY_MS = 500;
const DEFAULT_THUMB_WIDTH = 640;

/**
 * Module-level configured hover delay. Starts at DEFAULT_SHOW_DELAY_MS and can
 * be updated at runtime by calling setConfiguredHoverDelay().
 */
let _configuredDelayMs = DEFAULT_SHOW_DELAY_MS;

/**
 * Update the hover popup delay for all preview instances.
 * @param {number} ms  Delay in milliseconds.
 */
export function setConfiguredHoverDelay(ms) {
  _configuredDelayMs = ms;
}

// Async init: load any persisted delay setting from the settings API.
(async () => {
  try {
    const stored = await getSetting('ui_hover_delay_ms');
    if (stored !== null) {
      const ms = parseInt(stored, 10);
      if (Number.isFinite(ms) && ms >= 0 && ms <= 10000) _configuredDelayMs = ms;
    }
  } catch (_) {}
})();

/**
 * @param {{
 *   imageCache?: Map<string, { objectUrl: string, bboxes: object[] }>,
 *   thumbWidth?: number,
 *   showDelayMs?: number,
 * }} [opts]
 * @returns {{ attach(el: HTMLElement, meta: object): void, destroy(): void }}
 */
export function createHoverPreview({ imageCache, thumbWidth = DEFAULT_THUMB_WIDTH, showDelayMs = null } = {}) {
  // Created lazily on first hover (mirrors the old file-browser singleton) so
  // pages that never hover an image never grow a stray DOM node.
  let popover = null;
  let imgWrap = null;
  let sideWrap = null;

  let _timer = null;
  let _visible = false;
  let _token = 0;
  let _destroyed = false;

  function ensurePopover() {
    if (popover) return;
    popover = document.createElement('div');
    // Emit both legacy class names so the existing review/file-browser CSS
    // keeps applying without a stylesheet rewrite (see GRID_REWORK.md Phase 1).
    popover.className = 'thumb-hover-popover review-hover-popover fb-hover-popover';

    imgWrap = document.createElement('div');
    imgWrap.className = 'popover-img-wrap fb-popover-img-wrap';
    popover.appendChild(imgWrap);

    // Optional column beside the image, filled per-show from meta.buildSidePanel().
    // Empty (and display:none) for every caller that doesn't supply one.
    sideWrap = document.createElement('div');
    sideWrap.className = 'popover-side';
    popover.appendChild(sideWrap);

    document.body.appendChild(popover);
  }

  function hide() {
    clearTimeout(_timer);
    _timer = null;
    _token++;
    if (!_visible) return;
    popover.classList.remove('visible');
    _visible = false;
  }

  function positionNear(el) {
    const rect = el.getBoundingClientRect();
    const vw = window.innerWidth;
    const vh = window.innerHeight;
    const pw = popover.offsetWidth || (thumbWidth + 16);
    const ph = popover.offsetHeight || 200;

    // Try right of card, vertically centered-ish with card top.
    let left = rect.right + 8;
    let top = rect.top - 100;

    // Flip to left if it overflows the right edge.
    if (left + pw > vw) {
      left = rect.left - pw - 8;
    }
    if (left < 4) left = 4;

    // Flip up if it overflows the bottom edge.
    if (top + ph > vh) {
      top = Math.max(4, vh - ph - 8);
    }
    if (top < 4) top = 4;

    popover.style.left = `${left}px`;
    popover.style.top = `${top}px`;
  }

  function drawAnnotations(canvas, bboxes, hoveredBboxUuid) {
    const ctx = canvas.getContext('2d');
    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    for (const bbox of bboxes ?? []) {
      const x = bbox.x * w;
      const y = bbox.y * h;
      const bw = bbox.width * w;
      const bh = bbox.height * h;

      if (hoveredBboxUuid) {
        const isHovered = bbox.uuid === hoveredBboxUuid;
        if (isHovered) {
          // Highlighted bbox: colored fill + thick stroke.
          const color = bbox.label?.color ?? '#ffdd00';
          ctx.globalAlpha = 0.15;
          ctx.fillStyle = color;
          ctx.fillRect(x, y, bw, bh);
          ctx.globalAlpha = 1.0;
          ctx.strokeStyle = color;
          ctx.lineWidth = 2.5;
        } else {
          // Other bboxes: thin stroke at low opacity.
          ctx.globalAlpha = 0.55;
          ctx.strokeStyle = bbox.label?.color ?? '#ffffff';
          ctx.lineWidth = 1.5;
        }
        ctx.strokeRect(x, y, bw, bh);
        ctx.globalAlpha = 1.0;
      } else {
        // No highlight target: uniform strokes.
        ctx.strokeStyle = bbox.label?.color ?? '#ffdd00';
        ctx.lineWidth = 2;
        ctx.strokeRect(x, y, bw, bh);
      }
    }
  }

  function waitForImage(img) {
    if (img.complete && img.naturalWidth) return Promise.resolve(true);
    return new Promise((resolve) => {
      img.onload = () => resolve(true);
      img.onerror = () => resolve(false);
    });
  }

  async function resolveThumbnail(img, imagePath, imageUuid, allBboxes, token) {
    if (!imageCache) {
      img.src = getImageThumbnailUrl(imagePath, thumbWidth);
      return waitForImage(img);
    }

    const cached = imageCache.get(imageUuid);
    if (cached) {
      img.src = cached.objectUrl;
      return waitForImage(img);
    }

    try {
      const url = getImageThumbnailUrl(imagePath, thumbWidth);
      const res = await fetch(url);
      if (!res.ok || token !== _token) return false;
      const blob = await res.blob();
      if (token !== _token) return false;
      const objectUrl = URL.createObjectURL(blob);
      imageCache.set(imageUuid, { objectUrl, bboxes: allBboxes });
      img.src = objectUrl;
      return waitForImage(img);
    } catch {
      return false;
    }
  }

  async function resolveBboxes(imagePath, imageUuid, allBboxes) {
    if (allBboxes !== undefined) return allBboxes;

    const cached = imageCache?.get(imageUuid);
    if (cached) return cached.bboxes;

    const meta = await fetchImageMetadata(imagePath).catch(() => null);
    return meta?.bboxes;
  }

  async function show(el, meta) {
    if (_destroyed) return;
    const { imagePath, imageUuid, allBboxes, hoveredBboxUuid, buildSidePanel } = meta;
    ensurePopover();
    const token = ++_token;

    // Built at show time, not attach time, so a card whose state changed since it was
    // attached (e.g. gaps resolved) shows the current list rather than a stale capture.
    sideWrap.innerHTML = '';
    const sidePanel = buildSidePanel?.() ?? null;
    if (sidePanel) sideWrap.appendChild(sidePanel);
    sideWrap.hidden = !sidePanel;
    popover.classList.toggle('has-side', !!sidePanel);

    imgWrap.innerHTML = '';
    const img = document.createElement('img');
    img.className = 'popover-img fb-popover-img';
    const canvas = document.createElement('canvas');
    canvas.className = 'popover-canvas fb-popover-canvas';
    imgWrap.append(img, canvas);

    const [imgOk, bboxes] = await Promise.all([
      resolveThumbnail(img, imagePath, imageUuid, allBboxes, token),
      resolveBboxes(imagePath, imageUuid, allBboxes),
    ]);

    if (token !== _token || imgOk === false || !img.naturalWidth) return;
    if (!el.isConnected) return;  // element removed from DOM (e.g. modal closed during load)

    canvas.width = img.naturalWidth;
    canvas.height = img.naturalHeight;
    drawAnnotations(canvas, bboxes, hoveredBboxUuid);
    positionNear(el);
    popover.classList.add('visible');
    _visible = true;
  }

  function attach(el, meta) {
    el.addEventListener('mouseenter', () => {
      clearTimeout(_timer);
      // Read _configuredDelayMs at hover time so runtime changes take effect
      // immediately without needing to recreate hover-preview instances.
      _timer = setTimeout(() => show(el, meta), showDelayMs ?? _configuredDelayMs);
    });
    el.addEventListener('mouseleave', hide);
  }

  function destroy() {
    if (_destroyed) return;
    _destroyed = true;
    clearTimeout(_timer);
    _token++;
    if (imageCache) {
      for (const { objectUrl } of imageCache.values()) {
        URL.revokeObjectURL(objectUrl);
      }
      imageCache.clear();
    }
    popover?.remove();
  }

  return { attach, destroy };
}

const IMAGE_NAME_RE = /\.(jpe?g|png|gif|webp|bmp|tiff?)$/i;

/** @type {{ attach(el: HTMLElement, meta: object): void, destroy(): void }|null} */
let _singleton = null;

/**
 * Singleton adapter preserving the `attachFileEntryHoverPreview` call style
 * used by `FileBrowser`. No-op for directories and non-image files. Lazily
 * creates a shared popover (no `imageCache` — bboxes are fetched on hover).
 *
 * @param {HTMLElement} el - Element to listen for hover on.
 * @param {{ name: string, path: string, isDir: boolean, uuid?: string }} entry
 */
export function attachFileEntryHoverPreview(el, entry) {
  if (entry.isDir || !IMAGE_NAME_RE.test(entry.name)) return;
  _singleton ??= createHoverPreview();
  _singleton.attach(el, { imagePath: entry.path, imageUuid: entry.uuid ?? null });
}
