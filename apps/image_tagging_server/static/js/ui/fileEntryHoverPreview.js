import { fetchImageMetadata } from '/app-static/js/shared/api/images.js';
import { getImageThumbnailUrl } from '/app-static/js/shared/api/review.js';

/**
 * Hover-triggered popover that shows a thumbnail preview for image file entries
 * in the FileBrowser. When the entry is registered in the database (has a uuid),
 * its bboxes are fetched and drawn over the thumbnail.
 *
 * A single popover is shared across all FileBrowser instances (lazily created on
 * first use) so that repeatedly opening browser modals doesn't leak DOM nodes.
 */

const SHOW_DELAY_MS = 300;
const THUMBNAIL_WIDTH = 320;
const IMAGE_NAME_RE = /\.(jpe?g|png|gif|webp|bmp|tiff?)$/i;

/** @type {HTMLElement|null} */
let _popover = null;
/** @type {HTMLImageElement|null} */
let _img = null;
/** @type {HTMLCanvasElement|null} */
let _canvas = null;
let _timer = null;
let _visible = false;
let _token = 0;

function ensurePopover() {
  if (_popover) return;

  _popover = document.createElement('div');
  _popover.className = 'fb-hover-popover';

  const wrap = document.createElement('div');
  wrap.className = 'fb-popover-img-wrap';

  _img = document.createElement('img');
  _img.className = 'fb-popover-img';

  _canvas = document.createElement('canvas');
  _canvas.className = 'fb-popover-canvas';

  wrap.append(_img, _canvas);
  _popover.appendChild(wrap);
  document.body.appendChild(_popover);
}

function hide() {
  clearTimeout(_timer);
  _timer = null;
  _token++;
  if (!_visible) return;
  _popover?.classList.remove('visible');
  _visible = false;
}

function positionNear(el) {
  const rect = el.getBoundingClientRect();
  const vw = window.innerWidth;
  const vh = window.innerHeight;
  const pw = _popover.offsetWidth || 280;
  const ph = _popover.offsetHeight || 200;

  let left = rect.right + 8;
  let top = rect.top;

  if (left + pw > vw) left = rect.left - pw - 8;
  if (left < 4) left = 4;
  if (top + ph > vh) top = Math.max(4, vh - ph - 8);
  if (top < 4) top = 4;

  _popover.style.left = `${left}px`;
  _popover.style.top = `${top}px`;
}

function drawAnnotations(bboxes) {
  const ctx = _canvas.getContext('2d');
  ctx.clearRect(0, 0, _canvas.width, _canvas.height);
  for (const bbox of bboxes ?? []) {
    const x = bbox.x * _canvas.width;
    const y = bbox.y * _canvas.height;
    const w = bbox.width * _canvas.width;
    const h = bbox.height * _canvas.height;
    ctx.strokeStyle = bbox.label?.color ?? '#ffdd00';
    ctx.lineWidth = 2;
    ctx.strokeRect(x, y, w, h);
  }
}

/**
 * @param {HTMLElement} el
 * @param {{ path: string }} entry
 */
async function show(el, entry) {
  ensurePopover();
  const token = ++_token;

  const imgLoaded = new Promise(resolve => {
    _img.onload = resolve;
    _img.onerror = resolve;
  });
  _img.src = getImageThumbnailUrl(entry.path, THUMBNAIL_WIDTH);

  // Always attempt the fetch — `entry.uuid` is only populated when the browser
  // was opened with `showDbStatus`, but the metadata endpoint itself reports
  // whether the image is registered (returns null otherwise).
  const metaPromise = fetchImageMetadata(entry.path).catch(() => null);

  const [, meta] = await Promise.all([imgLoaded, metaPromise]);
  if (token !== _token || !_img.naturalWidth) return;

  _canvas.width = _img.naturalWidth;
  _canvas.height = _img.naturalHeight;
  drawAnnotations(meta?.bboxes);

  positionNear(el);
  _popover.classList.add('visible');
  _visible = true;
}

/**
 * Attach a hover-triggered annotated-thumbnail preview to a file-entry element.
 * No-op for directories and non-image files.
 *
 * @param {HTMLElement} el - Element to listen for hover on (typically the filename).
 * @param {{ name: string, path: string, isDir: boolean }} entry
 */
export function attachFileEntryHoverPreview(el, entry) {
  if (entry.isDir || !IMAGE_NAME_RE.test(entry.name)) return;

  el.addEventListener('mouseenter', () => {
    clearTimeout(_timer);
    _timer = setTimeout(() => show(el, entry), SHOW_DELAY_MS);
  });
  el.addEventListener('mouseleave', hide);
}
