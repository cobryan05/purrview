import { getImageThumbnailUrl } from '/app-static/js/shared/api/review.js';

const SHOW_DELAY_MS = 300;
const POPOVER_WIDTH = 500;

/**
 * Create a singleton hover-preview popover for review cards.
 *
 * @param {{ imageCache: Map<string, { objectUrl: string, bboxes: object[] }> }} opts
 * @returns {{ attach(card, opts): void, destroy(): void }}
 */
export function createHoverPreview({ imageCache }) {
  const popover = document.createElement('div');
  popover.className = 'review-hover-popover';
  popover.innerHTML = '<div class="popover-img-wrap"></div>';
  document.body.appendChild(popover);

  const imgWrap = popover.querySelector('.popover-img-wrap');

  let _timer = null;
  let _visible = false;
  let _currentObjectUrl = null; // track so we can revoke on cache clear

  function hide() {
    clearTimeout(_timer);
    _timer = null;
    if (!_visible) return;
    popover.classList.remove('visible');
    _visible = false;
  }

  function positionNearCard(card) {
    const rect = card.getBoundingClientRect();
    const vw = window.innerWidth;
    const vh = window.innerHeight;
    const pw = POPOVER_WIDTH + 8; // +padding
    const ph = popover.offsetHeight || 200;

    // Try right of card, vertically centered-ish with card top
    let left = rect.right + 8;
    let top = rect.top - 100;

    // Flip to left if it overflows right edge
    if (left + pw > vw) {
      left = rect.left - pw - 8;
    }
    // Keep on screen if left is negative
    if (left < 4) left = 4;

    // Flip below if it overflows top (popover would go negative)
    if (top + ph > vh) {
      top = Math.max(4, vh - ph - 8);
    }
    if (top < 4) top = 4;

    popover.style.left = `${left}px`;
    popover.style.top = `${top}px`;
  }

  function drawPopoverAnnotations(canvas, bboxes, hoveredBboxUuid) {
    const ctx = canvas.getContext('2d');
    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    for (const bbox of (bboxes ?? [])) {
      const x = bbox.x * w;
      const y = bbox.y * h;
      const bw = bbox.width * w;
      const bh = bbox.height * h;
      const isHovered = bbox.uuid === hoveredBboxUuid;

      if (isHovered) {
        // Highlighted bbox: colored fill + thick stroke
        const color = bbox.label?.color ?? '#ffdd00';
        ctx.globalAlpha = 0.15;
        ctx.fillStyle = color;
        ctx.fillRect(x, y, bw, bh);
        ctx.globalAlpha = 1.0;
        ctx.strokeStyle = color;
        ctx.lineWidth = 2.5;
      } else {
        // Other bboxes: thin white stroke at low opacity
        ctx.globalAlpha = 0.55;
        ctx.strokeStyle = bbox.label?.color ?? '#ffffff';
        ctx.lineWidth = 1.5;
      }
      ctx.strokeRect(x, y, bw, bh);
      ctx.globalAlpha = 1.0;
    }
  }

  async function show(card, { imagePath, imageUuid, allBboxes, hoveredBboxUuid }) {
    // Build/update popover content
    imgWrap.innerHTML = '';

    let cached = imageCache.get(imageUuid);
    if (!cached) {
      try {
        const url = getImageThumbnailUrl(imagePath, 640);
        const res = await fetch(url);
        if (!res.ok) return;
        const blob = await res.blob();
        const objectUrl = URL.createObjectURL(blob);
        cached = { objectUrl, bboxes: allBboxes };
        imageCache.set(imageUuid, cached);
      } catch {
        return;
      }
    }

    const img = document.createElement('img');
    img.className = 'popover-img';
    img.src = cached.objectUrl;

    const canvas = document.createElement('canvas');
    canvas.className = 'popover-canvas';

    imgWrap.appendChild(img);
    imgWrap.appendChild(canvas);

    // Set canvas resolution to natural image size; CSS handles display scaling via width/height 100%.
    function setupCanvas() {
      canvas.width = img.naturalWidth || 640;
      canvas.height = img.naturalHeight || 480;
      drawPopoverAnnotations(canvas, cached.bboxes, hoveredBboxUuid);
      positionNearCard(card);
      popover.classList.add('visible');
      _visible = true;
    }

    if (img.complete && img.naturalWidth) {
      setupCanvas();
    } else {
      img.onload = setupCanvas;
    }
  }

  function attach(card, meta) {
    card.addEventListener('mouseenter', () => {
      clearTimeout(_timer);
      _timer = setTimeout(() => show(card, meta), SHOW_DELAY_MS);
    });
    card.addEventListener('mouseleave', hide);
  }

  function destroy() {
    clearTimeout(_timer);
    // Revoke any created object URLs
    for (const { objectUrl } of imageCache.values()) {
      URL.revokeObjectURL(objectUrl);
    }
    imageCache.clear();
    popover.remove();
  }

  return { attach, destroy };
}
