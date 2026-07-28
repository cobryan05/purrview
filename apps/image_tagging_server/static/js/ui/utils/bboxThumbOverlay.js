/**
 * Draws normalized (0-1) bbox rects onto a canvas overlaid on an
 * `object-fit:contain` thumbnail `<img>`, accounting for letterboxing.
 * Shared by review/main.js's image-mode cards and the "Run Model on
 * Results" preview modal's new-box candidate thumbnails.
 */
export function drawLetterboxedBboxes(imgEl, canvasEl, boxes) {
  const natW = imgEl.naturalWidth;
  const natH = imgEl.naturalHeight;
  const cssW = imgEl.clientWidth;
  const cssH = imgEl.clientHeight;
  if (!natW || !natH || !cssW || !cssH) return;

  canvasEl.width = cssW;
  canvasEl.height = cssH;
  const ctx = canvasEl.getContext('2d');

  const scale = Math.min(cssW / natW, cssH / natH);
  const dispW = natW * scale;
  const dispH = natH * scale;
  const offsetX = (cssW - dispW) / 2;
  const offsetY = (cssH - dispH) / 2;

  ctx.clearRect(0, 0, cssW, cssH);
  ctx.lineWidth = Math.max(1, Math.round(cssW / 150));
  for (const { x, y, width, height, color } of boxes) {
    ctx.strokeStyle = color ?? '#ffdd00';
    ctx.strokeRect(offsetX + x * dispW, offsetY + y * dispH, width * dispW, height * dispH);
  }
}
