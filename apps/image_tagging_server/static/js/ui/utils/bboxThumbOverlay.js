/**
 * Draws normalized (0-1) bbox rects onto a canvas overlaid on an
 * `object-fit:contain` thumbnail `<img>`, accounting for letterboxing.
 * Shared by review/main.js's image-mode cards and the "Run Model on
 * Results" preview modal's new-box candidate thumbnails.
 *
 * A box may carry an optional short `label` string (the review grid passes the
 * model confidence, e.g. "82%"), drawn as a filled tab above the box in the
 * box's own colour so it reads as belonging to that box on a crowded frame.
 */

// Label tab geometry, as a fraction of the thumbnail's CSS width — the same
// overlay is drawn at 200px on a card and much larger in a hover preview, so
// fixed pixel sizes would be illegible at one end and gaudy at the other.
const LABEL_FONT_RATIO = 0.06;
const LABEL_MIN_FONT_PX = 8;
const LABEL_MAX_FONT_PX = 14;

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

  const fontPx = Math.round(
    Math.min(LABEL_MAX_FONT_PX, Math.max(LABEL_MIN_FONT_PX, cssW * LABEL_FONT_RATIO))
  );
  const padX = Math.max(2, Math.round(fontPx * 0.3));
  const tabH = fontPx + padX;

  for (const { x, y, width, height, color, label } of boxes) {
    const stroke = color ?? '#ffdd00';
    const px = offsetX + x * dispW;
    const py = offsetY + y * dispH;
    const pw = width * dispW;
    const ph = height * dispH;

    ctx.strokeStyle = stroke;
    ctx.strokeRect(px, py, pw, ph);

    if (!label) continue;

    ctx.font = `600 ${fontPx}px system-ui, sans-serif`;
    ctx.textBaseline = 'middle';
    const textW = ctx.measureText(label).width;
    const tabW = textW + padX * 2;
    // Sit the tab above the box, unless the box is hard against the top edge
    // (or is taller than the frame's top margin), in which case tuck it inside.
    const tabY = py - tabH >= 0 ? py - tabH : py;
    // Keep the tab inside the canvas horizontally so a box at the right edge
    // doesn't render its confidence half off-screen.
    const tabX = Math.max(0, Math.min(px, cssW - tabW));

    ctx.fillStyle = stroke;
    ctx.fillRect(tabX, tabY, tabW, tabH);
    // Fixed dark ink rather than a computed contrast colour: every label colour
    // in the app is a mid-to-bright hue, and a per-box luminance calculation
    // would run for every box on every redraw of every card.
    ctx.fillStyle = '#111';
    ctx.fillText(label, tabX + padX, tabY + tabH / 2);
  }
}
