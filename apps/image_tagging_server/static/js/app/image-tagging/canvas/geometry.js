/**
 * Pure bbox geometry helpers.
 *
 * Split out of canvas/image.js so that modules needing only the maths do not
 * have to import the whole image/inference/save surface with it — image.js
 * pulls in canvas/interaction.js, which pulls in canvas/contextTargets.js, and
 * an import from there back into image.js would close a cycle.
 */

/**
 * Intersection-over-union of two normalized (0-1) rects.
 * @param {{x:number,y:number,width:number,height:number}} a
 * @param {{x:number,y:number,width:number,height:number}} b
 * @returns {number}
 */
export function computeIou(a, b) {
  const ax2 = a.x + a.width, ay2 = a.y + a.height;
  const bx2 = b.x + b.width, by2 = b.y + b.height;
  const ix1 = Math.max(a.x, b.x), iy1 = Math.max(a.y, b.y);
  const ix2 = Math.min(ax2, bx2), iy2 = Math.min(ay2, by2);
  const iw = Math.max(0, ix2 - ix1), ih = Math.max(0, iy2 - iy1);
  const inter = iw * ih;
  if (inter <= 0) return 0;
  const union = a.width * a.height + b.width * b.height - inter;
  return union > 0 ? inter / union : 0;
}
