/**
 * What a watcher's frame looks like when it is drawn — shared by the Live card
 * thumbnails and the live-view drill-down.
 *
 * One module because the two surfaces show the *same* boxes at two sizes, and a colour
 * that means "detection" on the card while meaning "track" in the modal would make the
 * card actively misleading. The card is the overview; clicking it must not change what
 * anything means.
 *
 * Boxes are always drawn in the frame's own pixel space, on a canvas whose width/height
 * attributes are the frame's — CSS then scales the whole canvas down to whatever the
 * layout gives it. That is why nothing here scales coordinates: at any display size the
 * overlay lines up with the pixels the models were shown.
 */

/**
 * Fallbacks only.
 *
 * A box is stroked in **its label's** colour (`boxColour`) whenever the watcher knows one,
 * because "which animal is that?" is the first question asked of any box and a wall of
 * identically-coloured rectangles cannot answer it. These constants are what's left when
 * a detection has no label mapping at all — the configuration problem that state
 * represents is then visible as a box in the generic colour.
 *
 * The detection/track distinction the diagnostic view depends on (see liveViewModal's
 * docstring) therefore rides on *line style*, not hue: detections dashed, tracks solid.
 * Anchor and event tracks keep their meaning as a coloured glow around a label-coloured
 * box, so both facts are readable at once.
 */
export const DETECTION_COLOUR = '#ffb347';
export const TRACK_COLOUR = '#4ade80';
export const ANCHOR_COLOUR = '#60a5fa';
export const ZONE_COLOUR = '#c084fc';
/**
 * A track an open event is currently held on. Overrides the track colour rather than
 * decorating it: "which of these boxes is the watcher actually reacting to?" is the
 * question these views are opened to answer, and it has to be answerable at a glance
 * from across the room, not by cross-referencing a list with box coordinates.
 *
 * Kept in sync with `.watch-live-list__row--active` / `.nvr-preview-tile--event` in
 * app.css — the row, the badge and the box are one fact and must share one colour.
 */
export const EVENT_COLOUR = '#f87171';

/**
 * A readable caption for one box's label.
 *
 * The uuid is the *last* resort and is truncated when it does appear: a full uuid is
 * both meaningless to a human and wide enough to cover the object it is labelling.
 *
 * @param {?string} labelName
 * @param {?string} labelUuid
 * @param {string} [fallback]
 * @returns {string}
 */
export function labelText(labelName, labelUuid, fallback = '?') {
  if (labelName) return labelName;
  if (labelUuid) return `${labelUuid.slice(0, 8)}…`;
  return fallback;
}

/** True when an open event is currently held on this track. */
export function isEventTrack(track) {
  return (track?.eventRules ?? []).length > 0;
}

/**
 * The semantic colour of a track: event > anchor > plain track.
 *
 * Now the *glow* colour rather than the stroke — the stroke belongs to the label. Still
 * the stroke for a track whose label has no colour, so an unmapped track is never
 * invisible.
 */
export function trackColour(track) {
  if (isEventTrack(track)) return EVENT_COLOUR;
  return track?.isAnchor ? ANCHOR_COLOUR : TRACK_COLOUR;
}

/**
 * A CSS colour string we are willing to hand to `strokeStyle`.
 *
 * Canvas silently ignores an invalid colour and keeps whatever was set before, so a label
 * whose colour is empty, null or garbage would inherit the *previous* box's stroke rather
 * than fall back — boxes would be mis-coloured with no error anywhere. Hence a check
 * instead of a truthiness test.
 *
 * @param {*} value
 * @returns {boolean}
 */
export function isDrawableColour(value) {
  if (typeof value !== 'string') return false;
  const colour = value.trim();
  if (!colour) return false;
  if (/^#([0-9a-f]{3}|[0-9a-f]{4}|[0-9a-f]{6}|[0-9a-f]{8})$/i.test(colour)) return true;
  // Functional notations and plain keywords ("tomato"). Anything with whitespace or
  // punctuation outside those forms is rejected rather than guessed at.
  if (/^(rgb|rgba|hsl|hsla)\([^()]*\)$/i.test(colour)) return true;
  return /^[a-z]+$/i.test(colour);
}

/**
 * The colour to stroke one detection or track in: its label's, else the semantic fallback.
 *
 * @param {?{labelColor?: ?string}} item - a preview detection or track
 * @param {string} fallback - DETECTION_COLOUR, or `trackColour(track)`
 * @returns {string}
 */
export function boxColour(item, fallback) {
  return isDrawableColour(item?.labelColor) ? item.labelColor.trim() : fallback;
}

/**
 * Draw one box, optionally captioned.
 *
 * @param {CanvasRenderingContext2D} ctx
 * @param {?number[]} bbox - [x1, y1, x2, y2] in frame pixels
 * @param {string} colour - stroke and caption colour (the label's, normally)
 * @param {?string} label - omitted when null
 * @param {{dashed?: boolean, emphasis?: boolean, lineWidth?: number, fontPx?: number,
 *          glow?: ?string}} [opts] - `glow` is the halo colour: it carries the box's
 *          *status* (event, anchor) independently of the label colour on the stroke.
 */
export function drawBox(ctx, bbox, colour, label, { dashed = false, emphasis = false, lineWidth = 2, fontPx = 12, glow = null } = {}) {
  if (!ctx || !bbox || bbox.length !== 4) return;
  const [x1, y1, x2, y2] = bbox;
  ctx.save();
  ctx.strokeStyle = colour;
  ctx.lineWidth = emphasis ? lineWidth * 2 : lineWidth;
  if (emphasis || glow) {
    // Glow as well as weight: on a busy frame a thicker line alone reads as
    // "closer to the camera" rather than "this one matters".
    ctx.shadowColor = glow ?? colour;
    ctx.shadowBlur = emphasis ? 12 : 8;
  }
  if (dashed) ctx.setLineDash([5, 4]);
  ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
  ctx.shadowBlur = 0;
  if (label) {
    ctx.setLineDash([]);
    ctx.font = `${fontPx}px sans-serif`;
    const metrics = ctx.measureText(label);
    const boxTop = Math.max(0, y1 - fontPx - 3);
    ctx.fillStyle = 'rgba(0,0,0,0.65)';
    ctx.fillRect(x1, boxTop, metrics.width + 6, fontPx + 2);
    ctx.fillStyle = colour;
    ctx.fillText(label, x1 + 3, boxTop + fontPx - 1);
  }
  ctx.restore();
}

/**
 * Decode a base64 JPEG into something drawable.
 *
 * Kept as one place because the *bytes* are wanted too — "save frame for annotation"
 * stores exactly the JPEG the watcher analysed rather than a canvas re-encode, so the
 * saved image is byte-identical to what the models were shown.
 *
 * @param {string} base64
 * @returns {{blob: Blob, decode: () => Promise<HTMLImageElement>}}
 */
export function decodeFrame(base64) {
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  const blob = new Blob([bytes], { type: 'image/jpeg' });
  return {
    blob,
    decode: async () => {
      const url = URL.createObjectURL(blob);
      try {
        const img = new Image();
        img.src = url;
        await img.decode();
        return img;
      } finally {
        // Safe immediately after decode(): the bitmap is already in memory, and not
        // revoking here leaks one object URL per frame for as long as the view is open.
        URL.revokeObjectURL(url);
      }
    },
  };
}
