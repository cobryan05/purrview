/**
 * Draw what the watcher saw on top of the clip it saw it in.
 *
 * The boxes come from a JSON sidecar written beside the mp4 when the event closed (see
 * `apps/helpers/recordings/detectionTrack.py`), not from anything burnt into the video.
 * That is the whole design: the clip stays pristine, so "save frame for annotation"
 * cannot accidentally put a training image with boxes drawn on it into the dataset, and
 * the overlay can be switched off, restyled, or corrected without re-encoding anything.
 *
 * Two things are drawn. The **boxes** answer "what did the detector think it was looking
 * at", which is the question you ask when a rule fired on a shadow. The **red frame**
 * answers "when did the rule believe the event was happening", which the video alone
 * cannot show — pre-roll and post-roll look exactly like the event itself.
 *
 * ### Coordinates
 *
 * Sidecar boxes are normalised to the *analysis* frame, which is a downscale of the
 * camera's own resolution; the `<video>` is at the camera's resolution and is laid out
 * with `object-fit: contain`, so it is letterboxed inside its element. Three spaces,
 * therefore, and the canvas covers the element rather than the picture. `videoRect()`
 * does the contain math; skipping it draws every box offset by the letterbox bars, which
 * looks like a detector that is consistently wrong by a few percent.
 *
 * ### Time
 *
 * Samples are held until the next one, because the watcher analyses at a few frames a
 * second while the video plays at the camera's full rate — without the hold, boxes would
 * strobe. The hold is bounded by the sidecar's own `maxStalenessS`: past that the
 * watcher was not analysing, and drawing a gap is honest where drawing a frozen box
 * would invent a detection that never happened.
 */

/** Sidecar shapes this renderer understands. A newer one is ignored, not guessed at. */
const SUPPORTED_VERSION = 1;

/** Fallback hold when a sidecar omits it. Generous, but still bounded. */
const DEFAULT_STALENESS_S = 1.5;

/** The red frame's thickness, in CSS pixels. */
const EVENT_FRAME_WIDTH = 4;

/**
 * Fetch a clip's overlay sidecar.
 *
 * A missing overlay is the normal case, not an error: clips written before this existed
 * have none, the action's checkbox can be off, and a chunk-assembled clip is
 * deliberately given none. All of them resolve to null and the caller simply offers no
 * overlay.
 *
 * @param {string} recordingUuid
 * @returns {Promise<?Object>}
 */
export async function fetchClipOverlay(recordingUuid) {
  try {
    const res = await fetch(`/api/recordings/${recordingUuid}/overlay`);
    if (!res.ok) return null;
    const doc = await res.json();
    if (!doc || doc.version !== SUPPORTED_VERSION || !Array.isArray(doc.frames)) return null;
    return doc;
  } catch {
    // A clip that plays with no overlay is a far better outcome than a modal that fails
    // to open because a few tens of KB of JSON did not arrive.
    return null;
  }
}

/**
 * Where the picture actually is inside a letterboxed `<video>`, in CSS pixels relative
 * to the element's own box.
 *
 * @param {HTMLVideoElement} video
 * @returns {{left: number, top: number, width: number, height: number}}
 */
export function videoRect(video) {
  const boxW = video.clientWidth;
  const boxH = video.clientHeight;
  const natW = video.videoWidth;
  const natH = video.videoHeight;
  // Before metadata arrives there is no aspect ratio to honour. Filling the box is the
  // right answer for exactly one frame, and the first `loadedmetadata` redraw fixes it.
  if (!natW || !natH || !boxW || !boxH) return { left: 0, top: 0, width: boxW, height: boxH };
  const scale = Math.min(boxW / natW, boxH / natH);
  const width = natW * scale;
  const height = natH * scale;
  return { left: (boxW - width) / 2, top: (boxH - height) / 2, width, height };
}

/**
 * The sample in effect at `t` seconds into the clip, or null.
 *
 * The last sample at or before `t`, which is what "held until the next one" means. Null
 * once that sample is older than the hold allows, and null before the first sample —
 * a clip whose pre-roll predates the watcher starting has a genuinely unrecorded head.
 *
 * Linear from the front. The arrays are a few hundred entries and this runs on
 * `timeupdate` (~4 Hz), so a binary search would buy nothing measurable and cost the
 * reader something real.
 *
 * @param {Object} doc
 * @param {number} t
 * @returns {?Object}
 */
export function sampleAt(doc, t) {
  const frames = doc?.frames || [];
  const maxStale = Number.isFinite(doc?.maxStalenessS) ? doc.maxStalenessS : DEFAULT_STALENESS_S;
  let found = null;
  for (const frame of frames) {
    if (frame.t > t) break;
    found = frame;
  }
  if (!found) return null;
  return t - found.t <= maxStale ? found : null;
}

/**
 * Whether the rule was firing at `t`.
 *
 * @param {Object} doc
 * @param {number} t
 * @returns {boolean}
 */
export function eventActiveAt(doc, t) {
  const ev = doc?.event;
  if (!ev || !Number.isFinite(ev.startS)) return false;
  const end = Number.isFinite(ev.endS) ? ev.endS : Infinity;
  return t >= ev.startS && t <= end;
}

/**
 * Attach an overlay canvas to a `<video>`.
 *
 * The canvas is inserted as a sibling inside a positioned wrapper rather than drawn into
 * the video: a `<video>` cannot host children, and compositing into it would mean
 * re-encoding. It is `pointer-events: none` so the native controls underneath keep
 * working — an overlay that ate clicks on the scrubber would be worse than no overlay.
 *
 * @param {HTMLVideoElement} video
 * @param {Object} doc - a sidecar document from `fetchClipOverlay`
 * @param {Object} [options]
 * @param {boolean} [options.visible=true]
 * @param {Map<string,string>} [options.labelColours] - labelUuid → `#rrggbb`
 * @returns {{setVisible: (on: boolean) => void, redraw: () => void, destroy: () => void,
 *           boxesAt: (t: number) => Array<Object>}}
 */
export function attachClipOverlay(video, doc, { visible = true, labelColours = new Map() } = {}) {
  const wrapper = document.createElement('div');
  wrapper.className = 'clip-overlay';
  video.parentNode.insertBefore(wrapper, video);
  wrapper.appendChild(video);

  const canvas = document.createElement('canvas');
  canvas.className = 'clip-overlay__canvas';
  canvas.dataset.testid = 'clip-overlay-canvas';
  wrapper.appendChild(canvas);

  let showing = visible;

  function draw() {
    const rect = videoRect(video);
    // Backed at device resolution so a 1px stroke is a real pixel on a HiDPI screen,
    // but laid out in CSS pixels — hence the transform rather than doubling every
    // coordinate at the call sites.
    const dpr = window.devicePixelRatio || 1;
    const cssW = video.clientWidth;
    const cssH = video.clientHeight;
    if (canvas.width !== Math.round(cssW * dpr) || canvas.height !== Math.round(cssH * dpr)) {
      canvas.width = Math.round(cssW * dpr);
      canvas.height = Math.round(cssH * dpr);
    }
    canvas.style.width = `${cssW}px`;
    canvas.style.height = `${cssH}px`;

    const ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, cssW, cssH);
    if (!showing) return;

    const t = video.currentTime || 0;

    if (eventActiveAt(doc, t)) {
      ctx.strokeStyle = '#ef4444';
      ctx.lineWidth = EVENT_FRAME_WIDTH;
      ctx.strokeRect(
        rect.left + EVENT_FRAME_WIDTH / 2,
        rect.top + EVENT_FRAME_WIDTH / 2,
        Math.max(0, rect.width - EVENT_FRAME_WIDTH),
        Math.max(0, rect.height - EVENT_FRAME_WIDTH),
      );
      const name = doc.event?.ruleName;
      if (name) drawTag(ctx, `● ${name}`, rect.left + EVENT_FRAME_WIDTH + 4, rect.top + EVENT_FRAME_WIDTH + 4, '#ef4444');
    }

    const sample = sampleAt(doc, t);
    for (const box of sample?.boxes || []) {
      const colour = labelColours.get(box.labelUuid) || '#38bdf8';
      const x = rect.left + box.x * rect.width;
      const y = rect.top + box.y * rect.height;
      const w = box.w * rect.width;
      const h = box.h * rect.height;
      ctx.strokeStyle = colour;
      ctx.lineWidth = 2;
      ctx.strokeRect(x, y, w, h);
      const label = box.conf ? `${box.label || '?'} ${Math.round(box.conf * 100)}%` : box.label || '?';
      drawTag(ctx, label, x, Math.max(rect.top, y - 16), colour);
    }
  }

  function drawTag(ctx, text, x, y, colour) {
    ctx.font = '12px system-ui, sans-serif';
    const width = ctx.measureText(text).width + 8;
    ctx.fillStyle = 'rgba(0, 0, 0, 0.65)';
    ctx.fillRect(x, y, width, 16);
    ctx.fillStyle = colour;
    ctx.fillText(text, x + 4, y + 12);
  }

  // `timeupdate` alone fires about four times a second, which visibly lags the picture.
  // `requestVideoFrameCallback` redraws in step with the frames actually presented where
  // it exists; the listeners stay regardless so a paused, seeked or resized player still
  // updates, which rVFC does not cover.
  let rafHandle = null;
  function pump() {
    draw();
    if (video.requestVideoFrameCallback) {
      rafHandle = video.requestVideoFrameCallback(pump);
    }
  }
  if (video.requestVideoFrameCallback) rafHandle = video.requestVideoFrameCallback(pump);

  const redraw = () => draw();
  video.addEventListener('timeupdate', redraw);
  video.addEventListener('seeked', redraw);
  video.addEventListener('loadedmetadata', redraw);

  // Double-clicking a `<video>` is the browser's own shortcut for fullscreen, and it
  // fullscreens the VIDEO ELEMENT — which leaves this canvas, its sibling, behind on the
  // page: the picture fills the screen and the boxes silently vanish. That is the one
  // failure this wrapper exists to prevent, so the gesture is taken over here rather
  // than left to whichever browser happens to implement it. Handled on the wrapper so
  // the whole letterboxed area responds, not only the picture.
  const onDoubleClick = (e) => {
    e.preventDefault();
    if (document.fullscreenElement) document.exitFullscreen?.();
    // Not awaited: a refusal should leave the player exactly as it was.
    else wrapper.requestFullscreen?.().catch(() => {});
  };
  wrapper.addEventListener('dblclick', onDoubleClick);
  const observer = new ResizeObserver(redraw);
  observer.observe(video);
  draw();

  return {
    setVisible(on) {
      showing = !!on;
      draw();
    },
    redraw,
    /**
     * The boxes in effect at `t`, normalised. Exposed so "save frame for annotation" can
     * seed the captured frame with exactly the boxes drawn over it.
     */
    boxesAt(t) {
      return sampleAt(doc, t)?.boxes || [];
    },
    destroy() {
      observer.disconnect();
      video.removeEventListener('timeupdate', redraw);
      video.removeEventListener('seeked', redraw);
      video.removeEventListener('loadedmetadata', redraw);
      wrapper.removeEventListener('dblclick', onDoubleClick);
      if (rafHandle != null && video.cancelVideoFrameCallback) video.cancelVideoFrameCallback(rafHandle);
      canvas.remove();
      // Put the video back where it was before unwrapping, so a caller that kept a
      // reference to it (the frame capture does) is not left holding a detached node.
      wrapper.parentNode?.insertBefore(video, wrapper);
      wrapper.remove();
    },
  };
}
