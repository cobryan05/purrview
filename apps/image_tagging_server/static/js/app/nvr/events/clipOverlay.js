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

/**
 * Sidecar shapes this renderer understands. A newer one is ignored, not guessed at.
 *
 * Two, because v2 only *added* keys to each track — the per-frame detection behind a
 * track whose reported label became a vote across frames. Nothing this file reads moved,
 * so refusing a v2 document would blank the overlay on every new clip to no purpose.
 */
const SUPPORTED_VERSIONS = new Set([1, 2]);

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
    if (!doc || !SUPPORTED_VERSIONS.has(doc.version) || !Array.isArray(doc.frames)) return null;
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
 * How far past a sample's stated time to seek to land on the frame it describes.
 *
 * Sidecar times are written to four decimals (`_round`, `detectionTrack.py`), so a time
 * generated as `index / fps` can round to just *below* the boundary of its own frame —
 * 1/12 is written `0.0833` — and seeking there presents the previous frame.
 *
 * Playback never sees this: the overlay looks a sample *up* from the true `currentTime`,
 * so it stays self-consistent whatever the seek rounded to. A harvest does the opposite,
 * seeking *to* the stated time, and on a timelapse one frame is several seconds of real
 * time, so the picture it captures shows a moment the boxes never described.
 *
 * Two milliseconds clears the 5e-5 worst case by a wide margin while staying far inside
 * one frame at any rate a camera produces, so it can never reach the *next* frame.
 */
const SEEK_NUDGE_S = 0.002;

/**
 * The player time to seek to in order to capture the frame sample `t` describes.
 *
 * @param {number} t - a sidecar sample time
 * @param {number} [lead] - untrimmed head, as `leadSeconds()` reports it
 * @returns {number}
 */
export function seekTargetFor(t, lead = 0) {
  return Number(t) + (Number(lead) || 0) + SEEK_NUDGE_S;
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

/** Below this, a duration disagreement is muxer rounding rather than an untrimmed head. */
const LEAD_FLOOR_S = 0.25;

/**
 * Seconds of video playing *before* the sidecar's t=0, from the two durations.
 *
 * `declaredS` is how long the clip was meant to be (the sidecar's `durationS`);
 * `actualS` is how long this browser thinks it is. A browser that applied the clip's
 * edit list agrees with the first; one that did not reports the untrimmed head as extra
 * length, and that surplus is exactly how far ahead of the picture the overlay would
 * otherwise be drawn. Measuring it needs no user-agent sniffing.
 *
 * Clamped at 0 in both directions that matter:
 * - a *shorter* file than declared is a chunk-assembled clip whose duration was summed
 *   arithmetically instead of measured — a different fault, which shifting cannot fix
 *   and would only disguise;
 * - a sub-`LEAD_FLOOR_S` surplus is ordinary rounding (the final frame's own duration),
 *   and shifting the overlay by that noise is worse than leaving it alone.
 *
 * Declined outright for a chunk-assembled clip (`partCount > 1`). `concat_parts` runs
 * `-c copy -fflags +genpts`, which regenerates timestamps and discards each part's edit
 * list, so the file physically plays *every* part's untrimmed head — in every browser,
 * Chromium included. `declaredS` for that path is the sum of the logical spans, so the
 * surplus is the sum of all N heads rather than one, and shifting by it would
 * over-correct by N-1 heads on exactly the long events this path exists for.
 *
 * @param {number} declaredS
 * @param {number} actualS
 * @param {number} [partCount=1]
 * @returns {number} seconds, >= 0
 */
export function untrimmedLead(declaredS, actualS, partCount = 1) {
  const declared = Number(declaredS);
  const actual = Number(actualS);
  if (!Number.isFinite(declared) || !Number.isFinite(actual) || declared <= 0) return 0;
  if (Number(partCount) > 1) return 0;
  const surplus = actual - declared;
  return surplus > LEAD_FLOOR_S ? surplus : 0;
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

  /**
   * Seconds of video that play *before* the sidecar's t=0, or 0 when there are none.
   *
   * MediaMTX cannot cut a clip at an arbitrary instant: it starts at the keyframe at or
   * before the requested one and marks the excess with a negative PTS plus an mp4 edit
   * list. Measured on real clips that lead runs to several seconds — one sampled here
   * began 4.2 s early, another 8.8 s.
   *
   * Whether it is *played* is entirely up to the browser. ffmpeg and Chromium apply the
   * edit list, so their t=0 is the instant that was asked for and the sidecar lines up.
   * A browser that does not apply it starts at the earlier keyframe, and every box is
   * then drawn against a picture from `lead` seconds earlier — the overlay runs ahead of
   * the video by exactly that much, which is what a user on such a browser sees.
   *
   * So it is measured rather than assumed, and the measurement needs no user agent
   * sniffing: `durationS` is how long the clip was *meant* to be, and `video.duration`
   * is how long this browser thinks it is. A browser that trimmed agrees; one that did
   * not reports the lead as extra length, and that surplus IS the offset.
   *
   * Deliberately clamped at 0. A negative difference means the file is *shorter* than
   * the sidecar claims — a chunk-assembled clip whose duration was summed arithmetically
   * rather than measured — and that is a different fault, which shifting cannot repair
   * and would only obscure.
   */
  let lead = 0;

  function measureLead() {
    lead = untrimmedLead(doc?.durationS, video.duration, doc?.partCount);
  }

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

    // Sidecar time, which is only the same as player time on a browser that honoured the
    // clip's edit list — see `lead`.
    const t = (video.currentTime || 0) - lead;

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
  // The lead can only be measured once the browser knows the duration, and it must be
  // measured before the first paint that uses it — hence its own handler rather than a
  // bare redraw, and a call now for a video whose metadata already arrived.
  const onMetadata = () => { measureLead(); draw(); };
  measureLead();
  video.addEventListener('timeupdate', redraw);
  video.addEventListener('seeked', redraw);
  video.addEventListener('loadedmetadata', onMetadata);
  video.addEventListener('durationchange', onMetadata);

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
     *
     * `t` is PLAYER time — every caller passes `video.currentTime` — so the same lead
     * correction the drawing uses is applied here. Without it a frame saved on a browser
     * that plays the untrimmed head would be filed with the boxes belonging to a moment
     * several seconds away, which is far worse than a misdrawn overlay: it puts wrong
     * annotations into the training set.
     */
    boxesAt(t) {
      return sampleAt(doc, t - lead)?.boxes || [];
    },
    /** Seconds of untrimmed head this browser is playing; 0 when it honoured the trim. */
    leadSeconds() {
      return lead;
    },
    destroy() {
      observer.disconnect();
      video.removeEventListener('timeupdate', redraw);
      video.removeEventListener('seeked', redraw);
      video.removeEventListener('loadedmetadata', onMetadata);
      video.removeEventListener('durationchange', onMetadata);
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
