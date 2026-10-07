/**
 * Knowing whether a `<video>` is really showing the moment you asked for.
 *
 * Its own module so the decision can be tested without a playable clip — the two
 * failures it exists to prevent both looked like success on screen.
 */

/** How far the playhead may sit from the requested moment and still count as landed. */
export const SEEK_TOLERANCE_S = 0.25;

/**
 * Resolve when the video next *presents* a frame.
 *
 * **Arm this before moving the playhead.** A paused video presents exactly one frame
 * per completed seek, so the first callback after arming is that frame — which is the
 * whole signal, and needs no comparison against the requested time. An earlier version
 * compared `metadata.mediaTime` to the seek target and waited for a match; on any clip
 * with an untrimmed head those two live in different timebases (see `untrimmedLead`),
 * so the match never came, the wait ran out, and the caller captured whatever was
 * still on screen.
 *
 * `presented: false` means the picture was never confirmed — a backgrounded tab, a
 * hidden pane and a media error all land here, because `requestVideoFrameCallback`
 * only runs for a video the browser is compositing. It is not a reason to capture
 * anyway; see `frameIsTrustworthy`.
 *
 * @param {HTMLVideoElement} video
 * @param {Object} [opts]
 * @param {number} [opts.timeoutMs]
 * @returns {Promise<{supported: boolean, presented: boolean}>} `supported: false` on
 *   browsers without rVFC (Firefox, older Safari), where `seeked` is all there is.
 */
export function nextPresentedFrame(video, { timeoutMs = 2500 } = {}) {
  if (typeof video?.requestVideoFrameCallback !== 'function') {
    return Promise.resolve({ supported: false, presented: false });
  }
  return new Promise(resolve => {
    let done = false;
    const finish = (presented) => {
      if (done) return;
      done = true;
      clearTimeout(timer);
      resolve({ supported: true, presented });
    };
    const timer = setTimeout(() => finish(false), timeoutMs);
    video.requestVideoFrameCallback(() => finish(true));
  });
}

/**
 * Whether the picture on screen can be trusted to be the moment that was asked for.
 *
 * The rule this encodes: **an unverified frame is never captured.** Saving one costs
 * more than saving nothing, because the boxes written beside it are the *corrected*
 * labels for a moment the picture does not show — a training image that is confidently
 * wrong, filed as ground truth, and indistinguishable from a good one in review.
 *
 * Where rVFC is unavailable there is nothing to verify against but the playhead, so a
 * landed seek is accepted: that is the browser's own account of where it is, and
 * refusing every capture on Firefox would be worse than the risk.
 *
 * @param {{supported: boolean, presented: boolean}} frameState - from nextPresentedFrame
 * @param {number} currentTime - the playhead after the seek settled
 * @param {number} wantedTime - where it was sent, in the same (player) timebase
 * @param {number} [tolerance]
 * @returns {boolean}
 */
export function frameIsTrustworthy(frameState, currentTime, wantedTime, tolerance = SEEK_TOLERANCE_S) {
  if (!Number.isFinite(currentTime) || !Number.isFinite(wantedTime)) return false;
  if (Math.abs(currentTime - wantedTime) > tolerance) return false;
  return frameState?.supported ? frameState.presented === true : true;
}

/**
 * Resolve once the page is visible, or when the wait runs out.
 *
 * Capture is impossible while the tab is hidden — nothing is composited, so no frame
 * is ever presented and every capture would be a stale duplicate. Waiting beats both
 * alternatives: skipping silently throws away the harvest of someone who tabbed away
 * for a moment, and capturing anyway produces exactly the wrong images this module
 * exists to prevent.
 *
 * @param {number} [timeoutMs]
 * @returns {Promise<boolean>} whether the page is visible now
 */
export function whenPageVisible(timeoutMs = 60000) {
  if (typeof document === 'undefined' || !document.hidden) return Promise.resolve(true);
  return new Promise(resolve => {
    let done = false;
    const finish = () => {
      if (done) return;
      done = true;
      clearTimeout(timer);
      document.removeEventListener('visibilitychange', onChange);
      resolve(!document.hidden);
    };
    const onChange = () => { if (!document.hidden) finish(); };
    const timer = setTimeout(finish, timeoutMs);
    document.addEventListener('visibilitychange', onChange);
  });
}

/**
 * Greatest luma spread a frame may have and still count as blank, over 0–255.
 *
 * A decoder's uninitialised buffer is *exactly* uniform — every pixel identical, spread
 * 0.00 — so anything above zero is a picture. The margin is this thin because the spread
 * is measured on a 32x32 downsample: averaging ~2000 source pixels per output pixel
 * suppresses sensor noise by ~45x, so what keeps a real night frame above the threshold
 * is scene structure, not grain. A genuinely featureless dark frame can read as blank.
 */
export const BLANK_FRAME_MAX_SPREAD = 1;

/**
 * Standard deviation of a frame's luma.
 *
 * The one property every real picture has and an uninitialised decode buffer never
 * does: variation. Luma rather than RGB because the buffer that reaches the canvas is
 * uniform in *YUV* — all-zero chroma and luma renders as a flat green through the
 * conversion, which has a wide channel spread and would read as content to anything
 * comparing colours.
 *
 * @param {{data: Uint8ClampedArray}} imageData - from `getImageData`
 * @returns {number} 0 for a perfectly uniform frame
 */
export function lumaSpread(imageData) {
  const data = imageData?.data;
  if (!data?.length) return 0;
  let sum = 0;
  let sumSquares = 0;
  let count = 0;
  for (let i = 0; i < data.length; i += 4) {
    const luma = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
    sum += luma;
    sumSquares += luma * luma;
    count += 1;
  }
  if (!count) return 0;
  const mean = sum / count;
  // Clamped at 0: the identity is exact in reals, and in floats a uniform frame can
  // land a hair below zero and turn into NaN under the root.
  return Math.sqrt(Math.max(0, sumSquares / count - mean * mean));
}

/**
 * Whether a captured frame holds a picture at all.
 *
 * The third way a capture can be wrong, after "never presented" and "seek never
 * landed": presented, on time, and blank. A video seeked on a lossy clip can composite
 * an uninitialised buffer — `requestVideoFrameCallback` fires for it like any other
 * frame, so `frameIsTrustworthy` says yes and nothing downstream can tell. Nine images
 * reached the dataset that way, each carrying the corrected boxes of a moment it did
 * not show.
 *
 * @param {{data: Uint8ClampedArray}} imageData
 * @param {number} [maxSpread]
 * @returns {boolean}
 */
export function frameHasPicture(imageData, maxSpread = BLANK_FRAME_MAX_SPREAD) {
  return lumaSpread(imageData) > maxSpread;
}

/**
 * Move the playhead a frame and resolve once the browser has shown the result.
 *
 * A paused `<video>` the reader has never played presents frame 0 once and then
 * nothing ever again, so a capture that reads that frame back blank has no second
 * composite to wait for: `nextPresentedFrame` runs out its timeout and the retry reads
 * the same empty buffer. A seek is what forces a fresh decode and a fresh composite,
 * which is why the batch harvest — which seeks to every moment it captures — never hits
 * this and a single-frame save does.
 *
 * One frame's step keeps the picture the moment the reader is looking at. The playhead
 * is left where it lands: the caller records the time it actually captured, and seeking
 * back would cost another composite for no gain.
 *
 * Waits on `seeked` as well as `requestVideoFrameCallback` because rVFC is the half
 * that does not exist on Firefox, where an unsupported `nextPresentedFrame` resolves
 * immediately and would leave nothing waited on at all.
 *
 * @param {HTMLVideoElement} video
 * @param {Object} [opts]
 * @param {number} [opts.timeoutMs]
 * @param {number} [opts.stepS]
 * @returns {Promise<boolean>} whether the playhead could be moved at all
 */
export async function stepToFreshFrame(video, { timeoutMs = 1500, stepS = 1 / 30 } = {}) {
  const from = video?.currentTime;
  if (!Number.isFinite(from)) return false;
  const duration = video.duration;
  // A clip whose duration is not known yet (an unfinished or fragmented mp4 reports
  // Infinity) still seeks perfectly well — it just cannot be clamped, so step forward
  // and let the browser refuse an out-of-range target itself. Declining here instead
  // would leave the caller with the wait that does not work on a paused clip.
  const bounded = Number.isFinite(duration) && duration > 0;
  const to = !bounded || from + stepS < duration - 0.01
    ? from + stepS
    : Math.max(0, from - stepS);
  if (!Number.isFinite(to) || Math.abs(to - from) < 1e-6) return false;
  // Both armed before the playhead moves: a completed seek presents exactly one frame,
  // and a listener added afterwards is waiting for something that already happened.
  const presented = nextPresentedFrame(video, { timeoutMs });
  const seeked = new Promise(resolve => {
    const done = () => {
      clearTimeout(timer);
      video.removeEventListener('seeked', done);
      resolve();
    };
    // Bounded: `seeked` never fires if the media errors, and an unguarded await would
    // hang with both save buttons disabled and nothing on screen to say why.
    const timer = setTimeout(done, timeoutMs);
    video.addEventListener('seeked', done, { once: true });
  });
  video.currentTime = to;
  await seeked;
  await presented;
  return true;
}


/**
 * What a capture would see right now, for an error message that names the cause.
 *
 * The capture refusal has four causes behind one sentence — no intrinsic size, a flat
 * frame, a readback the browser refuses, and a media error — and which one fired is the
 * whole diagnosis. Cheap enough to run on the failure path only: it draws the same
 * 32x32 downsample `captureVideoFrame` already measures, and never encodes a blob.
 *
 * `spread` is measured on a thumbnail drawn straight off the video; `fullSpread` on a
 * full-size draw, downsampled after the fact. The pair is the diagnosis whenever they
 * disagree — a healthy `spread` beside a flat `fullSpread` is a browser whose
 * full-resolution video blit paints nothing, which no single measurement can show.
 *
 * @param {HTMLVideoElement} video
 * @returns {{width: number, height: number, spread: ?number, fullSpread: ?number,
 *   readyState: number, currentTime: number, paused: boolean, error: ?string,
 *   readback: ?string}}
 */
export function frameDiagnostics(video) {
  const out = {
    width: video?.videoWidth ?? 0,
    height: video?.videoHeight ?? 0,
    spread: null,
    fullSpread: null,
    readyState: video?.readyState ?? -1,
    currentTime: video?.currentTime ?? -1,
    paused: video?.paused !== false,
    error: video?.error ? `code ${video.error.code}: ${video.error.message}` : null,
    readback: null,
  };
  if (!out.width || !out.height) return out;
  const sample = (draw, w, h) => {
    const canvas = document.createElement('canvas');
    canvas.width = Math.max(1, w);
    canvas.height = Math.max(1, h);
    const context = canvas.getContext('2d', { willReadFrequently: true });
    draw(context, canvas.width, canvas.height);
    return Number(lumaSpread(context.getImageData(0, 0, canvas.width, canvas.height)).toFixed(2));
  };
  try {
    out.spread = sample((c, w, h) => c.drawImage(video, 0, 0, w, h), Math.min(32, out.width), Math.min(32, out.height));
    const full = document.createElement('canvas');
    full.width = out.width;
    full.height = out.height;
    full.getContext('2d').drawImage(video, 0, 0, out.width, out.height);
    out.fullSpread = sample((c, w, h) => c.drawImage(full, 0, 0, w, h), 32, 32);
  } catch (e) {
    // A tainted canvas throws here and nowhere else, so this is the only place that can
    // tell "the picture is blank" from "the browser will not let us read it".
    out.readback = String(e);
  }
  return out;
}

/**
 * The diagnosis as one short clause, for a toast the reader can report back.
 *
 * `reason` is what the capture itself refused on; the measurements only describe the
 * state it refused in. Deriving the reason from the measurements instead reads the
 * wrong one whenever the two disagree — which is exactly the interesting case, since a
 * frame measuring a healthy luma spread and still refusing to encode is a fault in the
 * encoder, not in the picture.
 *
 * @param {Object} d - from frameDiagnostics
 * @param {?string} [reason] - 'no-size' | 'blank' | 'encode'
 */
export function describeCaptureFailure(d, reason = null) {
  if (d?.error) return `the video errored (${d.error})`;
  if (d?.readback) return `the browser refused to read the frame back (${d.readback})`;
  const size = `${d?.width}x${d?.height}`;
  const where = Number.isFinite(d?.currentTime) ? ` at ${d.currentTime.toFixed(2)}s` : '';
  if (reason === 'no-size' || !d?.width || !d?.height) {
    return `the video reports no size yet (readyState ${d?.readyState})`;
  }
  const spreads = `luma spread ${d.spread} off the video, ${d.fullSpread} off a ${size} canvas`;
  if (reason === 'encode') {
    return `this browser produced no image data from a ${size} canvas (${spreads}${where})`;
  }
  if (reason === 'blank') {
    // Worth separating: one of these is a frame with nothing in it, the other a browser
    // that will not paint a frame there plainly is something in.
    const painted = d.spread !== null && d.spread > BLANK_FRAME_MAX_SPREAD;
    return painted
      ? `this browser painted nothing into a ${size} canvas from a frame that has a `
        + `picture in it (${spreads}${where})`
      : `the frame is blank (${size}, ${spreads}${where}, `
        + `readyState ${d.readyState}${d.paused ? ', paused' : ''})`;
  }
  return `the capture was refused (${size}, ${spreads}${where})`;
}
