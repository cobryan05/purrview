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
