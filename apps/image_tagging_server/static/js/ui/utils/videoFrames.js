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
