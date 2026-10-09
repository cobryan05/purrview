/**
 * The control bar for a clip that carries a detection overlay.
 *
 * A browser-drawn `<video controls>` bar has a fullscreen button that fullscreens the
 * VIDEO element, and only the fullscreen element's own subtree is painted — the overlay
 * canvas is the video's *sibling*, so the picture fills the screen with no boxes on it.
 * Recovering by re-requesting fullscreen on the wrapper needs transient user activation,
 * which a `fullscreenchange` handler does not have: Gecko denies it outright.
 * `controlsList="nofullscreen"` cannot close the hole either — it is Chromium-only.
 *
 * So a clip with boxes is given these controls and no native ones, which leaves exactly
 * one route to fullscreen: the one that takes the whole wrapper, canvas included. A clip
 * with no sidecar keeps the native bar, where a video-only fullscreen loses nothing.
 */

/** Idle time before the bar fades out. Only ever while playing — see `_poke`. */
const HIDE_AFTER_MS = 2500;

/** Seconds the arrow keys jump. */
const ARROW_SEEK_S = 5;

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text != null) node.textContent = text;
  return node;
}

/** `m:ss`, or `—:——` before a duration is known. */
export function clockTime(seconds) {
  if (!Number.isFinite(seconds) || seconds < 0) return '-:--';
  const whole = Math.floor(seconds);
  const mins = Math.floor(whole / 60);
  return `${mins}:${String(whole - mins * 60).padStart(2, '0')}`;
}

/**
 * Replace a video's native controls with ones that live inside the overlay wrapper.
 *
 * @param {HTMLElement} wrapper - the `.clip-overlay` that will be fullscreened
 * @param {HTMLVideoElement} video
 * @param {Object} options
 * @param {() => void} options.onToggleFullscreen
 * @returns {{destroy: () => void, element: HTMLElement, togglePlay: () => void}}
 */
export function attachClipControls(wrapper, video, { onToggleFullscreen } = {}) {
  // The native bar is the thing being replaced; leaving it on would restore the very
  // fullscreen button this exists to remove.
  video.controls = false;

  const bar = el('div', 'clip-controls');
  bar.dataset.testid = 'clip-controls';

  const playBtn = el('button', 'clip-controls__btn', '▶');
  playBtn.type = 'button';
  playBtn.dataset.testid = 'clip-controls-play';
  playBtn.title = 'Play / pause (Space)';

  const seek = document.createElement('input');
  seek.type = 'range';
  seek.className = 'clip-controls__seek';
  seek.min = '0';
  seek.max = '1';
  seek.step = '0.01';
  seek.value = '0';
  seek.dataset.testid = 'clip-controls-seek';
  seek.setAttribute('aria-label', 'Seek');

  const time = el('span', 'clip-controls__time', '-:-- / -:--');
  time.dataset.testid = 'clip-controls-time';

  const muteBtn = el('button', 'clip-controls__btn', '🔊');
  muteBtn.type = 'button';
  muteBtn.dataset.testid = 'clip-controls-mute';
  muteBtn.title = 'Mute (M)';

  const fsBtn = el('button', 'clip-controls__btn', '⛶');
  fsBtn.type = 'button';
  fsBtn.dataset.testid = 'clip-controls-fullscreen';
  fsBtn.title = 'Fullscreen (F, or double-click the picture)';

  bar.append(playBtn, seek, time, muteBtn, fsBtn);
  wrapper.appendChild(bar);

  // Focusable so the key handling below has somewhere to land, and so Tab reaches the
  // player at all now that the native bar is gone.
  if (!wrapper.hasAttribute('tabindex')) wrapper.tabIndex = 0;

  // ------------------------------------------------------------------ state → bar

  /** True while the pointer is down on the scrubber, when the video drives nothing. */
  let scrubbing = false;

  function syncPlay() {
    const playing = !video.paused && !video.ended;
    playBtn.textContent = playing ? '⏸' : '▶';
    playBtn.setAttribute('aria-label', playing ? 'Pause' : 'Play');
    bar.dataset.playing = playing ? 'true' : 'false';
    _poke();
  }

  function syncTime() {
    const duration = Number.isFinite(video.duration) ? video.duration : 0;
    if (duration > 0) seek.max = String(duration);
    if (!scrubbing) seek.value = String(Math.min(video.currentTime || 0, duration || 0));
    time.textContent = `${clockTime(video.currentTime)} / ${clockTime(duration)}`;
  }

  function syncMute() {
    muteBtn.textContent = video.muted || video.volume === 0 ? '🔇' : '🔊';
    muteBtn.setAttribute('aria-label', video.muted ? 'Unmute' : 'Mute');
  }

  // ------------------------------------------------------------------ auto-hide

  let hideTimer = null;

  /**
   * Show the bar, and arm its retreat only if the clip is playing.
   *
   * A paused player keeps its bar: this is a review surface, and the bar covers the
   * bottom strip of the picture where boxes are drawn, so it has to get out of the way
   * on its own — but hiding it while someone is stopped on a frame studying it just
   * means they cannot see where they are.
   */
  function _poke() {
    bar.classList.remove('clip-controls--hidden');
    if (hideTimer) clearTimeout(hideTimer);
    hideTimer = null;
    if (video.paused || video.ended) return;
    hideTimer = setTimeout(() => {
      // Re-checked rather than trusted: the clip may have been paused inside the window,
      // and a bar that vanishes from a paused player is the case above.
      if (!video.paused && !video.ended && !bar.contains(document.activeElement)) {
        bar.classList.add('clip-controls--hidden');
      }
    }, HIDE_AFTER_MS);
  }

  // ------------------------------------------------------------------ interaction

  function togglePlay() {
    if (video.paused || video.ended) void video.play().catch(() => {});
    else video.pause();
  }

  playBtn.addEventListener('click', togglePlay);
  muteBtn.addEventListener('click', () => {
    video.muted = !video.muted;
    syncMute();
  });
  fsBtn.addEventListener('click', () => onToggleFullscreen?.());

  // `input` rather than `change`: scrubbing has to move the picture while the thumb is
  // being dragged, which is the whole point of a scrubber on a review surface.
  seek.addEventListener('pointerdown', () => { scrubbing = true; });
  seek.addEventListener('pointerup', () => { scrubbing = false; });
  seek.addEventListener('input', () => {
    scrubbing = true;
    video.currentTime = Number(seek.value);
    syncTime();
  });
  seek.addEventListener('change', () => {
    scrubbing = false;
    video.currentTime = Number(seek.value);
    syncTime();
  });

  const onKeyDown = (e) => {
    if (e.altKey || e.ctrlKey || e.metaKey) return;
    // The scrubber is a range input and owns the arrow keys itself.
    const onSeek = e.target === seek;
    let handled = true;
    switch (e.key) {
      case ' ':
      case 'k':
        togglePlay();
        break;
      case 'ArrowLeft':
        if (onSeek) handled = false;
        else video.currentTime = Math.max(0, (video.currentTime || 0) - ARROW_SEEK_S);
        break;
      case 'ArrowRight':
        if (onSeek) handled = false;
        else video.currentTime = Math.min(video.duration || 0, (video.currentTime || 0) + ARROW_SEEK_S);
        break;
      case 'm':
        video.muted = !video.muted;
        syncMute();
        break;
      case 'f':
        onToggleFullscreen?.();
        break;
      default:
        handled = false;
    }
    if (!handled) return;
    // Space scrolls the page and the modal binds single letters of its own; a key the
    // player has already acted on must not also reach either.
    e.preventDefault();
    e.stopPropagation();
    _poke();
  };

  const onPoke = () => _poke();

  wrapper.addEventListener('keydown', onKeyDown);
  wrapper.addEventListener('pointermove', onPoke);
  wrapper.addEventListener('pointerdown', onPoke);
  wrapper.addEventListener('focusin', onPoke);
  video.addEventListener('play', syncPlay);
  video.addEventListener('pause', syncPlay);
  video.addEventListener('ended', syncPlay);
  video.addEventListener('timeupdate', syncTime);
  video.addEventListener('seeked', syncTime);
  video.addEventListener('loadedmetadata', syncTime);
  video.addEventListener('durationchange', syncTime);
  video.addEventListener('volumechange', syncMute);

  syncPlay();
  syncTime();
  syncMute();

  return {
    element: bar,
    togglePlay,
    destroy() {
      if (hideTimer) clearTimeout(hideTimer);
      wrapper.removeEventListener('keydown', onKeyDown);
      wrapper.removeEventListener('pointermove', onPoke);
      wrapper.removeEventListener('pointerdown', onPoke);
      wrapper.removeEventListener('focusin', onPoke);
      video.removeEventListener('play', syncPlay);
      video.removeEventListener('pause', syncPlay);
      video.removeEventListener('ended', syncPlay);
      video.removeEventListener('timeupdate', syncTime);
      video.removeEventListener('seeked', syncTime);
      video.removeEventListener('loadedmetadata', syncTime);
      video.removeEventListener('durationchange', syncTime);
      video.removeEventListener('volumechange', syncMute);
      bar.remove();
      // Handed back to the browser: whatever replaces this overlay is on its own, and a
      // clip with no controls at all is worse than a bar with a flawed fullscreen button.
      video.controls = true;
    },
  };
}
