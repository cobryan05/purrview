/**
 * Annotated thumbnails on the NVR Live cards.
 *
 * The Live pane's job is "is the house being watched?", and a card that answers it in
 * text alone ("running · 3 rules · 1 fps") is indistinguishable from a card whose camera
 * has been pointing at a wall since Tuesday. One frame with the boxes drawn on it answers
 * the question the text cannot: *this* is what the watcher is looking at, and *these* are
 * the objects it is currently reacting to.
 *
 * Refresh rate is deliberately slow (`REFRESH_MS`). This is an overview of every watcher
 * at once, not a video feed — the drill-down (`liveViewModal.js`) is where you go for
 * real-time, and it polls at 0.5–5 Hz for one source at a time. Three cost controls make
 * a wall of thumbnails affordable:
 *
 *  - one poll per source per `REFRESH_MS`, and each source's timer is staggered so eight
 *    cameras don't all encode a JPEG on the same tick;
 *  - `sinceSeq`, so a poll that outruns a 1 fps camera transfers no image bytes;
 *  - nothing polls while the tab is hidden, and everything stops when the pane
 *    re-renders — a card that has scrolled out of the DOM must not keep a timer alive.
 */
import { fetchTaskLivePreview } from '/app-static/js/shared/api/tasks.js';
import { Logger } from '/app-static/js/ui/utils/index.js';
import {
  DETECTION_COLOUR,
  boxColour,
  decodeFrame,
  drawBox,
  isEventTrack,
  labelText,
  trackColour,
} from '/app-static/js/app/nvr/overlay.js';

/** How often one source's thumbnail is refreshed. An overview, not a video feed. */
export const REFRESH_MS = 10000;

/**
 * Thumbnails shown per card before the rest are collapsed into a "+N more" note.
 *
 * A profile can watch a dozen cameras; a card showing all twelve would be a wall of
 * postage stamps that answers nothing. Four fits two-up on a narrow card and still shows
 * a multi-source profile as the multi-source thing it is.
 */
export const MAX_TILES = 4;

/** Lower than the drill-down's: this is a thumbnail scaled down by CSS anyway. */
const THUMB_QUALITY = 45;

/** On-screen size of a box caption, in CSS pixels, whatever the camera's resolution. */
const CAPTION_PX = 10;

/** Spacing between one card's tiles when they all refresh at once. */
const STAGGER_MS = 250;

/** Every live poller on the pane, so a re-render can stop all of them. */
const _active = new Set();

/**
 * Stop every card preview on the pane.
 *
 * Called before the Live pane re-renders. Without it each render would leave its timers
 * running against DOM that no longer exists — the classic invisible leak, where the pane
 * gets slower every time you switch tabs and nothing looks wrong.
 */
export function stopAllCardPreviews() {
  for (const stop of [..._active]) stop();
  _active.clear();
}

/**
 * Attach a live thumbnail strip for one running watcher to `host`.
 *
 * `sources` comes from the caller — one batched status-data call covers every card on
 * the pane — rather than being discovered here. Asking the live-preview endpoint for
 * the source list would work, but that endpoint's job is to return a *frame*: it would
 * encode and ship a JPEG purely to learn some names, once per card. The status payload
 * already carries the sources and encodes nothing.
 *
 * The names must come from the watcher rather than the profile row, which carries
 * source uuids only — a tile captioned with a uuid is the problem this exists to fix.
 *
 * @param {HTMLElement} host - element the strip is appended to
 * @param {{taskUuid: string, sources: Array<{sourceUuid: string, sourceName?: string}>,
 *          refreshMs?: number, maxTiles?: number}} opts
 * @returns {() => void} stop function (also registered for `stopAllCardPreviews`)
 */
export function attachCardPreviews(host, { taskUuid, sources = [], refreshMs = REFRESH_MS, maxTiles = MAX_TILES }) {
  // A watcher whose sources are not known yet gets no strip at all, rather than an
  // empty bordered box that reads as a broken camera.
  if (!sources.length) return () => {};

  const strip = document.createElement('div');
  strip.className = 'nvr-preview-strip';
  strip.dataset.testid = 'nvr-preview-strip';
  host.appendChild(strip);

  const timers = new Set();
  let stopped = false;

  function stop() {
    stopped = true;
    for (const timer of timers) clearTimeout(timer);
    timers.clear();
    document.removeEventListener('visibilitychange', onVisibility);
    _active.delete(stop);
  }
  _active.add(stop);

  function onVisibility() {
    // Nothing to do on hide — the scheduled poll checks `document.hidden` itself and
    // reschedules. On show, the thumbnails on screen may be up to refreshMs stale, so
    // this is the one place an immediate refresh is worth its cost — staggered, because
    // coming back to the browser tab must not fire every camera's encode on one tick.
    if (document.hidden) return;
    for (const tile of tiles) tile.refreshSoon();
  }
  document.addEventListener('visibilitychange', onVisibility);

  const tiles = [];

  /**
   * One source's thumbnail: canvas, caption, and its own polling loop.
   */
  function makeTile(source, index) {
    const tile = document.createElement('figure');
    tile.className = 'nvr-preview-tile';
    tile.dataset.testid = 'nvr-preview-tile';
    tile.dataset.sourceUuid = source.sourceUuid ?? '';

    const canvas = document.createElement('canvas');
    // Placeholder geometry only — replaced by the frame's own dimensions on the first
    // poll, so the overlay never needs to scale coordinates.
    canvas.width = 320;
    canvas.height = 180;
    tile.appendChild(canvas);
    const ctx = canvas.getContext('2d');

    const caption = document.createElement('figcaption');
    caption.className = 'nvr-preview-tile__caption';

    const nameEl = document.createElement('span');
    nameEl.className = 'nvr-preview-tile__name';
    nameEl.textContent = source.sourceName || source.sourceUuid || 'source';
    caption.appendChild(nameEl);

    const statEl = document.createElement('span');
    statEl.className = 'nvr-preview-tile__stat';
    statEl.dataset.testid = 'nvr-preview-stat';
    statEl.textContent = '…';
    caption.appendChild(statEl);

    tile.appendChild(caption);
    strip.appendChild(tile);

    const state = { frameSeq: null, image: null, inFlight: false, preview: null, timer: null };

    function placeholder(text) {
      ctx.fillStyle = '#111';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.fillStyle = '#888';
      ctx.font = '14px sans-serif';
      ctx.fillText(text, 10, 24);
    }
    placeholder('…');

    function draw() {
      const preview = state.preview;
      if (!ctx) return;
      const width = preview?.frameWidth || canvas.width;
      const height = preview?.frameHeight || canvas.height;
      if (canvas.width !== width || canvas.height !== height) {
        canvas.width = width;
        canvas.height = height;
      }
      ctx.clearRect(0, 0, width, height);
      if (state.image) ctx.drawImage(state.image, 0, 0, width, height);
      else placeholder('No frame yet');
      if (!preview) return;

      // Everything below is drawn in frame pixels and then scaled by CSS, so a fixed
      // 12px caption is 12px only by accident: on a 320-wide frame in a 176px tile it
      // renders at 6px, and on a small analysis frame blown up to fill a wide card it
      // swamps the picture. Dividing by the *displayed* size makes both the caption and
      // the line weight land at the same apparent size whatever the camera's resolution.
      const shown = canvas.clientWidth || width;
      const scale = width / shown;
      const fontPx = Math.round(CAPTION_PX * scale);

      // Detections are drawn *unlabelled*: at thumbnail scale a caption on every raw
      // detection covers the picture, and the diagnostic detections-vs-tracks split is
      // the drill-down's job. Here they are texture — "the model is seeing things" —
      // while the tracks carry the names.
      for (const det of preview.detections ?? []) {
        drawBox(ctx, det.bbox, boxColour(det, DETECTION_COLOUR), null, { dashed: true, lineWidth: scale });
      }
      let eventCount = 0;
      for (const track of preview.tracks ?? []) {
        const active = isEventTrack(track);
        if (active) eventCount += 1;
        const name = labelText(track.labelName, track.labelUuid, '');
        const status = trackColour(track);
        drawBox(ctx, track.bbox, boxColour(track, status), name || null, {
          emphasis: active,
          // Only when the glow says something the stroke does not: a plain track's
          // status colour is the default, and haloing every box says nothing.
          glow: active || track.isAnchor ? status : null,
          lineWidth: 1.5 * scale,
          fontPx,
        });
      }
      // The border echoes the box colour, so a card with something happening on it is
      // visible while scrolling past, before any caption is read.
      tile.classList.toggle('nvr-preview-tile--event', eventCount > 0);
    }

    function renderCaption(preview) {
      const tracks = (preview.tracks ?? []).length;
      const dets = (preview.detections ?? []).length;
      const events = (preview.openEvents ?? []).length;
      const age = preview.frameAgeS;
      if (preview.lastError) {
        statEl.textContent = 'source error';
        statEl.classList.add('nvr-preview-tile__stat--error');
        return;
      }
      // A frozen frame is the failure this whole strip exists to expose, and a still
      // picture cannot show it — the age has to be spelled out.
      const stale = age != null && age > Math.max(10, refreshMs / 1000);
      statEl.classList.toggle('nvr-preview-tile__stat--error', stale);
      statEl.textContent = stale
        ? `no frame for ${Math.round(age)}s`
        : [
          events ? `${events} event${events === 1 ? '' : 's'}` : '',
          `${tracks} track${tracks === 1 ? '' : 's'}`,
          // Only when it adds information: detections and tracks agreeing is the
          // normal case and printing both every time is noise.
          dets !== tracks ? `${dets} det` : '',
        ].filter(Boolean).join(' · ');
    }

    async function apply(preview) {
      state.preview = preview;
      if (preview.frame) {
        try {
          state.image = await decodeFrame(preview.frame).decode();
        } catch (err) {
          Logger.warn('nvr: could not decode card preview frame', err);
          state.image = null;
        }
        state.frameSeq = preview.frameSeq ?? null;
      } else if (preview.frameUnchanged) {
        // The metadata moved on, the pixels did not — redraw the overlay over the
        // frame already held, and decode nothing.
        state.frameSeq = preview.frameSeq ?? state.frameSeq;
      }
      if (stopped) return;
      draw();
      renderCaption(preview);
    }

    async function poll({ immediate = false } = {}) {
      if (stopped) return;
      if (state.inFlight) { schedule(); return; }
      if (document.hidden && !immediate) { schedule(); return; }
      state.inFlight = true;
      try {
        const preview = await fetchTaskLivePreview(taskUuid, {
          sourceUuid: source.sourceUuid,
          sinceSeq: state.frameSeq,
          quality: THUMB_QUALITY,
        });
        if (stopped || !preview) return;
        await apply(preview);
      } catch (err) {
        Logger.warn('nvr: card preview poll failed', err);
      } finally {
        state.inFlight = false;
        schedule();
      }
    }

    /**
     * Arm this tile's single timer, replacing whatever was pending.
     *
     * The replacement is the point. Three places start a new chain — poll()'s
     * `finally`, its `inFlight` early return, and refreshSoon() on every
     * visibilitychange — and none of them used to cancel the previous one, so
     * each tile accumulated one independent chain per tab switch. After N
     * switches a tile polled, and made the server encode a JPEG, N+1 times per
     * interval, for as long as the pane stayed open.
     */
    function armTimer(delay, run) {
      if (state.timer !== null) {
        clearTimeout(state.timer);
        timers.delete(state.timer);
      }
      const timer = setTimeout(() => {
        timers.delete(timer);
        state.timer = null;
        run();
      }, delay);
      state.timer = timer;
      timers.add(timer);
    }

    function schedule() {
      if (stopped) return;
      armTimer(refreshMs, () => void poll());
    }

    /**
     * Poll soon, but not at the same instant as this card's other tiles: without the
     * stagger every tile encodes a JPEG on the same tick, which on a multi-camera
     * profile is a visible hitch.
     */
    function refreshSoon() {
      if (stopped) return;
      armTimer(index * STAGGER_MS, () => void poll({ immediate: true }));
    }

    refreshSoon();

    return { poll, refreshSoon };
  }

  for (const [i, source] of sources.slice(0, maxTiles).entries()) tiles.push(makeTile(source, i));
  if (sources.length > maxTiles) {
    const hidden = sources.length - maxTiles;
    const more = document.createElement('div');
    more.className = 'nvr-preview-more';
    more.textContent = `+${hidden} more source${hidden === 1 ? '' : 's'}`;
    strip.appendChild(more);
  }

  return stop;
}
