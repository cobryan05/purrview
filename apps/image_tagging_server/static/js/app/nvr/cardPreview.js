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

/**
 * Frame age past which a tile stops being treated as live.
 *
 * Comfortably above one refresh interval, so an ordinary poll landing between two
 * frames of a slow camera never trips it, and well below the two minutes the backend
 * waits before calling a source failed — the picture should say something is wrong
 * before the task log does.
 */
const STALE_AFTER_S = 15;

/** Source states that mean "the picture on screen is not what the camera sees now". */
const DEAD_STATES = new Set(['failed', 'reconnecting']);

/** Coarse elapsed time. Mirrors `_format_duration` in `_watch_pipeline.py`. */
function formatAge(seconds) {
  const total = Math.max(0, Math.round(seconds));
  if (total < 60) return `${total}s`;
  const minutes = Math.floor(total / 60);
  if (minutes < 60) return `${minutes}m`;
  const hours = Math.floor(minutes / 60);
  const rem = minutes % 60;
  return rem ? `${hours}h ${rem}m` : `${hours}h`;
}

/**
 * Classify a source from the status table's `health` string.
 *
 * The counterpart to `livenessOf`, for the sources that have no tile. Only the first
 * `MAX_TILES` sources are polled, so on a six-camera profile the last two are invisible
 * to the preview strip entirely — and "invisible" must not mean "assumed fine". The
 * status payload already carries a health label for *every* source at no extra cost.
 *
 * Matches the labels built by `WatchPipeline._health_label` in `_watch_pipeline.py`.
 *
 * @param {?string} label
 * @returns {'ok'|'connecting'|'dead'}
 */
export function levelFromHealthLabel(label) {
  const text = String(label ?? '').toLowerCase();
  if (text.startsWith('connecting')) return 'connecting';
  if (text.startsWith('failed') || text.startsWith('reconnecting')) return 'dead';
  return 'ok';
}

/**
 * Classify one source's liveness from a preview payload.
 *
 * Three outcomes rather than a boolean, because they want different treatment: a
 * camera coming up is expected and should not shout, a camera that has stopped
 * delivering is a fault, and everything else is fine.
 *
 * @returns {{level: 'ok'|'connecting'|'dead', ageS: ?number, label: string}}
 */
export function livenessOf(preview) {
  const state = preview?.sourceState;
  // `frameAgeS` is null until the very first frame, and "never delivered" is exactly
  // the case a card must not render as healthy — the downtime stands in for the age.
  const ageS = preview?.frameAgeS ?? preview?.downtimeS ?? null;
  const age = ageS == null ? null : formatAge(ageS);

  if (state === 'connecting') {
    return { level: 'connecting', ageS, label: age ? `connecting ${age}` : 'connecting' };
  }
  // The state is authoritative when the watcher offers one; the age is the fallback
  // for a payload that predates it, and the backstop for a source the watcher still
  // believes is healthy while its frames have visibly stopped.
  const dead = DEAD_STATES.has(state) || (ageS != null && ageS > STALE_AFTER_S);
  if (!dead) return { level: 'ok', ageS, label: '' };
  return { level: 'dead', ageS, label: age ? `no signal · ${age}` : 'no signal' };
}

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
 * `onHealth` exists because the tiles are the only thing that learns a source has gone
 * dark, and the card — which is what someone actually scans down a wall of profiles —
 * had no way to know. It is called with one entry per polled source whenever any of
 * them changes level.
 *
 * @param {HTMLElement} host - element the strip is appended to
 * @param {{taskUuid: string, sources: Array<{sourceUuid: string, sourceName?: string}>,
 *          refreshMs?: number, maxTiles?: number,
 *          onHealth?: (states: Array<{sourceUuid: string, sourceName: string,
 *                                     level: string, label: string}>) => void}} opts
 * @returns {() => void} stop function (also registered for `stopAllCardPreviews`)
 */
export function attachCardPreviews(host, {
  taskUuid,
  sources = [],
  refreshMs = REFRESH_MS,
  maxTiles = MAX_TILES,
  onHealth = null,
}) {
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

  /** Latest level per source, so the card can be told without re-polling anything. */
  const health = new Map();

  function reportHealth(source, liveness) {
    if (!onHealth) return;
    const previous = health.get(source.sourceUuid);
    // Only on a change of *level*: the label ticks every poll ("58m" → "59m") and a
    // card that re-rendered its bar every ten seconds would fight the user's cursor.
    const changed = !previous || previous.level !== liveness.level;
    health.set(source.sourceUuid, {
      sourceUuid: source.sourceUuid,
      sourceName: source.sourceName || source.sourceUuid || 'source',
      level: liveness.level,
      label: liveness.label,
    });
    if (changed && !stopped) onHealth([...health.values()]);
  }

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

    /**
     * Stamp a dead or connecting source's frame so it cannot be read as live.
     *
     * The frame is kept rather than blanked — the last thing the camera saw is often
     * the most useful thing on the pane, and a black rectangle throws it away to say
     * something a caption already says. What it must not do is *look current*, so it
     * is dimmed and struck with a band naming the state and how long it has held.
     * Drawn into the canvas rather than layered in CSS so it survives a screenshot,
     * which is how these cards usually get reported.
     */
    function drawLivenessOverlay(liveness, width, height) {
      const dead = liveness.level === 'dead';
      // Darkened rather than hidden: enough to kill the "this is live" read at a
      // glance, not so much that the scene is lost.
      ctx.fillStyle = dead ? 'rgba(0, 0, 0, 0.55)' : 'rgba(0, 0, 0, 0.35)';
      ctx.fillRect(0, 0, width, height);

      const shown = canvas.clientWidth || width;
      const scale = width / shown;
      const bandH = Math.max(16, Math.round(18 * scale));
      const y = Math.round(height / 2 - bandH / 2);
      ctx.fillStyle = dead ? 'rgba(153, 27, 27, 0.85)' : 'rgba(120, 83, 8, 0.85)';
      ctx.fillRect(0, y, width, bandH);

      ctx.fillStyle = '#fff';
      ctx.font = `${Math.max(9, Math.round(11 * scale))}px sans-serif`;
      ctx.textBaseline = 'middle';
      const text = (dead ? 'NO SIGNAL' : 'CONNECTING') + (liveness.ageS != null ? ` · ${formatAge(liveness.ageS)}` : '');
      ctx.fillText(text, Math.round(6 * scale), y + bandH / 2);
      ctx.textBaseline = 'alphabetic';
    }

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

      const liveness = livenessOf(preview);
      // The boxes belong to the frozen frame, and drawing them over a dead camera
      // asserts the watcher is tracking something it stopped seeing an hour ago.
      if (liveness.level !== 'ok') {
        drawLivenessOverlay(liveness, width, height);
        tile.classList.toggle('nvr-preview-tile--stale', liveness.level === 'dead');
        tile.classList.toggle('nvr-preview-tile--connecting', liveness.level === 'connecting');
        tile.classList.remove('nvr-preview-tile--event');
        return;
      }
      tile.classList.remove('nvr-preview-tile--stale', 'nvr-preview-tile--connecting');

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
      const liveness = livenessOf(preview);

      // Duration first, error second. `lastError` used to short-circuit to a bare
      // "source error", which is how a camera that had been down for an hour read
      // identically to one that missed a single frame — the duration is the whole
      // difference between "it blinked" and "nobody has looked at this since Tuesday",
      // and it was the one thing not on screen.
      statEl.classList.toggle('nvr-preview-tile__stat--error', liveness.level === 'dead');
      statEl.classList.toggle('nvr-preview-tile__stat--warn', liveness.level === 'connecting');
      if (liveness.level !== 'ok') {
        statEl.textContent = liveness.label;
        // The reason belongs somewhere, just not in a caption sized for "3 tracks".
        tile.title = [liveness.label, preview.lastError].filter(Boolean).join(' — ');
        return;
      }
      tile.title = '';
      statEl.textContent = [
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
      reportHealth(source, livenessOf(preview));
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
