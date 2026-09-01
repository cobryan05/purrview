/**
 * Annotated thumbnails on the NVR Live cards.
 *
 * The Live pane's job is "is the house being watched?", and a card that answers it in
 * text alone ("running · 3 rules · 1 fps") is indistinguishable from a card whose camera
 * has been pointing at a wall since Tuesday. One frame with the boxes drawn on it answers
 * the question the text cannot: *this* is what the watcher is looking at, and *these* are
 * the objects it is currently reacting to.
 *
 * Refresh rate is per card and chosen by the user (`REFRESH_RATES`,
 * `DEFAULT_REFRESH_MS`). This is still an overview of every watcher at once rather than
 * a video feed — the drill-down (`liveViewModal.js`) is where you go for real-time — so
 * the offered rates top out well below it, and three cost controls make a wall of
 * thumbnails affordable:
 *
 *  - one poll per source per interval, and each source's timer is staggered so eight
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

/**
 * Refresh rates offered per card, slowest first.
 *
 * Expressed in fps because that is the unit the rest of the pane already talks in (the
 * card's own meta line reports the watcher's analysis fps), even though these are
 * thumbnail polls rather than analysed frames.
 *
 * The ceiling is deliberate. Every step costs one server-side JPEG encode per source
 * per interval, and a card can hold four tiles: at 2 fps a four-camera profile asks for
 * eight encodes a second on its own, and a pane of them multiplies that. Anything
 * faster belongs in the drill-down, which polls one source at a time.
 */
export const REFRESH_RATES = [
  { label: '0.1 fps', ms: 10000 },
  { label: '0.2 fps', ms: 5000 },
  { label: '0.5 fps', ms: 2000 },
  { label: '1 fps', ms: 1000 },
  { label: '2 fps', ms: 500 },
];

/** How often one source's thumbnail is refreshed, until the card's dropdown says otherwise. */
export const DEFAULT_REFRESH_MS = 5000;

/**
 * Thumbnails shown per card at once — one page of the profile's sources.
 *
 * A profile can watch a dozen cameras; a card showing all twelve would be a wall of
 * postage stamps that answers nothing. Four fits two-up on a narrow card and still shows
 * a multi-source profile as the multi-source thing it is. The rest are a page away
 * rather than merely counted, and which ones land on page one is the user's to decide
 * (see `onReorder`).
 *
 * Nine, in a 3×3 grid whose columns are fixed by CSS. Both halves of that matter: a
 * page of four on a seven-camera profile meant every other page was a *different size*,
 * because the grid was `auto-fit` and three tiles shared the width four had — the
 * pictures changed scale as you paged, which is the one thing a wall of cameras must
 * not do. A fixed 3-up grid makes a partial page leave a gap instead of stretching, and
 * nine is what fits before the tiles stop being worth looking at.
 */
export const MAX_TILES = 9;

/** Lower than the drill-down's: this is a thumbnail scaled down by CSS anyway. */
const THUMB_QUALITY = 45;

/**
 * Frame age past which a tile stops being treated as live — *only* for a source whose
 * provider offers no deadline of its own.
 *
 * Comfortably above one refresh interval, so an ordinary poll landing between two
 * frames of a moderately slow camera never trips it, and well below the two minutes the
 * backend waits before calling a source failed — the picture should say something is
 * wrong before the task log does.
 *
 * It is a *fallback*, not the rule. A flat threshold reported a perfectly healthy Ring
 * camera as "no signal" for three quarters of every snapshot cycle, because 15 s is a
 * fault on an RTSP stream and a normal Tuesday on a 30 s snapshot cadence. When
 * `frameDueInS` is present it is the provider's own answer to "should a frame have
 * arrived by now?", and it wins.
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
  // `frameDueInS` is the provider's own deadline for the next frame, so while it is
  // still counting down the source is on schedule *by definition* and its age says
  // nothing — a 30 s-old frame from a 90 s snapshot camera is not a fault. The flat
  // age threshold is kept only for providers with no opinion. Same rule as the
  // drill-down's status line, which had it right first.
  const dueIn = preview?.frameDueInS ?? null;
  const overdue = dueIn != null ? dueIn <= 0 : ageS != null && ageS > STALE_AFTER_S;
  // The state is authoritative when the watcher offers one; the deadline is the
  // backstop for a source the watcher still believes is healthy while its frames have
  // visibly stopped.
  const dead = DEAD_STATES.has(state) || overdue;
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
 * A small button for the strip's own chrome (pager, arrange toggle).
 *
 * Not `panel-btn`: those are the card's controls over the *watcher*, and a paging
 * button that looked like "Stop" would be a mis-click waiting to happen.
 *
 * @returns {HTMLButtonElement}
 */
function _navButton(text, title, onClick) {
  const btn = document.createElement('button');
  btn.type = 'button';
  btn.className = 'nvr-preview-nav__btn';
  btn.textContent = text;
  btn.title = title;
  btn.setAttribute('aria-label', title);
  btn.addEventListener('click', onClick);
  return btn;
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
 * `onOpenSource` makes each tile the way in to the drill-down for *that* camera. The
 * thumbnail is the thing someone is actually looking at when they decide they want a
 * closer look, and before this the only route was a card-level button that opened the
 * modal on whichever source happened to be first in the list.
 *
 * `onReorder` is called with the full uuid order whenever the user moves a tile, so the
 * caller can persist it. The order is the caller's to store — this module has no opinion
 * on where — but it is the reason paging is worth having: on a nine-camera profile the
 * four that matter are rarely the four the watcher happens to list first.
 *
 * @param {HTMLElement} host - element the strip is appended to
 * @param {{taskUuid: string, sources: Array<{sourceUuid: string, sourceName?: string}>,
 *          refreshMs?: number, maxTiles?: number,
 *          onHealth?: (states: Array<{sourceUuid: string, sourceName: string,
 *                                     level: string, label: string}>) => void,
 *          onOpenSource?: (sourceUuid: string) => void,
 *          onReorder?: (sourceUuids: Array<string>) => void}} opts
 * @returns {{stop: () => void, setRefreshMs: (ms: number) => void}}
 *          `stop` is also registered for `stopAllCardPreviews`
 */
export function attachCardPreviews(host, {
  taskUuid,
  sources = [],
  refreshMs = DEFAULT_REFRESH_MS,
  maxTiles = MAX_TILES,
  onHealth = null,
  onOpenSource = null,
  onReorder = null,
}) {
  // A watcher whose sources are not known yet gets no strip at all, rather than an
  // empty bordered box that reads as a broken camera.
  if (!sources.length) return { stop: () => {}, setRefreshMs: () => {} };

  /** Mutable: the card's dropdown changes it without tearing the strip down. */
  let currentRefreshMs = refreshMs;

  /** The profile's sources in display order — mutated in place by `moveSource`. */
  const order = [...sources];

  /** Which page of `maxTiles` sources is on screen. */
  let page = 0;

  /** Whether the per-tile move buttons are showing. Deliberately not persisted. */
  let arranging = false;

  const strip = document.createElement('div');
  strip.className = 'nvr-preview-strip';
  strip.dataset.testid = 'nvr-preview-strip';
  host.appendChild(strip);

  const nav = document.createElement('div');
  nav.className = 'nvr-preview-nav';
  nav.dataset.testid = 'nvr-preview-nav';
  host.appendChild(nav);

  const timers = new Set();
  let stopped = false;

  function stop() {
    stopped = true;
    for (const tile of tiles) tile.stop();
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

  /**
   * Gap between two tiles' polls, in ms.
   *
   * A flat step stopped working when a page grew to nine tiles: at 2 fps the last tile
   * would be armed 2.25 s after the first, well past the moment the first was due
   * again, so the stagger meant to spread the load became a source of drift. Spread the
   * page across at most one interval instead.
   */
  function _staggerStep() {
    return Math.min(STAGGER_MS, Math.max(40, currentRefreshMs / maxTiles));
  }

  /** The tiles currently on screen — one page's worth, rebuilt by `renderPage`. */
  const tiles = [];

  /** Latest level per source, so the card can be told without re-polling anything. */
  const health = new Map();

  /**
   * Last frame seen per source uuid, kept across page changes.
   *
   * Paging back to a camera must not show "…" for a whole refresh interval — the
   * picture was on screen a second ago and throwing it away makes the pager feel like
   * it reloaded the card. Carrying `frameSeq` with it also means the first poll after a
   * page change usually transfers no image bytes at all. Bounded by the profile's
   * source count, which is the same set the strip would hold if nothing were paged.
   */
  const lastFrames = new Map();

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

    // A `figure` is not focusable and does not fire on Enter, so the affordance has to
    // be spelled out — otherwise the drill-down is reachable with a mouse only, on the
    // one control that is now the primary way into it.
    if (onOpenSource && source.sourceUuid) {
      tile.classList.add('nvr-preview-tile--clickable');
      tile.setAttribute('role', 'button');
      tile.tabIndex = 0;
      tile.setAttribute('aria-label', `Live view — ${source.sourceName || source.sourceUuid}`);
      tile.addEventListener('click', () => onOpenSource(source.sourceUuid));
      tile.addEventListener('keydown', (ev) => {
        if (ev.key !== 'Enter' && ev.key !== ' ') return;
        // Space scrolls the pane otherwise, which on a wall of cards throws away the
        // card the user was on.
        ev.preventDefault();
        onOpenSource(source.sourceUuid);
      });
    }

    // The canvas is sized to the camera's own frame, so left to itself a 4:3 camera
    // next to a 16:9 one makes two different-height tiles and the row stops being a
    // grid. The frame box is a fixed aspect ratio and the canvas is letterboxed inside
    // it, which keeps every tile on the page the same size whatever the cameras do.
    const frame = document.createElement('div');
    frame.className = 'nvr-preview-tile__frame';

    const canvas = document.createElement('canvas');
    // Placeholder geometry only — replaced by the frame's own dimensions on the first
    // poll, so the overlay never needs to scale coordinates.
    canvas.width = 320;
    canvas.height = 180;
    frame.appendChild(canvas);
    tile.appendChild(frame);
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

    // Move buttons rather than drag-and-drop: dragging is a mouse-only affordance with
    // no keyboard equivalent and a miserable touch story, and the whole interaction here
    // is "this camera matters more than that one", which two buttons say exactly. They
    // are shown only in arrange mode so an ordinary card stays a wall of pictures.
    if (arranging && order.length > 1) {
      const position = order.findIndex(s => s.sourceUuid === source.sourceUuid);
      const moves = document.createElement('div');
      moves.className = 'nvr-preview-tile__arrange';
      for (const [dir, glyph, word] of [[-1, '◀', 'earlier'], [1, '▶', 'later']]) {
        const btn = document.createElement('button');
        btn.type = 'button';
        btn.className = 'nvr-preview-move';
        btn.dataset.testid = dir < 0 ? 'nvr-preview-move-earlier' : 'nvr-preview-move-later';
        btn.dataset.dir = String(dir);
        btn.textContent = glyph;
        const name = source.sourceName || source.sourceUuid || 'source';
        btn.title = `Move ${name} ${word}`;
        btn.setAttribute('aria-label', btn.title);
        btn.disabled = dir < 0 ? position <= 0 : position >= order.length - 1;
        btn.addEventListener('click', (ev) => {
          // The tile itself opens the drill-down; a move must not also do that.
          ev.stopPropagation();
          moveSource(source.sourceUuid, dir);
        });
        // Enter/Space on the button would otherwise also reach the tile's own handler.
        btn.addEventListener('keydown', ev => ev.stopPropagation());
        moves.appendChild(btn);
      }
      tile.appendChild(moves);
    }

    strip.appendChild(tile);

    // Seeded from the cache so a page change repaints the last known picture at once,
    // and asks the server only for what changed since.
    const cached = lastFrames.get(source.sourceUuid) ?? null;
    const state = {
      frameSeq: cached?.frameSeq ?? null,
      image: cached?.image ?? null,
      inFlight: false,
      preview: cached?.preview ?? null,
      timer: null,
      // Per tile, so paging away stops *this* tile without stopping the strip.
      dead: false,
    };

    /**
     * How wide the frame is *on screen*, in CSS pixels.
     *
     * Not `canvas.clientWidth`: the canvas fills a fixed 16:9 box with `object-fit:
     * contain`, so for any camera that is not 16:9 the painted picture is narrower (or
     * shorter) than the element. Overlay text and line weights are drawn in frame
     * pixels and divided by this, so using the element's width would make captions on a
     * 4:3 camera come out visibly smaller than its neighbours'.
     */
    function shownWidth() {
      const boxW = canvas.clientWidth || canvas.width;
      const boxH = canvas.clientHeight || canvas.height;
      return Math.min(boxW, boxH * (canvas.width / canvas.height)) || boxW;
    }

    function placeholder(text) {
      ctx.fillStyle = '#111';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.fillStyle = '#888';
      ctx.font = '14px sans-serif';
      ctx.fillText(text, 10, 24);
    }

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

      const shown = shownWidth();
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
      const shown = shownWidth();
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
      // `frameUnchanged` payloads carry no pixels, so the merge keeps the metadata from
      // this poll and the image from the last one — the same thing the tile does.
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
      lastFrames.set(source.sourceUuid, {
        frameSeq: state.frameSeq,
        image: state.image,
        preview: state.preview,
      });
      if (stopped || state.dead) return;
      draw();
      renderCaption(preview);
      reportHealth(source, livenessOf(preview));
    }

    async function poll({ immediate = false } = {}) {
      if (stopped || state.dead) return;
      if (state.inFlight) { schedule(); return; }
      if (document.hidden && !immediate) { schedule(); return; }
      state.inFlight = true;
      try {
        const preview = await fetchTaskLivePreview(taskUuid, {
          sourceUuid: source.sourceUuid,
          sinceSeq: state.frameSeq,
          quality: THUMB_QUALITY,
        });
        if (stopped || state.dead || !preview) return;
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
      if (stopped || state.dead) return;
      armTimer(currentRefreshMs, () => void poll());
    }

    /**
     * Retire this tile without touching the rest of the strip.
     *
     * Paging is the whole reason this exists: the sources you paged away from must stop
     * costing the server a JPEG encode each interval, and a poll already in flight must
     * not draw into a canvas that is no longer in the document.
     */
    function stopTile() {
      state.dead = true;
      if (state.timer !== null) {
        clearTimeout(state.timer);
        timers.delete(state.timer);
        state.timer = null;
      }
    }

    /**
     * Poll soon, but not at the same instant as this card's other tiles: without the
     * stagger every tile encodes a JPEG on the same tick, which on a multi-camera
     * profile is a visible hitch.
     */
    function refreshSoon() {
      if (stopped || state.dead) return;
      armTimer(index * _staggerStep(), () => void poll({ immediate: true }));
    }

    // Repaint whatever this source last showed before waiting on the network, so a page
    // change is instant for a camera that has been seen once already.
    if (state.preview) {
      draw();
      renderCaption(state.preview);
    } else {
      placeholder('…');
    }
    refreshSoon();

    return { poll, refreshSoon, stop: stopTile };
  }

  const pageCount = () => Math.max(1, Math.ceil(order.length / maxTiles));

  /**
   * Rebuild the strip for the current page.
   *
   * Tiles are destroyed and recreated rather than retargeted: a tile owns a canvas
   * sized to one camera's frames, a poll chain keyed to one source uuid, and a caption
   * built from one name. Repointing all of that at a different camera is the same work
   * as building a new tile, with an extra way to leave a stale frame on screen. What
   * *is* preserved is the picture itself (`lastFrames`) and the health map, which are
   * the two things a rebuild would otherwise visibly lose.
   */
  function renderPage() {
    for (const tile of tiles) tile.stop();
    tiles.length = 0;
    strip.replaceChildren();
    page = Math.min(Math.max(0, page), pageCount() - 1);
    const start = page * maxTiles;
    for (const [i, source] of order.slice(start, start + maxTiles).entries()) {
      tiles.push(makeTile(source, i));
    }
    renderNav();
  }

  function goToPage(next) {
    const clamped = Math.min(Math.max(0, next), pageCount() - 1);
    if (clamped === page) return;
    page = clamped;
    renderPage();
  }

  /**
   * Move one source through the profile's order and tell the caller to persist it.
   *
   * The view follows the source rather than staying on the page: moving a camera off
   * the end of page one is how you promote it *to* page one from page two, and a pager
   * that stayed put would make the tile appear to vanish.
   */
  function moveSource(sourceUuid, delta) {
    const from = order.findIndex(s => s.sourceUuid === sourceUuid);
    const to = from + delta;
    if (from < 0 || to < 0 || to >= order.length) return;
    [order[from], order[to]] = [order[to], order[from]];
    page = Math.floor(to / maxTiles);
    renderPage();
    // Keyboard users move a camera several places at a time; without this the focus
    // lands back on <body> after the rebuild and the next press does nothing.
    const moved = strip.querySelector(
      `[data-source-uuid="${CSS.escape(sourceUuid)}"] .nvr-preview-move[data-dir="${delta}"]`,
    );
    if (moved && !moved.disabled) moved.focus();
    else strip.querySelector(`[data-source-uuid="${CSS.escape(sourceUuid)}"] .nvr-preview-move:not(:disabled)`)?.focus();
    onReorder?.(order.map(s => s.sourceUuid));
  }

  /**
   * The pager and the arrange toggle.
   *
   * Rebuilt with the page rather than mutated in place — it is three controls and a
   * label, and keeping "which buttons are disabled" in sync by hand is how a pager ends
   * up offering a next page that does not exist.
   */
  function renderNav() {
    nav.replaceChildren();
    // Nothing to page and nothing to arrange: a single-source card gets no chrome.
    if (order.length <= 1) {
      nav.hidden = true;
      return;
    }
    nav.hidden = false;

    if (order.length > maxTiles) {
      const prev = _navButton('‹', 'Previous sources', () => goToPage(page - 1));
      prev.dataset.testid = 'nvr-preview-prev';
      prev.disabled = page === 0;
      nav.appendChild(prev);

      const start = page * maxTiles;
      const end = Math.min(order.length, start + maxTiles);
      const label = document.createElement('span');
      label.className = 'nvr-preview-nav__label';
      label.dataset.testid = 'nvr-preview-page';
      // Which sources, not which page: "5–6 of 6" answers "have I seen them all?",
      // and "page 2 of 2" does not.
      label.textContent = `${start + 1}–${end} of ${order.length}`;
      nav.appendChild(label);

      const next = _navButton('›', 'More sources', () => goToPage(page + 1));
      next.dataset.testid = 'nvr-preview-next';
      next.disabled = end >= order.length;
      nav.appendChild(next);
    }

    const arrange = _navButton(arranging ? 'Done' : 'Arrange', 'Reorder these cameras', () => {
      arranging = !arranging;
      renderPage();
      // The button is rebuilt by renderNav, so the focus has to be put back by hand or
      // toggling arrange mode throws the keyboard user out of the card.
      nav.querySelector('[data-testid="nvr-preview-arrange"]')?.focus();
    });
    arrange.dataset.testid = 'nvr-preview-arrange';
    arrange.classList.add('nvr-preview-nav__arrange');
    arrange.setAttribute('aria-pressed', String(arranging));
    nav.appendChild(arrange);
  }

  renderPage();

  /**
   * Change the polling interval of every tile on this strip.
   *
   * `refreshSoon` rather than only re-arming: someone who picks a faster rate is
   * asking to see the picture sooner, and waiting out the *old* interval before the
   * new one takes effect makes the dropdown look broken. The stagger is kept so a
   * four-camera card still doesn't fire four encodes on one tick.
   */
  function setRefreshMs(ms) {
    const next = Number(ms);
    if (!Number.isFinite(next) || next <= 0 || next === currentRefreshMs) return;
    currentRefreshMs = next;
    if (stopped) return;
    for (const tile of tiles) tile.refreshSoon();
  }

  return { stop, setRefreshMs };
}
