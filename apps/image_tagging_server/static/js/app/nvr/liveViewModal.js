/**
 * Watcher live view — the drill-down behind an NVR Live card.
 *
 * This is a **diagnostic** surface first and a pretty picture second. A watcher can
 * look perfectly healthy from its status card — frames analysed climbing, no errors —
 * while detecting nothing at all, and the card cannot tell you which link in the chain
 * is broken. So this view separates the links visually:
 *
 *   frame → detections (what the models returned) → tracks (what survived the tracker)
 *         → zones/open events (what the rules made of them)
 *
 * Detections and tracks are drawn distinguishably — dashed vs solid — and counted
 * separately on purpose. Both are stroked in their *label's* colour, so the overlay
 * answers "what is it?" and "which stage is it from?" at the same time.
 * "12 detections, 0 tracks" is a tracker problem; "0 detections" is a model,
 * threshold or frame problem; "tracks but no open events" is a rule problem. Collapsing
 * them into one overlay would hide exactly the distinction the view exists to make.
 *
 * Cost control (three layers, all needed for a 24/7 app):
 *  - the backend encodes a preview only when asked, so a watcher nobody has open is free;
 *  - `sinceSeq` means a poll faster than the camera transfers no image bytes;
 *  - polling stops entirely while the tab is hidden or the view is paused.
 */
import { createModalShell, Logger, toast } from '/app-static/js/ui/utils/index.js';
import { fetchTaskLivePreview, sendTaskCommand } from '/app-static/js/shared/api/tasks.js';
import { saveFrameForAnnotation, captureFilename, DEFAULT_CAPTURE_DIR } from '/app-static/js/ui/utils/saveFrameForAnnotation.js';
import { fetchLabels } from '/app-static/js/shared/api/labels.js';
import { events, EventTypes } from '/app-static/js/shared/events/index.js';
// Shared with the Live card thumbnails: the same box must mean the same thing in the
// overview and in this drill-down. See overlay.js.
import {
  DETECTION_COLOUR,
  ZONE_COLOUR,
  boxColour,
  decodeFrame,
  drawBox as drawOverlayBox,
  isEventTrack,
  labelText,
  trackColour,
} from '/app-static/js/app/nvr/overlay.js';

/** Poll rates offered, in Hz. */
const POLL_RATES = [
  { label: '0.5 /s', hz: 0.5 },
  { label: '1 /s', hz: 1 },
  { label: '2 /s', hz: 2 },
  { label: '5 /s', hz: 5 },
];
const DEFAULT_HZ = 2;

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text != null) node.textContent = text;
  return node;
}

function button(text, onClick, { testid = null, className = 'panel-btn' } = {}) {
  const btn = document.createElement('button');
  btn.type = 'button';
  btn.className = className;
  btn.textContent = text;
  if (testid) btn.dataset.testid = testid;
  btn.addEventListener('click', onClick);
  return btn;
}

function checkbox(labelText, checked, onChange, testid) {
  const wrap = el('label', 'watch-live-toggle');
  const input = document.createElement('input');
  input.type = 'checkbox';
  input.checked = checked;
  if (testid) input.dataset.testid = testid;
  input.addEventListener('change', () => onChange(input.checked));
  wrap.appendChild(input);
  wrap.appendChild(document.createTextNode(` ${labelText}`));
  return wrap;
}

/**
 * Open the live view for one running watcher.
 *
 * @param {Object} opts
 * @param {string} opts.taskUuid - the running StreamWatcherTask instance
 * @param {string} [opts.profileName]
 * @param {?string} [opts.sourceUuid] - open on this camera rather than the watcher's
 *   first. Set when the view is opened by clicking a card thumbnail: the picture the
 *   user clicked is the one they expect to land on.
 * @returns {{close: () => void}}
 */
export function openWatcherLiveView({ taskUuid, profileName = '', sourceUuid = null }) {
  const shell = createModalShell({ onClose: () => stop() });
  shell.dialog.classList.add('watch-live-dialog');
  shell.dialog.dataset.testid = 'watch-live-dialog';

  const state = {
    // Seeded before the first poll, so the requested camera is what gets fetched —
    // not the watcher's first source followed by a visible switch.
    sourceUuid: sourceUuid || null,
    hz: DEFAULT_HZ,
    paused: false,
    show: { detections: true, tracks: true, zones: true, labels: true },
    /** Serve the camera's resolution rather than the frame inference ran on. */
    fullRes: false,
    /** `fullRes` before fullscreen turned it on, restored on the way back out. */
    fullResBeforeFullscreen: null,
    saveWithBoxes: true,
    /** Last preview payload, whatever it contained. */
    preview: null,
    /** Sequence number of the frame currently drawn — the key to not re-fetching it. */
    frameSeq: null,
    /** The drawn frame's bytes, kept for "save frame". */
    frameBlob: null,
    image: null,
    timer: null,
    inFlight: false,
    stopped: false,
  };

  // ---------------------------------------------------------------- scaffold

  shell.dialog.appendChild(el('h3', null, `Live — ${profileName || 'watcher'}`));

  const bar = el('div', 'watch-live-bar');
  shell.dialog.appendChild(bar);

  const sourceSelect = document.createElement('select');
  sourceSelect.className = 'watch-live-source';
  sourceSelect.dataset.testid = 'watch-live-source';
  sourceSelect.addEventListener('change', () => {
    state.sourceUuid = sourceSelect.value || null;
    // A different camera's frame numbering is unrelated to this one's, so the
    // conditional fetch has to start over or the first frame would be skipped.
    state.frameSeq = null;
    state.image = null;
    void poll({ immediate: true });
  });
  bar.appendChild(sourceSelect);

  const rateSelect = document.createElement('select');
  rateSelect.className = 'watch-live-rate';
  rateSelect.dataset.testid = 'watch-live-rate';
  for (const rate of POLL_RATES) {
    const opt = document.createElement('option');
    opt.value = String(rate.hz);
    opt.textContent = rate.label;
    opt.selected = rate.hz === DEFAULT_HZ;
    rateSelect.appendChild(opt);
  }
  rateSelect.addEventListener('change', () => { state.hz = Number(rateSelect.value) || DEFAULT_HZ; });
  bar.appendChild(rateSelect);

  const pauseBtn = button('Pause', () => {
    state.paused = !state.paused;
    pauseBtn.textContent = state.paused ? 'Resume' : 'Pause';
    if (!state.paused) void poll({ immediate: true });
  }, { testid: 'watch-live-pause' });
  bar.appendChild(pauseBtn);

  // Only some sources have anything to go live *from*. A Ring camera idles on a cloud
  // snapshot every 30–90 s and can be woken onto real video on demand; an RTSP camera
  // is already live, and a button offering to make it live again would be noise. The
  // provider says which it is (`supportsGoLive`), so this stays hidden until a poll
  // has answered — never guessed from the source's name or type.
  const goLiveBtn = button('Go live', async () => {
    if (!state.sourceUuid) return;
    goLiveBtn.disabled = true;
    try {
      await sendTaskCommand({ taskUuid, command: 'request_live', params: { sourceUuid: state.sourceUuid } });
      // Deliberately not "now live": the request is a wake-up published to the camera,
      // and the stream takes several seconds to come up. Promising more than that is
      // how the button reads as broken.
      toast('Asked the camera for live video — it takes a few seconds to come up.', 4000);
    } catch (err) {
      Logger.error(err);
      toast(err.message || 'Could not request live video', 5000, 'error');
    } finally {
      goLiveBtn.disabled = false;
    }
  }, { testid: 'watch-live-go-live' });
  goLiveBtn.title = 'Wake this camera onto its live stream instead of waiting for the next snapshot.';
  goLiveBtn.hidden = true;
  bar.appendChild(goLiveBtn);

  const browseEventsBtn = button('Browse events', () => {
    if (!state.sourceUuid) return;
    // Closed first: left open, it sits on top of the Events tab we are about to
    // switch to, which reads as the navigation having failed.
    shell.close();
    events.publish(EventTypes.OPEN_NVR_EVENTS_FOR_SOURCE, { sourceUuid: state.sourceUuid });
  }, { testid: 'watch-live-browse-events' });
  browseEventsBtn.title = "Switch to this camera's recorded events.";
  bar.appendChild(browseEventsBtn);

  bar.appendChild(checkbox('Detections', true, v => { state.show.detections = v; draw(); }, 'watch-live-show-detections'));
  bar.appendChild(checkbox('Tracks', true, v => { state.show.tracks = v; draw(); }, 'watch-live-show-tracks'));
  bar.appendChild(checkbox('Zones', true, v => { state.show.zones = v; draw(); }, 'watch-live-show-zones'));
  bar.appendChild(checkbox('Labels', true, v => { state.show.labels = v; draw(); }, 'watch-live-show-labels'));

  // Off by default: the diagnostic reading of this view wants the frame the models were
  // actually shown, and a full-resolution frame costs a real encode and transfer on
  // every poll. It is opt-in, and fullscreen opts in for you (see below) because at
  // that size the downscaled frame is what people complain about.
  const fullResToggle = checkbox('Full res', false, v => {
    state.fullResBeforeFullscreen = null;
    setFullRes(v);
  }, 'watch-live-full-res');
  fullResToggle.title = 'Show the camera’s own resolution instead of the downscaled frame '
    + 'inference runs on. Boxes are still the model’s, scaled up to match.';
  bar.appendChild(fullResToggle);
  const fullResInput = fullResToggle.querySelector('input');

  /** Switch resolutions and re-fetch: the new pixels carry the frame seq we already hold. */
  function setFullRes(on) {
    if (state.fullRes === on) return;
    state.fullRes = on;
    if (fullResInput) fullResInput.checked = on;
    // Same frame, different pixels — without this the conditional fetch would answer
    // `frameUnchanged` and the picture would stay at the resolution we just left.
    state.frameSeq = null;
    void poll({ immediate: true });
  }

  const body = el('div', 'watch-live-body');
  shell.dialog.appendChild(body);

  const stage = el('div', 'watch-live-stage');
  stage.dataset.testid = 'watch-live-stage';
  body.appendChild(stage);

  const canvas = document.createElement('canvas');
  canvas.width = 640;
  canvas.height = 360;
  canvas.dataset.testid = 'watch-live-canvas';
  stage.appendChild(canvas);
  const ctx = canvas.getContext('2d');

  // ---------------------------------------------------------- fullscreen
  //
  // Fullscreen the *stage*, not the canvas and not the dialog. The canvas alone would
  // very nearly do — frame and boxes are drawn into the same one, so the overlay comes
  // along for free — but the stage carries the black letterbox background, and a bare
  // fullscreen canvas sits on whatever the UA paints behind it.

  function toggleFullscreen() {
    if (document.fullscreenElement) {
      document.exitFullscreen?.();
    } else {
      // Not awaited and not assumed to succeed: a browser can refuse (no user gesture
      // in some embeddings, or a policy block), and `fullscreenchange` is what actually
      // decides the label — so a refusal simply leaves the button as it was.
      stage.requestFullscreen?.().catch(() => {});
    }
  }

  // Over the picture's bottom-right corner rather than in the toolbar above: that is
  // where every other video player puts it, and so where people look — in the toolbar
  // it was only findable by reading the whole row.
  // Not `panel-btn`: this one sits ON the picture and is styled for legibility over a
  // moving frame rather than to match the toolbar it no longer lives in.
  const fullscreenBtn = button('⛶', toggleFullscreen, {
    testid: 'watch-live-fullscreen',
    className: 'watch-live-fullscreen',
  });
  fullscreenBtn.title = 'Fullscreen (or double-click the picture). Box overlays are drawn '
    + 'into the same canvas as the picture, so they scale with it.';
  fullscreenBtn.setAttribute('aria-label', 'Fullscreen');
  stage.appendChild(fullscreenBtn);

  // The other half of "where people look": double-clicking the video is the near
  // universal gesture for this, and costs nothing here — the stage has no other click
  // handling to disturb.
  stage.addEventListener('dblclick', (e) => {
    // The button is inside the stage, so a double-click ON it would toggle twice —
    // once from its own click handler, once from this — and land back where it started.
    if (e.target.closest('.watch-live-fullscreen')) return;
    toggleFullscreen();
  });

  function _syncFullscreenLabel() {
    const on = document.fullscreenElement === stage;
    // The glyph does not change — being fullscreen is self-evident from the screen, and
    // a second icon for it is one more thing to decode. Only the wording moves.
    fullscreenBtn.title = on
      ? 'Exit fullscreen (or double-click the picture, or press Esc).'
      : 'Fullscreen (or double-click the picture). Box overlays are drawn into the same '
        + 'canvas as the picture, so they scale with it.';
    fullscreenBtn.setAttribute('aria-label', on ? 'Exit fullscreen' : 'Fullscreen');

    // Fullscreen is the one place the analysis-resolution frame is indefensible: a
    // 640px picture blown up to a whole screen is mush. So going fullscreen turns full
    // res on, and leaving puts it back exactly as the user had it — a user who ticked
    // the box themselves keeps it ticked on the way out (that choice cleared the memo).
    if (on) {
      if (state.fullResBeforeFullscreen === null) state.fullResBeforeFullscreen = state.fullRes;
      setFullRes(true);
    } else if (state.fullResBeforeFullscreen !== null) {
      const restore = state.fullResBeforeFullscreen;
      state.fullResBeforeFullscreen = null;
      setFullRes(restore);
    }

    // The canvas is laid out by CSS, not by JS, but the *drawing* is sized from the
    // element's client box in a couple of places — redraw so the first fullscreen frame
    // is not the pre-fullscreen one stretched.
    draw();
  }
  document.addEventListener('fullscreenchange', _syncFullscreenLabel);

  const side = el('div', 'watch-live-side');
  side.dataset.testid = 'watch-live-side';
  body.appendChild(side);

  const statusLine = el('div', 'watch-live-status');
  statusLine.dataset.testid = 'watch-live-status';
  side.appendChild(statusLine);

  const statsBox = el('div', 'watch-live-stats');
  statsBox.dataset.testid = 'watch-live-stats';
  side.appendChild(statsBox);

  const listBox = el('div', 'watch-live-lists');
  listBox.dataset.testid = 'watch-live-lists';
  side.appendChild(listBox);

  const footer = el('div', 'watch-live-footer');
  shell.dialog.appendChild(footer);

  footer.appendChild(checkbox('with detected boxes', true, v => { state.saveWithBoxes = v; }, 'watch-live-save-boxes'));
  const saveBtn = button('Save For Annotation', () => void saveFrame(), { testid: 'watch-live-save' });
  footer.appendChild(saveBtn);
  const annotateBtn = button('Annotate Image', () => void saveFrame({ openCanvas: true }), { testid: 'watch-live-annotate' });
  footer.appendChild(annotateBtn);
  const savedHint = el('span', 'panel-hint');
  savedHint.dataset.testid = 'watch-live-saved-hint';
  footer.appendChild(savedHint);
  footer.appendChild(button('Close', () => shell.close(), { testid: 'watch-live-close' }));

  shell.open();

  // ---------------------------------------------------------------- polling

  function scheduleNext() {
    if (state.stopped) return;
    clearTimeout(state.timer);
    state.timer = setTimeout(() => void poll(), Math.max(100, 1000 / state.hz));
  }

  async function poll({ immediate = false } = {}) {
    if (state.stopped) return;
    if (state.paused && !immediate) return;
    // A chained timeout rather than an interval, plus this guard: a slow response
    // must not queue a second request behind it and turn a stall into a pile-up.
    if (state.inFlight) { scheduleNext(); return; }
    if (document.hidden && !immediate) { scheduleNext(); return; }

    state.inFlight = true;
    try {
      const preview = await fetchTaskLivePreview(taskUuid, {
        sourceUuid: state.sourceUuid,
        sinceSeq: state.frameSeq,
        full: state.fullRes,
      });
      if (state.stopped) return;
      if (!preview) {
        // The watcher stopped, or was never a watcher. Say so and stop polling a
        // task that will never answer differently.
        statusLine.textContent = 'This watcher is no longer running.';
        statusLine.classList.add('watch-live-status--error');
        state.paused = true;
        pauseBtn.textContent = 'Resume';
        return;
      }
      apply(preview);
    } catch (err) {
      Logger.warn('nvr: live preview poll failed', err);
      statusLine.textContent = 'Could not reach the watcher — retrying.';
      statusLine.classList.add('watch-live-status--error');
    } finally {
      state.inFlight = false;
      scheduleNext();
    }
  }

  async function apply(preview) {
    state.preview = preview;
    syncSourceOptions(preview.sources ?? []);

    if (preview.frame) {
      const { blob, decode } = decodeFrame(preview.frame);
      state.frameBlob = blob;
      try {
        state.image = await decode();
      } catch (err) {
        Logger.warn('nvr: could not decode preview frame', err);
        state.image = null;
      }
      state.frameSeq = preview.frameSeq ?? null;
    } else if (preview.frameUnchanged) {
      // Metadata moved on; the pixels did not. Nothing to decode.
      state.frameSeq = preview.frameSeq ?? state.frameSeq;
    } else {
      state.image = null;
      state.frameBlob = null;
      state.frameSeq = null;
    }

    draw();
    renderSide(preview);
  }

  function syncSourceOptions(sources) {
    const wanted = sources.map(s => s.sourceUuid).join('|');
    if (sourceSelect.dataset.sources === wanted) return;
    sourceSelect.dataset.sources = wanted;
    sourceSelect.innerHTML = '';
    for (const src of sources) {
      const opt = document.createElement('option');
      opt.value = src.sourceUuid;
      opt.textContent = src.sourceName || src.sourceUuid;
      sourceSelect.appendChild(opt);
    }
    if (state.sourceUuid) sourceSelect.value = state.sourceUuid;
    // Either nothing was requested, or what was requested is not one of this watcher's
    // sources — a caller passing a stale uuid must fall back to a real camera rather
    // than leaving the select blank and polling for a source that does not exist.
    if (!sourceSelect.value) sourceSelect.selectedIndex = sources.length ? 0 : -1;
    state.sourceUuid = sourceSelect.value || null;
  }

  // ---------------------------------------------------------------- drawing

  /**
   * Pixels of served frame per pixel of analysed frame.
   *
   * 1 unless a full-resolution frame is being shown: detections and tracks are always
   * reported in the *analysis* coordinate space, whatever picture the backend served,
   * so drawing them onto a full-res frame unscaled would cram every box into the
   * top-left corner. Zones are normalised and need none of this.
   */
  function overlayScale(preview) {
    const served = preview?.frameWidth;
    const analysed = preview?.analysisFrameWidth;
    if (!served || !analysed) return 1;
    return served / analysed;
  }

  const scaleBox = (bbox, k) => (k === 1 ? bbox : bbox.map(v => v * k));

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
    if (state.image) {
      ctx.drawImage(state.image, 0, 0, width, height);
    } else {
      ctx.fillStyle = '#111';
      ctx.fillRect(0, 0, width, height);
      ctx.fillStyle = '#888';
      ctx.font = '14px sans-serif';
      ctx.fillText('Waiting for a frame…', 12, 24);
    }
    if (!preview) return;

    const k = overlayScale(preview);
    // Strokes and captions are in canvas pixels, so a 4x bigger canvas needs 4x the
    // line weight and text to look the same on screen — otherwise full res trades a
    // sharp picture for hairline boxes and unreadable labels.
    const pen = { lineWidth: 2 * k, fontPx: 12 * k };

    if (state.show.zones) {
      for (const zone of preview.zones ?? []) drawZone(zone, width, height, pen);
    }
    if (state.show.detections) {
      for (const det of preview.detections ?? []) {
        drawBox(scaleBox(det.bbox, k), boxColour(det, DETECTION_COLOUR), state.show.labels
          ? `${labelText(det.labelName, det.labelUuid, 'det')} ${(det.confidence ?? 0).toFixed(2)}`
          : null, { dashed: true, ...pen });
      }
    }
    if (state.show.tracks) {
      for (const track of preview.tracks ?? []) {
        const rules = track.eventRules ?? [];
        const active = isEventTrack(track);
        // Stroke by label so the box says *what* it is; the halo keeps saying whether the
        // watcher is reacting to it (see overlay.js).
        const status = trackColour(track);
        const colour = boxColour(track, status);
        const caption = [
          `#${track.trackId}`,
          labelText(track.labelName, track.labelUuid, ''),
          track.confidence != null ? track.confidence.toFixed(2) : '',
          // The rule name, not just "active": with several rules on one profile,
          // knowing *which* one latched onto this object is the whole diagnosis.
          active ? `● ${rules.join(', ')}` : '',
        ].filter(Boolean).join(' ');
        drawBox(scaleBox(track.bbox, k), colour, state.show.labels ? caption : null, {
          emphasis: active,
          glow: active || track.isAnchor ? status : null,
          ...pen,
        });
      }
    }
  }

  /** Binds this view's canvas to the shared box renderer. */
  function drawBox(bbox, colour, label, opts = {}) {
    drawOverlayBox(ctx, bbox, colour, label, opts);
  }

  function drawZone(zone, width, height, { lineWidth = 2, fontPx = 12 } = {}) {
    const points = (zone.points ?? []).map(([x, y]) => [x * width, y * height]);
    if (points.length < 2) return;
    ctx.save();
    ctx.strokeStyle = ZONE_COLOUR;
    ctx.lineWidth = lineWidth;
    ctx.setLineDash([2, 3]);
    if (zone.kind === 'box') {
      const [[x1, y1], [x2, y2]] = points;
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
    } else {
      ctx.beginPath();
      ctx.moveTo(points[0][0], points[0][1]);
      for (const [x, y] of points.slice(1)) ctx.lineTo(x, y);
      ctx.closePath();
      ctx.stroke();
    }
    if (state.show.labels && zone.name) {
      ctx.setLineDash([]);
      ctx.fillStyle = ZONE_COLOUR;
      ctx.font = `${fontPx}px sans-serif`;
      ctx.fillText(zone.name, points[0][0] + 3, points[0][1] + fontPx + 1);
    }
    ctx.restore();
  }

  // ---------------------------------------------------------------- side panel

  function renderSide(preview) {
    const age = preview.frameAgeS;
    goLiveBtn.hidden = !preview.supportsGoLive;
    // A slow source is not a broken one. The old rule derived "late" from
    // `analysisFps`, which is how often the *watcher asks* — so an idle Ring camera,
    // 30 s into a snapshot cadence its own provider puts at 90 s, was reported as
    // "the source may be down" every single time anyone opened this view.
    //
    // `frameDueInS` is the provider's own deadline for the next frame, so while it is
    // still counting down the source is on schedule by definition. The fps guess is
    // kept only for providers with no opinion, which is what it was written for.
    const dueIn = preview.frameDueInS;
    const overdue = dueIn != null
      ? dueIn <= 0
      : age != null && preview.analysisFps && age > Math.max(5, 5 / preview.analysisFps);
    // The watcher's own health beats both: it decides a source is failing using the
    // same provider deadline, and it has seen every read, not just this poll's.
    const failing = preview.sourceState === 'failed' || preview.sourceState === 'reconnecting';
    const stale = overdue || failing;
    statusLine.classList.toggle('watch-live-status--error', Boolean(preview.lastError) || stale);
    const dueSuffix = dueIn != null && dueIn > 0 ? ` Next frame due in ${Math.round(dueIn)}s.` : '';
    statusLine.textContent = preview.lastError
      ? `Source error: ${preview.lastError}`
      : stale
        ? `No new frame for ${age}s — the source may be down.`
        : preview.sourceState === 'connecting'
          ? `Connecting to the source…${dueSuffix}`
          : preview.dryRun
            ? `Watching (dry run — no actions will fire).${dueSuffix}`
            : `Watching.${dueSuffix}`;

    // Counts sit next to each other because the *comparison* is the diagnosis:
    // detections without tracks, or tracks without events, each point somewhere
    // different. See the module docstring.
    const rows = [
      ['Detections (this frame)', (preview.detections ?? []).length],
      ['Tracks', (preview.tracks ?? []).length],
      ['Open events', (preview.openEvents ?? []).length],
      ['Frames analysed', preview.framesAnalysed ?? 0],
      ['Frame age (s)', age == null ? '—' : age.toFixed(2)],
      // Beside the age on purpose: the pair is the diagnosis. "40s old, due in 50s" is
      // a healthy slow camera; "40s old, due in 0s" is one that has stopped.
      ['Next frame due in (s)', dueIn == null ? '—' : dueIn.toFixed(1)],
      ['Analysis', preview.frameWidth ? `${preview.frameWidth}×${preview.frameHeight} @ ${preview.analysisFps} fps` : '—'],
      ['Models', (preview.modelNames ?? []).join(', ') || '— none configured —'],
      ['Buffer', preview.buffer ?? '—'],
    ];
    statsBox.innerHTML = '';
    for (const [label, value] of rows) {
      const row = el('div', 'watch-live-stat');
      row.appendChild(el('span', 'watch-live-stat__label', label));
      row.appendChild(el('span', 'watch-live-stat__value', String(value)));
      statsBox.appendChild(row);
    }

    listBox.innerHTML = '';
    listBox.appendChild(_list('Detections (this frame)', (preview.detections ?? []).map(d => ({
      swatch: d.labelColor,
      text: `${labelText(d.labelName, d.labelUuid, 'det')} · ${(d.confidence ?? 0).toFixed(2)}`,
    })), 'No detections in this frame.'));
    listBox.appendChild(_list('Tracks', (preview.tracks ?? []).map(t => ({
      swatch: t.labelColor,
      active: (t.eventRules ?? []).length > 0,
      text: [
        `#${t.trackId}`,
        labelText(t.labelName, t.labelUuid),
        (t.confidence ?? 0).toFixed(2),
        t.isAnchor ? 'anchor' : '',
        // Fixed decimals, not the raw number: JSON drops a trailing zero, so an age of
        // 0.10 arrives as "0.1" and the row is one character narrower for that frame.
        `${Number(t.ageS ?? 0).toFixed(2)}s`,
        (t.eventRules ?? []).length ? `● ${t.eventRules.join(', ')}` : '',
      ].filter(Boolean).join(' · '),
    })), 'No tracks. If detections above is non-zero, the tracker is discarding them — check its confidence threshold.'));
    listBox.appendChild(_list('Open events', (preview.openEvents ?? []).map(e => ({
      active: true,
      // The track ids tie a rule back to the highlighted boxes on the canvas; without
      // them an open event and a red box are two facts the user has to guess relate.
      text: `${e.ruleName} · ${e.state}${(e.trackIds ?? []).length ? ` · tracks ${e.trackIds.map(id => `#${id}`).join(' ')}` : ''}`,
    })), 'No open events.'));
  }

  /**
   * @param {string} title
   * @param {Array<{text: string, swatch?: string|null, active?: boolean}>} items
   * @param {string} emptyText
   */
  function _list(title, items, emptyText) {
    const box = el('div', 'watch-live-list');
    box.appendChild(el('strong', null, title));
    if (!items.length) {
      box.appendChild(el('div', 'panel-hint', emptyText));
      return box;
    }
    for (const item of items) {
      const row = el('div', 'watch-live-list__row');
      if (item.active) row.classList.add('watch-live-list__row--active');
      if (item.swatch) {
        const dot = el('span', 'watch-live-list__swatch');
        dot.style.background = item.swatch;
        row.appendChild(dot);
      }
      row.appendChild(el('span', 'watch-live-list__text', item.text));
      box.appendChild(row);
    }
    return box;
  }

  // ---------------------------------------------------------------- save

  async function saveFrame({ openCanvas = false } = {}) {
    if (!state.frameBlob) {
      toast('No frame to save yet', 3000, 'error');
      return;
    }
    const preview = state.preview ?? {};
    // The blob is whatever resolution was served, and `frameWidth`/`Height` below
    // describe that same picture — so the boxes have to be moved into its space too,
    // or a frame saved in full res gets its annotations in the top-left quarter.
    const k = overlayScale(preview);
    saveBtn.disabled = true;
    annotateBtn.disabled = true;
    try {
      const result = await saveFrameForAnnotation({
        blob: state.frameBlob,
        filename: captureFilename(preview.sourceName || 'source'),
        path: DEFAULT_CAPTURE_DIR,
        // Detections rather than tracks: a track's box can be a latched anchor or an
        // extrapolation from an earlier frame, and annotating a box that is not on
        // *this* frame is worse than annotating no box at all.
        boxes: state.saveWithBoxes
          ? (preview.detections ?? []).map(d => ({
            labelUuid: d.labelUuid,
            bbox: scaleBox(d.bbox, k),
            confidence: d.confidence,
          }))
          : [],
        frameWidth: preview.frameWidth,
        frameHeight: preview.frameHeight,
        // The camera actually requested, not something read back off `preview` — the
        // live-preview response describes the frame, not which source it was polled for.
        sourceUuid: state.sourceUuid || '',
        extra: {
          source_watcher: preview.profileName || '',
          source_camera: preview.sourceName || '',
        },
      });
      savedHint.textContent = `Saved ${result.path}${result.boxesSaved ? ` (${result.boxesSaved} boxes)` : ''}`;
      if (openCanvas) {
        await openOnCanvas(result);
      } else {
        toast(`Frame saved for annotation${result.boxesSaved ? ` with ${result.boxesSaved} boxes` : ''}`, 4000);
        saveBtn.disabled = false;
        annotateBtn.disabled = false;
      }
    } catch (err) {
      Logger.error(err);
      saveBtn.disabled = false;
      annotateBtn.disabled = false;
    }
  }

  /**
   * Opens a self-contained canvas modal in place, on top of the live view (which it
   * closes) — mirrors the event-detail drill-down's "Annotate Image" so a captured
   * frame is indistinguishable from one captured there.
   *
   * Boxes are not passed in directly — `showCanvasModal` loads the image by filename
   * and reads back whatever `saveFrameForAnnotation` just persisted, so the detections
   * saved above (tagged unverified) show up pre-placed, ready to confirm or correct.
   *
   * @param {{path: string, uuid: string, boxesSaved: number}} result
   */
  async function openOnCanvas(result) {
    if (!result?.path) return;
    // Closed first: the canvas is behind this modal, and the live view keeps polling
    // frames the user is no longer looking at until it is.
    shell.close();
    const [{ showCanvasModal }, labels] = await Promise.all([
      import('/app-static/js/app/review/main.js'),
      fetchLabels().catch(() => new Map()),
    ]);
    const img = { uuid: result.uuid, filename: result.path, bboxes: [] };
    const navContext = { items: [], startIndex: 0 };
    showCanvasModal(null, img, labels, async () => {}, navContext, { allowNavigation: false });
  }

  // ---------------------------------------------------------------- lifecycle

  function onVisibility() {
    // Resume promptly when the user comes back rather than waiting out a poll
    // interval on a tab they are now looking at.
    if (!document.hidden && !state.paused) void poll({ immediate: true });
  }
  document.addEventListener('visibilitychange', onVisibility);

  function stop() {
    state.stopped = true;
    clearTimeout(state.timer);
    document.removeEventListener('visibilitychange', onVisibility);
    document.removeEventListener('fullscreenchange', _syncFullscreenLabel);
    // Removing the stage would drop fullscreen anyway, but not before the modal has
    // already torn down behind it — the user gets a frame of fullscreen nothing. Ask
    // for the exit first so the two happen in the order they look like they happen.
    if (document.fullscreenElement === stage) document.exitFullscreen?.();
  }

  void poll({ immediate: true });

  return { close: () => shell.close() };
}
