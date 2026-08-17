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
import { fetchTaskLivePreview } from '/app-static/js/shared/api/tasks.js';
import { saveFrameForAnnotation, captureFilename, DEFAULT_CAPTURE_DIR } from '/app-static/js/ui/utils/saveFrameForAnnotation.js';
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
 * @returns {{close: () => void}}
 */
export function openWatcherLiveView({ taskUuid, profileName = '' }) {
  const shell = createModalShell({ onClose: () => stop() });
  shell.dialog.classList.add('watch-live-dialog');
  shell.dialog.dataset.testid = 'watch-live-dialog';

  const state = {
    sourceUuid: null,
    hz: DEFAULT_HZ,
    paused: false,
    show: { detections: true, tracks: true, zones: true, labels: true },
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

  bar.appendChild(checkbox('Detections', true, v => { state.show.detections = v; draw(); }, 'watch-live-show-detections'));
  bar.appendChild(checkbox('Tracks', true, v => { state.show.tracks = v; draw(); }, 'watch-live-show-tracks'));
  bar.appendChild(checkbox('Zones', true, v => { state.show.zones = v; draw(); }, 'watch-live-show-zones'));
  bar.appendChild(checkbox('Labels', true, v => { state.show.labels = v; draw(); }, 'watch-live-show-labels'));

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
  const saveBtn = button('Save frame for annotation', () => void saveFrame(), { testid: 'watch-live-save' });
  footer.appendChild(saveBtn);
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
    else state.sourceUuid = sourceSelect.value || null;
  }

  // ---------------------------------------------------------------- drawing

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

    if (state.show.zones) {
      for (const zone of preview.zones ?? []) drawZone(zone, width, height);
    }
    if (state.show.detections) {
      for (const det of preview.detections ?? []) {
        drawBox(det.bbox, boxColour(det, DETECTION_COLOUR), state.show.labels
          ? `${labelText(det.labelName, det.labelUuid, 'det')} ${(det.confidence ?? 0).toFixed(2)}`
          : null, { dashed: true });
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
        drawBox(track.bbox, colour, state.show.labels ? caption : null, {
          emphasis: active,
          glow: active || track.isAnchor ? status : null,
        });
      }
    }
  }

  /** Binds this view's canvas to the shared box renderer. */
  function drawBox(bbox, colour, label, opts = {}) {
    drawOverlayBox(ctx, bbox, colour, label, opts);
  }

  function drawZone(zone, width, height) {
    const points = (zone.points ?? []).map(([x, y]) => [x * width, y * height]);
    if (points.length < 2) return;
    ctx.save();
    ctx.strokeStyle = ZONE_COLOUR;
    ctx.lineWidth = 2;
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
      ctx.font = '12px sans-serif';
      ctx.fillText(zone.name, points[0][0] + 3, points[0][1] + 13);
    }
    ctx.restore();
  }

  // ---------------------------------------------------------------- side panel

  function renderSide(preview) {
    const age = preview.frameAgeS;
    const stale = age != null && preview.analysisFps && age > Math.max(5, 5 / preview.analysisFps);
    statusLine.classList.toggle('watch-live-status--error', Boolean(preview.lastError) || stale);
    statusLine.textContent = preview.lastError
      ? `Source error: ${preview.lastError}`
      : stale
        ? `No new frame for ${age}s — the source may be down.`
        : preview.dryRun
          ? 'Watching (dry run — no actions will fire).'
          : 'Watching.';

    // Counts sit next to each other because the *comparison* is the diagnosis:
    // detections without tracks, or tracks without events, each point somewhere
    // different. See the module docstring.
    const rows = [
      ['Detections (this frame)', (preview.detections ?? []).length],
      ['Tracks', (preview.tracks ?? []).length],
      ['Open events', (preview.openEvents ?? []).length],
      ['Frames analysed', preview.framesAnalysed ?? 0],
      ['Frame age (s)', age == null ? '—' : age.toFixed(2)],
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

  async function saveFrame() {
    if (!state.frameBlob) {
      toast('No frame to save yet', 3000, 'error');
      return;
    }
    const preview = state.preview ?? {};
    saveBtn.disabled = true;
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
            bbox: d.bbox,
            confidence: d.confidence,
          }))
          : [],
        frameWidth: preview.frameWidth,
        frameHeight: preview.frameHeight,
        extra: {
          source_watcher: preview.profileName || '',
          source_camera: preview.sourceName || '',
        },
      });
      savedHint.textContent = `Saved ${result.path}${result.boxesSaved ? ` (${result.boxesSaved} boxes)` : ''}`;
      toast(`Frame saved for annotation${result.boxesSaved ? ` with ${result.boxesSaved} boxes` : ''}`, 4000);
    } catch (err) {
      Logger.error(err);
    } finally {
      saveBtn.disabled = false;
    }
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
  }

  void poll({ immediate: true });

  return { close: () => shell.close() };
}
