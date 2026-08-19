/**
 * The drill-down behind an event card: everything known about one event, plus its clip.
 *
 * The clip is the reason this modal exists. An event's snapshot is a single instant
 * chosen by the watcher; the clip is the *context* around it, and it deliberately
 * begins before the event does — pre-roll is cut retroactively out of MediaMTX's
 * rolling buffer at the moment the event closes, so the video shows the approach, not
 * just the arrival. That lead-in is only legible if the player says where in it the
 * rule actually fired, which is what the marker on the scrubber is for.
 *
 * Scrubbing works because `/api/recordings/{uuid}/file` is served with Range support
 * and the proxy forwards it. Without that the `<video>` would still play, and the
 * scrubber would still move, but seeking would do nothing — a failure that looks like
 * a working feature.
 *
 * "Save frame for annotation" closes the loop the NVR exists for: the frames worth
 * training on are the ones where something happened, and this is where you are already
 * looking at them. It shares `saveFrameForAnnotation` with the Live view, so a frame
 * captured here is indistinguishable from one captured there.
 *
 * When the clip has a detection sidecar, `clipOverlay.js` draws the watcher's boxes over
 * the video and marks the stretch where the rule was firing. The video underneath stays
 * the untouched clip — nothing is ever composited into it — which is what lets the
 * capture above stay a clean training image while still being able to seed it with the
 * boxes that were on screen.
 */
import { createModalShell, Logger, showConfirmModal, toast } from '/app-static/js/ui/utils/index.js';
import { events, EventTypes } from '/app-static/js/shared/events/index.js';
import {
  deleteWatchEvent,
  fetchRecording,
  recordingFileUrl,
  watchEventSnapshotUrl,
} from '/app-static/js/shared/api/watchProfiles.js';
import { saveFrameForAnnotation, captureFilename, DEFAULT_CAPTURE_DIR } from '/app-static/js/ui/utils/saveFrameForAnnotation.js';
import { attachClipOverlay, fetchClipOverlay } from './clipOverlay.js';
import { eventHeadline, findRecording, formatDuration, formatEventTime, parseEventDate } from './eventCard.js';

/**
 * How a timelapse describes itself, from the metadata its action recorded.
 *
 * Worth the words because the duration alone is actively misleading: a scrubber saying
 * "0:14" over footage of a two-hour visit invites the reader to think the cat left after
 * fourteen seconds. Returns null when the row predates the metadata or the action did
 * not report a span — an unexplained video is better than an invented explanation.
 */
export function describeTimelapse(recording) {
  const meta = recording?.metadataJson || {};
  const span = Number(meta.sourceSpanS);
  const frames = Number(meta.frames);
  if (!Number.isFinite(span) || span <= 0 || !Number.isFinite(frames) || frames <= 0) return null;
  const played = Number(recording.durationS) || 0;
  const spanText = span >= 3600
    ? `${(span / 3600).toFixed(1)} hours`
    : span >= 60 ? `${Math.round(span / 60)} minutes` : `${Math.round(span)} seconds`;
  const speed = played > 0 ? ` (${Math.round(span / played)}x)` : '';
  return `${frames} frames over ${spanText}, played in ${played.toFixed(1)}s${speed}.`;
}

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

function detailRow(label, value) {
  const row = el('div', 'event-detail__row');
  row.appendChild(el('span', 'event-detail__row-label', label));
  row.appendChild(el('span', 'event-detail__row-value', value));
  return row;
}

/**
 * Where in the clip the event began, in seconds from the clip's first frame.
 *
 * Both timestamps are needed and neither substitutes for the other: the clip's start is
 * the event's start minus whatever pre-roll the buffer actually still held, which the
 * relay clamps and reports rather than guaranteeing.
 *
 * @param {Object} event
 * @param {?Object} recording
 * @returns {?number} null when it cannot be computed, so the caller can omit the marker
 *   rather than draw one at zero and imply there was no lead-in
 */
export function eventOffsetInClip(event, recording) {
  const eventStart = parseEventDate(event?.startedAt);
  const clipStart = parseEventDate(recording?.startedAt);
  if (!eventStart || !clipStart) return null;
  const offset = (eventStart - clipStart) / 1000;
  if (!Number.isFinite(offset) || offset < 0) return null;
  const duration = recording?.durationS ?? 0;
  // A marker past the end of the video would sit off the scrubber; that means the two
  // timestamps disagree, and showing nothing is more honest than showing it clamped.
  if (duration && offset > duration) return null;
  return offset;
}

/**
 * Grab the video's current frame as a JPEG blob.
 *
 * Reads at the video's *intrinsic* size, not its displayed size: the point of saving a
 * frame is to annotate it, and a training image downscaled to whatever the modal
 * happened to be sized at is worth much less than the frame itself.
 *
 * @param {HTMLVideoElement} video
 * @returns {Promise<?{blob: Blob, width: number, height: number}>}
 */
export async function captureVideoFrame(video) {
  const width = video.videoWidth;
  const height = video.videoHeight;
  if (!width || !height) return null;
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  canvas.getContext('2d').drawImage(video, 0, 0, width, height);
  const blob = await new Promise(resolve => canvas.toBlob(resolve, 'image/jpeg', 0.92));
  return blob ? { blob, width, height } : null;
}

/**
 * Open the detail modal for one event.
 *
 * @param {Object} event - a WatchEventRead
 * @param {Object} [ctx]
 * @param {Map<string, string>} [ctx.sourceNames]
 * @param {() => void} [ctx.onDeleted] - so the grid can drop the card
 * @returns {{close: () => void}}
 */
export function openEventDetailModal(event, { sourceNames = new Map(), onDeleted } = {}) {
  // The overlay owns a ResizeObserver and a requestVideoFrameCallback loop, neither of
  // which stops when the DOM node goes away. Opening a dozen events in a row would
  // otherwise leave a dozen redraw loops running against detached videos.
  const shell = createModalShell({ onClose: () => { overlay?.destroy(); overlay = null; } });
  shell.dialog.classList.add('event-detail');
  shell.dialog.dataset.testid = 'event-detail-modal';

  const started = parseEventDate(event.startedAt);

  // ---------------------------------------------------------------- header

  const header = el('div', 'event-detail__header');
  const titles = el('div', 'event-detail__titles');
  titles.appendChild(el('h3', 'event-detail__title', eventHeadline(event)));
  titles.appendChild(el('span', 'event-detail__subtitle', formatEventTime(started)));
  header.appendChild(titles);
  header.appendChild(button('Close', () => shell.close(), { testid: 'event-detail-close' }));
  shell.dialog.appendChild(header);

  const body = el('div', 'event-detail__body');
  shell.dialog.appendChild(body);

  // ---------------------------------------------------------------- media

  const mediaPane = el('div', 'event-detail__media');
  body.appendChild(mediaPane);

  const sidebar = el('div', 'event-detail__side');
  body.appendChild(sidebar);

  let video = null;
  let overlay = null;
  let saveBtn = null;
  let mediaSwitcher = null;
  const savedHint = el('div', 'event-detail__hint');

  const clipRow = findRecording(event, 'clip');
  const timelapseRow = findRecording(event, 'timelapse');
  // The clip first when there is one: it is the literal record, and the timelapse is a
  // summary of it. An event with only a timelapse opens on that.
  const mediaRows = [clipRow, timelapseRow].filter(Boolean);

  /**
   * Fetch a recording's full row and show it, degrading to the snapshot.
   *
   * The list only carries a summary of each recording — enough to draw a badge and
   * label a tab. The file path and the overlay live on the full row, which is fetched
   * lazily so an event with two recordings costs one request, not two, until the user
   * asks for the second.
   */
  async function showMedia(row) {
    try {
      const recording = await fetchRecording(row.uuid);
      // The row can outlive its file, or vice versa; either way the snapshot is still
      // worth showing, so this degrades rather than emptying the modal.
      if (!recording?.filePath) {
        renderSnapshotOnly(row.kind === 'timelapse'
          ? 'The timelapse for this event is no longer on disk.'
          : 'The clip for this event is no longer on disk.');
        return;
      }
      renderVideo({ ...recording, kind: recording.kind || row.kind });
    } catch (err) {
      Logger.error(err);
      renderSnapshotOnly('That recording could not be loaded.');
    }
  }

  /**
   * Two buttons, and only when there are two things to switch between.
   *
   * Deliberately not tabs-always: an event with one recording is the overwhelmingly
   * common case, and a single-tab tab strip is chrome that explains nothing.
   */
  function buildMediaSwitcher() {
    if (mediaRows.length < 2) return null;
    const bar = el('div', 'event-detail__media-tabs');
    bar.dataset.testid = 'event-detail-media-tabs';
    bar.setAttribute('role', 'tablist');
    for (const row of mediaRows) {
      const isTimelapse = row.kind === 'timelapse';
      const tab = button(isTimelapse ? 'Timelapse' : 'Clip', () => {
        for (const other of bar.children) other.setAttribute('aria-selected', String(other === tab));
        void showMedia(row);
      }, { testid: `event-detail-media-tab-${row.kind}`, className: 'panel-btn event-detail__media-tab' });
      tab.setAttribute('role', 'tab');
      tab.setAttribute('aria-selected', String(row === mediaRows[0]));
      bar.appendChild(tab);
    }
    return bar;
  }

  function renderSnapshotOnly(message) {
    // Before the wipe: the overlay's loops outlive the nodes they draw on, and
    // `innerHTML = ''` does not tell it anything.
    overlay?.destroy();
    overlay = null;
    video = null;
    mediaPane.innerHTML = '';
    // Kept above the message even when this recording cannot be shown: one of an
    // event's two videos being off disk is no reason to strand the user on it with no
    // way back to the other.
    if (mediaSwitcher) mediaPane.appendChild(mediaSwitcher);
    if (event.hasSnapshot) {
      const img = document.createElement('img');
      img.className = 'event-detail__image';
      img.alt = '';
      img.src = watchEventSnapshotUrl(event.uuid);
      img.dataset.testid = 'event-detail-snapshot';
      mediaPane.appendChild(img);
    }
    if (message) mediaPane.appendChild(el('p', 'event-detail__no-video', message));
  }

  function renderVideo(recording) {
    // Same wipe as `renderSnapshotOnly`, and for the same reason: switching between an
    // event's clip and its timelapse replaces the <video> the overlay's animation loop
    // is drawing onto, and a loop left running against a detached node keeps a whole
    // clip's worth of boxes alive for as long as the modal is open.
    overlay?.destroy();
    overlay = null;
    mediaPane.innerHTML = '';
    if (mediaSwitcher) mediaPane.appendChild(mediaSwitcher);

    const isTimelapse = recording.kind === 'timelapse';

    video = document.createElement('video');
    video.className = 'event-detail__video';
    video.controls = true;
    // No autoplay: this is a review surface, and a grid of events opened one after
    // another that each start talking and moving is hostile.
    video.preload = 'metadata';
    video.src = recordingFileUrl(recording.uuid);
    video.dataset.testid = 'event-detail-video';
    mediaPane.appendChild(video);

    if (isTimelapse) {
      // A timelapse has no pre-roll to mark and no detection sidecar to lay over it —
      // its frames are seconds apart, so there is no continuous track to draw. What it
      // does need is to say what it compressed.
      const note = describeTimelapse(recording);
      if (note) {
        const line = el('p', 'event-detail__timelapse-note', note);
        line.dataset.testid = 'event-detail-timelapse-note';
        mediaPane.appendChild(line);
      }
      const tools = el('div', 'event-detail__tools');
      saveBtn = button('Save frame for annotation', () => void saveFrame(), { testid: 'event-detail-save-frame' });
      tools.appendChild(saveBtn);
      tools.appendChild(savedHint);
      mediaPane.appendChild(tools);
      return;
    }

    const offset = eventOffsetInClip(event, recording);
    if (offset != null) {
      const marker = el('div', 'event-detail__marker');
      marker.dataset.testid = 'event-detail-marker';
      const jump = button(`Jump to the moment it fired (${offset.toFixed(1)}s in)`, () => {
        video.currentTime = offset;
      }, { testid: 'event-detail-jump' });
      marker.appendChild(jump);
      marker.appendChild(el('span', 'event-detail__marker-note',
        `${offset.toFixed(1)}s of lead-in was cut from the rewind buffer before the event.`));
      mediaPane.appendChild(marker);
    }

    if (recording.partialPreroll) {
      // Never present a truncated clip as though its lead-in were intact — the buffer
      // is RAM and had already evicted part of what was asked for.
      const warn = el('p', 'event-detail__warn',
        'The rewind buffer had already dropped part of the requested lead-in, so this clip starts later than intended.');
      warn.dataset.testid = 'event-detail-partial';
      mediaPane.appendChild(warn);
    }

    const tools = el('div', 'event-detail__tools');
    saveBtn = button('Save frame for annotation', () => void saveFrame(), { testid: 'event-detail-save-frame' });
    tools.appendChild(saveBtn);
    tools.appendChild(savedHint);
    mediaPane.appendChild(tools);

    // Fire and forget: the clip is playable the moment the element exists, and a modal
    // that waited on a sidecar before showing video would feel slower for a feature that
    // most clips do not even have.
    void attachOverlay(recording, tools);
  }

  /**
   * Load the clip's detection sidecar and draw it over the video.
   *
   * The toggle only appears when there is something to toggle. Rendering a permanently
   * disabled checkbox on every pre-existing clip would advertise a feature that clip can
   * never have — the boxes were never recorded and cannot be recovered.
   */
  async function attachOverlay(recording, tools) {
    const doc = await fetchClipOverlay(recording.uuid);
    // The modal can be closed, or re-rendered, while this is in flight.
    if (!doc || !video || !video.isConnected) return;
    overlay = attachClipOverlay(video, doc);

    const label = el('label', 'event-detail__toggle');
    const box = document.createElement('input');
    box.type = 'checkbox';
    box.checked = true;
    box.dataset.testid = 'event-detail-overlay-toggle';
    box.addEventListener('change', () => overlay.setVisible(box.checked));
    label.appendChild(box);
    label.appendChild(el('span', null, 'Show what the watcher saw'));
    tools.appendChild(label);
  }

  // ---------------------------------------------------------------- save

  /**
   * The overlay's boxes at the captured instant, in the captured frame's pixel space.
   *
   * Scaled to `frame`, not to the analysis frame they were recorded against: the capture
   * is at the video's intrinsic resolution, and normalised coordinates are exactly what
   * makes that conversion a multiplication rather than a guess about downscale ratios.
   *
   * @param {{width: number, height: number}} frame
   * @returns {Object} spread into the save call — empty when there is no overlay
   */
  function overlayBoxes(frame) {
    if (!overlay || !video) return {};
    const boxes = overlay.boxesAt(video.currentTime || 0)
      .filter(b => b.labelUuid)
      .map(b => ({
        labelUuid: b.labelUuid,
        bbox: [
          b.x * frame.width,
          b.y * frame.height,
          (b.x + b.w) * frame.width,
          (b.y + b.h) * frame.height,
        ],
        confidence: b.conf,
      }));
    if (!boxes.length) return {};
    return { boxes, frameWidth: frame.width, frameHeight: frame.height };
  }

  async function saveFrame() {
    if (!video) return;
    // Captured off the `<video>`, which is the raw clip — the boxes live on a canvas
    // above it and are never composited in. That is what makes this safe to feed
    // straight into the dataset: a training image with rectangles burnt into it is
    // worse than useless.
    const frame = await captureVideoFrame(video);
    if (!frame) {
      toast('The video has not loaded a frame yet', 3000, 'error');
      return;
    }
    saveBtn.disabled = true;
    try {
      const camera = sourceNames.get(event.sourceUuid) || 'camera';
      const result = await saveFrameForAnnotation({
        blob: frame.blob,
        filename: captureFilename(camera),
        path: DEFAULT_CAPTURE_DIR,
        // Seeded from the overlay, which is per-frame: these are the detections the
        // watcher had at *this* moment of the clip, not the ones that happened to open
        // the event. Without a sidecar there is nothing honest to seed with — the
        // event's own boxes belong to one analysed instant and would land around empty
        // space on any other frame — so an overlay-less clip still saves bare.
        ...overlayBoxes(frame),
        extra: {
          source_camera: camera,
          source_event: event.uuid,
          source_event_rule: event.ruleName || '',
        },
      });
      savedHint.textContent = `Saved ${result.path}`;
      toast('Frame saved for annotation', 4000);
      await offerToOpenOnCanvas(result);
    } catch (err) {
      Logger.error(err);
    } finally {
      saveBtn.disabled = false;
    }
  }

  /** Same offer the Live view makes — saving is nearly always the first half of
   * "…so I can annotate it", and Files is several steps away from here. */
  async function offerToOpenOnCanvas(result) {
    if (!result?.path) return;
    const open = await showConfirmModal({
      title: 'Open this frame?',
      message: `Saved ${result.path}. Open it on the annotation canvas now?`,
      confirmText: 'Open on canvas',
      cancelText: 'Keep reviewing',
    });
    if (!open) return;
    shell.close();
    events.publish(EventTypes.OPEN_IMAGE_ON_MAIN_CANVAS, { path: result.path });
  }

  // ---------------------------------------------------------------- details

  function renderSidebar() {
    sidebar.innerHTML = '';
    sidebar.appendChild(el('h4', 'event-detail__side-title', 'Details'));

    const camera = sourceNames.get(event.sourceUuid);
    if (camera) sidebar.appendChild(detailRow('Camera', camera));
    if (event.ruleName) sidebar.appendChild(detailRow('Event type', event.ruleName));
    if (event.subjectName) sidebar.appendChild(detailRow('Acting', event.subjectName));
    if (event.targetName) sidebar.appendChild(detailRow('Acted on', event.targetName));

    sidebar.appendChild(detailRow('Started', started ? started.toLocaleString() : 'Unknown'));
    const duration = formatDuration(event.startedAt, event.endedAt);
    sidebar.appendChild(detailRow('Duration', duration || 'Still open'));
    if (event.peakConfidence) {
      sidebar.appendChild(detailRow('Peak confidence', `${(event.peakConfidence * 100).toFixed(0)}%`));
    }
    if (event.trackIdsJson?.length) {
      sidebar.appendChild(detailRow('Tracks', event.trackIdsJson.join(', ')));
    }
    if (event.truncated) {
      sidebar.appendChild(detailRow('Ended by', 'The runaway-event cap, not the subject leaving'));
    }

    const actions = el('div', 'event-detail__side-actions');
    actions.appendChild(button('Delete event', () => void removeEvent(), {
      testid: 'event-detail-delete',
      className: 'panel-btn panel-btn--danger',
    }));
    sidebar.appendChild(actions);
  }

  async function removeEvent() {
    const confirmed = await showConfirmModal({
      title: 'Delete this event?',
      // Says what is destroyed and what is not: a `save_snapshot` action's training
      // images live under image_datasets and are the user's to curate.
      message: 'The event, its snapshot and any clip or timelapse of it will be removed. '
        + 'Frames you saved for annotation are not affected.',
      confirmText: 'Delete',
      cancelText: 'Keep',
    });
    if (!confirmed) return;
    try {
      await deleteWatchEvent(event.uuid);
      toast('Event deleted', 3000);
      shell.close();
      onDeleted?.();
    } catch (err) {
      Logger.error(err);
    }
  }

  // ---------------------------------------------------------------- lifecycle

  renderSidebar();
  mediaSwitcher = buildMediaSwitcher();
  renderSnapshotOnly(mediaRows.length ? 'Loading…' : 'Nothing was recorded for this event.');
  shell.open();

  if (mediaRows.length) void showMedia(mediaRows[0]);

  return { close: () => shell.close() };
}
