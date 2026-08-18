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
import { eventHeadline, formatDuration, formatEventTime, parseEventDate } from './eventCard.js';

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
  const shell = createModalShell();
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
  let saveBtn = null;
  const savedHint = el('div', 'event-detail__hint');

  function renderSnapshotOnly(message) {
    mediaPane.innerHTML = '';
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
    mediaPane.innerHTML = '';

    video = document.createElement('video');
    video.className = 'event-detail__video';
    video.controls = true;
    // No autoplay: this is a review surface, and a grid of events opened one after
    // another that each start talking and moving is hostile.
    video.preload = 'metadata';
    video.src = recordingFileUrl(recording.uuid);
    video.dataset.testid = 'event-detail-video';
    mediaPane.appendChild(video);

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
  }

  // ---------------------------------------------------------------- save

  async function saveFrame() {
    if (!video) return;
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
        // No seeded boxes, unlike the Live view: the detections that opened this event
        // belong to the analysed frame the watcher saw, not to whichever frame of the
        // clip is on screen now, and drawing them here would put boxes around empty
        // space more often than around the cat.
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
      message: 'The event, its snapshot and its clip will be removed. Frames you saved for annotation are not affected.',
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
  renderSnapshotOnly(event.recordingUuid ? 'Loading clip…' : 'No clip was recorded for this event.');
  shell.open();

  if (event.recordingUuid) {
    void (async () => {
      const recording = await fetchRecording(event.recordingUuid);
      // The row can outlive its file, or vice versa; either way the snapshot is still
      // worth showing, so this degrades rather than emptying the modal.
      if (!recording?.filePath) {
        renderSnapshotOnly('The clip for this event is no longer on disk.');
        return;
      }
      renderVideo(recording);
    })();
  }

  return { close: () => shell.close() };
}
