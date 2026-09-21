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
import { createModalShell, isTextEntryElement, Logger, showConfirmModal, toast } from '/app-static/js/ui/utils/index.js';
import { openDialogCount } from '/app-static/js/ui/utils/modalZIndex.js';
import { actionForCombo, comboLabel, combosForAction, eventToCombo } from '/app-static/js/app/image-tagging/hotkeyBindings.js';
import {
  deleteWatchEvent,
  fetchRecording,
  recordingFileUrl,
  updateWatchEvent,
  watchEventSnapshotUrl,
} from '/app-static/js/shared/api/watchProfiles.js';
import { fetchRunningWatcherUuid, fetchTimelapsePreview } from '/app-static/js/shared/api/tasks.js';
import { saveFrameForAnnotation, captureFilename, DEFAULT_CAPTURE_DIR } from '/app-static/js/ui/utils/saveFrameForAnnotation.js';
import { attachClipOverlay, fetchClipOverlay, sampleAt } from './clipOverlay.js';
import { fetchEventBlips } from '/app-static/js/shared/api/eventBlips.js';
import {
  applyLabelRemap,
  dedupeOverlappingBoxes,
  eventCorrectionFor,
  framesHoldingLabels,
  hardMomentsFor,
  labelTallyAcrossClip,
  NO_BBOX,
  openLabelCorrectionModal,
  pickTrainingFrames,
} from './labelCorrectionModal.js';
import { createBlipsPanel } from './blipsPanel.js';
import { fetchLabels } from '/app-static/js/shared/api/labels.js';
import { describeCloseReason, eventHeadline, findRecording, formatDuration, formatEventTime } from './eventCard.js';
import { parseApiDate } from '/app-static/js/shared/time.js';

/**
 * How a settled review state reads in the sidebar.
 *
 * `unreviewed` is deliberately absent rather than mapped to "Unreviewed": it is the
 * default every event carries, and the row exists to mark the ones somebody has acted
 * on. Keyed on the same strings as `WatchEventReviewState` in db/types.py.
 */
const REVIEW_STATE_LABELS = {
  confirmed: 'Reviewed',
  dismissed: 'Dismissed',
  flagged: 'Flagged — labels were wrong',
};

/**
 * How `metadataJson.flag` reads in the sidebar. `reviewState` alone can't tell "still
 * needs a look" from "already fixed" apart once both have gone through the same
 * `flagged`/`confirmed` values — this is the field that carries who raised it and
 * whether it's resolved.
 */
function describeFlag(flag) {
  if (!flag) return null;
  const origin = flag.source === 'home_assistant' ? 'Home Assistant' : 'in-app correction';
  return flag.resolvedAt ? `${origin} — fixed` : `${origin} — not yet fixed`;
}

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
  const eventStart = parseApiDate(event?.startedAt);
  const clipStart = parseApiDate(recording?.startedAt);
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
 * With `siblings`, the modal is a *walker* over that list rather than a view of one
 * row: reviewing a night's events means opening thirty of them in a row, and closing
 * the modal between each one costs a re-render of the whole grid and loses the reader's
 * place. Navigation swaps the modal's contents in place — same shell, same focus trap —
 * so only the parts that describe the event are rebuilt.
 *
 * @param {Object} initialEvent - a WatchEventRead
 * @param {Object} [ctx]
 * @param {Map<string, string>} [ctx.sourceNames]
 * @param {() => void} [ctx.onDeleted] - so the grid can drop the card
 * @param {Object[]} [ctx.siblings] - the events the modal can walk, in the grid's order
 * @returns {{close: () => void}}
 */
export function openEventDetailModal(
  initialEvent,
  { sourceNames = new Map(), onDeleted, siblings = [], seekToS = null } = {},
) {
  // Copied, not aliased: deleting from inside the modal splices this list so the walk
  // stays coherent, and doing that to the grid's own array would corrupt the selection
  // indices of the page behind us.
  const walk = siblings.length ? [...siblings] : [initialEvent];
  let index = Math.max(0, walk.findIndex(e => e.uuid === initialEvent.uuid));
  let event = walk[index] ?? initialEvent;
  // Bumped on every navigation. Everything that awaits — recording fetches, overlay
  // sidecars — captures it and drops out if it lost the race, so a slow clip cannot
  // paint itself into the modal three events later.
  let loadSeq = 0;

  // The overlay owns a ResizeObserver and a requestVideoFrameCallback loop, neither of
  // which stops when the DOM node goes away. Opening a dozen events in a row would
  // otherwise leave a dozen redraw loops running against detached videos.
  const shell = createModalShell({
    onClose: () => {
      dropOverlay();
      _dropFullscreenSyncs();
      // Leaving fullscreen behind a closing modal strands the user on a fullscreen
      // element that is about to be removed from the document.
      if (document.fullscreenElement) document.exitFullscreen?.();
      revokePreview();
      document.removeEventListener('keydown', onKeydown);
    },
  });
  shell.dialog.classList.add('event-detail');
  shell.dialog.dataset.testid = 'event-detail-modal';

  let started = parseApiDate(event.startedAt);

  // ---------------------------------------------------------------- header

  const header = el('div', 'event-detail__header');
  const titles = el('div', 'event-detail__titles');
  const titleEl = el('h3', 'event-detail__title');
  titleEl.dataset.testid = 'event-detail-title';
  const subtitleEl = el('span', 'event-detail__subtitle');
  titles.appendChild(titleEl);
  titles.appendChild(subtitleEl);
  header.appendChild(titles);

  const nav = el('div', 'event-detail__nav');
  nav.dataset.testid = 'event-detail-nav';
  const prevBtn = button('‹ Previous', () => goTo(index - 1), { testid: 'event-detail-prev' });
  const position = el('span', 'event-detail__position');
  position.dataset.testid = 'event-detail-position';
  const nextBtn = button('Next ›', () => goTo(index + 1), { testid: 'event-detail-next' });
  // Says which key does this, read from the user's own bindings rather than assuming
  // the defaults — a rebind that the tooltip still describes as `[` is worse than no
  // tooltip.
  const describeKey = (action, text) => {
    const combo = combosForAction(action)[0];
    return combo ? `${text} (${comboLabel(combo)})` : text;
  };
  prevBtn.title = describeKey('action:prevImage', 'Previous event');
  nextBtn.title = describeKey('action:nextImage', 'Next event');
  nav.appendChild(prevBtn);
  nav.appendChild(position);
  nav.appendChild(nextBtn);
  header.appendChild(nav);

  // Next to the title/nav rather than buried in the sidebar below the video — pinning
  // is a decision reviewers reach for immediately, and scrolling to find it next to
  // "Wrong label…"/"Delete event" was the actual complaint that moved it here.
  const pinBtn = button('Pin', () => void togglePinned(), { testid: 'event-detail-pin-toggle' });
  header.appendChild(pinBtn);
  header.appendChild(button('Close', () => shell.close(), { testid: 'event-detail-close' }));
  shell.dialog.appendChild(header);

  function renderHeader() {
    titleEl.textContent = eventHeadline(event);
    subtitleEl.textContent = formatEventTime(started);
    // Hidden rather than disabled for a lone event: two dead buttons and "1 / 1" is
    // chrome that only explains itself once you have already tried to use it.
    nav.hidden = walk.length < 2;
    position.textContent = `${index + 1} / ${walk.length}`;
    prevBtn.disabled = index <= 0;
    nextBtn.disabled = index >= walk.length - 1;
    pinBtn.textContent = event.pinned ? '📌 Unpin' : 'Pin';
    pinBtn.title = event.pinned
      ? "Stop protecting this event from PurrView's storage-quota and retention cleanup."
      : "Protect this event's video from PurrView's storage-quota and retention cleanup, and from delete.";
  }

  /**
   * Show the event at `i`, if there is one.
   *
   * Deliberately does not wrap: at the end of a page the honest answer is "that is the
   * last one", and silently jumping back to the top of the grid is how a reviewer ends
   * up watching the same ten events twice without noticing.
   */
  function goTo(i) {
    if (i < 0 || i >= walk.length || i === index) return;
    index = i;
    loadEvent();
  }

  /**
   * The user's own prev/next-image hotkeys, driving the walk.
   *
   * Read through `actionForCombo` rather than hard-coded to `[` / `]` so a rebind in
   * Configuration applies here too — and specifically *not* bound to the arrow keys,
   * which belong to the `<video>` element's own seeking once it has focus.
   */
  function onKeydown(e) {
    if (isTextEntryElement(document.activeElement)) return;
    // Something is stacked on top — the delete confirmation, or a canvas opened from
    // "save frame". Its keys are not ours.
    if (openDialogCount() > dialogBaseline) return;
    const action = actionForCombo(eventToCombo(e));
    if (action === 'action:prevImage') {
      e.preventDefault();
      goTo(index - 1);
    } else if (action === 'action:nextImage') {
      e.preventDefault();
      goTo(index + 1);
    }
  }

  const body = el('div', 'event-detail__body');
  shell.dialog.appendChild(body);

  // ---------------------------------------------------------------- media

  const mediaPane = el('div', 'event-detail__media');
  body.appendChild(mediaPane);

  const sidebar = el('div', 'event-detail__side');
  body.appendChild(sidebar);

  let video = null;
  let overlay = null;
  // The sidecar the overlay was built from, kept because the whole-event relabel reads
  // it directly: it needs every frame's boxes across the clip, not the one instant
  // `overlay.boxesAt` can answer for.
  let overlayDoc = null;
  //: `fullscreenchange` listeners belonging to fullscreen buttons that may already have
  //: been thrown away with their media pane. Unhooked together on close and on every
  //: re-render, in the same spirit as the overlay's loops above: this modal replaces its
  //: media pane wholesale, so nothing calls a per-button destroy.
  let _fullscreenSyncs = [];
  let saveBtn = null;
  let annotateBtn = null;
  let mediaSwitcher = null;
  //: Object URL of a mid-event timelapse preview, held so it can be revoked. The
  //: browser keeps a blob alive until it is, and these are whole videos.
  let previewUrl = null;
  const savedHint = el('div', 'event-detail__hint');

  // The clip first when there is one: it is the literal record, and the timelapse is a
  // summary of it. An event with only a timelapse opens on that. Recomputed per event
  // rather than captured, because the modal now outlives the event it opened on.
  let mediaRows = [];

  /**
   * Move the open clip to a moment, in clip-relative seconds.
   *
   * Shares the readiness dance with the arrive-at-a-timestamp path above and for the
   * same reason: `currentTime` written before the metadata is known is silently
   * discarded, and the player would sit at zero looking as though the button did
   * nothing. Timelapses are refused rather than seeked — a blip's seconds are the
   * clip's time base, and a timelapse compresses hours into them.
   */
  function seekVideoTo(seconds) {
    if (!video || video.dataset.kind === 'timelapse') return;
    const apply = () => {
      video.currentTime = Math.max(0, Math.min(seconds, (video.duration || seconds) - 0.01));
    };
    if (video.readyState >= 1) apply();
    else video.addEventListener('loadedmetadata', apply, { once: true });
  }

  // Mounted once and moved into each rebuild of the sidebar, rather than rebuilt with
  // it: the panel owns an in-flight fetch and a selection, and recreating it per event
  // would drop both on the floor mid-load.
  const blipsHost = el('div', 'event-detail__blips');
  const blipsPanel = createBlipsPanel(blipsHost, {
    onSeek: seekVideoTo,
    // Handed over in flight. The claims name labels by uuid, and the panel is drawn
    // from a fetch that usually answers first.
    labels: fetchLabels().catch(() => new Map()),
    // A reviewer who confirms an identity_switch was a real change (not a tracker
    // mistake) already knows the correction — opening the full picker again and asking
    // them to find it in a list would throw away the answer they just gave.
    onRelabel: (fromLabelUuid, toLabelUuid) => relabelEvent({ presetRemap: new Map([[fromLabelUuid, toLabelUuid]]) }),
  });

  /**
   * Fetch a recording's full row and show it, degrading to the snapshot.
   *
   * The list only carries a summary of each recording — enough to draw a badge and
   * label a tab. The file path and the overlay live on the full row, which is fetched
   * lazily so an event with two recordings costs one request, not two, until the user
   * asks for the second.
   */
  async function showMedia(row) {
    const seq = loadSeq;
    // Whatever was saved described the pane that is being replaced. A clip and a
    // timelapse both start at t=0, so a memo carried across the switch would call a
    // different picture "already saved" — see `alreadySavedFrame`.
    lastSavedFrame = null;
    try {
      const recording = await fetchRecording(row.uuid);
      // Navigated away while this was in flight — the modal is showing a different
      // event now, and rendering this one's clip into it would be a lie about which
      // event the user is looking at.
      if (seq !== loadSeq) return;
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
      if (seq !== loadSeq) return;
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

  /**
   * The snapshot, plus an offer to compile what the watcher has buffered so far.
   *
   * Only for an event that is still open. A finished event's timelapse is a recording
   * like any other; this exists for the hours-long event whose video does not exist
   * yet, which is the one a user is most likely to have the modal open for.
   *
   * The button is offered without first checking that a buffer exists, because finding
   * out costs the same request as fetching it: the watcher is asked, and "not yet" is a
   * perfectly good answer to show. Checking first would double the round trips to
   * decide whether to draw a button.
   */
  function renderInProgress() {
    renderSnapshotOnly('This event is still happening — no video has been compiled yet.');
    const preview = button('Preview what has been recorded so far', () => void showTimelapsePreview(), {
      testid: 'event-detail-timelapse-preview',
    });
    mediaPane.appendChild(preview);
  }

  /** Ask the running watcher to compile its buffer for this event, and play it. */
  async function showTimelapsePreview() {
    const seq = loadSeq;
    renderSnapshotOnly('Compiling what has been recorded so far…');
    try {
      const taskUuid = await fetchRunningWatcherUuid(event.profileUuid);
      if (seq !== loadSeq) return;
      if (!taskUuid) {
        renderInProgressMessage('This watcher is not running, so there is nothing buffered to preview.');
        return;
      }
      const preview = await fetchTimelapsePreview(taskUuid, {
        sourceUuid: event.sourceUuid,
        ruleUuid: event.ruleUuid,
        subjectRef: event.subjectRef || '',
        targetRef: event.targetRef || '',
      });
      // Navigated away while ffmpeg ran. Revoking here rather than leaking it is the
      // whole reason this is not fire-and-forget.
      if (seq !== loadSeq) {
        if (preview) URL.revokeObjectURL(preview.url);
        return;
      }
      if (!preview) {
        renderInProgressMessage('Not enough has been recorded yet to play. Try again in a minute.');
        return;
      }
      playPreview(preview);
    } catch (err) {
      if (seq !== loadSeq) return;
      Logger.error(err);
      renderInProgressMessage('That preview could not be built.');
    }
  }

  /** `renderInProgress` with a different explanation — the retry offer must survive. */
  function renderInProgressMessage(message) {
    renderSnapshotOnly(message);
    mediaPane.appendChild(button('Try the preview again', () => void showTimelapsePreview(), {
      testid: 'event-detail-timelapse-preview',
    }));
  }

  /**
   * Play a compiled preview, and say what it is.
   *
   * No detection overlay: the sidecar is written at the close, out of the frames that
   * were actually kept, and there is none for a session still being written. Drawing
   * the finished event's boxes over a partial video would put them on the wrong frames.
   */
  function playPreview(preview) {
    revokePreview();
    previewUrl = preview.url;
    dropOverlay();
    mediaPane.innerHTML = '';
    if (mediaSwitcher) mediaPane.appendChild(mediaSwitcher);

    video = document.createElement('video');
    video.className = 'event-detail__video';
    video.controls = true;
    video.src = preview.url;
    video.dataset.testid = 'event-detail-timelapse-preview-video';
    mediaPane.appendChild(video);

    const span = formatDuration(0, preview.sourceSpanS * 1000);
    const note = preview.partial
      // Says so explicitly: a tail presented as the whole thing reads as the event
      // having started much later than it did.
      ? `The most recent ${preview.frames} frames, still recording.`
      : `${preview.frames} frames${span ? ` covering ${span}` : ''} so far — still recording.`;
    mediaPane.appendChild(el('p', 'event-detail__no-video', note));
    mediaPane.appendChild(button('Refresh', () => void showTimelapsePreview(), {
      testid: 'event-detail-timelapse-refresh',
    }));
  }

  /** A blob URL is held by the browser until revoked; a modal walked across a dozen
   * open events would otherwise hold a dozen videos in memory. */
  function revokePreview() {
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    previewUrl = null;
  }

  function renderSnapshotOnly(message) {
    // Before the wipe: the overlay's loops outlive the nodes they draw on, and
    // `innerHTML = ''` does not tell it anything.
    dropOverlay();
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

  /** Unhook every fullscreen-button listener whose button is going away. */
  function _dropFullscreenSyncs() {
    for (const sync of _fullscreenSyncs) document.removeEventListener('fullscreenchange', sync);
    _fullscreenSyncs = [];
  }

  function renderVideo(recording) {
    // Same wipe as `renderSnapshotOnly`, and for the same reason: switching between an
    // event's clip and its timelapse replaces the <video> the overlay's animation loop
    // is drawing onto, and a loop left running against a detached node keeps a whole
    // clip's worth of boxes alive for as long as the modal is open.
    dropOverlay();
    _dropFullscreenSyncs();
    mediaPane.innerHTML = '';
    if (mediaSwitcher) mediaPane.appendChild(mediaSwitcher);

    const isTimelapse = recording.kind === 'timelapse';

    video = document.createElement('video');
    video.className = 'event-detail__video';
    video.controls = true;
    // The control bar's own fullscreen button fullscreens the <video> ELEMENT, which
    // leaves the overlay canvas — its sibling inside `.clip-overlay` — behind on the
    // page: the picture fills the screen and the boxes silently vanish. Suppressed where
    // the browser honours it, in favour of the tools-row button that fullscreens the
    // wrapper and keeps them. Advisory only (Chromium-family today), so the wrapper
    // button is the real answer and this just removes a wrong-looking route to it.
    video.setAttribute('controlsList', 'nofullscreen');
    // No autoplay: this is a review surface, and a grid of events opened one after
    // another that each start talking and moving is hostile.
    video.preload = 'metadata';
    video.src = recordingFileUrl(recording.uuid);
    video.dataset.testid = 'event-detail-video';
    // Read back by `seekVideoTo`: a blip's timestamps are the clip's time base, so the
    // one thing they must never be applied to is the timelapse tab.
    video.dataset.kind = isTimelapse ? 'timelapse' : 'clip';
    mediaPane.appendChild(video);

    // Arrived from a harvested frame asking about one instant. Seeking has to wait for
    // metadata: `currentTime` set before the duration is known is silently discarded,
    // and the clip would open at zero with no sign anything was asked for.
    if (seekToS != null && !isTimelapse) {
      const seek = () => {
        // Clamped rather than trusted. The moment comes from a blip found in this
        // clip's own sidecar, so it should always be inside it — but a clip capped by
        // `max_length_s` after the sidecar was written would put it past the end, and
        // seeking past the end leaves the player showing nothing at all.
        video.currentTime = Math.max(0, Math.min(seekToS, (video.duration || seekToS) - 0.01));
      };
      if (video.readyState >= 1) seek();
      else video.addEventListener('loadedmetadata', seek, { once: true });
    }

    if (isTimelapse) {
      // A timelapse has no pre-roll to mark — it begins when its event does. It does
      // have a sidecar: one entry per kept frame, at that frame's position in the
      // *video*, which is exact rather than interpolated (see build_timelapse_overlay).
      // What it also needs is to say what it compressed.
      const note = describeTimelapse(recording);
      if (note) {
        const line = el('p', 'event-detail__timelapse-note', note);
        line.dataset.testid = 'event-detail-timelapse-note';
        mediaPane.appendChild(line);
      }
      const tools = buildFrameTools();
      mediaPane.appendChild(tools);
      void attachOverlay(recording, tools);
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

    const tools = buildFrameTools();
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
    const seq = loadSeq;
    const doc = await fetchClipOverlay(recording.uuid);
    // The modal can be closed, navigated, or re-rendered while this is in flight. The
    // `isConnected` check alone does not cover navigation: the next event's <video> is
    // connected too, and these boxes would be drawn over the wrong clip.
    if (seq !== loadSeq || !doc || !video || !video.isConnected) return;
    overlay = attachClipOverlay(video, doc);
    overlayDoc = doc;
    // The relabel is only offered once there is a sidecar to drive it — see relabelEvent.
    renderSidebar();

    const label = el('label', 'event-detail__toggle');
    const box = document.createElement('input');
    box.type = 'checkbox';
    box.checked = true;
    box.dataset.testid = 'event-detail-overlay-toggle';
    box.addEventListener('change', () => overlay.setVisible(box.checked));
    label.appendChild(box);
    label.appendChild(el('span', null, 'Show Overlays'));
    tools.appendChild(label);

    tools.appendChild(_fullscreenButton());
  }

  /**
   * Fullscreen for a clip, without losing the boxes drawn over it.
   *
   * The native control-bar button fullscreens the `<video>` element itself, which leaves
   * the overlay canvas — a *sibling* of the video inside `.clip-overlay` — behind on the
   * page, so the picture fills the screen and the boxes vanish. `controlsList` suppresses
   * that button where the browser honours it (see `renderVideo`); this one fullscreens
   * the wrapper, so the canvas comes with it and the overlay's own resize loop refits it.
   *
   * @returns {HTMLElement}
   */
  function _fullscreenButton() {
    const btn = button('⛶ Fullscreen', () => {
      const wrapper = video?.closest('.clip-overlay');
      if (!wrapper) return;
      if (document.fullscreenElement) document.exitFullscreen?.();
      // Not awaited: a refusal must leave the label alone, and `fullscreenchange` is
      // what sets it either way.
      else wrapper.requestFullscreen?.().catch(() => {});
    }, { testid: 'event-detail-fullscreen' });
    btn.title = 'Fill the screen with the clip, keeping the detection overlay on it.';
    const sync = () => {
      btn.textContent = document.fullscreenElement ? '⛶ Exit fullscreen' : '⛶ Fullscreen';
    };
    document.addEventListener('fullscreenchange', sync);
    // The modal replaces `mediaPane` wholesale when switching clip/timelapse, so the
    // button is discarded without anything calling a destroy hook — the listener has to
    // unhook itself when its button is no longer on the page.
    _fullscreenSyncs.push(sync);
    return btn;
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
   * @param {?number} [atS] - clip offset in seconds; defaults to where the video is now.
   *   Passed explicitly when capturing a batch of frames, which reads several offsets
   *   without ever moving `video.currentTime` to them.
   * @returns {Object} spread into the save call — empty when there is no overlay
   */
  function overlayBoxes(frame, atS = null) {
    if (!overlay || !video) return {};
    return scaledBoxes(overlay.boxesAt(atS ?? video.currentTime ?? 0), frame);
  }

  /**
   * Sidecar boxes scaled into `frame`'s pixel space. The one place that owns the
   * normalised→pixel convention; both the toolbar and the correction harvest read it.
   *
   * @param {Array<Object>} sidecarBoxes - `{labelUuid, x, y, w, h, conf}`, normalised
   * @param {{width: number, height: number}} frame
   * @returns {Object} spread into the save call — empty when there is nothing to save
   */
  function scaledBoxes(sidecarBoxes, frame) {
    const boxes = (sidecarBoxes ?? [])
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

  /**
   * The frame-capture toolbar, identical for a clip and for a timelapse.
   *
   * Built once rather than twice: the two media paths differ in what they say about the
   * footage, not in what you can do with a frame of it, and the pair drifted apart the
   * first time a third button was added to only one of them.
   */
  function buildFrameTools() {
    const tools = el('div', 'event-detail__tools');
    saveBtn = button('Save For Annotation', () => void saveFrame(), { testid: 'event-detail-save-frame' });
    annotateBtn = button('Annotate Image', () => void saveFrame({ openCanvas: true }), { testid: 'event-detail-annotate-frame' });
    tools.appendChild(saveBtn);
    tools.appendChild(annotateBtn);
    tools.appendChild(savedHint);
    return tools;
  }

  /** Drop the overlay and the sidecar together — they are only ever valid as a pair. */
  function dropOverlay() {
    overlay?.destroy();
    overlay = null;
    overlayDoc = null;
  }

  /**
   * "These labels are wrong, they were actually…" — correct the frame, keep the boxes.
   *
   * Only ever offered against the overlay, because a correction needs something to
   * correct: the event's own subject label describes one analysed instant and would land
   * on empty space anywhere else in the clip, so a clip without a sidecar has no
   * detections on *this* frame to speak for. Said out loud rather than silently doing
   * nothing, which is how a dead-looking button gets reported as a bug.
   */
  /**
   * "The watcher called this the wrong thing" — a correction about the whole event.
   *
   * Deliberately not a frame operation, and deliberately not next to "Save For
   * Annotation": the mistake is one the watcher made about the event, the tracker
   * followed the box across the whole clip, and correcting only the frame that happens
   * to be on screen would throw all of that away. So it lives with the event's own
   * actions, reads the entire sidecar, and harvests a spread from across the clip.
   *
   * Requires the sidecar, which is why the button only appears once one has loaded: the
   * event's own subject label describes a single analysed instant and cannot say where
   * in the clip the box was, so without per-frame boxes there is nothing to relabel.
   */
  /**
   * @param {Object} [opts]
   * @param {Map<string, string>} [opts.presetRemap] - pre-select a correction the caller
   *   already knows, e.g. from confirming an identity_switch finding was a real change
   *   rather than a tracker mistake — see `blipsPanel.js`'s `onRelabel`.
   */
  async function relabelEvent({ presetRemap = null } = {}) {
    // The button is rendered when the sidecar is loaded, but the sidebar is not redrawn
    // when the media pane switches to the timelapse or to a snapshot — both of which drop
    // the overlay. Said out loud rather than returning silently, which is how a
    // dead-looking button gets reported as a bug.
    if (!video || !overlayDoc) {
      toast('Switch back to the event clip to correct its labels', 4000);
      return;
    }
    // The analyser's own findings for this clip — where the detector flipped, dropped
    // out or merged two animals — alongside the labels, not after them: the dialog
    // waits on both, and a serial second round trip is what the user would feel.
    // Fetched here rather than borrowed from the blips panel because the panel filters
    // to what it can harvest, and a correction wants all of them. Optional: a clip with
    // no findings still gets an even spread.
    const blipsReq = fetchEventBlips({ eventUuid: event.uuid })
      .then(r => r.blips)
      .catch(err => { Logger.warn(err); return []; });
    let labels;
    try {
      labels = await fetchLabels();
    } catch (err) {
      Logger.error(err);
      toast(err.message || 'Could not load the labels', 5000, 'error');
      return;
    }
    const detected = labelTallyAcrossClip(overlayDoc, labels);
    if (!detected.length) {
      toast('The watcher recorded no labelled detections on this clip', 4000);
      return;
    }
    const picker = framePickerFor(await blipsReq);
    openLabelCorrectionModal({
      detected,
      labels,
      // Handed the event so the dialog can offer to correct the row itself, not just
      // the training data harvested from it.
      event,
      presetRemap,
      estimateFrames: labelUuids => {
        const { times, reasons } = picker(labelUuids);
        return { frames: times.length, hard: times.filter(t => reasons.has(t)).length };
      },
      onConfirm: (remap, { fixEvent } = {}) => saveCorrectedFrames(remap, labels, { fixEvent, picker }),
    });
  }

  /**
   * A picker that puts the frames the detector was caught out on ahead of the even
   * spread — see `hardMomentsFor` and `pickTrainingFrames`.
   *
   * @param {Array<Object>} blips
   * @returns {(labelUuids: Iterable<string>) => {times: number[], reasons: Map<number, string>}}
   *   `reasons` names why a time was a hard moment; a time absent from it came from the
   *   spread
   */
  function framePickerFor(blips) {
    return labelUuids => {
      const wanted = [...labelUuids];
      const hard = hardMomentsFor(overlayDoc, wanted, blips);
      const times = pickTrainingFrames(framesHoldingLabels(overlayDoc, wanted), { priority: hard });
      return { times, reasons: new Map(hard.map(m => [m.t, m.why])) };
    };
  }

  /**
   * Rewrite the event's own subject/target to what the correction says they were.
   *
   * The row, not the frames: an event whose subject was wrong reads wrong in the grid,
   * groups under the wrong identity and answers the wrong filter, and the correction
   * that fixed the training data used to leave all of that untouched.
   *
   * The local copy is patched from the server's answer rather than from what was sent,
   * so the modal cannot drift from the row — and `onDeleted` is the grid's only "your
   * data changed" hook, so it is reused here to re-run the search behind us. Failure is
   * reported and swallowed: the frame harvest that follows is independently valuable
   * and must still run.
   */
  async function correctEventRow(remap, labels) {
    const correction = eventCorrectionFor(event, remap, labels);
    if (!correction) return;
    try {
      const updated = await updateWatchEvent(event.uuid, correction.patch);
      Object.assign(event, updated || correction.patch);
      renderHeader();
      renderSidebar();
      toast(`Event corrected: ${correction.changes.map(c => `${c.from} → ${c.to}`).join(', ')}`, 4000);
      onDeleted?.();
    } catch (err) {
      Logger.error(err);
      toast(err.message || 'Could not correct this event', 5000, 'error');
    }
  }

  /**
   * Wait until the video has actually *presented* a frame at or after `atS`, not merely
   * finished the seek operation.
   *
   * `seeked` fires once the seek completes, but for compressed video the newly decoded
   * picture is not guaranteed to be composited before the next paint — back-to-back
   * seeks in a tight loop can `drawImage` the *previous* frame's pixels onto a canvas
   * meant to hold this one. The visible symptom is a batch of "corrected" frames that
   * all look identical (and carry boxes for a moment the picture doesn't actually show),
   * which is worse than a slow capture: it looks like it worked.
   *
   * Bounded by `maxAttempts` rather than looping until an exact match — `requestVideoFrameCallback`
   * reports the frame's own presentation time, which will not land exactly on `atS`, and
   * "close enough" is what the hold-until-next-sample overlay logic already assumes.
   *
   * @param {HTMLVideoElement} video
   * @param {number} atS
   * @returns {Promise<void>} resolves immediately where the API is unsupported
   *   (Firefox, older Safari) — the pre-existing `seeked`-only behaviour there rather
   *   than a hang.
   */
  function waitForPresentedFrame(video, atS, { maxAttempts = 3 } = {}) {
    if (typeof video.requestVideoFrameCallback !== 'function') return Promise.resolve();
    return new Promise(resolve => {
      let attempts = 0;
      const check = (_now, metadata) => {
        attempts += 1;
        if ((metadata?.mediaTime ?? atS) >= atS - 0.02 || attempts >= maxAttempts) {
          resolve();
          return;
        }
        video.requestVideoFrameCallback(check);
      };
      video.requestVideoFrameCallback(check);
    });
  }

  /**
   * Seek the clip to each chosen time and capture the frame there.
   *
   * Sequential rather than parallel because there is one `<video>` and one playhead;
   * "seek to all of these at once" is not a thing it can do. Each capture waits for the
   * `seeked` event, then for an actually-presented frame (`waitForPresentedFrame`) —
   * `seeked` alone is not enough; see that function for why a decode is not instant and
   * why a `drawImage` issued too early silently paints the *previous* frame, which would
   * put the corrected boxes on the wrong picture.
   *
   * The playhead is put back where the user left it: this runs behind a modal they are
   * about to return to, and finding the clip parked somewhere else would read as the
   * correction having done something to the video.
   *
   * ⚠️ `times` are *sidecar* times and the playhead runs in *player* time; the two differ
   * by the untrimmed head this browser is playing (`leadSeconds`, measured at several
   * seconds on real clips). Seeking straight to a sidecar time lands `lead` seconds early
   * — and `boxesAt` subtracts the same lead again, so the early picks sample before t=0
   * and come back empty, silently dropping most of a harvest the dialog had already
   * promised a count for. Both the seek and the box lookup are done in player time here.
   *
   * @returns {Promise<Array<{at: number, sidecarAt: number, frame: Object}>>} `at` is
   *   player time, for `overlayBoxes`; `sidecarAt` is the clip offset the pick names.
   */
  async function captureFramesAt(times) {
    const lead = overlay?.leadSeconds?.() ?? 0;
    const resumeAt = video.currentTime;
    const wasPlaying = !video.paused;
    video.pause();
    const frames = [];
    for (const t of times) {
      const seeked = new Promise(resolve => {
        // `once` so a later seek in this same loop cannot resolve an earlier promise.
        const done = () => {
          clearTimeout(timer);
          video.removeEventListener('seeked', done);
          resolve();
        };
        // A bounded wait, not an open-ended one. `seeked` does not fire when the
        // playhead is already at `t` (the first pick often is, since the harvest starts
        // where the user left it) and it never fires at all if the media errors mid-run
        // — either way an unguarded await hangs here forever, with Save and Annotate
        // left disabled and no error anywhere to say why.
        const timer = setTimeout(done, 2000);
        video.addEventListener('seeked', done, { once: true });
      });
      const playerT = t + lead;
      video.currentTime = playerT;
      await seeked;
      await waitForPresentedFrame(video, playerT);
      const frame = await captureVideoFrame(video);
      if (frame) frames.push({ at: playerT, sidecarAt: t, frame });
    }
    video.currentTime = resumeAt;
    if (wasPlaying) void video.play().catch(() => {});
    return frames;
  }

  /**
   * Save a spread of corrected frames from across the clip, then flag the event.
   *
   * A failure part-way through keeps what already saved rather than unwinding: each
   * frame is an independent training image, and eight-of-eight is better than nothing
   * when the ninth 500s. The count reported at the end is what actually landed.
   */
  async function saveCorrectedFrames(remap, labels, { fixEvent = false, picker = framePickerFor([]) } = {}) {
    // Before the harvest, and before the early return below it. The row correction is
    // the half the user sees immediately — the headline, the sidebar, the grouping —
    // and it must not be conditional on eight video seeks succeeding, nor skipped
    // entirely for a clip whose sidecar yields no frames worth saving.
    if (fixEvent) await correctEventRow(remap, labels);
    const { times, reasons } = picker(remap.keys());
    if (!times.length) {
      // Said out loud: a correction that ends in silence reads as a button that did
      // nothing, and "the frames were never there" is the answer to that.
      toast('No frames of this clip hold the corrected labels, so nothing was saved', 5000);
      return;
    }
    saveBtn.disabled = true;
    annotateBtn.disabled = true;
    savedHint.textContent = `Saving ${times.length} frame${times.length === 1 ? '' : 's'}…`;
    const camera = sourceNames.get(event.sourceUuid) || 'camera';
    const nameOf = uuid => labels?.get(uuid)?.name || uuid;
    const correctedFrom = [...remap.entries()]
      .map(([from, to]) => `${nameOf(from)} → ${to === NO_BBOX ? 'no bbox' : nameOf(to)}`)
      .join(', ');
    const saved = [];
    try {
      const captured = await captureFramesAt(times);
      for (const { sidecarAt, frame } of captured) {
        // Looked up by the sidecar time the pick names, not by converting the player
        // time back. `sidecarAt` IS a sample's `t`, so this lands on it exactly; the
        // round trip through `t + lead - lead` can come back a float below `t`, which
        // `sampleAt` resolves to the *previous* sample — empty at the first frame and
        // after any dropout gap, precisely the frames a correction most wants.
        const { boxes = [] } = scaledBoxes(sampleAt(overlayDoc, sidecarAt)?.boxes, frame);
        if (!boxes.length) continue;
        const result = await saveFrameForAnnotation({
          blob: frame.blob,
          // Per-frame stamps, so eight captures in the same second do not all collide on
          // one name and get uniquified into a confusing sequence by the server.
          // Stamped with the *sidecar* offset, which is the clip's own timebase and the
          // one the event's times are quoted in — not the player time the seek used.
          filename: captureFilename(`${camera}_t${sidecarAt.toFixed(1)}`),
          path: DEFAULT_CAPTURE_DIR,
          boxes: dedupeOverlappingBoxes(applyLabelRemap(boxes, remap)),
          frameWidth: frame.width,
          frameHeight: frame.height,
          sourceUuid: event.sourceUuid || '',
          extra: {
            source_camera: camera,
            source_event: event.uuid,
            source_event_rule: event.ruleName || '',
            source_clip_offset_s: sidecarAt.toFixed(2),
            // Which guess this replaced: a reviewer meeting the frame later can see it
            // was a correction rather than a detection the model got right, and it is
            // the record of what the model is bad at.
            corrected_from: correctedFrom,
            // Why this frame and not its neighbour: a blip kind, a confidence extreme,
            // or the even spread. Tells a reviewer which frames are the hard cases.
            pick_reason: reasons.get(sidecarAt) ?? 'spread',
          },
        });
        saved.push(result);
      }
    } catch (err) {
      Logger.error(err);
      toast(err.message || 'Could not save every corrected frame', 5000, 'error');
    }
    savedHint.textContent = saved.length ? `Saved ${saved.length} frames` : '';
    saveBtn.disabled = false;
    annotateBtn.disabled = false;
    if (!saved.length) {
      toast('No frames could be saved', 5000, 'error');
      return;
    }
    try {
      // Correcting an event *resolves* it, whether the flag came from Home Assistant
      // or this is the first anyone's touched it — either way `reviewState: 'flagged'`
      // would mean the same "still needs a look" that untouched HA flags mean, and the
      // two would be indistinguishable in the Review filter. `metadataJson.flag` carries
      // the distinction: `source` says who raised it, `resolvedAt` says whether it's
      // still open.
      const existingFlag = event.metadataJson?.flag ?? null;
      const flag = { ...(existingFlag ?? { source: 'app' }), resolvedAt: new Date().toISOString() };
      const metadataJson = { ...(event.metadataJson ?? {}), flag };
      const updated = await updateWatchEvent(event.uuid, { reviewState: 'confirmed', metadataJson });
      event.reviewState = updated.reviewState;
      event.metadataJson = updated.metadataJson;
      renderSidebar();
    } catch (err) {
      // Reported, but not allowed to swallow the offer below: the frames — the thing
      // actually worth reviewing — were saved either way.
      Logger.error(err);
      toast('The corrected frames were saved, but the event could not be marked reviewed', 6000, 'error');
    }
    await offerToReview(saved);
  }

  /**
   * Offer to open what was just auto-tagged, rather than only saying it happened.
   *
   * A toast is the wrong instrument here. The boxes are written Unverified, which means
   * the save is explicitly *not* the end of the job — something still has to confirm the
   * rectangles — and a notification that fades after four seconds leaves the user with
   * nothing to act on and a frame filed somewhere they would have to go and find. The
   * one moment they are certainly willing to review it is the moment they asked for it.
   *
   * Declining is a real answer and the default one: correcting a run of events in a row
   * is the other real workflow, and a modal that hijacked the screen every time would
   * make the fast path the annoying one. "Not now" leaves the frame in the queue, which
   * is where it was always going.
   */
  async function offerToReview(saved) {
    if (!saved?.length) return;
    const frames = `${saved.length} frame${saved.length === 1 ? '' : 's'}`;
    const boxes = saved.reduce((sum, r) => sum + (r.boxesSaved ?? 0), 0);
    const wantsReview = await showConfirmModal({
      title: `${frames} tagged`,
      message: `${boxes} box${boxes === 1 ? '' : 'es'} relabelled across ${frames} of this clip, `
        + 'saved unverified. Review them now, or leave them in the annotation queue?',
      confirmText: 'Review now',
      cancelText: 'Not now',
    });
    if (!wantsReview) {
      toast(`${frames} saved for review`, 4000);
      return;
    }
    await openOnCanvas(saved);
  }

  /**
   * The frame most recently saved from this clip, so the same one is never saved twice.
   *
   * "Save For Annotation" then "Annotate Image" is the obvious way to do the obvious
   * thing — save it, now let me draw on it — and it used to produce *two* dataset
   * images of one instant, the second of which is what the canvas opened. The first was
   * left behind in the queue as a duplicate nobody asked for.
   *
   * Keyed on the event *and* the playhead: a different event, or a seek to a different
   * moment, is a genuinely different picture. Cleared on navigation (`loadEvent`) and
   * whenever the media pane changes what is on screen, since neither leaves the memo
   * describing anything that is still visible.
   *
   * @type {?{eventUuid: string, at: number, result: Object}}
   */
  let lastSavedFrame = null;

  /** The saved image for what is on screen right now, or null if this frame is new. */
  function alreadySavedFrame() {
    if (!lastSavedFrame || !video) return null;
    if (lastSavedFrame.eventUuid !== event.uuid) return null;
    // Exact, not approximate: a paused `currentTime` does not drift, and while the clip
    // is *playing* every read is a different moment — which is a different frame and
    // must be saved as one.
    return lastSavedFrame.at === video.currentTime ? lastSavedFrame.result : null;
  }

  async function saveFrame({ openCanvas = false } = {}) {
    if (!video) return;
    const already = alreadySavedFrame();
    if (already) {
      // Both buttons land here. Annotate opens the image that already exists; a second
      // Save says so rather than silently doing nothing, which reads as a dead button.
      if (openCanvas) {
        await openOnCanvas(already);
      } else {
        toast('This frame is already saved for annotation', 4000);
      }
      return;
    }
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
    annotateBtn.disabled = true;
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
        sourceUuid: event.sourceUuid || '',
        extra: {
          source_camera: camera,
          source_event: event.uuid,
          source_event_rule: event.ruleName || '',
        },
      });
      savedHint.textContent = `Saved ${result.path}`;
      // Recorded before the canvas opens: opening it tears this modal down and rebuilds
      // it through `navContext.onClose`, and a memo written afterwards would be written
      // onto the modal that is going away.
      lastSavedFrame = { eventUuid: event.uuid, at: video.currentTime, result };
      if (openCanvas) {
        await openOnCanvas(result);
      } else {
        toast('Frame saved for annotation', 4000);
        saveBtn.disabled = false;
        annotateBtn.disabled = false;
      }
    } catch (err) {
      Logger.error(err);
      saveBtn.disabled = false;
      annotateBtn.disabled = false;
    }
  }

  /** Opens a self-contained canvas modal in place, rather than switching the main
   * tab to Annotate: this modal sits on top of the event detail modal (which it
   * closes), and closing it drops the user straight back onto the Event Review
   * grid at whatever scroll position they left it — switching tabs used to lose
   * that place entirely.
   *
   * The event modal itself is torn down while the canvas is open (it owns a
   * ResizeObserver / rVFC loop that can't sit paused underneath), so closing the
   * canvas modal — save, cancel, or delete-image — reopens this same event via
   * `navContext.onClose`, reseeked to the moment the user was watching. Without
   * that, "annotate one frame" would silently strand the reviewer back on the
   * grid instead of where the clip was playing.
   *
   * Boxes are not passed in directly — `showCanvasModal` loads the image by
   * filename and reads back whatever was just persisted by `saveFrameForAnnotation`,
   * so the detections saved above (tagged unverified) show up pre-placed. */
  async function openOnCanvas(result) {
    // One result or a whole harvest: "Annotate Image" saves a single frame, the
    // whole-event relabel saves a spread of them, and both land here.
    const results = (Array.isArray(result) ? result : [result]).filter(r => r?.path);
    if (!results.length) return;
    const resumeAtS = video?.currentTime ?? null;
    shell.close();
    const [{ showCanvasModal }, labels] = await Promise.all([
      import('/app-static/js/app/review/main.js'),
      fetchLabels().catch(() => new Map()),
    ]);
    const images = results.map(r => ({ uuid: r.uuid, filename: r.path, bboxes: [] }));
    const navContext = {
      // Navigable only when there is somewhere to navigate: a lone frame with prev/next
      // wired is two dead buttons, which is what `allowNavigation` already avoids for
      // the single-frame case.
      items: images.map(img => ({ img, bbox: null })),
      startIndex: 0,
      onClose: () => openEventDetailModal(event, { sourceNames, onDeleted, siblings: walk, seekToS: resumeAtS }),
    };
    showCanvasModal(null, images[0], labels, async () => {}, navContext, {
      allowNavigation: images.length > 1,
    });
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
      sidebar.appendChild(detailRow('Ended by', describeCloseReason(event.closeReason).title));
    }
    // Only once it is something other than the default. Every event starts unreviewed,
    // so a row saying so on all of them is a column of noise that makes the events
    // somebody *has* acted on harder to spot, not easier.
    if (REVIEW_STATE_LABELS[event.reviewState]) {
      sidebar.appendChild(detailRow('Review', REVIEW_STATE_LABELS[event.reviewState]));
    }
    const flagDesc = describeFlag(event.metadataJson?.flag);
    if (flagDesc) {
      sidebar.appendChild(detailRow('Flag', flagDesc));
    }

    // Above the delete button, below the facts: the findings are something to act on,
    // and burying them under the one destructive control in the modal would be an odd
    // place to put the feature's only entry point.
    sidebar.appendChild(blipsHost);

    const actions = el('div', 'event-detail__side-actions');
    // With the event's own actions rather than the frame toolbar under the video: this
    // is a statement about what the watcher decided, not about the frame on screen.
    // Only once the sidecar has loaded — without per-frame boxes there is nothing to
    // relabel, and a button that explains that after you press it is worse than absent.
    if (overlayDoc) {
      actions.appendChild(button('Wrong label…', () => void relabelEvent(), {
        testid: 'event-detail-relabel-event',
      }));
    }
    const deleteBtn = button('Delete event', () => void removeEvent(), {
      testid: 'event-detail-delete',
      className: 'panel-btn panel-btn--danger',
    });
    // A pin is a promise the event will not be deleted, automatically or by hand —
    // see `WatchEvent.pinned`. Disabling here rather than confirming-then-refusing
    // makes that promise visible before the click, not after.
    if (event.pinned) {
      deleteBtn.disabled = true;
      deleteBtn.title = 'Unpin this event before deleting it.';
    }
    actions.appendChild(deleteBtn);
    sidebar.appendChild(actions);
  }

  async function togglePinned() {
    const next = !event.pinned;
    try {
      const updated = await updateWatchEvent(event.uuid, { pinned: next });
      event.pinned = updated.pinned;
      renderHeader();
      renderSidebar();
      // The grid's only "your data changed" hook — see `correctEventRow` above.
      // Without it the card behind this modal keeps its stale badge and the
      // Pinned filter until the next manual search.
      onDeleted?.();
    } catch (err) {
      Logger.error(err);
      toast(err.message || 'Could not update that event', 5000, 'error');
    }
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
      // Deleting one of thirty events being reviewed should leave you reviewing the
      // other twenty-nine, not back at the grid — that is the whole point of walking
      // the page from in here. The deleted row leaves the walk so "next" cannot land
      // back on it; only an empty walk closes the modal.
      walk.splice(index, 1);
      if (!walk.length) {
        shell.close();
      } else {
        index = Math.min(index, walk.length - 1);
        loadEvent();
      }
      onDeleted?.();
    } catch (err) {
      Logger.error(err);
    }
  }

  // ---------------------------------------------------------------- lifecycle

  /** Point the whole modal at `walk[index]`. */
  function loadEvent() {
    loadSeq++;
    event = walk[index];
    started = parseApiDate(event.startedAt);
    mediaRows = [findRecording(event, 'clip'), findRecording(event, 'timelapse')].filter(Boolean);
    // Both belong to the frame that was saved, not to the modal.
    savedHint.textContent = '';
    lastSavedFrame = null;
    renderHeader();
    renderSidebar();
    // Fire and forget, like the overlay: the findings are a second opinion about the
    // clip, and making the modal wait on them would slow every event down for a feature
    // most of them have nothing to show for.
    void blipsPanel.show(event);
    mediaSwitcher = buildMediaSwitcher();
    // Walking to another event must not leave the previous one's preview holding a
    // whole video in memory.
    revokePreview();
    if (mediaRows.length) {
      renderSnapshotOnly('Loading…');
      void showMedia(mediaRows[0]);
    } else if (!event.endedAt) {
      // Still running: its video does not exist yet, but the watcher is holding the
      // frames for it and can compile them on request.
      renderInProgress();
    } else {
      renderSnapshotOnly('Nothing was recorded for this event.');
    }
  }

  loadEvent();
  shell.open();
  // Captured after open(), so the modal's own layer is part of the baseline and only a
  // dialog stacked on top of it silences the walk hotkeys (same rule as hotkeys.js).
  const dialogBaseline = openDialogCount();
  document.addEventListener('keydown', onKeydown);

  return { close: () => shell.close() };
}
