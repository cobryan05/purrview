/**
 * Identity switches, reviewed one after another across a whole selection of events.
 *
 * An `identity_switch` is the one tracker-fault finding with a real question behind it:
 * the track's settled identity changed and never changed back, which is usually the
 * tracker handing one id to a second object — but sometimes it is two animals the
 * watcher genuinely confused. Nothing about the frames is harvestable either way (the
 * detector was right), so the outcome is a judgement, not training data.
 *
 * **Why a queue rather than the per-event modal repeated.** The decision takes one look
 * at two crops, and the crops are the expensive part — each is a seek and a decode in
 * db_server. Reviewing twenty of them through the event modal means twenty navigations
 * for twenty two-second decisions. Here the next item's crops are fetched while the
 * current one is on screen, so the queue runs at the speed of the judgement.
 *
 * **"Different animal" does not relabel from here.** Correcting a label is an
 * event-scoped operation over the whole clip (see `labelCorrectionModal`), and dragging
 * that modal into the middle of a queue spanning nine events would end the queue every
 * time the interesting case came up. Instead the blip is marked `confirmed` — a human
 * looked and it is real — and the events holding one are listed at the end, where each
 * opens its own correction flow.
 */
import { Logger, toast } from '/app-static/js/ui/utils/index.js';
import { createModalShell } from '/app-static/js/ui/utils/modalShell.js';
import { fetchBlipCropPreview, setEventBlipStates } from '/app-static/js/shared/api/eventBlips.js';
import { parseApiDate } from '/app-static/js/shared/time.js';

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text != null) node.textContent = text;
  return node;
}

/** `4.2–7.8s`, or `4.2s` for a single instant. Mirrors the blips panel's own span text. */
function formatSpan(startS, endS) {
  const start = Number(startS) || 0;
  const end = Number(endS) || 0;
  return end > start ? `${start.toFixed(1)}–${end.toFixed(1)}s` : `${start.toFixed(1)}s`;
}

/**
 * Walk a list of `identity_switch` blips, deciding each.
 *
 * @param {Object} opts
 * @param {Array<Object>} opts.blips - EventBlipRead rows, camelCased, switches only.
 * @param {Map<string, {name: string}>} [opts.labels] - to name the two sides.
 * @param {Map<string, Object>} [opts.events] - uuid → watch event, to date-stamp an item.
 * @param {Map<string, {name: string}>} [opts.sourceNames] - uuid → camera.
 * @param {?(eventUuid: string) => void} [opts.onOpenEvent] - open an event for correcting.
 * @param {?() => void} [opts.onFinished] - called once, after the summary is closed, when
 *   anything was actually decided. The caller refreshes its grid from here.
 */
export function openSwitchReviewQueue({
  blips,
  labels = new Map(),
  events = new Map(),
  sourceNames = new Map(),
  onOpenEvent = null,
  onFinished = null,
} = {}) {
  const queue = blips.filter(b => b?.uuid);
  if (!queue.length) return;

  let index = 0;
  let dismissed = 0;
  let skipped = 0;
  /** Event uuids whose switch the reviewer called a real identity change. */
  const needsCorrection = new Set();
  /** blip uuid → in-flight or settled crop fetch, so the one-ahead prefetch is reused. */
  const crops = new Map();

  const shell = createModalShell({ onClose: () => { if (decided()) onFinished?.(); } });
  shell.dialog.classList.add('event-blips__review-dialog', 'blip-queue');
  shell.dialog.dataset.testid = 'blip-queue-modal';

  const progress = el('p', 'event-blips__note');
  progress.dataset.testid = 'blip-queue-progress';
  const heading = el('h3', null, 'Was this a real identity change?');
  const claim = el('p', 'blip-queue__claim');
  claim.dataset.testid = 'blip-queue-claim';
  const body = el('div', 'event-blips__review-body');
  body.dataset.testid = 'blip-queue-body';
  const actions = el('div', 'event-blips__review-actions');

  const glitchBtn = el('button', 'panel-btn', 'Same animal — tracker mistake');
  glitchBtn.type = 'button';
  glitchBtn.dataset.testid = 'blip-queue-glitch';
  glitchBtn.title = 'Dismiss this finding: the detector was right, the tracker lost the thread. (1)';
  const realBtn = el('button', 'panel-btn', 'Different animal');
  realBtn.type = 'button';
  realBtn.dataset.testid = 'blip-queue-real';
  realBtn.title = 'Keep this finding and list its event at the end, to correct the labels there. (2)';
  const skipBtn = el('button', 'panel-btn', 'Skip');
  skipBtn.type = 'button';
  skipBtn.dataset.testid = 'blip-queue-skip';
  skipBtn.title = 'Leave this one undecided and move on. (3)';
  const closeBtn = el('button', 'panel-btn', 'Close');
  closeBtn.type = 'button';
  closeBtn.dataset.testid = 'blip-queue-close';
  closeBtn.addEventListener('click', () => shell.close());
  actions.append(glitchBtn, realBtn, skipBtn, closeBtn);

  shell.dialog.append(progress, heading, claim, body, actions);

  function decided() {
    return dismissed > 0 || needsCorrection.size > 0;
  }

  function current() {
    return queue[index] ?? null;
  }

  function sideNames(blip) {
    const detail = blip?.detailJson || {};
    return {
      detail,
      modalName: labels.get(detail.modalLabel)?.name || detail.modalLabel || 'the earlier label',
      sawName: labels.get(detail.sawLabel)?.name || detail.sawLabel || 'the new label',
    };
  }

  /** Where in the history this finding sits, so a decision is not made on crops alone. */
  function provenance(blip) {
    const event = events.get(blip.eventUuid);
    const parts = [];
    const when = parseApiDate(event?.startedAt);
    if (when) {
      parts.push(when.toLocaleString([], {
        day: 'numeric', month: 'short', hour: '2-digit', minute: '2-digit',
      }));
    }
    const camera = sourceNames.get(event?.sourceUuid)?.name;
    if (camera) parts.push(camera);
    parts.push(`at ${formatSpan(blip.startS, blip.endS)}`);
    return parts.join(' · ');
  }

  function cropsFor(blip) {
    if (!blip) return null;
    if (!crops.has(blip.uuid)) {
      // Stored as the promise, not awaited here: the whole point is that the next
      // item's decode starts while this one is still being looked at.
      crops.set(blip.uuid, fetchBlipCropPreview(blip.uuid).catch(err => {
        Logger.error(err);
        return { error: err?.message || 'Could not load a preview.' };
      }));
    }
    return crops.get(blip.uuid);
  }

  function buildCropColumn(title, crop, labelName) {
    const col = el('div', 'event-blips__review-col');
    col.appendChild(el('h4', 'event-blips__review-col-title', `${title} · ${labelName}`));
    if (crop?.dataUrl) {
      const img = document.createElement('img');
      img.className = 'event-blips__review-img';
      img.src = crop.dataUrl;
      img.alt = `${title.toLowerCase()} the switch, ${labelName}`;
      col.appendChild(img);
    } else {
      col.appendChild(el('p', 'event-blips__note', 'No frame available.'));
    }
    return col;
  }

  async function render() {
    const blip = current();
    if (!blip) {
      renderSummary();
      return;
    }
    const { detail, modalName, sawName } = sideNames(blip);
    progress.textContent = `${index + 1} of ${queue.length} — ${provenance(blip)}`;
    claim.textContent = `Carried ${modalName} until ${formatSpan(blip.startS, blip.endS)}, `
      + `then showed ${sawName}. One frame from just before, one from just after.`;
    realBtn.textContent = `Different animal — really ${sawName}`;
    // A finding missing either side cannot be corrected to anything, so the outcome
    // that means "correct it later" has nothing to point at.
    realBtn.disabled = !detail.modalLabel || !detail.sawLabel;

    body.innerHTML = '';
    body.appendChild(el('p', 'event-blips__note', 'Loading…'));

    const pending = cropsFor(blip);
    // Started before the await so the decode overlaps the one being looked at.
    cropsFor(queue[index + 1]);
    const result = await pending;
    // Answered after the reviewer already moved on — drop it rather than painting
    // the previous item's frames over the current one.
    if (current() !== blip) return;

    body.innerHTML = '';
    if (result?.error) {
      // The claim above still stands on its own: a clip that can no longer be read
      // must not trap a finding that is otherwise decidable.
      body.appendChild(el('p', 'event-blips__note', result.error));
      return;
    }
    body.appendChild(buildCropColumn('Before', result?.before, modalName));
    body.appendChild(buildCropColumn('After', result?.after, sawName));
  }

  function renderSummary() {
    progress.textContent = `${queue.length} finding${queue.length === 1 ? '' : 's'} reviewed`;
    heading.textContent = 'Done';
    claim.textContent = [
      dismissed ? `${dismissed} put down to the tracker` : '',
      needsCorrection.size ? `${needsCorrection.size} event${needsCorrection.size === 1 ? '' : 's'} to correct` : '',
      skipped ? `${skipped} skipped` : '',
    ].filter(Boolean).join(' · ') || 'Nothing was decided.';

    body.innerHTML = '';
    for (const eventUuid of needsCorrection) {
      const row = el('div', 'blip-queue__todo');
      row.dataset.testid = 'blip-queue-todo';
      row.appendChild(el('span', null, provenanceForEvent(eventUuid)));
      if (onOpenEvent) {
        const open = el('button', 'panel-btn', 'Correct labels…');
        open.type = 'button';
        open.dataset.testid = 'blip-queue-todo-open';
        open.addEventListener('click', () => {
          shell.close();
          onOpenEvent(eventUuid);
        });
        row.appendChild(open);
      }
      body.appendChild(row);
    }

    actions.innerHTML = '';
    const done = el('button', 'panel-btn', 'Close');
    done.type = 'button';
    done.dataset.testid = 'blip-queue-done';
    done.addEventListener('click', () => shell.close());
    actions.appendChild(done);
  }

  function provenanceForEvent(eventUuid) {
    const event = events.get(eventUuid);
    const when = parseApiDate(event?.startedAt);
    const camera = sourceNames.get(event?.sourceUuid)?.name;
    return [
      event?.eventLabel || event?.ruleName || 'event',
      when ? when.toLocaleString([], { day: 'numeric', month: 'short', hour: '2-digit', minute: '2-digit' }) : '',
      camera || '',
    ].filter(Boolean).join(' · ');
  }

  async function settle(state) {
    const blip = current();
    if (!blip) return;
    if (state) {
      try {
        await setEventBlipStates([blip.uuid], state);
      } catch (err) {
        Logger.error(err);
        toast('Could not update that finding', 4000, 'error');
        return;
      }
    }
    if (state === 'dismissed') dismissed += 1;
    else if (state === 'confirmed') needsCorrection.add(blip.eventUuid);
    else skipped += 1;
    index += 1;
    await render();
  }

  glitchBtn.addEventListener('click', () => void settle('dismissed'));
  realBtn.addEventListener('click', () => void settle('confirmed'));
  skipBtn.addEventListener('click', () => void settle(null));

  // Number keys rather than arrows: the three outcomes are a choice, not a position,
  // and ← / → in a modal over a grid already means "previous/next card" elsewhere.
  function onKeydown(e) {
    if (e.metaKey || e.ctrlKey || e.altKey) return;
    if (e.key === '1') { e.preventDefault(); void settle('dismissed'); }
    else if (e.key === '2' && !realBtn.disabled) { e.preventDefault(); void settle('confirmed'); }
    else if (e.key === '3') { e.preventDefault(); void settle(null); }
  }
  shell.dialog.addEventListener('keydown', onKeydown);

  shell.open();
  void render();
}
