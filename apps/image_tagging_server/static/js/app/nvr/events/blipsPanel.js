/**
 * The tracking problems found in one event's clip, and the button that harvests them.
 *
 * This is where the blip pipeline becomes usable by a human. Analysis runs by itself
 * when a clip lands, so by the time anyone opens an event the findings already exist —
 * but turning them into training frames is deliberately *not* automatic, and until this
 * panel existed there was no way to ask for it short of calling the API by hand.
 *
 * Two things make the list judgeable rather than just a list:
 *
 * * **The claim is stated.** "carried cat_a, showed cat_b · 9 of 121 frames" is
 *   something a reviewer can agree or disagree with while watching the clip. A bare
 *   "identity flip at 4.2s" is not.
 * * **The clip is one press away.** Every row seeks the video that is already open
 *   above it. Judging a tracking mistake from a still is the same ambiguity that caused
 *   the mistake.
 *
 * Most tracker-fault blips (the tracker lost or swapped a track the detector never got
 * wrong) are excluded entirely — fetched with `harvestable: true` below, they carry no
 * actionable content and listing them would only be a "check off" with nothing behind
 * it. `identity_switch` is the one exception: it gets its own review action (a
 * before/after crop compare — was this really the same animal, or a genuine identity
 * change?) instead of the harvest checkbox, so it's fetched too — see `buildRow` and
 * `openIdentitySwitchReview`. A harvestable blip that is currently blocked for some
 * other reason (its clip expired, or it was already harvested) is still shown with the
 * *reason*, not hidden — those are real findings, just not ones this panel can act on
 * right now.
 */
import { Logger, toast } from '/app-static/js/ui/utils/index.js';
import { createModalShell } from '/app-static/js/ui/utils/modalShell.js';
import { fetchBlipCropPreview, fetchEventBlips, setEventBlipStates } from '/app-static/js/shared/api/eventBlips.js';
import {
  KIND_IDENTITY_SWITCH,
  blipKindText,
  describeBlip,
  harvestBlocker,
  switchReviewable,
} from '/app-static/js/shared/domain/blips.js';
import { startHarvest } from './harvestRun.js';

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text != null) node.textContent = text;
  return node;
}

/** `4.2–7.8s`, or `4.2s` when the incident is a single instant. */
export function formatSpan(startS, endS) {
  const start = Number(startS) || 0;
  const end = Number(endS) || 0;
  return end > start ? `${start.toFixed(1)}–${end.toFixed(1)}s` : `${start.toFixed(1)}s`;
}

/**
 * Mount the panel. It renders nothing until `show` is called with an event.
 *
 * @param {HTMLElement} container
 * @param {Object} [opts]
 * @param {?(seconds: number) => void} [opts.onSeek] - move the open clip to a moment.
 *   Omitted when there is no player, in which case rows carry no jump button rather
 *   than a dead one.
 * @param {Map<string, {name: string}>|Promise<Map>} [opts.labels] - to name label uuids
 *   in claims. A promise is accepted and awaited before the first render, so the modal
 *   can hand over an in-flight fetch rather than either blocking on it or rendering a
 *   claim with the names missing and no second chance to fill them in.
 * @param {?(fromLabelUuid: string, toLabelUuid: string) => void} [opts.onRelabel] - open
 *   the wrong-label flow with this correction pre-selected, offered after confirming an
 *   `identity_switch` was a real change rather than a tracker mistake. Omitted where
 *   there is no relabel flow to hand off to, in which case that option is not offered.
 * @returns {{show: (event: Object) => Promise<void>, clear: () => void}}
 */
export function createBlipsPanel(container, { onSeek = null, labels: labelsInput = new Map(), onRelabel = null } = {}) {
  let labels = labelsInput instanceof Map ? labelsInput : new Map();
  // Bumped on every `show`. The modal walks events with `[` / `]`, so a slow fetch for
  // the event you just left must not paint its findings onto the one you are looking at
  // — the same race the modal guards its own loads against.
  let seq = 0;
  let event = null;
  let blips = [];
  const selected = new Set();

  const root = el('div', 'event-blips');
  root.dataset.testid = 'event-blips';
  container.appendChild(root);

  async function show(nextEvent) {
    const mine = ++seq;
    event = nextEvent;
    blips = [];
    selected.clear();
    // Nothing at all until the fetch answers. A "Tracking problems" heading over an
    // empty list reads as "none found", which is a different statement from "not asked
    // yet" — and it is the one that would be wrong here.
    root.innerHTML = '';
    try {
      // Two calls, not one unfiltered fetch client-filtered afterwards: `harvestable`
      // is a real server-side filter (see `_attach_event_blip_counts`'s own reasoning),
      // and asking for exactly the two groups this panel shows keeps that rule in one
      // place rather than re-deciding it here from raw `fault`/`kind` values.
      const [harvestable, identitySwitches, resolvedLabels] = await Promise.all([
        fetchEventBlips({ eventUuid: nextEvent.uuid, harvestable: true }),
        fetchEventBlips({ eventUuid: nextEvent.uuid, kinds: [KIND_IDENTITY_SWITCH] }),
        Promise.resolve(labelsInput).catch(() => new Map()),
      ]);
      if (mine !== seq) return;
      // A claim reads "carried cat_a, showed cat_b" or it reads nothing — never the raw
      // uuids, which say less than silence does.
      if (resolvedLabels instanceof Map) labels = resolvedLabels;
      // Each call is already worst-first on its own; re-sort the merge so the combined
      // list still reads worst-first rather than "every harvestable one, then every
      // identity switch" regardless of how sure the detector or tracker was.
      blips = [...harvestable.blips, ...identitySwitches.blips]
        .sort((a, b) => (Number(b.severity) || 0) - (Number(a.severity) || 0));
    } catch (err) {
      if (mine !== seq) return;
      Logger.error(err);
      render({ error: 'The tracking analysis for this event could not be loaded.' });
      return;
    }
    render();
  }

  function clear() {
    seq++;
    event = null;
    blips = [];
    selected.clear();
    root.innerHTML = '';
  }

  /** Blips a harvest could actually act on, in the order they are shown. */
  function selectable() {
    return blips.filter(b => harvestBlocker(b) === null);
  }

  function render({ error = null } = {}) {
    root.innerHTML = '';
    if (error) {
      root.appendChild(el('p', 'event-blips__note', error));
      return;
    }
    // Silence when there is nothing, rather than a heading announcing a clean bill of
    // health: an event with no row here is either clean or never analysed, and the two
    // are not distinguishable from the outside. Claiming the first would be a guess.
    if (!blips.length) return;

    const title = el('h4', 'event-detail__side-title', `Tracking problems (${blips.length})`);
    title.dataset.testid = 'event-blips-title';
    root.appendChild(title);

    for (const blip of blips) root.appendChild(buildRow(blip));

    root.appendChild(buildActions());
  }

  function buildRow(blip) {
    const row = el('div', 'event-blips__row');
    row.dataset.testid = 'event-blip-row';
    row.dataset.blipUuid = blip.uuid;
    row.dataset.kind = blip.kind;

    const isIdentitySwitch = blip.kind === KIND_IDENTITY_SWITCH;
    const head = el('div', 'event-blips__head');
    const blocker = harvestBlocker(blip);
    if (isIdentitySwitch) {
      // Its own review action rather than the harvest checkbox — harvestable is always
      // false for this kind (the tracker was at fault), but unlike every other
      // tracker-fault kind, this one has a real question a reviewer can answer.
      if (switchReviewable(blip)) {
        const reviewBtn = el('button', 'panel-btn event-blips__review', 'Review…');
        reviewBtn.type = 'button';
        reviewBtn.dataset.testid = 'event-blip-review';
        reviewBtn.title =
          'Compare a frame from just before and just after the switch — was this really '
          + 'the same animal (a tracker mistake), or a genuine identity change worth correcting?';
        reviewBtn.addEventListener('click', () => void openIdentitySwitchReview(blip));
        head.appendChild(reviewBtn);
      }
    } else if (blocker === null) {
      const check = document.createElement('input');
      check.type = 'checkbox';
      check.dataset.testid = 'event-blip-check';
      check.checked = selected.has(blip.uuid);
      check.addEventListener('change', () => {
        if (check.checked) selected.add(blip.uuid);
        else selected.delete(blip.uuid);
        syncActions();
      });
      head.appendChild(check);
    }
    head.appendChild(el('span', 'event-blips__kind', blipKindText(blip.kind)));
    head.appendChild(el('span', 'event-blips__span', formatSpan(blip.startS, blip.endS)));

    // Only offered when there is a player to move. The panel is also shown for an event
    // whose clip has been swept, where seeking is exactly the thing that cannot happen.
    if (onSeek && blip.state !== 'expired') {
      const jump = el('button', 'panel-btn event-blips__jump', '▶');
      jump.type = 'button';
      jump.title = `Play the clip from ${formatSpan(blip.startS, blip.endS)}`;
      jump.dataset.testid = 'event-blip-jump';
      jump.addEventListener('click', () => onSeek(Number(blip.startS) || 0));
      head.appendChild(jump);
    }
    row.appendChild(head);

    const claim = describeBlip(blip, labels);
    if (claim) {
      const line = el('div', 'event-blips__claim', claim);
      line.dataset.testid = 'event-blip-claim';
      row.appendChild(line);
    }

    if (isIdentitySwitch) {
      if (blip.state === 'dismissed') {
        const note = el('div', 'event-blips__blocked', 'reviewed');
        note.dataset.testid = 'event-blip-blocked';
        row.appendChild(note);
      } else if (blip.state === 'expired') {
        const note = el('div', 'event-blips__blocked', 'the clip has been deleted, so this can no longer be reviewed');
        note.dataset.testid = 'event-blip-blocked';
        row.appendChild(note);
      }
    } else if (blocker) {
      const why = el('div', 'event-blips__blocked', blocker);
      why.dataset.testid = 'event-blip-blocked';
      row.appendChild(why);
    } else if (blip.state === 'dismissed') {
      // Selectable all the same. Dismissing is a judgement, not a deletion, and a
      // reviewer who changes their mind should not have to undo it first.
      const note = el('div', 'event-blips__blocked', 'dismissed — harvest anyway to undo');
      note.dataset.testid = 'event-blip-blocked';
      row.appendChild(note);
    }
    return row;
  }

  /**
   * "Was this really the same animal?" — a before/after crop either side of the switch,
   * with the two outcomes a reviewer can reach: it was the tracker's mistake (dismiss),
   * or it was a genuine change worth correcting (dismiss *and* hand the correction to
   * the wrong-label flow, pre-filled, via `onRelabel`).
   */
  async function openIdentitySwitchReview(blip) {
    const detail = blip.detailJson || {};
    const modalName = labels.get(detail.modalLabel)?.name || detail.modalLabel || 'the earlier label';
    const sawName = labels.get(detail.sawLabel)?.name || detail.sawLabel || 'the new label';

    const shell = createModalShell();
    shell.dialog.classList.add('event-blips__review-dialog');
    shell.dialog.dataset.testid = 'blip-review-modal';
    shell.dialog.appendChild(el('h3', null, 'Was this a real identity change?'));
    shell.dialog.appendChild(el(
      'p', 'event-blips__note',
      `Carried ${modalName} until ${formatSpan(blip.startS, blip.endS)}, then showed ${sawName}. `
      + 'One frame from just before, one from just after.',
    ));

    const body = el('div', 'event-blips__review-body');
    body.dataset.testid = 'blip-review-body';
    body.appendChild(el('p', 'event-blips__note', 'Loading…'));
    shell.dialog.appendChild(body);

    const actions = el('div', 'event-blips__review-actions');
    const glitchBtn = el('button', 'panel-btn', 'Same animal — tracker mistake');
    glitchBtn.type = 'button';
    glitchBtn.dataset.testid = 'blip-review-glitch';
    const relabelBtn = el('button', 'panel-btn', `Different animal — really ${sawName}`);
    relabelBtn.type = 'button';
    relabelBtn.dataset.testid = 'blip-review-relabel';
    relabelBtn.disabled = !onRelabel || !detail.modalLabel || !detail.sawLabel;
    relabelBtn.title = relabelBtn.disabled
      ? 'This finding is missing a label to correct to.'
      : `Open the wrong-label flow with ${modalName} → ${sawName} already chosen.`;
    const cancelBtn = el('button', 'panel-btn', 'Cancel');
    cancelBtn.type = 'button';
    cancelBtn.addEventListener('click', () => shell.close());
    actions.append(glitchBtn, relabelBtn, cancelBtn);
    shell.dialog.appendChild(actions);

    async function resolve(relabel) {
      shell.close();
      try {
        await setEventBlipStates([blip.uuid], 'dismissed');
      } catch (err) {
        Logger.error(err);
        toast('Could not update that finding', 4000, 'error');
      }
      // After the dismiss, not before: the relabel flow opens its own modal on top of
      // this one, and closing that dismiss request behind it would race the repaint
      // below against whichever one answers last.
      if (relabel) onRelabel?.(detail.modalLabel, detail.sawLabel);
      if (event) await show(event);
    }
    glitchBtn.addEventListener('click', () => void resolve(false));
    relabelBtn.addEventListener('click', () => void resolve(true));

    shell.open();

    try {
      const { before, after } = await fetchBlipCropPreview(blip.uuid);
      body.innerHTML = '';
      body.appendChild(buildCropColumn('Before', before, modalName));
      body.appendChild(buildCropColumn('After', after, sawName));
    } catch (err) {
      Logger.error(err);
      body.innerHTML = '';
      // Still lets the reviewer decide from the claim above alone — a clip that can no
      // longer be read must not trap a finding that could otherwise still be resolved.
      body.appendChild(el('p', 'event-blips__note', err.message || 'Could not load a preview.'));
    }
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

  let harvestBtn = null;
  let dismissBtn = null;

  function buildActions() {
    const actions = el('div', 'event-blips__actions');

    const all = el('button', 'panel-btn', 'Select all');
    all.type = 'button';
    all.dataset.testid = 'event-blips-select-all';
    all.addEventListener('click', () => {
      const eligible = selectable();
      // Toggle, so the same button undoes a full selection. A separate "clear" button
      // for a list this short is chrome.
      const turningOn = selected.size < eligible.length;
      selected.clear();
      if (turningOn) for (const blip of eligible) selected.add(blip.uuid);
      render();
    });
    if (selectable().length > 1) actions.appendChild(all);

    harvestBtn = el('button', 'panel-btn event-blips__harvest', 'Harvest frames');
    harvestBtn.type = 'button';
    harvestBtn.dataset.testid = 'event-blips-harvest';
    harvestBtn.addEventListener('click', () => void harvest());
    actions.appendChild(harvestBtn);

    dismissBtn = el('button', 'panel-btn', 'Dismiss');
    dismissBtn.type = 'button';
    dismissBtn.title = 'Mark the selected findings as not worth acting on';
    dismissBtn.dataset.testid = 'event-blips-dismiss';
    dismissBtn.addEventListener('click', () => void dismiss());
    actions.appendChild(dismissBtn);

    syncActions();
    return actions;
  }

  /** The buttons say how many they would act on, and are dead when that is none. */
  function syncActions() {
    if (!harvestBtn) return;
    const n = selected.size;
    harvestBtn.textContent = n ? `Harvest frames from ${n}` : 'Harvest frames';
    harvestBtn.disabled = n === 0;
    dismissBtn.disabled = n === 0;
  }

  async function harvest() {
    const uuids = [...selected];
    if (!uuids.length) return;
    harvestBtn.disabled = true;
    try {
      await startHarvest({ blipUuids: uuids, name: `Blip frames · ${event?.ruleName || 'event'}` });
      // Not marked harvested here. The task sets that itself once frames have actually
      // landed, so a run that fails does not leave behind a record claiming otherwise.
      // Re-reading now shows the truth at this instant, which is still `new`.
      if (event) await show(event);
    } catch (err) {
      Logger.error(err);
      toast('Could not start the harvest', 5000, 'error');
      harvestBtn.disabled = false;
    }
  }

  async function dismiss() {
    const uuids = [...selected];
    if (!uuids.length) return;
    dismissBtn.disabled = true;
    try {
      await setEventBlipStates(uuids, 'dismissed');
      toast(`Dismissed ${uuids.length} finding(s)`, 3000);
      if (event) await show(event);
    } catch (err) {
      Logger.error(err);
      toast('Could not dismiss those findings', 5000, 'error');
      dismissBtn.disabled = false;
    }
  }

  return { show, clear };
}
