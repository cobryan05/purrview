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
 * A blip that cannot be harvested is shown with the *reason*, not hidden and not merely
 * greyed out. The commonest one — the tracker was at fault — is not a lesser finding;
 * it means the detector already got those frames right, and training on them would pad
 * the set with cases the model handles. That is worth one sentence in front of a user
 * who is about to wonder why the checkbox is missing.
 */
import { Logger, toast } from '/app-static/js/ui/utils/index.js';
import { launchRun } from '/app-static/js/ui/utils/launchRun.js';
import { fetchEventBlips, setEventBlipStates } from '/app-static/js/shared/api/eventBlips.js';
import { blipKindText, describeBlip, harvestBlocker } from '/app-static/js/shared/domain/blips.js';

const HARVEST_TYPENAME = 'HarvestBlipFramesTask';

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
 * @returns {{show: (event: Object) => Promise<void>, clear: () => void}}
 */
export function createBlipsPanel(container, { onSeek = null, labels: labelsInput = new Map() } = {}) {
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
      const [result, resolvedLabels] = await Promise.all([
        fetchEventBlips({ eventUuid: nextEvent.uuid }),
        Promise.resolve(labelsInput).catch(() => new Map()),
      ]);
      if (mine !== seq) return;
      // A claim reads "carried cat_a, showed cat_b" or it reads nothing — never the raw
      // uuids, which say less than silence does.
      if (resolvedLabels instanceof Map) labels = resolvedLabels;
      blips = result.blips;
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

    const head = el('div', 'event-blips__head');
    const blocker = harvestBlocker(blip);
    if (blocker === null) {
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

    if (blocker) {
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
      await launchRun({
        typename: HARVEST_TYPENAME,
        name: `Blip frames · ${event?.ruleName || 'event'}`,
        params: { blip_uuids: uuids },
      });
      // Says where the result goes, because it does not appear here: the run writes a
      // preview run, and the frames are triaged in Review. A toast that only said
      // "started" would leave the user watching this panel for something that is never
      // going to happen in it.
      toast(`Harvesting frames from ${uuids.length} incident(s) — they will appear in Review`, 6000);
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
