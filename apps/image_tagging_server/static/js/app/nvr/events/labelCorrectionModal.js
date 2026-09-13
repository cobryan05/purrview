/**
 * NVR → Events: "this event's label is wrong, it was actually…".
 *
 * The correction half of the loop the NVR exists for. A watcher that calls the tabby
 * "Clover" will keep calling her Clover, and the only thing that fixes it is training
 * data saying otherwise — so the moment you *notice* is the moment worth capturing.
 *
 * **This is an event-level operation, not a frame one.** It is deliberately not a
 * variant of "save this frame for annotation": the mistake being corrected is one the
 * watcher made about *the whole event*, and the tracker followed that box across the
 * entire clip. Correcting only the frame that happens to be on screen would throw away
 * everything the tracker already did, and would hand the trainer a single image of a
 * mistake that lasted ninety seconds.
 *
 * So the correction harvests a **spread** across the clip rather than one frame — and,
 * equally deliberately, not *every* frame. Consecutive frames of a tracked animal are
 * near-duplicates; a hundred of them is not a hundred times the training signal, it is
 * one pose weighted a hundredfold, plus a hundred images for the dedup pass to throw
 * away. `pickTrainingFrames` is where that judgement lives.
 *
 * The remap is per detected label rather than one label for the whole event. A clip
 * routinely holds more than one thing: correcting "Clover → Luna" must not also rewrite
 * the correctly-detected water bowl beside her, and a single-label form would do exactly
 * that while looking like it had worked.
 *
 * What it does *not* do is decide anything about the boxes' geometry. The detections'
 * rectangles are kept as they are and written Unverified, exactly as
 * `saveFrameForAnnotation` writes any model output: a corrected label on a model's box
 * is still a guess about *where*, and the review queue is where that gets settled.
 */
import { createModalShell } from '/app-static/js/ui/utils/modalShell.js';

/**
 * Most frames one correction will harvest from a clip.
 *
 * A ceiling rather than a target: a two-second visit has three frames and should yield
 * three. It exists because the cost is paid twice — once seeking and encoding each frame
 * in the browser, once by whoever reviews them — and the tenth near-identical picture of
 * the same cat at the same bowl earns neither.
 */
export const MAX_TRAINING_FRAMES = 8;

/**
 * Closest two harvested frames may sit in clip time.
 *
 * The sidecar samples at the analysis rate, so at 5 fps a burst of frames spans barely a
 * second and shows one pose. Spacing them is what makes the set a *spread* — different
 * positions, different poses, occasionally a different part of the room.
 */
export const MIN_FRAME_GAP_S = 0.75;

/** The select option meaning "this label was right, leave its boxes alone". */
const KEEP = '';

/**
 * The select option meaning "the detector saw something that wasn't there" — a false
 * positive, not a mislabel. Distinct from every real label uuid so `applyLabelRemap`
 * can tell "call this something else" from "this box shouldn't exist" apart, and from
 * `KEEP` so "leave it alone" and "remove it" are never the same choice.
 */
export const NO_BBOX = '__no_bbox__';

/**
 * What a remap would do to the event row's own subject and target.
 *
 * The event carries the watcher's verdict about the *whole* visit — "Maple drinks
 * water" — as `subjectRef`/`targetRef`, which are label uuids (a zone-kind rule's
 * target is a zone uuid, which no label remap can match, so it simply never appears
 * here). Correcting the boxes without correcting these leaves every list, group and
 * filter still answering with the mistake that was just corrected.
 *
 * Returns null when the remap touches neither slot — the common case of correcting a
 * label that was never the event's own subject, where offering to "fix the event"
 * would be offering to change nothing.
 *
 * A `NO_BBOX` target never touches the event row: "this box shouldn't exist" says
 * nothing about what actually happened, so there is no corrected identity to write —
 * unlike a relabel, which is itself the answer to "what was it really?".
 *
 * @param {?Object} event - a watch event row
 * @param {Map<string, string>} remap - old label uuid → new label uuid (or `NO_BBOX`)
 * @param {Map<string, Object>} [labels] - uuid → label, for the new names
 * @returns {?{patch: Object, changes: Array<{slot: string, from: string, to: string}>}}
 */
export function eventCorrectionFor(event, remap, labels) {
  if (!event || !remap?.size) return null;
  const nameOf = uuid => labels?.get(uuid)?.name || uuid;
  const patch = {};
  const changes = [];
  for (const [slot, refKey, nameKey] of [
    ['subject', 'subjectRef', 'subjectName'],
    ['target', 'targetRef', 'targetName'],
  ]) {
    const ref = event[refKey];
    if (!ref || !remap.has(ref) || remap.get(ref) === NO_BBOX) continue;
    const to = remap.get(ref);
    patch[refKey] = to;
    // The name goes with the ref, always — see `WatchEventUpdate` on why a row that
    // filters as one label and reads as another is the worst of the three states.
    patch[nameKey] = nameOf(to);
    changes.push({ slot, from: event[nameKey] || nameOf(ref), to: nameOf(to) });
  }
  return changes.length ? { patch, changes } : null;
}

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text != null) node.textContent = text;
  return node;
}

/**
 * Tally every label the watcher put on this clip, across all of it.
 *
 * Both counts are shown, because they answer different questions: *boxes* is how much
 * the model insisted on the label, *frames* is how much training data correcting it
 * would actually produce. Two boxes in one frame and two boxes across a minute are the
 * same number and very different situations.
 *
 * @param {Object} doc - a clip overlay sidecar
 * @param {Map<string, Object>} labels - uuid → label, as `fetchLabels()` returns
 * @returns {Array<{labelUuid: string, name: string, boxes: number, frames: number}>} biggest first
 */
export function labelTallyAcrossClip(doc, labels) {
  const tally = new Map();
  for (const frame of doc?.frames ?? []) {
    const seenHere = new Set();
    for (const box of frame?.boxes ?? []) {
      if (!box?.labelUuid) continue;
      let row = tally.get(box.labelUuid);
      if (!row) tally.set(box.labelUuid, (row = { boxes: 0, frames: 0 }));
      row.boxes += 1;
      // Counted once per frame however many boxes of it are in that frame — otherwise
      // "frames" would say 12 for a clip that only has 4.
      if (!seenHere.has(box.labelUuid)) {
        seenHere.add(box.labelUuid);
        row.frames += 1;
      }
    }
  }
  return [...tally.entries()]
    .map(([labelUuid, row]) => ({
      labelUuid,
      // A label the DB has never heard of is still worth a row: it is precisely the kind
      // of stale detection someone wants to correct, and hiding it would leave the event
      // uncorrectable for the one label that is wrong.
      name: labels?.get(labelUuid)?.name || labelUuid,
      ...row,
    }))
    .sort((a, b) => b.boxes - a.boxes || a.name.localeCompare(b.name));
}

/**
 * The times of every frame holding at least one of these labels.
 *
 * @param {Object} doc - a clip overlay sidecar
 * @param {Iterable<string>} labelUuids
 * @returns {number[]} clip-relative seconds, ascending
 */
export function framesHoldingLabels(doc, labelUuids) {
  const wanted = new Set(labelUuids ?? []);
  const times = [];
  for (const frame of doc?.frames ?? []) {
    if (typeof frame?.t !== 'number') continue;
    if ((frame.boxes ?? []).some(b => wanted.has(b?.labelUuid))) times.push(frame.t);
  }
  return times.sort((a, b) => a - b);
}

/**
 * Choose the subset of frames actually worth saving.
 *
 * Two passes, in this order, because they answer different objections:
 *
 *  1. **Thin by time.** Frames closer together than `minGapS` show the same pose. This
 *     pass runs first so the cap is spent on *distinct* frames rather than on whichever
 *     burst happened to be densest.
 *  2. **Spread to the cap.** If more survive than the cap allows, take evenly spaced
 *     ones across the whole span — first and last included, since the beginning and end
 *     of a visit are where the animal is entering and leaving, and those are the poses
 *     the middle of the clip does not have.
 *
 * Taking the *first* N instead would hand the trainer the first two seconds of every
 * event and nothing else, which is a worse set than a random sample and looks fine.
 *
 * @param {number[]} times - ascending, as from framesHoldingLabels
 * @param {Object} [opts]
 * @param {number} [opts.maxFrames]
 * @param {number} [opts.minGapS]
 * @returns {number[]}
 */
export function pickTrainingFrames(times, { maxFrames = MAX_TRAINING_FRAMES, minGapS = MIN_FRAME_GAP_S } = {}) {
  const spaced = [];
  for (const t of times ?? []) {
    if (!spaced.length || t - spaced[spaced.length - 1] >= minGapS) spaced.push(t);
  }
  // The greedy pass always keeps the *first* frame but silently drops the *last* one
  // whenever the tail of the clip falls inside the gap behind the previous keep — and
  // the end of a visit, where the animal is leaving, is precisely the pose the middle
  // of the clip does not have. Swapped in over the final keep rather than appended, so
  // the minimum-gap invariant still holds: the frame it replaces was already at least
  // `minGapS` past the one before it, and this one is later still.
  const last = times?.length ? times[times.length - 1] : null;
  if (spaced.length >= 2 && last != null && spaced[spaced.length - 1] !== last) {
    spaced[spaced.length - 1] = last;
  }
  if (spaced.length <= maxFrames) return spaced;
  if (maxFrames <= 1) return spaced.slice(0, Math.max(0, maxFrames));
  const picked = [];
  // Fenceposts, not buckets: dividing by (maxFrames - 1) is what puts the last pick on
  // the final frame rather than one step short of it.
  const step = (spaced.length - 1) / (maxFrames - 1);
  for (let i = 0; i < maxFrames; i++) picked.push(spaced[Math.round(i * step)]);
  return [...new Set(picked)];
}

/**
 * Apply a remap to a frame's boxes.
 *
 * A box mapped to `NO_BBOX` is dropped rather than relabelled — a false positive isn't
 * a mislabel, there is no correct label to give it. A frame left with no boxes at all
 * is still filed: the caller's own "skip empty frames" guard exists for a frame that
 * never had anything worth saving, not for one that had exactly one box and this is
 * what removing it looks like.
 *
 * Returns a new array — the caller's overlay boxes belong to the running redraw loop
 * and must not be rewritten under it.
 *
 * @param {Array<Object>} boxes
 * @param {Map<string, string>} remap - old label uuid → new label uuid (or `NO_BBOX`)
 * @returns {Array<Object>}
 */
export function applyLabelRemap(boxes, remap) {
  return (boxes ?? [])
    .filter(box => !(box?.labelUuid && remap.get(box.labelUuid) === NO_BBOX))
    .map(box => (
      box?.labelUuid && remap.has(box.labelUuid)
        ? { ...box, labelUuid: remap.get(box.labelUuid) }
        : box
    ));
}

/**
 * Ask which of this event's labels were wrong, and what they should have been.
 *
 * @param {Object} opts
 * @param {Array<{labelUuid: string, name: string, boxes: number, frames: number}>} opts.detected -
 *   from labelTallyAcrossClip
 * @param {Map<string, Object>} opts.labels - every label the DB knows, uuid → label
 * @param {?Object} [opts.event] - the event being corrected. Given, the dialog can also
 *   offer to fix the event's own subject/target — see `eventCorrectionFor`.
 * @param {(labelUuids: string[]) => number} [opts.estimateFrames] - how many frames a
 *   given selection would harvest, for the live count
 * @param {(remap: Map<string, string>, opts: {fixEvent: boolean}) => (void | Promise<void>)}
 *   opts.onConfirm - non-empty remap, plus whether the event row itself should be corrected
 * @param {?Map<string, string>} [opts.presetRemap] - a correction the caller already
 *   knows (old label uuid → new label uuid), pre-selected on open rather than left for
 *   the reviewer to find again in a list they just answered a question about — see
 *   `eventDetailModal.js`'s `onRelabel`. Ignored for any `from` not present in `detected`.
 * @returns {{close: () => void}}
 */
export function openLabelCorrectionModal({ detected, labels, event = null, estimateFrames, onConfirm, presetRemap = null }) {
  const shell = createModalShell();
  shell.dialog.classList.add('label-correction');
  shell.dialog.dataset.testid = 'label-correction-modal';

  shell.dialog.appendChild(el('h3', 'label-correction__title', 'What did the watcher get wrong?'));
  shell.dialog.appendChild(el('p', 'label-correction__note',
    'A spread of frames from across this clip is saved, left unverified for review — '
    + 'not every frame, because consecutive frames of the same subject are near-duplicates. '
    + 'A relabelled box keeps its geometry; "No bbox" removes the box instead, for a '
    + 'detection that should never have been made. The event is flagged so you can find it '
    + 'again.'));

  const rows = el('div', 'label-correction__rows');
  /** @type {Array<{from: string, select: HTMLSelectElement}>} */
  const pickers = [];
  // Sorted by name so the dropdown is scannable; the detected label itself is excluded
  // from its own row's options, where it would mean "correct this to what it already
  // is" — which is what leaving it alone already says.
  const allLabels = [...(labels?.values() ?? [])].sort((a, b) => String(a.name).localeCompare(String(b.name)));

  for (const row of detected) {
    const line = el('div', 'label-correction__row');
    line.dataset.testid = 'label-correction-row';
    // Both counts: how insistent the model was, and how much training data correcting it
    // would produce. They are routinely very different numbers.
    line.appendChild(el('span', 'label-correction__from',
      `${row.name} — ${row.boxes} box${row.boxes === 1 ? '' : 'es'} `
      + `in ${row.frames} frame${row.frames === 1 ? '' : 's'}`));

    const select = document.createElement('select');
    select.className = 'label-correction__to';
    select.dataset.testid = 'label-correction-select';
    select.dataset.labelUuid = row.labelUuid;
    const keep = document.createElement('option');
    keep.value = KEEP;
    keep.textContent = '(leave as-is)';
    select.appendChild(keep);
    const noBbox = document.createElement('option');
    noBbox.value = NO_BBOX;
    noBbox.textContent = 'No bbox — false positive';
    select.appendChild(noBbox);
    for (const label of allLabels) {
      if (label.uuid === row.labelUuid) continue;
      const option = document.createElement('option');
      option.value = label.uuid;
      option.textContent = label.name;
      select.appendChild(option);
    }
    // Preset before the row is appended, not after — so the estimate and the "also fix
    // this event" line both read the real starting state on their very first render
    // rather than flashing from "nothing chosen" once `syncConfirm()` runs below.
    const preset = presetRemap?.get(row.labelUuid);
    if (preset && [...select.options].some(o => o.value === preset)) select.value = preset;
    line.appendChild(select);
    rows.appendChild(line);
    pickers.push({ from: row.labelUuid, select });
  }
  shell.dialog.appendChild(rows);

  // The number the user is actually deciding about. Live rather than stated afterwards:
  // "correcting this will produce two frames" is worth knowing *before* agreeing to it,
  // and it is the honest answer to "why did I only get two?" for a short event.
  const estimate = el('p', 'label-correction__estimate');
  estimate.dataset.testid = 'label-correction-estimate';
  shell.dialog.appendChild(estimate);

  // ── Fix the event row itself ──────────────────────────────────────────────
  // Training data is the long game; this is the short one. An event whose subject was
  // wrong reads wrong in every list, groups under the wrong identity and answers the
  // wrong filter, and until now the only correction on offer rewrote frames and left
  // the row saying what it always said.
  //
  // Opt-*out*, not opt-in: someone who has just told the dialog the label was wrong
  // has already made this decision. It appears only when the chosen remap actually
  // touches the event's own subject or target — offering to change nothing is worse
  // than not offering.
  const fixRow = el('label', 'label-correction__fix-event');
  fixRow.dataset.testid = 'label-correction-fix-event';
  const fixBox = document.createElement('input');
  fixBox.type = 'checkbox';
  fixBox.checked = true;
  fixBox.dataset.testid = 'label-correction-fix-event-box';
  const fixText = el('span', null, '');
  fixRow.append(fixBox, fixText);
  fixRow.hidden = true;
  shell.dialog.appendChild(fixRow);

  const actions = el('div', 'label-correction__actions');
  const confirm = el('button', 'panel-btn label-correction__confirm', 'Flag & save for review');
  confirm.type = 'button';
  confirm.dataset.testid = 'label-correction-confirm';
  confirm.disabled = true;
  const cancel = el('button', 'panel-btn', 'Cancel');
  cancel.type = 'button';
  cancel.dataset.testid = 'label-correction-cancel';
  cancel.addEventListener('click', () => shell.close());

  /** A correction with nothing corrected would flag the event and save duplicate frames
   * for no reason, so the button stays dead until something is actually chosen. */
  /** The remap as it stands, which both the estimate and the event-fix row read. */
  function currentRemap() {
    const remap = new Map();
    for (const { from, select } of pickers) {
      if (select.value !== KEEP) remap.set(from, select.value);
    }
    return remap;
  }

  function syncFixEventRow() {
    const correction = eventCorrectionFor(event, currentRemap(), labels);
    fixRow.hidden = !correction;
    if (!correction) return;
    // Named per slot, because "subject" and "target" are not interchangeable and a
    // rule like "Maple near Luna" can have both corrected at once.
    fixText.textContent = 'Also fix this event: '
      + correction.changes.map(c => `${c.slot} ${c.from} → ${c.to}`).join(', ');
  }

  function syncConfirm() {
    const chosen = pickers.filter(p => p.select.value !== KEEP).map(p => p.from);
    confirm.disabled = !chosen.length;
    syncFixEventRow();
    if (!chosen.length) {
      estimate.textContent = '';
      return;
    }
    const frames = estimateFrames?.(chosen);
    estimate.textContent = frames == null
      ? ''
      : `Will save ${frames} frame${frames === 1 ? '' : 's'} from this clip.`;
  }
  for (const picker of pickers) picker.select.addEventListener('change', syncConfirm);
  syncConfirm();

  confirm.addEventListener('click', async () => {
    const remap = currentRemap();
    if (!remap.size) return;
    // A hidden row is not an unchecked one: the box stays checked while irrelevant, so
    // the answer has to come from whether the row applies at all.
    const fixEvent = !fixRow.hidden && fixBox.checked;
    // Closed before the work rather than after: saving uploads a frame and writes
    // boxes, and a dialog left open over it invites a second click on a button whose
    // first click is still in flight.
    shell.close();
    await onConfirm?.(remap, { fixEvent });
  });

  actions.appendChild(cancel);
  actions.appendChild(confirm);
  shell.dialog.appendChild(actions);

  shell.open();
  return { close: () => shell.close() };
}
