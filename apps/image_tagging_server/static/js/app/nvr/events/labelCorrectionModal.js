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
 * The select option meaning "this label was right, but the detector was barely sure of
 * it — collect the frames it hedged on".
 *
 * The third answer to "what did the watcher get wrong?", and the one the other two
 * cannot express: the identity is correct and the box is real, so there is nothing to
 * remap and nothing to delete, yet a label the model only ever scrapes past the floor
 * on is one missed frame away from the event never firing. Those frames are training
 * data of exactly the kind a correct-but-weak detection never otherwise produces.
 *
 * Carried in the same remap as a relabel so one dialog, one spread and one save covers
 * a clip where one label was wrong and another was merely weak. `applyLabelRemap`
 * leaves its boxes untouched; the picker aims only at the weak frames.
 */
export const WEAK = '__low_conf__';

/**
 * Confidence at or above which a box is not worth collecting for being weak.
 *
 * The watcher's own inference floor is far lower (the tracker needs the weak boxes),
 * so everything between the floor and this is a detection the model made and hedged
 * on. Above it the box tells the trainer nothing it does not already know.
 */
export const WEAK_CONF_MAX = 0.6;

/**
 * The lowest confidence any of these labels' boxes carries in a frame.
 *
 * The *lowest*, not the highest: a frame holding a confident box and a hedged one of
 * the same label still holds a hedged one, and that box is the reason to collect it.
 *
 * @param {Object} frame - a sidecar frame
 * @param {Set<string>} wanted
 * @returns {?number} null when the frame holds none of them, or none carries a conf
 */
function weakestConf(frame, wanted) {
  let worst = null;
  for (const box of frame?.boxes ?? []) {
    if (!wanted.has(box?.labelUuid)) continue;
    const c = Number(box.conf);
    if (Number.isFinite(c) && (worst == null || c < worst)) worst = c;
  }
  return worst;
}

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
 * A `WEAK` target never touches the event row either, for the opposite reason: the
 * identity it already carries is the one being confirmed.
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
    if (!ref || !remap.has(ref)) continue;
    if (remap.get(ref) === NO_BBOX || remap.get(ref) === WEAK) continue;
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
 * `weakFrames` is how many of those frames hold a box the detector hedged on, which is
 * the only number saying whether "collect its low-confidence frames" has anything to
 * collect — and the reason that option is offered on one row and not another.
 *
 * @returns {Array<{labelUuid: string, name: string, boxes: number, frames: number, weakFrames: number}>} biggest first
 */
export function labelTallyAcrossClip(doc, labels, { weakConfMax = WEAK_CONF_MAX } = {}) {
  const tally = new Map();
  for (const frame of doc?.frames ?? []) {
    const seenHere = new Set();
    for (const box of frame?.boxes ?? []) {
      if (!box?.labelUuid) continue;
      let row = tally.get(box.labelUuid);
      if (!row) tally.set(box.labelUuid, (row = { boxes: 0, frames: 0, weakFrames: 0 }));
      row.boxes += 1;
      // Counted once per frame however many boxes of it are in that frame — otherwise
      // "frames" would say 12 for a clip that only has 4.
      if (!seenHere.has(box.labelUuid)) {
        seenHere.add(box.labelUuid);
        row.frames += 1;
      }
    }
    // After the boxes, not inside them: a frame counts as weak if *any* of the label's
    // boxes in it is weak, and the hedged one is routinely not the first.
    for (const labelUuid of seenHere) {
      const worst = weakestConf(frame, new Set([labelUuid]));
      if (worst != null && worst < weakConfMax) tally.get(labelUuid).weakFrames += 1;
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
 * With `maxConf`, only the frames holding one the detector hedged on — the candidate
 * set for a `WEAK` collection, which must never reach for the frames where the model
 * was sure.
 *
 * @param {Object} doc - a clip overlay sidecar
 * @param {Iterable<string>} labelUuids
 * @param {Object} [opts]
 * @param {number} [opts.maxConf] - keep only frames whose weakest box of these labels
 *   is below this; a box carrying no confidence at all never qualifies
 * @returns {number[]} clip-relative seconds, ascending
 */
export function framesHoldingLabels(doc, labelUuids, { maxConf = null } = {}) {
  const wanted = new Set(labelUuids ?? []);
  const times = [];
  for (const frame of doc?.frames ?? []) {
    if (typeof frame?.t !== 'number') continue;
    if (maxConf == null) {
      if ((frame.boxes ?? []).some(b => wanted.has(b?.labelUuid))) times.push(frame.t);
      continue;
    }
    const worst = weakestConf(frame, wanted);
    if (worst != null && worst < maxConf) times.push(frame.t);
  }
  return times.sort((a, b) => a - b);
}

/**
 * The weak frames of these labels, weakest first.
 *
 * The `WEAK` counterpart to `hardMomentsFor`: there is no incident to rank by, so the
 * ranking is the confidence itself. Handed to `pickTrainingFrames` as priority, it
 * ensures the frames the model nearly missed survive the even spread rather than being
 * thinned away by whichever neighbour sat nearer the fencepost.
 *
 * @param {Object} doc - a clip overlay sidecar
 * @param {Iterable<string>} labelUuids
 * @param {Object} [opts]
 * @param {number} [opts.maxConf]
 * @returns {Array<{t: number, weight: number, why: string}>} weakest first
 */
export function weakMomentsFor(doc, labelUuids, { maxConf = WEAK_CONF_MAX } = {}) {
  const wanted = new Set(labelUuids ?? []);
  const moments = [];
  for (const frame of doc?.frames ?? []) {
    if (typeof frame?.t !== 'number') continue;
    const worst = weakestConf(frame, wanted);
    if (worst == null || worst >= maxConf) continue;
    moments.push({ t: frame.t, weight: 1 + (maxConf - worst), why: 'low_confidence' });
  }
  return moments.sort((a, b) => b.weight - a.weight || a.t - b.t);
}

/**
 * Slots the spread pass always keeps, however many hard moments there are.
 *
 * The first and last frames of a visit are where the animal is entering and leaving —
 * poses no blip will ever point at — so hard moments may fill the cap only this far.
 */
export const SPREAD_RESERVE = 2;

/**
 * How many of the most- and least-confident wrong-label frames count as hard moments.
 *
 * Both ends, deliberately. A detector that was *sure* and wrong is the most damaging
 * case and the strongest correction; one that was barely sure is the decision
 * boundary. The middle is where the even spread already looks.
 */
export const CONFIDENCE_EXTREMES = 2;

/** A blip moment must land within this of a frame holding the label to count. */
const SNAP_TOLERANCE_S = 1.0;

/**
 * The moments in a clip where correcting these labels teaches the model the most.
 *
 * Two sources, ranked so a blip always outranks a confidence pick:
 *
 *  * **Blips** (`blips.py`): identity flips, dropouts, occlusion merges — the frames the
 *    analyser already decided the detector failed on. Onset, middle and recovery of each
 *    incident, the same moments `harvest_times` takes, because the boundaries are where
 *    the decision actually changed. Weighted by `severity`: a blip the detector was
 *    sure about is worse than one it hedged on. Tracker-fault blips are included too —
 *    the harvest excludes them because *the detector* was right, but a correction is
 *    the user saying it was not.
 *  * **Confidence extremes** of the wrong-label boxes themselves, `CONFIDENCE_EXTREMES`
 *    from each end.
 *
 * Every moment is snapped to a frame that actually holds one of the labels, so the
 * picker can return it as-is and the capture lands on a sample with boxes to relabel.
 *
 * @param {Object} doc - a clip overlay sidecar
 * @param {Iterable<string>} labelUuids - the labels being corrected
 * @param {Array<Object>} [blips] - this event's blips, as `fetchEventBlips` returns them
 * @returns {Array<{t: number, weight: number, why: string}>} heaviest first, one per frame
 */
export function hardMomentsFor(doc, labelUuids, blips = []) {
  const wanted = new Set(labelUuids ?? []);
  const times = framesHoldingLabels(doc, wanted);
  if (!times.length) return [];

  const snap = t => {
    let best = null;
    for (const cand of times) {
      if (best == null || Math.abs(cand - t) < Math.abs(best - t)) best = cand;
    }
    return best != null && Math.abs(best - t) <= SNAP_TOLERANCE_S ? best : null;
  };

  /** @type {Map<number, {t: number, weight: number, why: string}>} */
  const byFrame = new Map();
  const offer = (t, weight, why) => {
    const at = snap(t);
    if (at == null) return;
    const have = byFrame.get(at);
    if (!have || have.weight < weight) byFrame.set(at, { t: at, weight, why });
  };

  for (const blip of blips ?? []) {
    const detail = blip?.detail ?? {};
    const named = [detail.sawLabel, detail.modalLabel, detail.absorbedByLabel].filter(Boolean);
    // A blip that names labels is only about this correction if one of them is being
    // corrected; one that names none (a raw-detection ghost) is offered regardless.
    if (named.length && !named.some(u => wanted.has(u))) continue;
    const start = Number(blip.startS);
    const end = Number(blip.endS);
    if (!Number.isFinite(start)) continue;
    const severity = Number(blip.severity) || 0;
    const weight = 2 + severity;
    const moments = Number.isFinite(end) && end > start ? [start, (start + end) / 2, end] : [start];
    for (const t of moments) offer(t, weight, blip.kind || 'blip');
  }

  // Per frame, the most confident box of a corrected label — that is the claim being
  // corrected, and the number that says how badly.
  const confAt = [];
  for (const frame of doc?.frames ?? []) {
    if (typeof frame?.t !== 'number') continue;
    let conf = null;
    for (const box of frame.boxes ?? []) {
      if (!wanted.has(box?.labelUuid)) continue;
      const c = Number(box.conf);
      if (Number.isFinite(c) && (conf == null || c > conf)) conf = c;
    }
    if (conf != null) confAt.push({ t: frame.t, conf });
  }
  // Strictly beyond the median, or it is not an extreme: on a clip where every box
  // scored the same, "the top two" would be whichever came first, filed as if the
  // model had been unusually sure of them.
  const byConf = [...confAt].sort((a, b) => b.conf - a.conf);
  const median = byConf.length ? byConf[Math.floor(byConf.length / 2)].conf : 0;
  for (const { t, conf } of byConf.slice(0, CONFIDENCE_EXTREMES)) {
    if (conf > median) offer(t, 1 + conf, 'confident_wrong');
  }
  for (const { t, conf } of byConf.slice(-CONFIDENCE_EXTREMES).reverse()) {
    if (conf < median) offer(t, 1 + (1 - conf), 'unsure');
  }

  return [...byFrame.values()].sort((a, b) => b.weight - a.weight || a.t - b.t);
}

/**
 * Choose the subset of frames actually worth saving.
 *
 * Four passes, in this order, because they answer different objections:
 *
 *  1. **Hard moments first.** Frames `priority` points at — where the detector was
 *     caught being wrong — are taken heaviest-first, as long as each sits `minGapS`
 *     from the ones already taken. They may fill the cap only up to `SPREAD_RESERVE`
 *     short of it: a set that is nothing but the incident is a set with no idea what
 *     the rest of the visit looked like.
 *  2. **Thin by time.** Frames closer together than `minGapS` — to each other or to a
 *     hard moment — show the same pose. This runs before the cap so the cap is spent
 *     on *distinct* frames rather than on whichever burst happened to be densest.
 *  3. **Spread to the cap.** If more survive than the remaining cap allows, take evenly
 *     spaced ones across the whole span — first and last included, since the beginning
 *     and end of a visit are where the animal is entering and leaving, and those are the
 *     poses the middle of the clip does not have.
 *  4. **Restore the endpoints.** A hard moment within `minGapS` of the true first or
 *     last frame can keep both of the above passes from ever picking it. Swapped back
 *     in over whichever kept frame sits closest, so this is the one guarantee that
 *     holds regardless of what competed for the slot next to it.
 *
 * Taking the *first* N instead would hand the trainer the first two seconds of every
 * event and nothing else, which is a worse set than a random sample and looks fine.
 *
 * @param {number[]} times - ascending, as from framesHoldingLabels
 * @param {Object} [opts]
 * @param {number} [opts.maxFrames]
 * @param {number} [opts.minGapS]
 * @param {Array<{t: number, weight: number}>} [opts.priority] - as from hardMomentsFor;
 *   `t` must be one of `times`
 * @returns {number[]} ascending
 */
export function pickTrainingFrames(times, { maxFrames = MAX_TRAINING_FRAMES, minGapS = MIN_FRAME_GAP_S, priority = [] } = {}) {
  const all = times ?? [];
  const have = new Set(all);
  const picked = [];
  const clear = t => picked.every(p => Math.abs(p - t) >= minGapS);

  // A short clip has room for everything; the reserve only matters once frames compete.
  const priorityCap = all.length > maxFrames ? Math.max(0, maxFrames - SPREAD_RESERVE) : maxFrames;
  const ranked = [...(priority ?? [])].sort((a, b) => b.weight - a.weight || a.t - b.t);
  for (const { t } of ranked) {
    if (picked.length >= priorityCap) break;
    if (have.has(t) && clear(t)) picked.push(t);
  }

  const spaced = [];
  for (const t of all) {
    if (!clear(t)) continue;
    if (!spaced.length || t - spaced[spaced.length - 1] >= minGapS) spaced.push(t);
  }

  const room = Math.max(0, maxFrames - picked.length);
  let spread;
  if (spaced.length <= room) {
    spread = spaced;
  } else if (room <= 1) {
    spread = spaced.slice(0, room);
  } else {
    // Fenceposts, not buckets: dividing by (room - 1) is what puts the last pick on the
    // final frame rather than one step short of it.
    const step = (spaced.length - 1) / (room - 1);
    spread = [];
    for (let i = 0; i < room; i++) spread.push(spaced[Math.round(i * step)]);
  }
  const result = [...new Set([...picked, ...spread])].sort((a, b) => a - b);

  // Guarantee the entering/exiting pose even when a hard moment claimed the slot next
  // to it, crowding the endpoint out of both `clear()` checks above. Swap in over the
  // nearest kept frame within `minGapS` — that frame was already at least `minGapS`
  // from its other neighbor, and the true endpoint is at least as far past it, so the
  // invariant survives the swap — or append outright if the cap still has room.
  const restoreEndpoint = t => {
    if (t == null || result.includes(t)) return;
    let nearest = -1;
    let nearestDist = Infinity;
    result.forEach((p, i) => {
      const d = Math.abs(p - t);
      if (d < nearestDist) { nearestDist = d; nearest = i; }
    });
    if (nearest !== -1 && nearestDist < minGapS) result[nearest] = t;
    else if (result.length < maxFrames) result.push(t);
    else return;
    result.sort((a, b) => a - b);
  };
  restoreEndpoint(all.length ? all[0] : null);
  restoreEndpoint(all.length ? all[all.length - 1] : null);

  return result;
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
      // `WEAK` is a collection instruction, not a correction: the label it names is the
      // right one, so its boxes pass through exactly as the detector drew them.
      box?.labelUuid && remap.has(box.labelUuid) && remap.get(box.labelUuid) !== WEAK
        ? { ...box, labelUuid: remap.get(box.labelUuid) }
        : box
    ));
}

function boxIou(a, b) {
  const [ax1, ay1, ax2, ay2] = a;
  const [bx1, by1, bx2, by2] = b;
  const ix1 = Math.max(ax1, bx1), iy1 = Math.max(ay1, by1);
  const ix2 = Math.min(ax2, bx2), iy2 = Math.min(ay2, by2);
  const iw = Math.max(0, ix2 - ix1), ih = Math.max(0, iy2 - iy1);
  const inter = iw * ih;
  if (inter <= 0) return 0;
  const areaA = Math.max(0, ax2 - ax1) * Math.max(0, ay2 - ay1);
  const areaB = Math.max(0, bx2 - bx1) * Math.max(0, by2 - by1);
  const union = areaA + areaB - inter;
  return union > 0 ? inter / union : 0;
}

/**
 * Backstop against the tracker occasionally running two live tracks on one physical
 * animal (a brief detector dropout breaks association and a new track starts beside
 * the old one instead of continuing it — see `merge_duplicate_tracks` on the Python
 * side, which fixes this upstream but can't cover every existing/undeployed watcher).
 * Applied AFTER the remap, so two tracks that disagreed on identity before correction
 * (one "Maple", one "Willow") and both got corrected to the same final label are caught
 * too, not just tracks that already agreed.
 *
 * Greedy NMS by confidence: among same-final-label boxes, a box overlapping an
 * already-kept one at or above `iouThreshold` is dropped rather than kept, since at
 * that overlap it is essentially always the same animal, not two side by side.
 *
 * @param {Array<Object>} boxes - `{labelUuid, bbox:[x1,y1,x2,y2], confidence}` entries
 * @param {number} [iouThreshold]
 * @returns {Array<Object>}
 */
export function dedupeOverlappingBoxes(boxes, iouThreshold = 0.75) {
  const ordered = [...(boxes ?? [])].sort((a, b) => (b.confidence ?? 0) - (a.confidence ?? 0));
  const kept = [];
  for (const box of ordered) {
    const dupe = box.labelUuid
      && kept.some(k => k.labelUuid === box.labelUuid && boxIou(box.bbox, k.bbox) >= iouThreshold);
    if (!dupe) kept.push(box);
  }
  // Restore the caller's original ordering rather than the confidence sort above.
  const keptSet = new Set(kept);
  return (boxes ?? []).filter(box => keptSet.has(box));
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
 * @param {(remap: Map<string, string>) => (number | {frames: number, hard: number})} [opts.estimateFrames] -
 *   how many frames a given selection would harvest, for the live count; `hard` is how
 *   many of them sit on a moment `hardMomentsFor` or `weakMomentsFor` pointed at. Given
 *   the whole remap, not only its keys: a `WEAK` label harvests a different set of
 *   frames from a relabelled one
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
    + 'detection that should never have been made; "right label, collect weak frames" '
    + 'keeps both and harvests only the moments the detector hedged on. The event is '
    + 'flagged so you can find it again.'));

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
      + `in ${row.frames} frame${row.frames === 1 ? '' : 's'}`
      // The number that decides whether the weak-frame option is worth choosing, said
      // on the row rather than discovered from an estimate of zero after choosing it.
      + (row.weakFrames ? `, ${row.weakFrames} low-confidence` : '')));

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
    // Offered only where there is something to collect. A row whose every box scored
    // high can choose it and harvest nothing, and an option that can only ever produce
    // an estimate of zero teaches the reader that the dialog lies.
    if (row.weakFrames) {
      const weak = document.createElement('option');
      weak.value = WEAK;
      weak.textContent = `Right label — collect its ${row.weakFrames} low-confidence frame`
        + (row.weakFrames === 1 ? '' : 's');
      select.appendChild(weak);
    }
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
    const remap = currentRemap();
    confirm.disabled = !remap.size;
    syncFixEventRow();
    if (!remap.size) {
      estimate.textContent = '';
      return;
    }
    const est = estimateFrames?.(remap);
    const frames = typeof est === 'number' ? est : est?.frames;
    const hard = typeof est === 'number' ? 0 : (est?.hard ?? 0);
    // A clip routinely stops short of the event its sidecar describes. Said here rather
    // than discovered afterwards: "why only 3?" is asked at the moment of agreeing to it.
    const beyond = typeof est === 'number' ? 0 : (est?.beyondClip ?? 0);
    estimate.textContent = frames == null
      ? ''
      : `Will save ${frames} frame${frames === 1 ? '' : 's'} from this clip`
        + (hard ? `, ${hard} at moments the detector struggled.` : '.')
        + (beyond ? ` ${beyond} labelled moment${beyond === 1 ? ' is' : 's are'} past the end of the recording — `
          + 'the clip is shorter than the event, so they cannot be captured.' : '');
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
