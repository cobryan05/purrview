/**
 * How a tracking blip describes itself, in words.
 *
 * Shared rather than duplicated because two surfaces now name the same findings — the
 * event modal, where you decide whether a blip is worth harvesting, and the preview
 * panel, where you judge the frames it produced. Those two have to agree: "detector
 * dropout" in one and "lost by the detector" in the other reads as two different
 * problems, and the whole point of the claim text is that a reviewer can carry it from
 * the proposal back to the clip.
 *
 * The kind strings themselves are the `KIND_*` constants in
 * `apps/helpers/recordings/blips.py`. An unknown kind falls through to the raw string
 * rather than to a placeholder: a finding the UI has no wording for is still a finding,
 * and showing "unknown" would hide it.
 */

/**
 * The one kind the UI branches on by name.
 *
 * `identity_switch` is attributed to the tracker and never harvested, so it gets a
 * before/after compare — "was this the same animal, or a real identity change?" —
 * where every other kind gets a harvest checkbox.
 */
export const KIND_IDENTITY_SWITCH = 'identity_switch';

/** Human-readable names for the blip kinds the analyser emits. */
export const BLIP_KIND_TEXT = {
  identity_flip: 'identity flip',
  identity_switch: 'identity switch',
  detector_dropout: 'detector dropout',
  full_dropout: 'lost entirely',
  tracker_dropout: 'tracker dropout',
  occlusion_merge: 'merged while overlapping',
  ghost: 'false positive',
  confidence_dip: 'confidence dip',
};

/** @param {string} kind @returns {string} */
export function blipKindText(kind) {
  return BLIP_KIND_TEXT[kind] ?? kind ?? 'tracking issue';
}

/**
 * What the analyser is claiming about one blip, as a sentence a human can check.
 *
 * This is the load-bearing half of the review: a cropped still judged on its own is the
 * same ambiguity that produced the mistake, whereas "carried cat_a for 121 samples and
 * showed cat_b for 9" is a claim you can agree or disagree with while watching the clip.
 *
 * Everything here is optional, because `detail_json` is display detail and the analyser
 * fills in only what its detector actually knows. Nothing is invented to fill a gap —
 * a claim with a number missing is weaker than one without, and a made-up number is
 * worse than either.
 *
 * @param {Object} blip - an EventBlipRead, camelCased by the API
 * @param {Map<string, {name: string}>} [labels] - to name the label uuids in `detail`
 * @returns {string} '' when there is nothing honest to say
 */
export function describeBlip(blip, labels = new Map()) {
  const detail = blip?.detailJson ?? {};
  const name = uuid => labels.get(uuid)?.name ?? null;
  const modal = name(detail.modalLabel);
  const saw = name(detail.sawLabel);
  const parts = [];

  switch (blip?.kind) {
    case 'identity_flip':
    case 'identity_switch':
      // Only stated with both sides known. "9 of 121 frames disagree" is a different
      // story from "9 frames disagree", and the total is what carries the weight.
      if (modal && saw) parts.push(`carried ${modal}, showed ${saw}`);
      if (detail.samples != null && detail.trackSamples != null) {
        parts.push(`${detail.samples} of ${detail.trackSamples} frames`);
      }
      break;
    case 'occlusion_merge':
      if (modal) parts.push(`${modal} disappeared into another track`);
      if (detail.absorbedByTrackId != null) {
        const into = name(detail.absorbedByLabel);
        parts.push(`absorbed by track #${detail.absorbedByTrackId}${into ? ` (${into})` : ''}`);
      }
      break;
    default:
      // The dropouts, the ghost and the dip all say the same thing: this label, for
      // this many frames.
      if (modal) parts.push(modal);
      if (detail.samples != null) parts.push(`${detail.samples} frames`);
      break;
  }
  return parts.join(' · ');
}

/**
 * Why a blip cannot be harvested, in the user's terms — or null when it can.
 *
 * Says the reason rather than just disabling the checkbox. A tracker fault is not an
 * error and not a lesser finding; it is a frame the detector already got right, and
 * training on it would teach the model something it knows while diluting the cases it
 * does not. A user who selected one deserves that sentence, not a greyed-out box.
 *
 * @param {Object} blip
 * @returns {?string}
 */
export function harvestBlocker(blip) {
  if (blip?.state === 'expired') return 'the clip has been deleted, so there are no frames left to pull';
  if (blip?.state === 'harvested') return 'already harvested';
  if (!blip?.harvestable) {
    return blip?.fault === 'tracker'
      ? 'the tracker was at fault, so the detector already got these frames right'
      : 'these frames are diagnostic only';
  }
  return null;
}

/**
 * Whether an `identity_switch` still has a question worth asking.
 *
 * Shared by the per-event panel and the cross-event queue so one rule decides what
 * counts as pending: a switch already decided is not re-offered, and an expired one has
 * no frames left to compare. `harvestBlocker` cannot answer this — every switch is
 * unharvestable by construction, which is exactly why it needs its own review.
 *
 * @param {Object} blip
 * @returns {boolean}
 */
export function switchReviewable(blip) {
  return blip?.kind === KIND_IDENTITY_SWITCH
    && blip?.state !== 'dismissed'
    && blip?.state !== 'harvested'
    && blip?.state !== 'expired';
}
