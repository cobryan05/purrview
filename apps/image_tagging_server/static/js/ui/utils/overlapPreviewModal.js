/**
 * A sandbox for a rule's match threshold and margin.
 *
 * A rule's "match 0.6" is the one number in the watcher editor with no intuition behind
 * it: it is a fraction of an area the user never sees, under a relation whose choice
 * already confused them, and the only way to find out what it meant was to point a
 * camera at a cat and wait. This modal makes it a thing you drag: two boxes, the live
 * score for every relation, and a verdict that flips as they touch.
 *
 * Every number here is computed by the same formulas as `apps/helpers/watch/geometry.py`
 * — they are duplicated rather than fetched because the point is a readout that updates
 * while the pointer moves, and a round trip per frame would make the drag feel dead.
 * Keep the two in step; the Playwright test pins the values this file produces.
 */

import { createModalShell } from '/app-static/js/ui/utils/modalShell.js';
import { fetchReviewQueue, getRegionThumbnailUrl } from '/app-static/js/shared/api/review.js';

/**
 * The virtual frame the preview measures in. Pixel units, like the engine's, so the
 * formulas below are the Python ones unchanged; boxes render as percentages of the
 * stage so the modal can be any size.
 */
const FRAME_W = 1600;
const FRAME_H = 900;

/** Smallest box a drag may leave behind, so a box can never be lost by resizing. */
const MIN_SIDE = 40;

// Relations that compare geometry. `present` is arity 1 — there is nothing to preview,
// and the rule row hides the button for it.
const PREVIEW_RELATIONS = [
  { key: 'contains', text: 'is inside' },
  { key: 'overlaps', text: 'overlaps' },
  { key: 'centroid_in', text: 'has its centre in' },
  { key: 'iou', text: 'closely matches (same size)' },
  { key: 'distance', text: 'is near' },
];

/** Relations whose score is a fraction of an area, and so read as a percentage. */
const AREA_RELATIONS = new Set(['contains', 'overlaps', 'iou']);

// ── Geometry (mirrors apps/helpers/watch/geometry.py) ────────────────────────

const area = b => Math.max(0, b.x2 - b.x1) * Math.max(0, b.y2 - b.y1);

function intersection(a, b) {
  const x1 = Math.max(a.x1, b.x1);
  const y1 = Math.max(a.y1, b.y1);
  const x2 = Math.min(a.x2, b.x2);
  const y2 = Math.min(a.y2, b.y2);
  return x2 > x1 && y2 > y1 ? { x1, y1, x2, y2 } : null;
}

const intersectionArea = (a, b) => { const i = intersection(a, b); return i ? area(i) : 0; };

const iou = (a, b) => {
  const inter = intersectionArea(a, b);
  const union = area(a) + area(b) - inter;
  return union > 0 ? inter / union : 0;
};

/** How much of A is inside B, over A's own area. */
const containment = (a, b) => (area(a) > 0 ? intersectionArea(a, b) / area(a) : 0);

/** Overlap over the *smaller* box's area — symmetric, and the water-bowl relation. */
const overlapRatio = (a, b) => {
  const smaller = Math.min(area(a), area(b));
  return smaller > 0 ? intersectionArea(a, b) / smaller : 0;
};

const centre = b => ({ x: (b.x1 + b.x2) / 2, y: (b.y1 + b.y2) / 2 });

const boxScale = b => (Math.max(0, b.x2 - b.x1) + Math.max(0, b.y2 - b.y1)) / 2;

/**
 * Pixels to grow each edge by. Measured against the *smaller* box so the margin means
 * the same thing at any distance from the camera — see `geometry.margin_pad` for the
 * full argument. Each box takes half, so the number is the whole gap it closes.
 */
const marginPad = (margin, a, b) => (margin * Math.min(boxScale(a), boxScale(b))) / 2;

function inflate(box, pad) {
  let { x1, y1, x2, y2 } = box;
  x1 -= pad; x2 += pad; y1 -= pad; y2 += pad;
  if (x1 > x2) { x1 = x2 = (box.x1 + box.x2) / 2; }
  if (y1 > y2) { y1 = y2 = (box.y1 + box.y2) / 2; }
  return { x1, y1, x2, y2 };
}

const normalisedDistance = (a, b) => {
  const ca = centre(a); const cb = centre(b);
  return Math.hypot(ca.x - cb.x, ca.y - cb.y) / Math.hypot(FRAME_W, FRAME_H);
};

/**
 * Score a pair of boxes exactly as `evaluate_predicate` would, margin included.
 *
 * @returns {{score: number, fires: boolean, padded: {a: object, b: object}, pad: number}}
 */
function evaluate(relation, boxA, boxB, threshold, margin) {
  // `distance` compares centre points, which do not grow, so it takes no margin — the
  // backend rejects one rather than accepting a control that does nothing.
  const pad = relation === 'distance' ? 0 : marginPad(margin, boxA, boxB);
  const a = inflate(boxA, pad);
  const b = inflate(boxB, pad);

  if (relation === 'distance') {
    const dist = normalisedDistance(boxA, boxB);
    return { score: dist, fires: dist <= threshold, padded: { a: boxA, b: boxB }, pad: 0 };
  }
  if (relation === 'centroid_in') {
    // The subject contributes a point, so the whole margin widens the target.
    const target = inflate(boxB, 2 * pad);
    const c = centre(boxA);
    const inside = c.x >= target.x1 && c.x <= target.x2 && c.y >= target.y1 && c.y <= target.y2;
    return { score: inside ? 1 : 0, fires: inside, padded: { a: boxA, b: target }, pad };
  }
  const score = relation === 'iou' ? iou(a, b) : relation === 'overlaps' ? overlapRatio(a, b) : containment(a, b);
  return { score, fires: score >= threshold, padded: { a, b }, pad };
}

// ── DOM helpers ──────────────────────────────────────────────────────────────

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text != null) node.textContent = text;
  return node;
}

const pct = v => `${(v * 100).toFixed(0)}%`;

/**
 * The first bbox of a label, as a crop URL plus its real shape.
 *
 * Best-effort by design: no samples, no boxes drawn from the DB and the preview falls
 * back to plain rectangles. A failure here must never stop the modal opening — the
 * geometry is the feature and the picture is the garnish.
 *
 * @returns {Promise<?{url: string, aspect: number, name: string}>}
 */
async function fetchLabelSample(labelUuid) {
  if (!labelUuid) return null;
  try {
    const { images } = await fetchReviewQueue({
      labelUuids: [labelUuid],
      showAllBboxes: true,
      requireBboxes: true,
      tagUuids: [],
      limit: 5,
    });
    for (const img of images || []) {
      const bbox = (img.bboxes || []).find(b => b.labelUuid === labelUuid);
      if (!bbox || !img.filename) continue;
      // Guard the ratio: a zero-height box would collapse the preview box to a line.
      const aspect = bbox.height > 0 && bbox.width > 0 ? bbox.width / bbox.height : 1;
      // The region crop with no padding and no stroke, rather than the bbox thumbnail
      // the review grid uses. That one adds 20% context and draws the box's outline
      // inside the picture, so the preview showed a *second* smaller box within the
      // rectangle being dragged — and which edge the overlap was measured from stopped
      // being obvious, which is the entire job of this modal.
      const url = getRegionThumbnailUrl(
        img.filename,
        { x: bbox.x, y: bbox.y, width: bbox.width, height: bbox.height },
        { width: 300, pad: 0, outline: false },
      );
      return { url, aspect };
    }
  } catch {
    // Swallowed on purpose — see above.
  }
  return null;
}

// ── The modal ────────────────────────────────────────────────────────────────

/**
 * Open the match preview for one rule.
 *
 * @param {object} opts
 * @param {string} opts.relation - the rule's current relation.
 * @param {number} opts.threshold
 * @param {number} opts.margin
 * @param {?{uuid: string, name: string}} opts.subjectA - label to sample for box A.
 * @param {?{uuid: string, name: string}} opts.subjectB - label for box B, or null when
 *   subject B is a zone (a zone has no crops, and draws as a plain region).
 * @param {string} [opts.subjectBName] - what to call box B when it is not a label.
 * @param {(applied: {relation: string, threshold: number, margin: number}) => void} opts.onApply
 * @returns {{close: () => void}}
 */
export function openOverlapPreviewModal({
  relation = 'contains',
  threshold = 0.6,
  margin = 0,
  subjectA = null,
  subjectB = null,
  subjectBName = 'Target',
  onApply = null,
} = {}) {
  const shell = createModalShell();
  shell.dialog.classList.add('overlap-preview');
  shell.dialog.dataset.testid = 'overlap-preview';

  const state = {
    relation: PREVIEW_RELATIONS.some(r => r.key === relation) ? relation : 'contains',
    threshold,
    margin,
    // Two boxes that start overlapping a little: an empty stage with the boxes apart
    // reads as "nothing works", and the first thing anyone does is drag them together.
    a: { x1: 420, y1: 300, x2: 720, y2: 620 },
    b: { x1: 640, y1: 260, x2: 1140, y2: 700 },
  };

  shell.dialog.appendChild(el('h3', 'overlap-preview__title', 'Match preview'));
  shell.dialog.appendChild(el(
    'p',
    'overlap-preview__hint',
    'Drag the boxes, or their bottom-right corners to resize. The verdict below updates live.',
  ));

  // ── stage ──
  const stage = el('div', 'overlap-preview__stage');
  stage.dataset.testid = 'overlap-preview-stage';
  const interBox = el('div', 'overlap-preview__intersection');
  const haloA = el('div', 'overlap-preview__halo overlap-preview__halo--a');
  const haloB = el('div', 'overlap-preview__halo overlap-preview__halo--b');
  const centreDot = el('div', 'overlap-preview__centre');
  stage.append(haloA, haloB, interBox, centreDot);

  function buildBox(key, name) {
    const node = el('div', `overlap-preview__box overlap-preview__box--${key}`);
    node.dataset.testid = `overlap-preview-box-${key}`;
    node.appendChild(el('span', 'overlap-preview__box-name', name));
    const handle = el('span', 'overlap-preview__handle');
    handle.dataset.testid = `overlap-preview-handle-${key}`;
    node.appendChild(handle);
    stage.appendChild(node);
    attachDrag(node, handle, key);
    return node;
  }

  /**
   * Wire pointer drags for one box.
   *
   * Deltas are converted through the stage's live rect rather than cached: the modal is
   * resizable with the viewport, and a cached scale would make the box drift away from
   * the pointer after a window resize.
   */
  function attachDrag(node, handle, key) {
    let mode = null;
    let start = null;

    const toFrame = ev => {
      const rect = stage.getBoundingClientRect();
      return { x: ((ev.clientX - rect.left) / rect.width) * FRAME_W, y: ((ev.clientY - rect.top) / rect.height) * FRAME_H };
    };

    const onMove = ev => {
      if (!mode) return;
      const p = toFrame(ev);
      const box = state[key];
      if (mode === 'move') {
        const w = start.x2 - start.x1;
        const h = start.y2 - start.y1;
        // Clamped to the stage: a box dragged off the edge is unreachable, and its
        // score would keep changing from somewhere the user cannot see.
        const x1 = Math.min(Math.max(start.x1 + (p.x - start.px), 0), FRAME_W - w);
        const y1 = Math.min(Math.max(start.y1 + (p.y - start.py), 0), FRAME_H - h);
        Object.assign(box, { x1, y1, x2: x1 + w, y2: y1 + h });
      } else {
        box.x2 = Math.min(Math.max(p.x, box.x1 + MIN_SIDE), FRAME_W);
        box.y2 = Math.min(Math.max(p.y, box.y1 + MIN_SIDE), FRAME_H);
      }
      render();
    };

    const onUp = ev => {
      mode = null;
      node.releasePointerCapture?.(ev.pointerId);
      window.removeEventListener('pointermove', onMove);
      window.removeEventListener('pointerup', onUp);
    };

    const begin = (ev, which) => {
      ev.preventDefault();
      ev.stopPropagation();
      mode = which;
      const p = toFrame(ev);
      start = { ...state[key], px: p.x, py: p.y };
      node.setPointerCapture?.(ev.pointerId);
      window.addEventListener('pointermove', onMove);
      window.addEventListener('pointerup', onUp);
    };

    node.addEventListener('pointerdown', ev => begin(ev, 'move'));
    handle.addEventListener('pointerdown', ev => begin(ev, 'resize'));
  }

  const nameA = subjectA?.name || 'Subject';
  const nameB = subjectB?.name || subjectBName;
  const boxAEl = buildBox('a', nameA);
  const boxBEl = buildBox('b', nameB);
  shell.dialog.appendChild(stage);

  // ── verdict ──
  const verdict = el('div', 'overlap-preview__verdict');
  verdict.dataset.testid = 'overlap-preview-verdict';
  const scoreEl = el('strong', 'overlap-preview__score');
  scoreEl.dataset.testid = 'overlap-preview-score';
  const badge = el('span', 'overlap-preview__badge');
  badge.dataset.testid = 'overlap-preview-badge';
  const scoreLine = el('div', 'overlap-preview__score-line');
  scoreLine.append(scoreEl, badge);
  const meter = el('div', 'overlap-preview__meter');
  const meterFill = el('div', 'overlap-preview__meter-fill');
  const meterMark = el('div', 'overlap-preview__meter-mark');
  meter.append(meterFill, meterMark);
  const others = el('div', 'overlap-preview__others');
  others.dataset.testid = 'overlap-preview-others';
  verdict.append(scoreLine, meter, others);

  // ── controls ──
  const controls = el('div', 'overlap-preview__controls');

  const relationSelect = document.createElement('select');
  relationSelect.dataset.testid = 'overlap-preview-relation';
  for (const r of PREVIEW_RELATIONS) {
    const opt = document.createElement('option');
    opt.value = r.key;
    opt.textContent = r.text;
    relationSelect.appendChild(opt);
  }
  relationSelect.value = state.relation;
  relationSelect.addEventListener('change', () => { state.relation = relationSelect.value; render(); });

  function slider({ testid, min, max, step, value, onInput }) {
    const input = document.createElement('input');
    input.type = 'range';
    input.min = String(min);
    input.max = String(max);
    input.step = String(step);
    input.value = String(value);
    input.dataset.testid = testid;
    input.addEventListener('input', () => onInput(Number(input.value)));
    return input;
  }

  const thresholdValue = el('span', 'overlap-preview__value');
  thresholdValue.dataset.testid = 'overlap-preview-threshold-value';
  const thresholdSlider = slider({
    testid: 'overlap-preview-threshold',
    min: 0, max: 1, step: 0.01, value: state.threshold,
    onInput: v => { state.threshold = v; render(); },
  });

  const marginValue = el('span', 'overlap-preview__value');
  marginValue.dataset.testid = 'overlap-preview-margin-value';
  // -1 erodes a box to nothing and +2 fires across two subject-widths; the same bounds
  // the write model enforces, so the slider cannot produce a payload it would reject.
  const marginSlider = slider({
    testid: 'overlap-preview-margin',
    min: -1, max: 2, step: 0.05, value: state.margin,
    onInput: v => { state.margin = v; render(); },
  });

  function controlRow(labelText, input, valueEl, help) {
    const row = el('label', 'overlap-preview__control');
    row.append(el('span', 'overlap-preview__control-label', labelText), input, valueEl);
    if (help) row.title = help;
    return row;
  }

  controls.append(
    controlRow('Relation', relationSelect, el('span')),
    controlRow('Match', thresholdSlider, thresholdValue, 'How strongly the relation must hold before the rule fires'),
    controlRow(
      'Nearby',
      marginSlider,
      marginValue,
      'Widens both boxes before matching, measured in subject-widths, so “near” means '
      + 'the same thing close to the camera and far from it. Negative demands a deeper overlap.',
    ),
  );

  const marginNote = el('p', 'overlap-preview__hint overlap-preview__margin-note');
  marginNote.dataset.testid = 'overlap-preview-margin-note';
  controls.appendChild(marginNote);

  shell.dialog.append(verdict, controls);

  // ── footer ──
  const footer = el('div', 'overlap-preview__footer');
  const cancel = el('button', 'secondary', 'Close');
  cancel.type = 'button';
  cancel.addEventListener('click', () => shell.close());
  const apply = el('button', null, 'Use these settings');
  apply.type = 'button';
  apply.dataset.testid = 'overlap-preview-apply';
  apply.addEventListener('click', () => {
    onApply?.({ relation: state.relation, threshold: state.threshold, margin: marginApplies() ? state.margin : 0 });
    shell.close();
  });
  footer.append(cancel, apply);
  shell.dialog.appendChild(footer);

  /** Whether the current relation uses the margin at all. */
  const marginApplies = () => state.relation !== 'distance';

  // ── rendering ──

  /** Place one absolutely-positioned element from frame-unit coordinates. */
  function place(node, box) {
    node.style.left = `${(box.x1 / FRAME_W) * 100}%`;
    node.style.top = `${(box.y1 / FRAME_H) * 100}%`;
    node.style.width = `${((box.x2 - box.x1) / FRAME_W) * 100}%`;
    node.style.height = `${((box.y2 - box.y1) / FRAME_H) * 100}%`;
  }

  function render() {
    place(boxAEl, state.a);
    place(boxBEl, state.b);

    const result = evaluate(state.relation, state.a, state.b, state.threshold, state.margin);

    // Halos show what the margin actually did — the padded boxes the score was
    // measured on, which is the whole explanation of "nearby" in one glance.
    const showHalo = marginApplies() && Math.abs(state.margin) > 0.001;
    haloA.hidden = !showHalo || state.relation === 'centroid_in';
    haloB.hidden = !showHalo;
    if (showHalo) {
      place(haloA, result.padded.a);
      place(haloB, result.padded.b);
      haloA.classList.toggle('overlap-preview__halo--eroded', state.margin < 0);
      haloB.classList.toggle('overlap-preview__halo--eroded', state.margin < 0);
    }

    // The shaded intersection is of the *padded* boxes, because that is what the score
    // divides; shading the raw overlap would contradict the number beside it.
    const inter = state.relation === 'distance' ? null : intersection(result.padded.a, result.padded.b);
    interBox.hidden = !inter;
    if (inter) place(interBox, inter);

    // The centre dot is the whole subject of `centroid_in`, and noise elsewhere.
    centreDot.hidden = state.relation !== 'centroid_in';
    if (!centreDot.hidden) {
      const c = centre(state.a);
      centreDot.style.left = `${(c.x / FRAME_W) * 100}%`;
      centreDot.style.top = `${(c.y / FRAME_H) * 100}%`;
    }

    if (AREA_RELATIONS.has(state.relation)) {
      scoreEl.textContent = `${pct(result.score)} match`;
      meter.hidden = false;
      meterFill.style.width = pct(Math.min(1, result.score));
      meterMark.style.left = pct(Math.min(1, state.threshold));
    } else if (state.relation === 'distance') {
      scoreEl.textContent = `${pct(result.score)} of the frame apart`;
      meter.hidden = false;
      meterFill.style.width = pct(Math.min(1, result.score));
      meterMark.style.left = pct(Math.min(1, state.threshold));
    } else {
      scoreEl.textContent = result.fires ? `${nameA}’s centre is inside` : `${nameA}’s centre is outside`;
      meter.hidden = true;
    }

    badge.textContent = result.fires ? 'FIRES' : 'no match';
    badge.classList.toggle('is-firing', result.fires);

    // The other relations, always visible: the commonest tuning mistake is a rule on
    // `contains` that will never fire, and seeing `overlaps` at 60% beside it is the
    // fastest way to find that out.
    const pad = marginApplies() ? marginPad(state.margin, state.a, state.b) : 0;
    const pa = inflate(state.a, pad);
    const pb = inflate(state.b, pad);
    others.innerHTML = '';
    for (const [label, value] of [
      [`inside (${nameA} in ${nameB})`, containment(pa, pb)],
      ['overlaps', overlapRatio(pa, pb)],
      ['IoU', iou(pa, pb)],
    ]) {
      const chip = el('span', 'overlap-preview__chip');
      chip.append(el('span', 'overlap-preview__chip-label', label), el('span', null, pct(value)));
      others.appendChild(chip);
    }

    thresholdValue.textContent = pct(state.threshold);
    marginValue.textContent = state.margin === 0 ? 'off' : `${state.margin > 0 ? '+' : ''}${state.margin.toFixed(2)}×`;

    const marginRow = marginSlider.closest('.overlap-preview__control');
    if (marginRow) marginRow.hidden = !marginApplies();
    marginNote.textContent = !marginApplies()
      ? '“is near” already measures distance, so it takes no nearby margin.'
      : state.margin === 0
        ? 'Boxes are matched exactly as detected.'
        : state.margin > 0
          ? `Fires when the two are within about ${state.margin.toFixed(2)}× the smaller box’s width `
            + `— ${Math.round(pad * 2)}px apart at this size, and proportionally less for a smaller subject.`
          : `Demands a deeper overlap: each box is eroded by ${Math.round(-pad)}px at this size.`;
  }

  render();
  shell.open();

  // Sample crops are decoration, so they load after the modal is already interactive.
  // Stretched rather than letterboxed: the box is a geometry handle the user resizes at
  // will, and a crop that keeps its own aspect would leave the picture disagreeing with
  // the rectangle being measured.
  (async () => {
    const [sampleA, sampleB] = await Promise.all([
      fetchLabelSample(subjectA?.uuid),
      subjectB ? fetchLabelSample(subjectB.uuid) : Promise.resolve(null),
    ]);
    for (const [sample, node, key] of [[sampleA, boxAEl, 'a'], [sampleB, boxBEl, 'b']]) {
      if (!sample) continue;
      node.style.backgroundImage = `url("${sample.url}")`;
      node.classList.add('has-sample');
      // Seed the box with the sample's real shape — a cat is taller than it is wide,
      // and a square stand-in makes every overlap reading subtly unrepresentative.
      const box = state[key];
      const h = box.y2 - box.y1;
      box.x2 = Math.min(box.x1 + Math.max(MIN_SIDE, h * sample.aspect), FRAME_W);
      render();
    }
  })();

  return { close: shell.close };
}
