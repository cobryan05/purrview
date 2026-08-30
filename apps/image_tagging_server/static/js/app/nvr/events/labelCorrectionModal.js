/**
 * NVR → Events: "these labels are wrong, they were actually…".
 *
 * The correction half of the loop the NVR exists for. A watcher that calls the tabby
 * "Clover" will keep calling her Clover, and the only thing that fixes it is training
 * data saying otherwise — so the moment you *notice* is the moment worth capturing,
 * with the frame already in front of you and the boxes already drawn.
 *
 * The remap is per detected label rather than one label for the whole frame. A frame
 * routinely holds more than one thing: correcting "Clover → Luna" must not also rewrite
 * the correctly-detected water bowl next to her, and a single-label form would do
 * exactly that while looking like it had worked.
 *
 * What it does *not* do is decide anything about the boxes' geometry. The detections'
 * rectangles are kept as they are and written Unverified, exactly as
 * `saveFrameForAnnotation` writes any model output: a corrected label on a model's box
 * is still a guess about *where*, and the review queue is where that gets settled.
 */
import { createModalShell } from '/app-static/js/ui/utils/modalShell.js';

/** The select option meaning "this label was right, leave its boxes alone". */
const KEEP = '';

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text != null) node.textContent = text;
  return node;
}

/**
 * Group a frame's detections by the label the model gave them.
 *
 * Exported for its own sake: the counts are what make the dialog readable ("Clover
 * (2 boxes)"), and a frame where one label appears three times must offer one row, not
 * three identical ones.
 *
 * @param {Array<{labelUuid: ?string}>} boxes - overlay boxes for the current frame
 * @param {Map<string, Object>} labels - uuid → label, as `fetchLabels()` returns
 * @returns {Array<{labelUuid: string, name: string, count: number}>} biggest first
 */
export function groupDetectionsByLabel(boxes, labels) {
  const counts = new Map();
  for (const box of boxes ?? []) {
    if (!box?.labelUuid) continue;
    counts.set(box.labelUuid, (counts.get(box.labelUuid) ?? 0) + 1);
  }
  return [...counts.entries()]
    .map(([labelUuid, count]) => ({
      labelUuid,
      // A label the DB has never heard of is still worth showing as a row: it is
      // precisely the kind of stale detection someone wants to correct, and hiding it
      // would leave the frame uncorrectable for the one box that is wrong.
      name: labels?.get(labelUuid)?.name || labelUuid,
      count,
    }))
    .sort((a, b) => b.count - a.count || a.name.localeCompare(b.name));
}

/**
 * Apply a remap to a frame's boxes.
 *
 * Returns a new array — the caller's overlay boxes belong to the running redraw loop
 * and must not be rewritten under it.
 *
 * @param {Array<Object>} boxes
 * @param {Map<string, string>} remap - old label uuid → new label uuid
 * @returns {Array<Object>}
 */
export function applyLabelRemap(boxes, remap) {
  return (boxes ?? []).map(box => (
    box?.labelUuid && remap.has(box.labelUuid)
      ? { ...box, labelUuid: remap.get(box.labelUuid) }
      : box
  ));
}

/**
 * Ask which of this frame's detected labels were wrong, and what they should have been.
 *
 * @param {Object} opts
 * @param {Array<{labelUuid: string, name: string, count: number}>} opts.detected - from groupDetectionsByLabel
 * @param {Map<string, Object>} opts.labels - every label the DB knows, uuid → label
 * @param {(remap: Map<string, string>) => (void | Promise<void>)} opts.onConfirm - non-empty remap
 * @returns {{close: () => void}}
 */
export function openLabelCorrectionModal({ detected, labels, onConfirm }) {
  const shell = createModalShell();
  shell.dialog.classList.add('label-correction');
  shell.dialog.dataset.testid = 'label-correction-modal';

  shell.dialog.appendChild(el('h3', 'label-correction__title', 'Which labels are wrong?'));
  shell.dialog.appendChild(el('p', 'label-correction__note',
    'The frame is saved with these boxes, relabelled and left unverified for review. '
    + 'The event is flagged so you can find it again.'));

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
    line.appendChild(el('span', 'label-correction__from',
      `${row.name} (${row.count} box${row.count === 1 ? '' : 'es'})`));

    const select = document.createElement('select');
    select.className = 'label-correction__to';
    select.dataset.testid = 'label-correction-select';
    select.dataset.labelUuid = row.labelUuid;
    const keep = document.createElement('option');
    keep.value = KEEP;
    keep.textContent = '(leave as-is)';
    select.appendChild(keep);
    for (const label of allLabels) {
      if (label.uuid === row.labelUuid) continue;
      const option = document.createElement('option');
      option.value = label.uuid;
      option.textContent = label.name;
      select.appendChild(option);
    }
    line.appendChild(select);
    rows.appendChild(line);
    pickers.push({ from: row.labelUuid, select });
  }
  shell.dialog.appendChild(rows);

  const actions = el('div', 'label-correction__actions');
  const confirm = el('button', 'panel-btn label-correction__confirm', 'Flag & save for review');
  confirm.type = 'button';
  confirm.dataset.testid = 'label-correction-confirm';
  confirm.disabled = true;
  const cancel = el('button', 'panel-btn', 'Cancel');
  cancel.type = 'button';
  cancel.dataset.testid = 'label-correction-cancel';
  cancel.addEventListener('click', () => shell.close());

  /** A correction with nothing corrected would flag the event and save a duplicate
   * frame for no reason, so the button stays dead until something is actually chosen. */
  function syncConfirm() {
    confirm.disabled = !pickers.some(p => p.select.value !== KEEP);
  }
  for (const picker of pickers) picker.select.addEventListener('change', syncConfirm);

  confirm.addEventListener('click', async () => {
    const remap = new Map();
    for (const { from, select } of pickers) {
      if (select.value !== KEEP) remap.set(from, select.value);
    }
    if (!remap.size) return;
    // Closed before the work rather than after: saving uploads a frame and writes
    // boxes, and a dialog left open over it invites a second click on a button whose
    // first click is still in flight.
    shell.close();
    await onConfirm?.(remap);
  });

  actions.appendChild(cancel);
  actions.appendChild(confirm);
  shell.dialog.appendChild(actions);

  shell.open();
  return { close: () => shell.close() };
}
