/**
 * Copy / paste of bboxes between images.
 *
 * The same subjects sit in the same places across long runs of frames from one
 * camera, so the fastest way to annotate frame N+1 is to carry frame N's boxes
 * over and only draw what is new.
 *
 * Coordinates are stored normalized (0..1), which is what makes pasting across
 * differently-sized images work at all. Across a different *aspect ratio* the
 * boxes land proportionally rather than pixel-identically — correct for frames
 * from the same source, and the only thing that could be correct without
 * knowing the two images are related.
 *
 * The clipboard is a module-level singleton, so it is shared by every canvas on
 * the page: copy on the main canvas, paste inside the review modal's editor.
 * It is deliberately in-memory only — it does not survive a reload, and it is
 * not the OS clipboard.
 */

import { events, EventTypes } from '/app-static/js/shared/events/index.js';
import { createUIBBbox } from '/app-static/js/shared/domain/image/mapper.js';
import { generateUUID } from '/app-static/js/ui/utils/utils.js';
import { Logger, toast } from '/app-static/js/ui/utils/index.js';
import { fetchLabelByUuid } from '/app-static/js/shared/api/labels.js';
import { BboxView } from './views/bboxView.js';
import { computeIou } from './geometry.js';
import { getSelectedViews, setSelection } from './selection.js';
import { syncBboxDraggable } from './tools.js';

/**
 * A pasted box is considered "already there" when an existing box on the target
 * image carries the same label and overlaps it at least this much.
 *
 * Lower than INFERENCE_DUPLICATE_IOU_THRESHOLD (0.7) on purpose: that one
 * decides whether a detector re-found the same object in the *same* image,
 * where agreement is tight. Here the two boxes come from different frames, so
 * the subject has usually drifted a little between them and 0.7 would call an
 * obvious repeat a new box.
 */
export const PASTE_DUPLICATE_IOU_THRESHOLD = 0.5;

/** @type {Array<object>} Detached copies of the last copied bboxes. */
let _clipboard = [];

/** How many bboxes are on the clipboard. */
export function clipboardSize() {
  return _clipboard.length;
}

/**
 * Snapshot one bbox into a clipboard entry.
 *
 * Everything is deep-copied: the entry has to survive the source image being
 * navigated away from and its BboxView destroyed.
 *
 * `confidence` is deliberately dropped. It describes one detector's opinion
 * about one crop of one frame, and carrying it to a box the user placed by hand
 * on a different image would put a fabricated percentage in the label text.
 */
export function toClipboardEntry(uiBBox) {
  return {
    x: uiBBox.x,
    y: uiBBox.y,
    width: uiBBox.width,
    height: uiBBox.height,
    label: uiBBox.label ? { ...uiBBox.label } : null,
    tags: (uiBBox.tags ?? []).map(t => ({ ...t })),
    metadataJson: uiBBox.metadataJson ? { ...uiBBox.metadataJson } : undefined,
  };
}

/**
 * Copy the current selection to the clipboard.
 * @param {import('@image_tagging_types').ImageTaggingRuntime} runtime
 * @returns {number} how many bboxes were copied
 */
export function copySelectedBboxes(runtime) {
  const views = getSelectedViews(runtime);
  if (!views.length) {
    toast('Nothing selected to copy', 'warning');
    return 0;
  }
  return setClipboard(views.map(v => toClipboardEntry(v.uiBBox)));
}

/**
 * Replace the clipboard with the given entries (built by `toClipboardEntry`).
 *
 * The context-menu path uses this rather than `copySelectedBboxes`, because the
 * menu's targets are not always the selection — right-clicking an unselected
 * box acts on that box alone.
 *
 * @param {Array<object>} entries
 * @returns {number} how many entries were stored
 */
export function setClipboard(entries) {
  _clipboard = entries.filter(Boolean);
  if (!_clipboard.length) return 0;
  toast(`Copied ${_clipboard.length} bbox${_clipboard.length === 1 ? '' : 'es'}`);
  return _clipboard.length;
}

/**
 * True when `entry` duplicates something already on the image: same label, and
 * overlapping enough to be the same subject.
 *
 * Label identity is by uuid. Two unlabeled boxes count as the same label, so
 * pasting a bare box onto another bare box in the same spot is still a
 * duplicate — that is the case where a stacked pair is hardest to notice.
 */
function _isDuplicate(entry, existingBboxes) {
  const entryLabel = entry.label?.uuid ?? null;
  for (const existing of existingBboxes) {
    if ((existing.label?.uuid ?? null) !== entryLabel) continue;
    if (computeIou(entry, existing) >= PASTE_DUPLICATE_IOU_THRESHOLD) return true;
  }
  return false;
}

/**
 * Paste the clipboard onto the runtime's current image.
 *
 * @param {import('@image_tagging_types').ImageTaggingRuntime} runtime
 * @param {{force?: boolean}} [opts] - `force` skips the duplicate check and
 *   pastes every entry, however much it overlaps what is already there.
 * @returns {Promise<number>} how many bboxes were actually added
 */
export async function pasteBboxes(runtime, { force = false } = {}) {
  const layer = runtime?.canvas?.layer;
  const image = runtime?.state?.image;
  if (!layer || !image) return 0;

  if (!_clipboard.length) {
    toast('Clipboard is empty — select some bboxes and copy them first', 'warning');
    return 0;
  }

  const viewport = runtime.canvas.viewport;
  const existing = [...image.bboxes.values()];
  const toPaste = force ? _clipboard : _clipboard.filter(entry => !_isDuplicate(entry, existing));
  const skipped = _clipboard.length - toPaste.length;

  if (!toPaste.length) {
    toast(`All ${skipped} copied bbox${skipped === 1 ? ' is' : 'es are'} already on this image`, 'warning');
    return 0;
  }

  const created = [];
  for (const entry of toPaste) {
    // Re-resolve the label rather than trusting the copied text: a box that
    // came back from live inference has its confidence baked into label.text
    // ("cat (82%)"), and pasting that verbatim would carry a percentage that
    // means nothing on this image. Falls back to the copied text if the label
    // has since been deleted.
    let label = entry.label ? { ...entry.label } : undefined;
    if (entry.label?.uuid) {
      const fresh = await fetchLabelByUuid(entry.label.uuid);
      if (fresh) label = { uuid: entry.label.uuid, text: fresh.name, color: fresh.color };
    }

    const uiBBox = createUIBBbox({
      uuid: generateUUID(),
      label,
      // Tags carry over exactly as copied, verification state included: a
      // pasted box is the same claim about the same subject as the one it came
      // from, so re-deciding it here would be this module inventing an opinion.
      tags: entry.tags.map(t => ({ ...t })),
      x: entry.x,
      y: entry.y,
      width: entry.width,
      height: entry.height,
      dirty: true,
      ...(entry.metadataJson ? { metadataJson: { ...entry.metadataJson } } : {}),
    });

    image.bboxes.set(uiBBox.uuid, uiBBox);
    const view = new BboxView(uiBBox, viewport);
    layer.add(view);
    created.push(view);
  }

  syncBboxDraggable(runtime);
  // One undo step for the whole paste — Ctrl+Z should take back the paste, not
  // one box of it.
  runtime.pushState();
  // Select what was just pasted, so it can be nudged or deleted as a unit.
  setSelection(runtime, created);
  events.publish(EventTypes.CANVAS_BBOXES_CHANGED, { runtimeId: runtime.id });
  layer.batchDraw();

  const noun = `bbox${created.length === 1 ? '' : 'es'}`;
  toast(skipped
    ? `Pasted ${created.length} ${noun} (${skipped} skipped — already present)`
    : `Pasted ${created.length} ${noun}`);
  Logger.debug('Pasted bboxes', { pasted: created.length, skipped, force });
  return created.length;
}
