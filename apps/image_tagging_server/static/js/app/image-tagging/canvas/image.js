import { BboxView } from '/app-static/js/app/image-tagging/canvas/views/bboxView.js';
import { fetchImage, updateImage } from '/app-static/js/shared/api/images.js';
import { fetchLabels } from '/app-static/js/shared/api/labels.js';
import { createUIBBbox } from '/app-static/js/shared/domain/image/mapper.js';
import { CONFIDENCE_KEY } from '/app-static/js/shared/domain/image/confidence.js';
import { Logger, toast, showUnsavedChangesModal } from '/app-static/js/ui/utils/index.js';
import { events, EventTypes } from '/app-static/js/shared/events/index.js';
import { UNVERIFIED_TAG_UUID, VERIFIED_TAG_UUID } from '/app-static/js/shared/api/review.js';
import { fetchTags } from '/app-static/js/shared/api/tags.js';
import { syncBboxDraggable } from './tools.js';
import { applyVerifiedTag } from './interaction.js';

/**
 * Prompt about unsaved changes before navigating away from the current image.
 * Returns true if navigation should proceed (nothing dirty, discarded, or saved).
 * Must be called BEFORE advancing any file-navigation index so Cancel is a no-op.
 * @param {import('@image_tagging_types').ImageTaggingRuntime} runtime
 * @returns {Promise<boolean>}
 */
export async function guardUnsavedChanges(runtime) {
  if (!runtime?.isDirty) return true;
  const choice = await showUnsavedChangesModal();
  if (choice === 'cancel') return false;
  if (choice === 'save') return await saveAnnotations(runtime);
  return true; // discard
}

export async function reloadImage(runtime) {
  const imageName = runtime.state.image?.image_path;
  if (!imageName) {
    Logger.notify('No image loaded to reload');
    return;
  }
  await loadImageOntoCanvas(runtime, imageName);
}

/**
 * @param {import('@image_tagging_types').ImageTaggingRuntime} runtime
 * @param {string} imageName
 */
export async function loadImageOntoCanvas(runtime, imageName) {
  Logger.debug('loadImageOntoCanvas called with imageName:', imageName);

  if (!runtime.canvas) return;

  // Unsaved-changes guarding happens at navigation triggers (files.js) *before*
  // the file index moves, so this loader stays a pure load.

  // Create a new load id and invalidate all previous loads for this canvas instance
  const loadId = ++runtime.canvas.activeImageLoadId;
  /** @type {import('@domain_image').UIImage} */
  const imageInfo = await fetchImage(imageName);

  // Abort if canvas was destroyed or a newer load started while we waited
  if (!runtime.canvas || loadId !== runtime.canvas.activeImageLoadId) {
    Logger.debug('Discarding stale image load for', imageName);
    return;
  }

  runtime.setImage(imageInfo);
  events.publish(EventTypes.IMAGE_LOADED_ONTO_CANVAS, { path: imageName, runtimeId: runtime.id });
}


/**
 * @param {import('@image_tagging_types').ImageTaggingRuntime} runtime
 */
export function clearAnnotations(runtime) {
  if (!runtime.canvas) return;
  const { layer, transformer } = runtime.canvas;
  transformer?.nodes([]);
  runtime.clearBboxes();

  const children = [...layer.getChildren()];
  children.forEach(child => {
    if (child.name() === 'bbox') {
      child.destroy();
    }
  });
  layer.draw();
  runtime.pushState();
  events.publish(EventTypes.CANVAS_BBOXES_CHANGED, { runtimeId: runtime.id });
}

/**
 * Mark every bbox on the canvas as verified (strips UNVERIFIED/other verification
 * tags, adds VERIFIED). Bboxes already carrying VERIFIED are left untouched.
 * @param {import('@image_tagging_types').ImageTaggingRuntime} runtime
 */
export function verifyAllBboxes(runtime) {
  if (!runtime.canvas) return;
  const { layer } = runtime.canvas;

  let changed = 0;
  [...layer.getChildren()].forEach(child => {
    if (child.name() !== 'bbox' || !(child instanceof BboxView)) return;
    if (child.uiBBox.tags?.some(t => t.uuid === VERIFIED_TAG_UUID)) return;
    applyVerifiedTag(child.uiBBox);
    child.refreshIcons(child.uiBBox.tags);
    changed++;
  });

  if (!changed) {
    toast('All bboxes are already verified.');
    return;
  }

  runtime.pushState();
  events.publish(EventTypes.CANVAS_BBOXES_CHANGED, { runtimeId: runtime.id });
  toast(`Verified ${changed} bbox${changed === 1 ? '' : 'es'}.`);
}

/**
 * Remove every bbox that has no resolved label (rendered grey) — i.e. inference
 * detections whose class never matched a real label. Bboxes carrying a valid
 * `label.uuid` are kept.
 * @param {import('@image_tagging_types').ImageTaggingRuntime} runtime
 */
export function clearUnmatched(runtime) {
  if (!runtime.canvas) return;
  const { layer, transformer } = runtime.canvas;

  let removed = 0;
  [...layer.getChildren()].forEach(child => {
    if (child.name() !== 'bbox' || !(child instanceof BboxView)) return;
    if (child.uiBBox.label?.uuid) return;
    transformer?.nodes(transformer.nodes().filter(n => n !== child && n.getParent() !== child));
    runtime.state.image?.bboxes.delete(child.uiBBox.uuid);
    child.destroy();
    removed++;
  });

  if (!removed) {
    toast('No unmatched labels to clear.');
    return;
  }

  layer.draw();
  runtime.pushState();
  events.publish(EventTypes.CANVAS_BBOXES_CHANGED, { runtimeId: runtime.id });
}

/**
 * @param {import('@image_tagging_types').ImageTaggingRuntime} runtime
 */
export async function saveAnnotations(runtime) {
  if (!runtime.state?.image) {
    toast('No image loaded to save annotations!');
    return false;
  }

  try {
    await updateImage(runtime.state.image);
    runtime.markClean();
    events.publish(EventTypes.CANVAS_BBOXES_CHANGED, { runtimeId: runtime.id });
    events.publish(EventTypes.CANVAS_ANNOTATIONS_SAVED, { runtimeId: runtime.id });
    toast('Annotations saved successfully!');
    return true;
  }
  catch (err) {
    Logger.error('Failed to save annotations:', err);
    return false;
  }
}

/**
 * @param {import('@image_tagging_types').ImageTaggingRuntime} runtime
 */
export function deleteSelected(runtime) {
  const transformer = runtime.canvas?.transformer;
  const layer = runtime.canvas?.layer;
  if (!transformer || !layer) return;

  const selectedNodes = transformer.nodes();
  if (!selectedNodes.length) return;

  // Walk up from the selected node to find the BboxView group
  let node = selectedNodes[0];
  let bboxView = null;
  while (node && node !== layer) {
    if (node instanceof BboxView) {
      bboxView = node;
      break;
    }
    node = node.getParent();
  }

  if (bboxView) {
    runtime.state.image?.bboxes.delete(bboxView.uiBBox.uuid);
    bboxView.destroy();
    runtime.pushState();
    events.publish(EventTypes.CANVAS_BBOXES_CHANGED, { runtimeId: runtime.id });
  }

  transformer.nodes([]);
  layer.draw();
}

/**
 * @param {import('@image_tagging_types').ImageTaggingState} state
 */
export function exportAnnotations(state) {
  const layer = state.canvas.layer;
  const shapes = layer.getChildren().filter(s => s.name() === 'annotation');

  const annotations = shapes
    .map(group => {
      const rect = group.findOne('.box');
      if (!rect) return null;

      return {
        type: 'rect',
        x: group.x(),
        y: group.y(),
        width: rect.width(),
        height: rect.height(),
        metadata: group.metadata || {},
      };
    })
    .filter(Boolean);

  const pre = document.getElementById('annotation-json');
  if (pre) {
    pre.textContent = JSON.stringify(annotations, null, 2);
  }
}

/**
 * Inject not-yet-persisted candidate bboxes into the current canvas so they are
 * editable (the `bboxUuid:'new'` preview case). Each seed is a UIBBox-shaped
 * object (as produced by `createUIBBbox`); anything else is coerced through it.
 * Seeds become part of the clean baseline (via `pushState` + `markClean`) so the
 * modal opens with them present but not flagged dirty — mirroring how a freshly
 * loaded image with those boxes would look.
 * @param {import('@image_tagging_types').ImageTaggingRuntime} runtime
 * @param {Array<object>} seedBboxes
 */
export function seedRuntimeBboxes(runtime, seedBboxes) {
  if (!runtime.canvas || !runtime.state?.image || !seedBboxes?.length) return;

  const { layer, viewport } = runtime.canvas;
  for (const seed of seedBboxes) {
    // Accept an already-built UIBBox or a plain descriptor; createUIBBbox is
    // idempotent enough for our fields and fills in defaults either way.
    const uiBbox = createUIBBbox(seed);
    runtime.state.image.bboxes.set(uiBbox.uuid, uiBbox);
    layer.add(new BboxView(uiBbox, viewport));
  }

  syncBboxDraggable(runtime);
  layer.draw();
  runtime.pushState();
  runtime.markClean();
}

// A new detection whose IOU against an existing bbox meets or exceeds this is
// treated as "the same box" for conflict resolution rather than a new one.
export const INFERENCE_DUPLICATE_IOU_THRESHOLD = 0.7;

// Glow colors for the temporary flash drawn on conflicting bboxes:
export const GLOW_MATCHED = '#ffd60a'; // duplicate skipped, or an existing box replaced
export const GLOW_CONFLICT_KEPT = '#ff453a'; // different label detected but left untouched

/**
 * Intersection-over-union of two normalized (0-1) rects.
 * @param {{x:number,y:number,width:number,height:number}} a
 * @param {{x:number,y:number,width:number,height:number}} b
 * @returns {number}
 */
export function computeIou(a, b) {
  const ax2 = a.x + a.width, ay2 = a.y + a.height;
  const bx2 = b.x + b.width, by2 = b.y + b.height;
  const ix1 = Math.max(a.x, b.x), iy1 = Math.max(a.y, b.y);
  const ix2 = Math.min(ax2, bx2), iy2 = Math.min(ay2, by2);
  const iw = Math.max(0, ix2 - ix1), ih = Math.max(0, iy2 - iy1);
  const inter = iw * ih;
  if (inter <= 0) return 0;
  const union = a.width * a.height + b.width * b.height - inter;
  return union > 0 ? inter / union : 0;
}

/**
 * Identifies a bbox's label for "same label" comparisons even before it has a
 * resolved uuid (grey/unmatched inference results carry only display text).
 * @param {import('@domain_image').UITag|undefined} label
 * @returns {string|null}
 */
export function labelKey(label) {
  if (!label) return null;
  if (label.uuid) return `uuid:${label.uuid}`;
  const text = (label.text || '').replace(/\s*\(\d+%\)\s*$/, '');
  return text ? `text:${text}` : null;
}

/**
 * Adds recognition results to the current canvas
 * @param {import('@image_tagging_types').ImageTaggingRuntime} runtime
 * @param {import('@web_api').InferenceResultModel} results - The inference results to add to the current canvas
 * @param {{ replaceExisting?: boolean }} [opts] - When a detection significantly overlaps an
 *   existing bbox with a different label, replace it instead of leaving it untouched. Only
 *   applies when the new detection itself resolved to a real label (`bbox.labelUuid` mapped
 *   to a known label) — an unmatched/grey detection never replaces an existing labeled bbox,
 *   regardless of this flag.
 */
export async function addInferenceResults(runtime, results, { replaceExisting = false } = {}) {
  if (!runtime.canvas || !runtime.state?.image) {
    Logger.warn('addInferenceResults: no canvas or image loaded');
    return;
  }

  const { layer, viewport } = runtime.canvas;
  const [labelsMap, tagsMap] = await Promise.all([fetchLabels(), fetchTags()]);
  const unverifiedIcon = tagsMap.get(UNVERIFIED_TAG_UUID)?.icon ?? null;

  // Snapshot pre-existing bboxes so newly-added detections in this same batch
  // don't get matched against each other.
  const existingBboxes = [...runtime.state.image.bboxes.values()];
  let changed = false;

  for (const detection of results.detections) {
    const { bbox } = detection;

    const labelData = bbox.labelUuid ? labelsMap.get(bbox.labelUuid) : null;
    // The confidence rides on the bbox rather than being baked into the label
    // text: BboxView appends it for display, and metadataJson carries it into
    // the save so it is still there after a reload. Baking it into the text
    // meant the percentage existed only until you navigated away.
    const label = labelData
      ? { uuid: labelData.uuid, text: labelData.name, color: labelData.color }
      : bbox.labelUuid
        ? { uuid: bbox.labelUuid, text: bbox.classStr ?? bbox.labelUuid }
        : bbox.classStr
          ? { text: bbox.classStr }
          : undefined;

    // Always include the UNVERIFIED tag on inference results
    const baseTags = (bbox.tagUuids ?? []).map(id => ({ uuid: id, icon: tagsMap.get(id)?.icon ?? null }));
    const tags = baseTags.some(t => t.uuid === UNVERIFIED_TAG_UUID)
      ? baseTags
      : [{ uuid: UNVERIFIED_TAG_UUID, icon: unverifiedIcon }, ...baseTags];

    const uiBbox = createUIBBbox({
      uuid: bbox.uuid,
      label,
      tags,
      x: bbox.x,
      y: bbox.y,
      width: bbox.width,
      height: bbox.height,
      selected: false,
      dirty: true,
      confidence: detection.confidence,
      // Persisted on save, so the percentage survives a reload instead of
      // living only in this canvas session. Stringified because the save
      // payload's `extra` is Dict[str, str] server-side — a raw float is
      // rejected. parseConfidence reads it back.
      metadataJson: { [CONFIDENCE_KEY]: String(detection.confidence) },
    });

    let bestMatch = null;
    let bestIou = 0;
    for (const existing of existingBboxes) {
      const iou = computeIou(existing, uiBbox);
      if (iou > bestIou) {
        bestIou = iou;
        bestMatch = existing;
      }
    }

    if (bestMatch && bestIou >= INFERENCE_DUPLICATE_IOU_THRESHOLD) {
      const matchView = layer.findOne(`#${bestMatch.uuid}`);
      const sameLabel = labelKey(bestMatch.label) !== null && labelKey(bestMatch.label) === labelKey(uiBbox.label);

      if (sameLabel) {
        // Same box, same label already present — don't add a duplicate.
        matchView?.flash(GLOW_MATCHED);
        continue;
      }

      if (!replaceExisting || !uiBbox.label?.uuid) {
        // Conflicting label detected on an existing box, but left untouched.
        // Also applies when the new detection's class has no mapped label at
        // all (grey/unmatched) — an unresolved detection must never evict an
        // existing, resolved label just because replaceExisting is on for
        // this model run. With multiple models running back-to-back, this
        // stops a later model's unmapped class from silently wiping out a
        // good label an earlier model already placed.
        matchView?.flash(GLOW_CONFLICT_KEPT);
        continue;
      }

      // Replace: drop the old box, fall through to add the new one. Detach it
      // from the transformer first — Konva caches the bounds of whatever was
      // passed to transformer.nodes([...]) and destroying a node does not
      // clear or reposition that cached selection rectangle. Also clear any
      // stale selectedBbox reference so a later click doesn't operate on a
      // destroyed node.
      runtime.state.image.bboxes.delete(bestMatch.uuid);
      const { transformer, interactionState } = runtime.canvas;
      transformer?.nodes(transformer.nodes().filter(n => n !== matchView && n.getParent() !== matchView));
      if (interactionState?.selectedBbox === matchView) {
        matchView.off('transformend.transformer');
        interactionState.selectedBbox = null;
      }
      matchView?.destroy();
    }

    runtime.state.image.bboxes.set(uiBbox.uuid, uiBbox);
    const bboxView = new BboxView(uiBbox, viewport);
    layer.add(bboxView);
    if (bestMatch && bestIou >= INFERENCE_DUPLICATE_IOU_THRESHOLD) bboxView.flash(GLOW_MATCHED);
    changed = true;
  }

  syncBboxDraggable(runtime);
  layer.draw();
  if (changed) runtime.pushState();
  events.publish(EventTypes.CANVAS_BBOXES_CHANGED, { runtimeId: runtime.id });
}
