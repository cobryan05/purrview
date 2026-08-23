/**
 * Canvas bbox selection — the single owner of the Konva Transformer.
 *
 * The selection is a *set* of BboxViews. `interactionState.selectedBbox` is
 * still maintained as the "primary" (most recently added) member, because the
 * inspector and the double-click tool-switch are single-box concepts and read
 * it directly.
 *
 * ⚠️ Nothing outside this module may call `transformer.nodes(...)`. Doing so
 * renders live resize anchors while skipping the `transformend` wiring below,
 * so dragging them mutates the box without pushing undo state or marking the
 * canvas dirty — the edit looks applied, Save stays disabled, and it is thrown
 * away on close. `rerenderBboxViews` is the one exception: it destroys every
 * view, so it must detach the transformer *and* empty this module's set, which
 * it does via `clearSelectionState`.
 *
 * Resize is deliberately disabled while more than one box is selected. Konva
 * can transform several nodes at once, but it does so by scaling them about a
 * shared origin, and each BboxView normalizes its own geometry back to 0..1 on
 * `transformend` — the two do not compose into anything a user would predict.
 * Multi-selection is for copy / delete / label, not for resizing.
 */

import { events, EventTypes } from '/app-static/js/shared/events/index.js';

/**
 * The currently selected views, in selection order.
 * @param {import('@image_tagging_types').ImageTaggingRuntime} runtime
 * @returns {import('./views/bboxView.js').BboxView[]}
 */
export function getSelectedViews(runtime) {
  return runtime?.canvas?.interactionState?.selectedBboxes ?? [];
}

/**
 * @param {import('@image_tagging_types').ImageTaggingRuntime} runtime
 * @param {import('./views/bboxView.js').BboxView} view
 */
export function isSelected(runtime, view) {
  return getSelectedViews(runtime).includes(view);
}

/**
 * Replace the selection with exactly `views`.
 *
 * @param {import('@image_tagging_types').ImageTaggingRuntime} runtime
 * @param {import('./views/bboxView.js').BboxView[]} views
 */
export function setSelection(runtime, views) {
  if (!runtime?.canvas) return;
  const { transformer, layer } = runtime.canvas;
  const ist = runtime.canvas.interactionState;

  // Drop duplicates and anything already destroyed (a stale reference held
  // across a delete or an undo rebuild).
  const next = [];
  for (const view of views ?? []) {
    if (!view || next.includes(view)) continue;
    if (!view.getLayer()) continue;
    next.push(view);
  }

  for (const prev of ist.selectedBboxes ?? []) {
    if (next.includes(prev)) continue;
    prev.off('transformend.transformer');
    prev.setSelected?.(false);
  }

  // The per-box marker is only drawn for a multi-selection. With one box the
  // Transformer's own border already identifies it exactly, and a second
  // outline on top of it is just noise.
  const showMarkers = next.length > 1;

  for (const view of next) {
    // Konva's namespaced .on() appends rather than replaces, so re-selecting an
    // already-selected box would stack a second handler (N undo pushes per
    // resize). Clear unconditionally before wiring.
    view.off('transformend.transformer');
    view.on('transformend.transformer', () => {
      transformer.forceUpdate();
      runtime.pushState();
      events.publish(EventTypes.CANVAS_BBOXES_CHANGED, { runtimeId: runtime.id });
    });
    view.setSelected?.(showMarkers);
  }

  ist.selectedBboxes = next;
  ist.selectedBbox = next.length ? next[next.length - 1] : null;

  transformer.resizeEnabled(next.length === 1);
  transformer.nodes(next);
  transformer.moveToTop();
  layer?.batchDraw();

  events.publish(EventTypes.CANVAS_SELECTION_CHANGED, {
    runtimeId: runtime.id,
    bboxIds: next.map(v => v.uiBBox?.uuid).filter(Boolean),
  });
}

/**
 * Add `view` to the selection, or remove it if already present.
 * @param {import('@image_tagging_types').ImageTaggingRuntime} runtime
 * @param {import('./views/bboxView.js').BboxView} view
 * @returns {boolean} true if the view ended up selected
 */
export function toggleSelection(runtime, view) {
  if (!view) return false;
  const current = getSelectedViews(runtime);
  const wasSelected = current.includes(view);
  setSelection(runtime, wasSelected ? current.filter(v => v !== view) : [...current, view]);
  return !wasSelected;
}

/**
 * Add views to the selection without removing anything (marquee drag).
 * @param {import('@image_tagging_types').ImageTaggingRuntime} runtime
 * @param {import('./views/bboxView.js').BboxView[]} views
 */
export function addToSelection(runtime, views) {
  setSelection(runtime, [...getSelectedViews(runtime), ...views]);
}

/** @param {import('@image_tagging_types').ImageTaggingRuntime} runtime */
export function clearBboxSelection(runtime) {
  setSelection(runtime, []);
}

/**
 * Select every bbox on the canvas.
 * @param {import('@image_tagging_types').ImageTaggingRuntime} runtime
 */
export function selectAllBboxes(runtime) {
  const layer = runtime?.canvas?.layer;
  if (!layer) return;
  setSelection(runtime, layer.find('.bbox'));
}

/**
 * Forget the selection without touching the views.
 *
 * For callers that are about to destroy every BboxView (`rerenderBboxViews`):
 * the normal deselect path would call `setSelected(false)` on nodes that are
 * being thrown away, and leaving the set populated would strand references to
 * destroyed nodes in `interactionState` — the exact stale-selection bug that
 * the inference-conflict path already guards against by hand.
 *
 * @param {import('@image_tagging_types').CanvasState} canvas
 */
export function clearSelectionState(canvas) {
  const ist = canvas?.interactionState;
  if (!ist) return;
  ist.selectedBboxes = [];
  ist.selectedBbox = null;
}
