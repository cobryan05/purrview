import {
  handleClick,
  handleContextMenu,
  handleMouseDown,
  handleMouseMove,
  handleMouseUp,
  handleResize,
  handleWheel
} from './interaction.js';
import { events, EventTypes } from '/app-static/js/shared/events/index.js';

/**
 * Wires stage event handlers and a ResizeObserver for a canvas instance.
 * App-specific event subscriptions (LOAD_IMAGE_ONTO_CANVAS, RUN_INFERENCE_ON_CANVAS)
 * and window globals belong in the calling context, not here.
 *
 * @param {import('@image_tagging_types').ImageTaggingRuntime} runtime
 */
export async function initCanvasStage(runtime) {
  const stage = runtime.canvas?.stage;
  if (!stage) {
    return;
  }

  stage.on('mousedown', e => handleMouseDown(e, runtime));
  stage.on('mousemove', e => handleMouseMove(e, runtime));
  stage.on('mouseup', e => handleMouseUp(e, runtime));
  stage.on('wheel', e => handleWheel(e, runtime));
  stage.on('click', e => handleClick(e, runtime));
  stage.on('contextmenu', e => handleContextMenu(e, runtime));

  // Snapshot bbox state after drag/transform completes. BboxView's own handlers
  // mutate uiBBox first, so by the time the event bubbles here the data is final.
  stage.on('dragend', e => {
    if (e.target.name() === 'bbox') {
      runtime.pushState();
      events.publish(EventTypes.CANVAS_BBOXES_CHANGED, { runtimeId: runtime.id });
    }
  });
  stage.on('transformend', e => {
    if (e.target.name() === 'bbox') {
      runtime.pushState();
      events.publish(EventTypes.CANVAS_BBOXES_CHANGED, { runtimeId: runtime.id });
    }
  });

  const observer = new ResizeObserver(() => handleResize({}, runtime));
  observer.observe(runtime.canvas.container);
  runtime.canvas.resizeObserver = observer;
}
