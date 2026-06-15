/**
 * Canvas-side context adapters for the right-click context menu.
 *
 * makeCanvasBboxAdapter(bboxView, runtime) → CtxBboxAdapter
 * makeCanvasSurfaceAdapter(runtime)        → stub for the 'canvas' type
 *
 * Note: this module intentionally avoids importing from interaction.js to prevent
 * a circular dependency (interaction.js imports contextTargets.js at module level).
 * The mutation helpers below mirror the equivalent functions in interaction.js.
 */

import { fetchLabelByUuid } from '/app-static/js/shared/api/labels.js';
import { fetchTagByUuid } from '/app-static/js/shared/api/tags.js';
import { events, EventTypes } from '/app-static/js/shared/events/index.js';
import {
  UNVERIFIED_TAG_UUID,
  VERIFIED_TAG_UUID,
  VERIFICATION_TAG_UUIDS,
  REJECTION_TAG_UUID,
} from '/app-static/js/shared/api/review.js';

/** All UUIDs that count as "verification indicators" for the Clear action. */
const ALL_VERIFICATION_UUIDS = new Set([
  UNVERIFIED_TAG_UUID,
  ...VERIFICATION_TAG_UUIDS,
  REJECTION_TAG_UUID,
]);

// ── Local helpers (mirrors interaction.js, no import cycle) ───────────────────

/** Strip all verification tags in-place and add VERIFIED. */
function _applyVerifiedTag(uiBBox) {
  uiBBox.tags = [
    ...(uiBBox.tags ?? []).filter(t => !VERIFICATION_TAG_UUIDS.has(t.uuid)),
    { uuid: VERIFIED_TAG_UUID, icon: null },
  ];
}

/**
 * Toggle a tag on a bbox (same logic as interaction.js toggleBboxTag).
 * Verification-group tags are mutually exclusive.
 */
async function _toggleBboxTag(bboxView, tagUuid, runtime) {
  const tag = await fetchTagByUuid(tagUuid);
  if (!tag) return;
  const tags = bboxView.uiBBox.tags ?? [];
  const has = tags.some(t => t.uuid === tagUuid);
  let next;
  if (has) {
    next = tags.filter(t => t.uuid !== tagUuid);
  } else {
    const base = VERIFICATION_TAG_UUIDS.has(tagUuid)
      ? tags.filter(t => !VERIFICATION_TAG_UUIDS.has(t.uuid))
      : tags;
    next = [...base, { uuid: tagUuid, icon: tag.icon ?? null }];
  }
  bboxView.uiBBox.tags = next;
  bboxView.refreshIcons(next);
  runtime.pushState();
  events.publish(EventTypes.CANVAS_BBOXES_CHANGED, { runtimeId: runtime.id });
}

// ── Adapters ──────────────────────────────────────────────────────────────────

/**
 * @param {import('./views/bboxView.js').BboxView} bboxView
 * @param {import('../runtime.js').ImageTaggingRuntime} runtime
 */
export function makeCanvasBboxAdapter(bboxView, runtime) {
  return {
    getBboxUuid() {
      return bboxView.uiBBox.uuid ?? null;
    },

    getImageUuid() {
      return runtime.state.image?.uuid ?? null;
    },

    getBboxCoords() {
      // Return null for clean (saved, unmodified) bboxes so callers can use the
      // UUID path, which guarantees the source bbox appears at distance 0.
      // For dirty bboxes (moved/resized since last save, or newly drawn), return
      // live coords so the search reflects the current crop position.
      if (!bboxView.uiBBox.dirty) return null;
      const { x, y, width, height } = bboxView.uiBBox;
      return { x, y, w: width, h: height };
    },

    getLabelUuid() {
      return bboxView.uiBBox.label?.uuid ?? null;
    },

    getTagUuids() {
      return (bboxView.uiBBox.tags ?? []).map(t => t.uuid);
    },

    async applyLabel(uuid) {
      if (uuid) {
        const label = await fetchLabelByUuid(uuid);
        bboxView.uiBBox.label = label
          ? { uuid, text: label.name, color: label.color }
          : null;
        if (label) _applyVerifiedTag(bboxView.uiBBox);
        bboxView.updateAppearance(label?.name ?? '', label?.color ?? '#888888');
      } else {
        bboxView.uiBBox.label = null;
        bboxView.updateAppearance('', '#888888');
      }
      bboxView.refreshIcons(bboxView.uiBBox.tags);
      runtime.pushState();
      events.publish(EventTypes.CANVAS_BBOXES_CHANGED, { runtimeId: runtime.id });
    },

    async toggleTag(uuid) {
      await _toggleBboxTag(bboxView, uuid, runtime);
    },

    async setVerification(uuid) {
      const tags = bboxView.uiBBox.tags ?? [];
      let next;
      if (uuid === null) {
        next = tags.filter(t => !ALL_VERIFICATION_UUIDS.has(t.uuid));
      } else if (uuid === REJECTION_TAG_UUID) {
        // REJECTION has no exclusive_group — add without swapping
        const tag = await fetchTagByUuid(uuid);
        const has = tags.some(t => t.uuid === uuid);
        next = has ? tags : [...tags, { uuid, icon: tag?.icon ?? null }];
      } else {
        // UNVERIFIED or FALSE_POS — swap within verification exclusive group
        const tag = await fetchTagByUuid(uuid);
        const base = tags.filter(t => !VERIFICATION_TAG_UUIDS.has(t.uuid));
        next = [...base, { uuid, icon: tag?.icon ?? null }];
      }
      bboxView.uiBBox.tags = next;
      bboxView.refreshIcons(next);
      runtime.pushState();
      events.publish(EventTypes.CANVAS_BBOXES_CHANGED, { runtimeId: runtime.id });
    },

    remove() {
      const layer = runtime.canvas?.layer;
      const transformer = runtime.canvas?.transformer;
      runtime.state.image?.bboxes.delete(bboxView.uiBBox.uuid);
      bboxView.destroy();
      if (transformer) {
        const nodes = transformer.nodes().filter(n => n !== bboxView);
        transformer.nodes(nodes);
      }
      runtime.pushState();
      events.publish(EventTypes.CANVAS_BBOXES_CHANGED, { runtimeId: runtime.id });
      layer?.draw();
    },
  };
}

/**
 * Stub adapter for the 'canvas' surface type.
 * No actions are registered for this type initially, so right-clicking an empty
 * canvas falls through to the original draw-tool toggle.
 */
export function makeCanvasSurfaceAdapter(runtime) {
  return {
    getImageUuid() {
      return runtime.state.image?.uuid ?? null;
    },
  };
}
