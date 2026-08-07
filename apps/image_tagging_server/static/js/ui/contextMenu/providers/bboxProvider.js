/**
 * Context-menu provider for the 'bbox' type.
 *
 * Adapter interface (CtxBboxAdapter):
 *   getLabelUuid()        → string | null
 *   getTagUuids()         → string[]
 *   applyLabel(uuid)      → Promise<void>   (null = no label)
 *   toggleTag(uuid)       → Promise<void>
 *   setVerification(uuid, opts?) → Promise<void>   (null = clear all verification indicators;
 *                                                   opts.refreshImageCard = false to batch)
 *   remove()              → Promise<void>
 */

import { registerContextProvider } from '../registry.js';
import { asyncPanel } from '../menu.js';
import { fetchLabels } from '/app-static/js/shared/api/labels.js';
import { createDynamicLabelPicker } from '/app-static/js/ui/utils/dynamicLabelPicker.js';
import { showConfirmModal } from '/app-static/js/ui/utils/confirmModal.js';
import { openSimilarImagesModal, openSimilarBboxesModal } from '/app-static/js/ui/similarSearch.js';
import { openDissimilarImagesModal, openDissimilarBboxesModal } from '/app-static/js/ui/dissimilarSearch.js';
import {
  REJECTION_TAG_UUID,
  UNVERIFIED_TAG_UUID,
  VERIFIED_TAG_UUID,
} from '/app-static/js/shared/api/review.js';
import { runAll, buildTagPanel } from '../tagPanelBuilder.js';

/**
 * Build a label picker flyout panel.
 *
 * @param {(uuid: string|null) => Promise<void>} onSelect
 * @returns {Promise<HTMLElement>}
 */
async function _buildLabelPanel(onSelect) {
  const labels = await fetchLabels();
  // Only object-scoped labels can be assigned to bounding boxes.
  const objectLabels = new Map([...labels].filter(([, l]) => l.scope !== 'image'));
  const picker = createDynamicLabelPicker(objectLabels, { onSelect });
  const wrap = document.createElement('div');
  wrap.style.cssText = 'min-width:240px;';
  wrap.appendChild(picker.element);
  setTimeout(() => picker.element.querySelector('input')?.focus(), 30);
  return wrap;
}

// ── Individual menu-item builders ───────────────────────────────────────────
// Exported so a trimmed context type (e.g. 'candidateBbox' in the Run-Model-on-
// Results preview) can compose a subset of these items instead of forking the
// whole provider. Each takes `(adapters, ctx)` — the same args the provider fn
// receives — and returns a single menu-item descriptor.

export function buildApplyLabelItem(adapters, ctx) {
  const labelSuffix = ctx.count > 1 ? ` to ${ctx.count} boxes` : '';
  return {
    label: `Apply label${labelSuffix}`,
    submenu: () => asyncPanel('240px', () => _buildLabelPanel(async uuid => {
      await runAll(adapters, a => a.applyLabel(uuid));
      ctx.close();
    })),
  };
}

export function buildTagsItem(adapters, ctx) {
  const labelSuffix = ctx.count > 1 ? ` to ${ctx.count} boxes` : '';
  return {
    label: `Tags${labelSuffix}`,
    submenu: () => asyncPanel('220px', () => buildTagPanel(adapters)),
  };
}

export function buildEditGeometryItem(adapters, ctx) {
  // Preview-only: opens the shared canvas editor on the (single) target box so
  // its geometry can be tweaked. Disabled for multi-select — the canvas editor
  // is single-image. Wired via the adapter's `openEditor()` method (candidate
  // adapter only; the review grid uses double-click instead).
  return {
    label: 'Edit geometry…',
    disabled: ctx.count > 1,
    onSelect: () => {
      const open = adapters[0].openEditor;
      ctx.close();
      open?.();
    },
  };
}

export function buildDeleteItem(adapters, ctx, { label } = {}) {
  const count = ctx.count;
  return {
    label: label ?? (count > 1 ? `Delete ${count} boxes` : 'Delete'),
    danger: true,
    onSelect: async () => {
      if (count > 1) {
        const confirmed = await showConfirmModal({
          title: 'Delete boxes',
          message: `Delete ${count} boxes?`,
          confirmText: 'Delete',
          danger: true,
        });
        if (!confirmed) return;
      }
      await runAll(adapters, a => a.remove());
    },
  };
}

export function buildSimilarBboxesItem(adapters, ctx) {
  return {
    label: 'Search for similar bboxes…',
    onSelect: () => {
      const bboxes = adapters
        .map(a => ({ bboxUuid: a.getBboxUuid?.() ?? null, bboxCoords: a.getBboxCoords?.() ?? null, imageUuid: a.getImageUuid?.() ?? null }))
        .filter(b => b.bboxUuid || (b.bboxCoords && b.imageUuid));
      ctx.close();
      if (bboxes.length) openSimilarBboxesModal({ bboxes });
    },
  };
}

registerContextProvider('bbox', async (adapters, ctx) => {
  const count = ctx.count;

  const items = [
    // ── Apply label ────────────────────────────────────────────────────────
    buildApplyLabelItem(adapters, ctx),

    // ── Tags (toggle on/off, checkbox + indeterminate for multi-select) ────
    buildTagsItem(adapters, ctx),

    // ── Verification ───────────────────────────────────────────────────────
    {
      label: 'Verification',
      submenu: () => [
        {
          label: 'Mark Verified',
          onSelect: async () => { await runAll(adapters, a => a.setVerification(VERIFIED_TAG_UUID)); ctx.close(); },
        },
        {
          label: 'Mark Unverified',
          onSelect: async () => { await runAll(adapters, a => a.setVerification(UNVERIFIED_TAG_UUID)); ctx.close(); },
        },
        {
          label: 'Mark Rejection',
          onSelect: async () => { await runAll(adapters, a => a.setVerification(REJECTION_TAG_UUID)); ctx.close(); },
        },
        {
          label: 'Clear verification',
          onSelect: async () => { await runAll(adapters, a => a.setVerification(null)); ctx.close(); },
        },
      ],
    },

    // ── Delete (danger) ────────────────────────────────────────────────────
    buildDeleteItem(adapters, ctx),

    // ── Similar search ─────────────────────────────────────────────────────
    buildSimilarBboxesItem(adapters, ctx),
    {
      label: 'Search for similar images…',
      submenu: () => {
        const imageUuids = [...new Set(adapters.map(a => a.getImageUuid?.()).filter(Boolean))];
        return [
          {
            label: 'Search all images',
            onSelect: () => { ctx.close(); if (imageUuids.length) openSimilarImagesModal({ imageUuids }); },
          },
          {
            label: 'Search with current filters',
            onSelect: async () => {
              ctx.close();
              if (!imageUuids.length) return;
              const candidateUuids = await adapters[0].getFilteredImageUuids?.();
              openSimilarImagesModal({ imageUuids, candidateUuids: candidateUuids ?? null });
            },
          },
        ];
      },
    },
    {
      label: 'Search for dissimilar bboxes…',
      disabled: count > 1,
      onSelect: () => {
        // Label filtering used to seed the modal's now-removed bespoke Labels
        // dropdown (server-side push); M1 moved that to the panel's shared
        // curated result-filter surface, so only the anchor bbox is needed here.
        const bboxUuid = adapters[0].getBboxUuid?.() ?? null;
        ctx.close();
        if (bboxUuid) openDissimilarBboxesModal({ bboxUuid });
      },
    },
    {
      label: 'Search for dissimilar images…',
      disabled: count > 1,
      submenu: () => {
        const imageUuid = adapters[0].getImageUuid?.() ?? null;
        return [
          {
            label: 'Search all images',
            onSelect: () => { ctx.close(); if (imageUuid) openDissimilarImagesModal({ imageUuid }); },
          },
          {
            label: 'Search with current filters',
            onSelect: async () => {
              ctx.close();
              if (!imageUuid) return;
              const candidateUuids = await adapters[0].getFilteredImageUuids?.();
              openDissimilarImagesModal({ imageUuid, candidateUuids: candidateUuids ?? null });
            },
          },
        ];
      },
    },
  ];

  return [{ items }];
});

/**
 * Context-menu provider for the 'candidateBbox' type — the Run-Model-on-Results
 * preview's proposed/new boxes. Reuses the 'bbox' provider's item builders but
 * exposes only Relabel / Edit-geometry / Remove / Similar-search (scope decision
 * 3: no per-candidate verification or tag editing). The candidate adapter
 * mutates in-memory preview state only; nothing hits the DB until Apply.
 *
 * Similar-search is naturally disabled for `isNew` candidates because the
 * builder `?.`-guards `getBboxUuid()` (which returns null for new boxes) and
 * filters them out, so no menu wiring change is needed here.
 */
registerContextProvider('candidateBbox', (adapters, ctx) => {
  const items = [
    buildApplyLabelItem(adapters, ctx),
    buildEditGeometryItem(adapters, ctx),
    buildSimilarBboxesItem(adapters, ctx),
    buildDeleteItem(adapters, ctx, { label: ctx.count > 1 ? `Remove ${ctx.count} from preview` : 'Remove from preview' }),
  ];
  return [{ items }];
});

/**
 * Context-menu provider for the 'candidateImage' type — the classifier branch
 * of the Run-Model-on-Results preview (`runClassifierOnResults.js`). Classifier
 * candidates are IMAGE-LEVEL (whole-image predictions, no bbox geometry), so
 * this provider exposes ONLY Relabel and Remove-from-preview — deliberately no
 * geometry / similar-search / verification / tags (those are box-only concepts).
 * The candidate adapter mutates in-memory preview state only; nothing hits the
 * DB until Apply.
 *
 * `buildApplyLabelItem`'s label picker filters to object-scoped labels via a
 * shared `_buildLabelPanel`, so image-scoped predictions are relabeled from the
 * same curated label set the rest of the preview uses.
 */
registerContextProvider('candidateImage', (adapters, ctx) => {
  const items = [
    buildApplyLabelItem(adapters, ctx),
    buildDeleteItem(adapters, ctx, { label: ctx.count > 1 ? `Remove ${ctx.count} from preview` : 'Remove from preview' }),
  ];
  return [{ items }];
});
