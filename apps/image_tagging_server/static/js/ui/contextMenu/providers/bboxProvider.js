/**
 * Context-menu provider for the 'bbox' type.
 *
 * Adapter interface (CtxBboxAdapter):
 *   getLabelUuid()        → string | null
 *   getTagUuids()         → string[]
 *   applyLabel(uuid)      → Promise<void>   (null = no label)
 *   toggleTag(uuid)       → Promise<void>
 *   setVerification(uuid) → Promise<void>   (null = clear all verification indicators)
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
  FALSE_POS_TAG_UUID,
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

registerContextProvider('bbox', async (adapters, ctx) => {
  const count = ctx.count;
  const labelSuffix = count > 1 ? ` to ${count} boxes` : '';

  const items = [
    // ── Apply label ────────────────────────────────────────────────────────
    {
      label: `Apply label${labelSuffix}`,
      submenu: () => asyncPanel('240px', () => _buildLabelPanel(async uuid => {
        await runAll(adapters, a => a.applyLabel(uuid));
        ctx.close();
      })),
    },

    // ── Tags (toggle on/off, checkbox + indeterminate for multi-select) ────
    {
      label: `Tags${labelSuffix}`,
      submenu: () => asyncPanel('220px', () => buildTagPanel(adapters)),
    },

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
          label: 'Mark False Positive',
          onSelect: async () => { await runAll(adapters, a => a.setVerification(FALSE_POS_TAG_UUID)); ctx.close(); },
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
    {
      label: count > 1 ? `Delete ${count} boxes` : 'Delete',
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
    },

    // ── Similar search ─────────────────────────────────────────────────────
    {
      label: 'Search for similar bboxes…',
      onSelect: () => {
        const bboxes = adapters
          .map(a => ({ bboxUuid: a.getBboxUuid?.() ?? null, bboxCoords: a.getBboxCoords?.() ?? null, imageUuid: a.getImageUuid?.() ?? null }))
          .filter(b => b.bboxUuid || (b.bboxCoords && b.imageUuid));
        ctx.close();
        if (bboxes.length) openSimilarBboxesModal({ bboxes });
      },
    },
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
