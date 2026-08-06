/**
 * Context-menu provider for the 'image' type.
 *
 * Adapter interface (CtxImageAdapter):
 *   getPath()                           → string
 *   openOnCanvas()                      → void
 *   setLabelPresence(uuid, status)      → Promise<{ skipped?: boolean }|void>
 *                                          status = 'present'|'absent'|'uncertain'
 *                                          `skipped: true` = the server deliberately
 *                                          refused the write (verified-bbox conflict);
 *                                          not an error, but the state did not change.
 *   getTagUuids()                       → string[]        whole-image tags
 *   toggleTag(uuid)                     → Promise<void>
 *   setTagStatus(uuid, status)          → Promise<void>  status = 'present'|'absent'|'uncertain'
 */

import { registerContextProvider } from '../registry.js';
import { asyncPanel } from '../menu.js';
import { fetchLabels } from '/app-static/js/shared/api/labels.js';
import { openSimilarImagesModal } from '/app-static/js/ui/similarSearch.js';
import { openDissimilarImagesModal } from '/app-static/js/ui/dissimilarSearch.js';
import { buildTagPanel } from '../tagPanelBuilder.js';
import { toast } from '/app-static/js/ui/utils/logging.js';

/**
 * Apply a label presence status across every selected image and report the
 * server's deliberate refusals.
 *
 * POST /api/images/labels/set answers `status: success, skipped: true` when it
 * declines the write (marking a label ABSENT on an image that has a verified
 * bbox for that label). That is a success response, so nothing throws and the
 * menu just closes — without this the user gets no feedback at all and the
 * label silently stays as it was. Mirrors ui/labelList.js's aggregate info
 * toast for the same server contract.
 *
 * @param {object[]} adapters
 * @param {string} uuid
 * @param {'present'|'absent'|'uncertain'} status
 */
async function _applyLabelPresence(adapters, uuid, status) {
  let skipped = 0;
  for (const a of adapters) {
    const res = await a.setLabelPresence(uuid, status);
    if (res?.skipped) skipped += 1;
  }
  if (skipped) {
    toast(
      skipped === 1
        ? 'Label left unchanged: image has a verified bbox for it'
        : `${skipped} image(s) left unchanged: verified bboxes exist`,
      4000,
      'info'
    );
  }
}

/**
 * Build the "search all images" / "search with current filters" submenu for a
 * similar/dissimilar image search action.
 *
 * @param {object[]} adapters
 * @param {object} ctx
 * @param {(imageUuids: string[], candidateUuids: string[]|null) => void} openFn
 */
function _buildScopedSearchSubmenu(adapters, ctx, openFn) {
  const imageUuids = adapters.map(a => a.getUuid?.()).filter(Boolean);
  return [
    {
      label: 'Search all images',
      onSelect: () => {
        ctx.close();
        if (imageUuids.length) openFn(imageUuids, null);
      },
    },
    {
      label: 'Search with current filters',
      onSelect: async () => {
        ctx.close();
        if (!imageUuids.length) return;
        const candidateUuids = await adapters[0].getFilteredImageUuids?.();
        openFn(imageUuids, candidateUuids ?? null);
      },
    },
  ];
}

/**
 * Build a label-list panel for the Mark-label flyouts.
 *
 * @param {(uuid: string) => Promise<void>} onSelect
 * @returns {Promise<HTMLElement>}
 */
async function _buildLabelListPanel(onSelect) {
  const labels = await fetchLabels();

  const wrap = document.createElement('div');
  wrap.style.cssText = 'min-width:200px;max-height:260px;overflow-y:auto;padding:4px 0;';

  for (const [uuid, label] of labels) {
    if (label.scope !== 'image') continue;

    const row = document.createElement('div');
    row.style.cssText = [
      'display:flex', 'align-items:center', 'gap:6px',
      'padding:5px 10px', 'cursor:pointer', 'font-size:0.875em',
    ].join(';');
    row.addEventListener('mouseenter', () => (row.style.background = 'var(--pico-primary-background,rgba(255,255,255,.08))'));
    row.addEventListener('mouseleave', () => (row.style.background = ''));

    if (label.color) {
      const dot = document.createElement('span');
      dot.style.cssText = `display:inline-block;width:8px;height:8px;border-radius:50%;background:${label.color};flex-shrink:0;`;
      row.appendChild(dot);
    }
    const nm = document.createElement('span');
    nm.textContent = label.name ?? label.uuid;
    row.appendChild(nm);

    row.addEventListener('mousedown', e => { e.preventDefault(); onSelect(uuid); });
    wrap.appendChild(row);
  }

  return wrap;
}


registerContextProvider('image', async (adapters, ctx) => {
  const count = ctx.count;

  const items = [
    // ── Open on main canvas ────────────────────────────────────────────────
    {
      label: 'Open on main canvas',
      disabled: count > 1,
      onSelect: () => { adapters[0].openOnCanvas(); },
    },

    // ── Mark label present ────────────────────────────────────────────────
    {
      label: 'Mark label present',
      submenu: () => asyncPanel('200px', () => _buildLabelListPanel(async uuid => {
        await _applyLabelPresence(adapters, uuid, 'present');
        ctx.close();
      })),
    },

    // ── Mark label absent ─────────────────────────────────────────────────
    {
      label: 'Mark label absent',
      submenu: () => asyncPanel('200px', () => _buildLabelListPanel(async uuid => {
        await _applyLabelPresence(adapters, uuid, 'absent');
        ctx.close();
      })),
    },

    // ── Clear label status ────────────────────────────────────────────────
    {
      label: 'Clear label status (unreviewed)',
      submenu: () => asyncPanel('200px', () => _buildLabelListPanel(async uuid => {
        await _applyLabelPresence(adapters, uuid, 'uncertain');
        ctx.close();
      })),
    },

    // ── Tags (toggle on/off, checkbox + indeterminate for multi-select) ────
    {
      label: 'Tags',
      submenu: () => asyncPanel('220px', () => buildTagPanel(adapters)),
    },

    // ── Similar search ─────────────────────────────────────────────────────
    {
      label: 'Search for similar images…',
      submenu: () => _buildScopedSearchSubmenu(adapters, ctx,
        (imageUuids, candidateUuids) => openSimilarImagesModal({ imageUuids, candidateUuids })),
    },
    {
      label: 'Search for dissimilar images…',
      submenu: () => _buildScopedSearchSubmenu(adapters, ctx,
        (imageUuids, candidateUuids) => openDissimilarImagesModal({ imageUuids, candidateUuids })),
    },
  ];

  return [{ items }];
});
