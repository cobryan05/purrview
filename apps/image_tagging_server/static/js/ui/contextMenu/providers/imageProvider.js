/**
 * Context-menu provider for the 'image' type.
 *
 * Adapter interface (CtxImageAdapter):
 *   getPath()                           → string
 *   openOnCanvas()                      → void
 *   setLabelPresence(uuid, status)      → Promise<void>  status = 'present'|'absent'|'uncertain'
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
        for (const a of adapters) await a.setLabelPresence(uuid, 'present');
        ctx.close();
      })),
    },

    // ── Mark label absent ─────────────────────────────────────────────────
    {
      label: 'Mark label absent',
      submenu: () => asyncPanel('200px', () => _buildLabelListPanel(async uuid => {
        for (const a of adapters) await a.setLabelPresence(uuid, 'absent');
        ctx.close();
      })),
    },

    // ── Clear label status ────────────────────────────────────────────────
    {
      label: 'Clear label status (unreviewed)',
      submenu: () => asyncPanel('200px', () => _buildLabelListPanel(async uuid => {
        for (const a of adapters) await a.setLabelPresence(uuid, 'uncertain');
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
      onSelect: () => {
        const imageUuids = adapters.map(a => a.getUuid?.()).filter(Boolean);
        ctx.close();
        if (imageUuids.length) openSimilarImagesModal({ imageUuids });
      },
    },
    {
      label: 'Search for dissimilar images…',
      onSelect: () => {
        const imageUuids = adapters.map(a => a.getUuid?.()).filter(Boolean);
        ctx.close();
        if (imageUuids.length) openDissimilarImagesModal({ imageUuids });
      },
    },
  ];

  return [{ items }];
});
