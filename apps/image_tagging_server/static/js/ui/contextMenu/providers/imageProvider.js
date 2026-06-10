/**
 * Context-menu provider for the 'image' type.
 *
 * Adapter interface (CtxImageAdapter):
 *   getPath()                           → string
 *   openOnCanvas()                      → void
 *   setLabelPresence(uuid, status)      → Promise<void>  status = 'absent'|'uncertain'
 */

import { registerContextProvider } from '../registry.js';
import { asyncPanel } from '../menu.js';
import { fetchLabels } from '/app-static/js/shared/api/labels.js';
import { openSimilarImagesModal } from '/app-static/js/ui/similarSearch.js';
import { buildSimilarPanel } from '../similarPanelBuilder.js';

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

    // ── Mark label absent ──────────────────────────────────────────────────
    {
      label: 'Mark label absent',
      submenu: () => asyncPanel('200px', () => _buildLabelListPanel(async uuid => {
        for (const a of adapters) await a.setLabelPresence(uuid, 'absent');
        ctx.close();
      })),
    },

    // ── Mark label present (uncertain) ────────────────────────────────────
    {
      label: 'Mark label present (uncertain)',
      submenu: () => asyncPanel('200px', () => _buildLabelListPanel(async uuid => {
        for (const a of adapters) await a.setLabelPresence(uuid, 'uncertain');
        ctx.close();
      })),
    },

    // ── Similar search ─────────────────────────────────────────────────────
    {
      label: 'Search for similar images…',
      disabled: count > 1,
      submenu: () => buildSimilarPanel((threshold) => {
        const imageUuid = adapters[0].getUuid?.();
        ctx.close();
        if (imageUuid) openSimilarImagesModal({ imageUuid, threshold });
      }),
    },
  ];

  return [{ items }];
});
