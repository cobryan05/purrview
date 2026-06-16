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
import { fetchTags, createNewTag } from '/app-static/js/shared/api/tags.js';
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


/** Run an async fn over every adapter in sequence. */
async function runAll(adapters, fn) {
  for (const a of adapters) await fn(a);
}

/**
 * Build a tag-list panel (HTMLElement) for the Add tag flyout.
 *
 * @param {(uuid: string) => Promise<void>} onToggle
 * @returns {Promise<HTMLElement>}
 */
async function _buildTagPanel(onToggle) {
  const tagsMap = await fetchTags();

  const wrap = document.createElement('div');
  wrap.style.cssText = 'min-width:200px;max-height:260px;overflow-y:auto;padding:4px 0;';

  const filterInput = document.createElement('input');
  filterInput.type = 'text';
  filterInput.placeholder = '🔍 Filter tags…';
  filterInput.style.cssText = 'width:calc(100% - 16px);margin:4px 8px;box-sizing:border-box;font-size:0.85em;';
  wrap.appendChild(filterInput);

  const listEl = document.createElement('div');
  wrap.appendChild(listEl);

  function _makeTagRow(uuid, tag) {
    const row = document.createElement('div');
    row.style.cssText = 'display:flex;align-items:center;gap:6px;padding:5px 10px;cursor:pointer;font-size:0.875em;';
    row.addEventListener('mouseenter', () => (row.style.background = 'var(--pico-primary-background,rgba(255,255,255,.08))'));
    row.addEventListener('mouseleave', () => (row.style.background = ''));
    if (tag.icon) {
      const ic = document.createElement('span');
      ic.textContent = tag.icon;
      row.appendChild(ic);
    }
    const nm = document.createElement('span');
    nm.textContent = tag.name ?? '';
    row.appendChild(nm);
    row.addEventListener('mousedown', async e => { e.preventDefault(); await onToggle(uuid); });
    return row;
  }

  function renderList(filter) {
    listEl.innerHTML = '';
    const lower = (filter ?? '').toLowerCase();
    for (const [uuid, tag] of tagsMap) {
      if (lower && !(tag.name ?? '').toLowerCase().includes(lower)) continue;
      listEl.appendChild(_makeTagRow(uuid, tag));
    }

    // + Create tag
    const createRow = document.createElement('div');
    createRow.style.cssText = [
      'padding:5px 10px', 'cursor:pointer', 'font-size:0.875em',
      'color:var(--pico-primary,#6bc)',
      'border-top:1px solid var(--pico-muted-border-color,#444)',
      'margin-top:4px',
    ].join(';');
    createRow.textContent = '+ Create tag…';
    createRow.addEventListener('mouseenter', () => (createRow.style.background = 'var(--pico-primary-background,rgba(255,255,255,.08))'));
    createRow.addEventListener('mouseleave', () => (createRow.style.background = ''));
    createRow.addEventListener('mousedown', async e => {
      e.preventDefault();
      const name = prompt('New tag name:');
      if (!name?.trim()) return;
      try {
        const tag = await createNewTag(name.trim(), '#888888');
        await onToggle(tag.uuid);
      } catch (err) {
        console.error('[bboxProvider] createNewTag failed:', err);
      }
    });
    listEl.appendChild(createRow);
  }

  renderList('');
  filterInput.addEventListener('input', () => renderList(filterInput.value));
  return wrap;
}

/**
 * Build a label picker flyout panel.
 *
 * @param {(uuid: string|null) => Promise<void>} onSelect
 * @returns {Promise<HTMLElement>}
 */
async function _buildLabelPanel(onSelect) {
  const labels = await fetchLabels();
  const picker = createDynamicLabelPicker(labels, { onSelect });
  const wrap = document.createElement('div');
  wrap.style.cssText = 'min-width:240px;';
  wrap.appendChild(picker.element);
  // Auto-focus after mount
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

    // ── Add tag ────────────────────────────────────────────────────────────
    {
      label: `Add tag${labelSuffix}`,
      submenu: () => asyncPanel('200px', () => _buildTagPanel(async uuid => {
        await runAll(adapters, a => a.toggleTag(uuid));
      })),
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
      onSelect: () => {
        const imageUuids = [...new Set(adapters.map(a => a.getImageUuid?.()).filter(Boolean))];
        ctx.close();
        if (imageUuids.length) openSimilarImagesModal({ imageUuids });
      },
    },
    {
      label: 'Search for dissimilar bboxes…',
      disabled: count > 1,
      onSelect: () => {
        const bboxUuid = adapters[0].getBboxUuid?.() ?? null;
        const labelUuid = adapters[0].getLabelUuid?.() ?? null;
        ctx.close();
        if (bboxUuid) openDissimilarBboxesModal({ bboxUuid, labelUuid });
      },
    },
    {
      label: 'Search for dissimilar images…',
      disabled: count > 1,
      onSelect: () => {
        const imageUuid = adapters[0].getImageUuid?.() ?? null;
        ctx.close();
        if (imageUuid) openDissimilarImagesModal({ imageUuid });
      },
    },
  ];

  return [{ items }];
});
