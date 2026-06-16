/**
 * Shared tag-list flyout panel for context menus.
 *
 * Works for any adapter exposing:
 *   getTagUuids()    → string[]
 *   toggleTag(uuid)  → Promise<void>
 *
 * Used by both the 'bbox' and 'image' context-menu providers so the picker
 * UI (checkboxes, indeterminate state for multi-select, filter, "+ Create
 * tag…") isn't duplicated.
 */

import { fetchTags, createNewTag } from '/app-static/js/shared/api/tags.js';
import { filterVisibleTags } from '/app-static/js/shared/activeModel.js';

/** Run an async fn over every adapter in sequence. */
export async function runAll(adapters, fn) {
  for (const a of adapters) await fn(a);
}

/**
 * Build a tag-list panel (HTMLElement) for a Tags flyout.
 * Shows a checkbox per tag reflecting current state across all adapters.
 * Indeterminate = mixed selection; first click adds to all; second click removes.
 *
 * @param {Array<{getTagUuids: () => string[], toggleTag: (uuid: string) => Promise<void>}>} adapters
 * @returns {Promise<HTMLElement>}
 */
export async function buildTagPanel(adapters) {
  const tagsMap = await fetchTags();
  const visibleTags = filterVisibleTags(tagsMap.values());

  const wrap = document.createElement('div');
  wrap.style.cssText = 'min-width:220px;max-height:300px;overflow-y:auto;padding:4px 0;';

  const filterInput = document.createElement('input');
  filterInput.type = 'text';
  filterInput.placeholder = '🔍 Filter tags…';
  filterInput.style.cssText = 'width:calc(100% - 16px);margin:4px 8px;box-sizing:border-box;font-size:0.85em;';
  wrap.appendChild(filterInput);

  const listEl = document.createElement('div');
  wrap.appendChild(listEl);

  /** Returns 'checked' | 'unchecked' | 'indeterminate' for a tag across all adapters. */
  function _tagState(uuid) {
    const count = adapters.filter(a => a.getTagUuids().includes(uuid)).length;
    if (count === 0) return 'unchecked';
    if (count === adapters.length) return 'checked';
    return 'indeterminate';
  }

  /** Build one row; cb element is returned so callers can update it after a toggle. */
  function _makeTagRow(tag) {
    const row = document.createElement('div');
    row.style.cssText = 'display:flex;align-items:center;gap:6px;padding:5px 10px;cursor:pointer;font-size:0.875em;';
    row.addEventListener('mouseenter', () => (row.style.background = 'var(--pico-primary-background,rgba(255,255,255,.08))'));
    row.addEventListener('mouseleave', () => (row.style.background = ''));

    const cb = document.createElement('input');
    cb.type = 'checkbox';
    cb.style.cssText = 'flex-shrink:0;cursor:pointer;pointer-events:none;';
    const state = _tagState(tag.uuid);
    cb.checked = state === 'checked';
    cb.indeterminate = state === 'indeterminate';
    row.appendChild(cb);

    if (tag.icon) {
      const ic = document.createElement('span');
      ic.textContent = tag.icon;
      row.appendChild(ic);
    }

    const nm = document.createElement('span');
    nm.textContent = tag.name ?? '';
    nm.style.flex = '1';
    row.appendChild(nm);

    row.addEventListener('mousedown', async e => {
      e.preventDefault();
      const current = _tagState(tag.uuid);
      if (current === 'checked') {
        // Remove from all adapters that have it
        await runAll(adapters, async a => {
          if (a.getTagUuids().includes(tag.uuid)) await a.toggleTag(tag.uuid);
        });
      } else {
        // Add to all adapters that don't have it (indeterminate → sync to all; unchecked → add to all)
        await runAll(adapters, async a => {
          if (!a.getTagUuids().includes(tag.uuid)) await a.toggleTag(tag.uuid);
        });
      }
      const next = _tagState(tag.uuid);
      cb.checked = next === 'checked';
      cb.indeterminate = next === 'indeterminate';
    });

    return row;
  }

  function renderList(filter) {
    listEl.innerHTML = '';
    const lower = (filter ?? '').toLowerCase();
    for (const tag of visibleTags) {
      if (lower && !(tag.name ?? '').toLowerCase().includes(lower)) continue;
      listEl.appendChild(_makeTagRow(tag));
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
        await runAll(adapters, a => a.toggleTag(tag.uuid));
        renderList(filterInput.value);
      } catch (err) {
        console.error('[tagPanelBuilder] createNewTag failed:', err);
      }
    });
    listEl.appendChild(createRow);
  }

  renderList('');
  filterInput.addEventListener('input', () => renderList(filterInput.value));
  return wrap;
}

/**
 * Build a simple pick-one tag-list panel for flyouts that need a single
 * selection (no checkboxes, no filter input, no "+ Create tag" row).
 * Clicking a tag calls onSelect(uuid); the caller is responsible for
 * closing the menu afterward.
 *
 * @param {(uuid: string) => Promise<void>} onSelect
 * @returns {Promise<HTMLElement>}
 */
export async function buildPickTagPanel(onSelect) {
  const tagsMap = await fetchTags();
  const visibleTags = filterVisibleTags(tagsMap.values());

  const wrap = document.createElement('div');
  wrap.style.cssText = 'min-width:220px;max-height:260px;overflow-y:auto;padding:4px 0;';

  for (const tag of visibleTags) {
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
    nm.style.flex = '1';
    row.appendChild(nm);

    row.addEventListener('mousedown', e => { e.preventDefault(); onSelect(tag.uuid); });
    wrap.appendChild(row);
  }

  return wrap;
}
