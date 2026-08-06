// tags.js
//
// The tag list uses the same compact row as the label tree (see
// ui/utils/treeRow.js and the "Label tree" block in app.css) — swatch, icon,
// name, then an action cluster that only appears on hover. Tags are flat, so
// there are no rails and no caret.

import { clearTagCache, createNewTag, deleteTag, fetchTags, updateTag } from '/app-static/js/shared/api/tags.js';
import { groupTagsByKind } from '/app-static/js/shared/activeModel.js';
import { showPaneError, toast } from '/app-static/js/ui/utils/index.js';
import { openAnchoredPopup } from '/app-static/js/ui/utils/anchoredPopup.js';
import { iconAction, makeColorWell, makeSectionHeader, openRowEditor } from '/app-static/js/ui/utils/treeRow.js';

const DEFAULT_COLOR = '#cccccc';

/** First entry means "no icon". Mirrors the palette the old TagField offered. */
const ICON_PALETTE = ['—', '?', '!', '★', '⚠', '✓', '✗'];

/**
 * Compact icon control for the inline editor: a trigger showing the current
 * icon, opening a palette plus a free-text box for anything not in it.
 *
 * @param {string|null} initial
 * @returns {HTMLButtonElement & { getValue: () => string|null }}
 */
function makeIconPicker(initial) {
  const trigger = /** @type {any} */ (document.createElement('button'));
  trigger.type = 'button';
  trigger.className = 'lt-icon-trigger';
  trigger.title = 'Tag icon (drawn on the bbox in the canvas)';
  trigger.setAttribute('aria-label', 'Tag icon');

  let icon = initial ?? null;
  const sync = () => { trigger.textContent = icon ?? 'icon'; };
  sync();

  let popup = null;
  trigger.addEventListener('click', (e) => {
    e.stopPropagation();
    if (popup) { popup.close(); popup = null; return; }

    const menu = document.createElement('div');
    menu.className = 'lt-icon-menu';

    for (const symbol of ICON_PALETTE) {
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'lt-icon-option';
      btn.textContent = symbol;
      btn.title = symbol === '—' ? 'No icon' : symbol;
      btn.addEventListener('click', () => {
        icon = symbol === '—' ? null : symbol;
        sync();
        popup?.close();
        popup = null;
      });
      menu.appendChild(btn);
    }

    const custom = document.createElement('input');
    custom.type = 'text';
    custom.className = 'lt-icon-custom';
    custom.maxLength = 2;
    custom.placeholder = 'custom';
    custom.setAttribute('aria-label', 'Custom icon character');
    if (icon && !ICON_PALETTE.includes(icon)) custom.value = icon;
    custom.addEventListener('input', () => {
      icon = custom.value.trim() || null;
      sync();
    });
    menu.appendChild(custom);

    popup = openAnchoredPopup(trigger, menu);
  });

  trigger.getValue = () => icon;
  return trigger;
}

/**
 * Render one tag row into `container`.
 *
 * System tags are owned by the app, not the user: rather than rendering them as
 * ordinary rows whose buttons are silently missing, they carry a "locked" chip,
 * so the absence of controls reads as deliberate.
 */
function renderTag(tag, container, editable, onSelectCallback, rerender) {
  const { name, color, icon, uuid } = tag;
  const isSystem = tag.kind === 'system';
  const canEdit = editable && !isSystem;

  const row = document.createElement('div');
  row.className = 'lt-row';
  row.dataset.uuid = uuid;
  row.dataset.testid = 'tag-list-row';

  const swatch = document.createElement('span');
  swatch.className = 'lt-swatch';
  swatch.style.background = color ?? DEFAULT_COLOR;
  row.appendChild(swatch);

  const iconEl = document.createElement('span');
  iconEl.className = 'lt-tagicon';
  iconEl.textContent = icon ?? '';
  row.appendChild(iconEl);

  const nameEl = document.createElement('span');
  nameEl.className = `lt-name${isSystem ? ' lt-name--muted' : ''}`;
  nameEl.textContent = name;
  row.appendChild(nameEl);

  const startEdit = () => {
    const picker = makeIconPicker(icon ?? null);
    openRowEditor(row, {
      name,
      color: color ?? DEFAULT_COLOR,
      uuid,
      placeholder: 'Tag name',
      extras: [picker],
      onSave: async ({ name: newName, color: newColor }) => {
        if (!newName) { toast('Error: Name required', 5000, 'error'); return; }
        try {
          await updateTag(uuid, newName, newColor, picker.getValue());
          rerender();
        } catch (err) {
          toast(err.message, 5000, 'error');
        }
      },
      onCancel: () => rerender(),
    });
  };

  if (canEdit) {
    nameEl.classList.add('lt-name--action');
    nameEl.title = 'Click to rename';
    nameEl.addEventListener('click', (e) => { e.stopPropagation(); startEdit(); });

    const acts = document.createElement('div');
    acts.className = 'lt-acts';
    acts.append(
      iconAction('edit', 'Edit', { onClick: startEdit }),
      iconAction('trash', 'Delete', {
        className: 'lt-act--danger',
        onClick: async () => {
          if (!window.confirm(`Delete tag "${name}"?`)) return;
          try {
            await deleteTag(uuid);
            rerender();
          } catch (err) {
            toast(err.message, 5000, 'error');
          }
        },
      }),
    );
    row.appendChild(acts);
  } else if (isSystem) {
    const locked = document.createElement('span');
    locked.className = 'lt-locked';
    locked.textContent = 'locked';
    locked.title = 'System tag — created and managed by the app';
    row.appendChild(locked);
  }

  if (onSelectCallback) {
    row.style.cursor = 'pointer';
    row.addEventListener('click', () => onSelectCallback(uuid));
  }

  container.appendChild(row);
}

/** The always-visible "new tag" row: colour, icon, name, Add, Refresh. */
function renderCreateRow(container, rerender) {
  const row = document.createElement('div');
  row.className = 'lt-newrow';
  row.dataset.testid = 'tag-list-new-row';

  const well = makeColorWell(DEFAULT_COLOR);
  const picker = makeIconPicker(null);
  const input = document.createElement('input');
  input.type = 'text';
  input.placeholder = 'New Tag Name';
  input.setAttribute('aria-label', 'New Tag Name');

  const submit = async () => {
    const name = input.value.trim();
    if (!name) { toast('Error: Name required', 5000, 'error'); return; }
    try {
      await createNewTag(name, well.getValue() || DEFAULT_COLOR, picker.getValue());
      rerender();
    } catch (err) {
      toast(err.message, 5000, 'error');
    }
  };
  input.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') { e.preventDefault(); submit(); }
  });

  row.append(
    well,
    picker,
    input,
    iconAction('add', 'Add tag', { onClick: submit }),
    iconAction('refresh', 'Refresh', { onClick: () => { clearTagCache(); rerender(); } }),
  );
  container.appendChild(row);
}

/**
 * Renders all tags into a container.
 *
 * @param {{ target?: string, editable?: boolean, onSelectCallback?: ((uuid: string) => void)|null }} [opts]
 */
export function renderTagList({ target = 'tags-list', editable = true, onSelectCallback = null } = {}) {
  try {
    const targetElement = document.getElementById(target);
    if (!targetElement) {
      console.error('Tag container not found');
      return;
    }
    targetElement.innerHTML = '';
    const container = document.createElement('div');
    container.className = 'label-tree';
    targetElement.appendChild(container);

    const rerender = () => renderTagList({ target, editable, onSelectCallback });

    fetchTags().then(tags => {
      const byKind = groupTagsByKind(tags.values(), ['generic', 'system', 'task_generated']);

      for (const { label, tags: group } of byKind) {
        container.appendChild(makeSectionHeader(label, group.length));
        for (const tag of group) {
          renderTag(tag, container, editable, onSelectCallback, rerender);
        }
      }

      if (editable) renderCreateRow(container, rerender);
    }).catch(err => {
      showPaneError(targetElement, err.message);
    });
  } catch (err) {
    console.error('Failed to render tag list:', err);
  }
}
