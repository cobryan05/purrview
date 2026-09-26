/**
 * Building blocks for the compact list row shared by ui/labelList.js (labels,
 * in all three of its modes) and ui/tags.js.
 *
 * The row grammar is:
 *
 *     [rails][caret][swatch][…][name][…] ……… [hover action cluster]
 *
 * They live here rather than in labelList.js because labelList.js imports the
 * canvas tool/image modules, and ui/tags.js — which renders on the configuration
 * page with no canvas in sight — must not pull those in transitively.
 *
 * Styling lives in app.css under "Label tree: the shared compact row". The
 * `title` on every action button is load-bearing: playwright_tests select on
 * title ("Add Child", "Move", "Edit", "Delete", "Save", "Cancel", "Add label",
 * "Add tag", "Refresh"), never on class names.
 */

/** 16×16 stroked glyphs (caret is 10×10 filled). Sized by CSS, not by markup. */
export const ICONS = Object.freeze({
  caret:    '<svg viewBox="0 0 10 10" fill="currentColor" aria-hidden="true"><path d="M1 3h8L5 8z"/></svg>',
  addChild: '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" aria-hidden="true"><path d="M4 2v7a2 2 0 0 0 2 2h3"/><path d="M11.5 8.5v5M9 11h5"/></svg>',
  add:      '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" aria-hidden="true"><path d="M8 3.5v9M3.5 8h9"/></svg>',
  move:     '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M8 2.5v11M5.5 5 8 2.5 10.5 5M5.5 11 8 13.5 10.5 11"/></svg>',
  edit:     '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M11 2.5 13.5 5 6 12.5l-3 .5.5-3z"/></svg>',
  trash:    '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M3 4.5h10M6.5 4.5V3h3v1.5M4.5 4.5l.6 8.5h5.8l.6-8.5"/></svg>',
  refresh:  '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" aria-hidden="true"><path d="M13.5 8a5.5 5.5 0 1 1-1.7-4"/><path d="M13.5 2v3.2h-3.2"/></svg>',
  save:     '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="m3 8.5 3.5 3.5L13 4.5"/></svg>',
  cancel:   '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" aria-hidden="true"><path d="M4 4l8 8M12 4l-8 8"/></svg>',
  home:     '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M2.5 7 8 2.5 13.5 7"/><path d="M4 6.5v7h8v-7"/></svg>',
  variants: '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" aria-hidden="true"><circle cx="6" cy="8" r="4.5"/><circle cx="10" cy="8" r="4.5"/></svg>',
});

/**
 * A borderless icon button for a row's hover action cluster or a section header.
 *
 * The handler is awaited with the button disabled, so a slow PATCH can't be
 * double-fired. The handler usually re-renders and detaches the button; the
 * re-enable in `finally` is then a harmless no-op on a detached node.
 *
 * @param {keyof ICONS} icon
 * @param {string} title  also the accessible name, and what tests select on
 * @param {{ className?: string, active?: boolean, onClick?: (e: MouseEvent) => any, tour?: string }} [opts]
 *   `tour` sets the `data-tour` anchor id a guided tour points its spotlight at.
 * @returns {HTMLButtonElement}
 */
export function iconAction(icon, title, { className = '', active = false, onClick = null, tour = '' } = {}) {
  const btn = document.createElement('button');
  btn.type = 'button';
  btn.className = `lt-act${active ? ' lt-act--active' : ''}${className ? ` ${className}` : ''}`;
  btn.title = title;
  btn.setAttribute('aria-label', title);
  if (tour) btn.dataset.tour = tour;
  btn.innerHTML = ICONS[icon] ?? '';
  if (onClick) {
    btn.addEventListener('click', async (e) => {
      e.stopPropagation();
      if (btn.disabled) return;
      btn.disabled = true;
      try {
        await onClick(e);
      } finally {
        btn.disabled = false;
      }
    });
  }
  return btn;
}

/**
 * One hairline column per ancestor level. The row's own level gets an elbow;
 * ancestors that still have siblings below get a full-height vertical line, so a
 * deeply nested row stays traceable back to its parent.
 *
 * @param {Array<'line'|'elbow'|'elbow-cont'|'blank'>} kinds
 * @returns {HTMLSpanElement}
 */
export function makeRails(kinds) {
  const rails = document.createElement('span');
  rails.className = 'lt-rails';
  for (const kind of kinds) {
    const rail = document.createElement('span');
    rail.className = 'lt-rail';
    if (kind === 'line') rail.classList.add('lt-rail--line');
    else if (kind === 'elbow') rail.classList.add('lt-rail--elbow');
    else if (kind === 'elbow-cont') rail.classList.add('lt-rail--elbow', 'lt-rail--cont');
    rails.appendChild(rail);
  }
  return rails;
}

/**
 * A colour well: a swatch that *is* the `<input type="color">`.
 *
 * `onChange` fires on 'change' (picker committed), not 'input' (fires
 * continuously while dragging) — the callers PATCH on every call.
 *
 * @param {string} color
 * @param {((color: string) => any)|null} [onChange]
 * @returns {HTMLLabelElement & { getValue: () => string }}
 */
export function makeColorWell(color, onChange = null) {
  const well = /** @type {any} */ (document.createElement('label'));
  well.className = 'lt-color-well';
  well.style.background = color;
  well.title = 'Change colour';

  const input = document.createElement('input');
  input.type = 'color';
  input.value = color;
  input.setAttribute('aria-label', 'Colour');
  input.addEventListener('input', () => { well.style.background = input.value; });
  if (onChange) input.addEventListener('change', () => onChange(input.value));
  well.appendChild(input);

  well.getValue = () => input.value;
  return well;
}

/**
 * Swap a row into its editing state: colour well, optional extra controls, name
 * input, and a Save/Cancel pair — replacing everything after the rails/caret so
 * the editor lines up with the row it came from.
 *
 * @param {HTMLElement} row
 * @param {{
 *   name?: string,
 *   color?: string,
 *   uuid?: string|null,
 *   placeholder: string,
 *   extras?: HTMLElement[],          inserted between the colour well and the input
 *   onSave: (v: { name: string, color: string }) => any,
 *   onCancel: () => any,
 * }} opts
 */
export function openRowEditor(row, {
  name = '',
  color = '#cccccc',
  uuid = null,
  placeholder,
  extras = [],
  onSave,
  onCancel,
}) {
  if (row.dataset.editing === 'true') return;
  row.dataset.editing = 'true';

  for (const child of [...row.children]) {
    if (!child.classList.contains('lt-rails') && !child.classList.contains('lt-caret')) child.remove();
  }

  const editor = document.createElement('div');
  editor.className = 'lt-editor';

  const well = makeColorWell(color);
  const input = document.createElement('input');
  input.type = 'text';
  input.value = name;
  input.placeholder = placeholder;
  input.setAttribute('aria-label', placeholder);

  editor.append(well, ...extras, input);

  // The UUID used to occupy a permanent second line under every row. It's only
  // ever wanted while you're looking at that one item, so it lives here instead.
  if (uuid) {
    const hint = document.createElement('span');
    hint.className = 'lt-hint';
    hint.textContent = uuid;
    hint.title = 'UUID';
    editor.appendChild(hint);
  }

  const acts = document.createElement('div');
  acts.className = 'lt-acts lt-acts--open';
  const commit = () => onSave({ name: input.value.trim(), color: well.getValue() });
  acts.append(
    iconAction('save', 'Save', { onClick: commit }),
    iconAction('cancel', 'Cancel', { onClick: () => onCancel() }),
  );

  input.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') { e.preventDefault(); commit(); }
    else if (e.key === 'Escape') { e.preventDefault(); onCancel(); }
  });

  row.append(editor, acts);
  input.focus();
  input.select();
}

/**
 * Section header — "OBJECT LABELS ──────── 22 ↻".
 *
 * The rule is a real element, not a ::after, because the count and any controls
 * have to sit on the far side of it.
 *
 * @param {string} title
 * @param {number|null} count  omit with null to show no count
 * @param {...(HTMLElement|null)} controls
 * @returns {HTMLDivElement}
 */
export function makeSectionHeader(title, count, ...controls) {
  const header = document.createElement('div');
  header.className = 'lt-section';

  const titleEl = document.createElement('span');
  titleEl.className = 'lt-section__title';
  titleEl.dataset.testid = 'lt-section-title';
  titleEl.textContent = title;
  header.appendChild(titleEl);

  const rule = document.createElement('span');
  rule.className = 'lt-section__rule';
  header.appendChild(rule);

  if (count != null) {
    const countEl = document.createElement('span');
    countEl.className = 'lt-section__count';
    countEl.textContent = String(count);
    header.appendChild(countEl);
  }
  controls.filter(Boolean).forEach((c) => header.appendChild(c));
  return header;
}
