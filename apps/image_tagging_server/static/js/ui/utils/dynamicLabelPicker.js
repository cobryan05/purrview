import { createNewLabel } from '/app-static/js/shared/api/labels.js';

/**
 * Creates a combobox-style label picker that:
 *  - Filters all labels (standard + dynamic) as the user types
 *  - Offers a "Create as dynamic label" option for unmatched names
 *  - Offers a "No label" option for generic false positives
 *
 * @param {Map<string, object>} allLabels - Map of uuid → LabelRead (ALL labels, not pre-filtered)
 * @param {{ onSelect?: (uuid: string|null) => void }} [opts]
 * @returns {{ element: HTMLElement, getValue: () => string|null, destroy: () => void }}
 */
export function createDynamicLabelPicker(allLabels, { onSelect } = {}) {
  let _selectedUuid = null;
  let _outsideClickHandler = null;

  const wrap = document.createElement('div');
  wrap.style.cssText = 'position:relative;';

  const input = document.createElement('input');
  input.type = 'text';
  input.placeholder = 'Search or create label…';
  input.style.cssText = 'width:100%;box-sizing:border-box;';

  const results = document.createElement('div');
  results.style.cssText = [
    'display:none',
    'position:absolute',
    'z-index:1000',
    'left:0',
    'right:0',
    'max-height:200px',
    'overflow-y:auto',
    'border:1px solid var(--pico-muted-border-color, #444)',
    'background:var(--pico-card-background-color, #1a1a1a)',
    'border-radius:var(--pico-border-radius, 4px)',
    'margin-top:2px',
    'box-shadow:0 4px 12px rgba(0,0,0,0.4)',
  ].join(';');

  function _makeRow(content, onClick, extraStyle = '') {
    const row = document.createElement('div');
    row.style.cssText = `padding:6px 10px;cursor:pointer;font-size:0.875em;${extraStyle}`;
    row.addEventListener('mouseenter', () => (row.style.background = 'var(--pico-primary-background, rgba(255,255,255,0.08))'));
    row.addEventListener('mouseleave', () => (row.style.background = ''));
    if (typeof content === 'string') {
      row.textContent = content;
    } else {
      row.appendChild(content);
    }
    row.addEventListener('mousedown', (e) => {
      // Prevent the input from losing focus (which would close the dropdown
      // before the click event fires). Do NOT call onClick here — hiding the
      // results in mousedown causes the subsequent click to re-target the
      // overlay and triggers its outside-click close handler.
      e.preventDefault();
    });
    row.addEventListener('click', () => {
      onClick();
    });
    return row;
  }

  function _selectLabel(uuid, displayName) {
    _selectedUuid = uuid;
    input.value = displayName ?? '';
    _hideResults();
    onSelect?.(uuid);
  }

  function _showResults() {
    results.style.display = 'block';
  }

  function _hideResults() {
    results.style.display = 'none';
  }

  function _render(query) {
    results.innerHTML = '';

    // "No label" option
    results.appendChild(
      _makeRow(
        '— No label / generic false positive —',
        () => _selectLabel(null, ''),
        'color:var(--pico-muted-color, #888);font-style:italic;'
      )
    );

    const q = query.trim().toLowerCase();
    let hasExactMatch = false;

    for (const lbl of allLabels.values()) {
      if (q && !lbl.name.toLowerCase().includes(q)) continue;
      if (lbl.name.toLowerCase() === q) hasExactMatch = true;

      const isDynamic = lbl.labelType === 'dynamic';

      const inner = document.createElement('span');
      inner.style.cssText = 'display:flex;align-items:center;gap:6px;';

      const swatch = document.createElement('span');
      swatch.style.cssText = `width:8px;height:8px;border-radius:50%;flex-shrink:0;background:${lbl.color ?? '#888'};display:inline-block;`;

      const name = document.createElement('span');
      name.textContent = lbl.name;

      inner.append(swatch, name);

      if (isDynamic) {
        const badge = document.createElement('span');
        badge.textContent = '[dynamic]';
        badge.style.cssText = 'font-size:0.75em;color:var(--pico-muted-color,#888);margin-left:auto;';
        inner.appendChild(badge);
      }

      results.appendChild(_makeRow(inner, () => _selectLabel(lbl.uuid, lbl.name)));
    }

    // "Create" option when typed name has no exact match
    const trimmed = query.trim();
    if (trimmed && !hasExactMatch) {
      const createRow = _makeRow(
        `+ Create "${trimmed}" as dynamic label`,
        async () => {
          createRow.textContent = 'Creating…';
          try {
            const newLabel = await createNewLabel(trimmed, '#888888', null, 'dynamic');
            if (newLabel) {
              allLabels.set(newLabel.uuid, newLabel);
              _selectLabel(newLabel.uuid, newLabel.name);
            }
          } catch (err) {
            console.error('Failed to create dynamic label:', err);
            createRow.textContent = 'Error creating label';
          }
        },
        'color:var(--pico-primary, #6af);font-style:italic;border-top:1px solid var(--pico-muted-border-color,#333);'
      );
      results.appendChild(createRow);
    }

    _showResults();
  }

  input.addEventListener('input', () => _render(input.value));
  input.addEventListener('focus', () => _render(input.value));
  input.addEventListener('blur', () => {
    // Small delay so mousedown on a result row fires first
    setTimeout(_hideResults, 150);
  });

  _outsideClickHandler = (e) => {
    if (!wrap.contains(e.target)) _hideResults();
  };
  document.addEventListener('click', _outsideClickHandler);

  wrap.appendChild(input);
  wrap.appendChild(results);

  return {
    element: wrap,
    getValue: () => _selectedUuid,
    /** Call when the containing UI is removed to clean up the document listener. */
    destroy() {
      if (_outsideClickHandler) {
        document.removeEventListener('click', _outsideClickHandler);
        _outsideClickHandler = null;
      }
    },
  };
}
