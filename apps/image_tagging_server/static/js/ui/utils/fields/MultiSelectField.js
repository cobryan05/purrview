// @ts-check
import { Field } from './Field.js';
import { createMultiSelectDropdown } from '/app-static/js/ui/utils/createMultiSelectDropdown.js';

/**
 * Multi-select list field — the Field-shaped wrapper around
 * `createMultiSelectDropdown`, the compact trigger-plus-panel list used by the
 * review filter toolbar.
 *
 * Replaces the old `CheckboxField`, which drew every option as a full-width
 * `<label><input></label><br>` row: a 30-tag vocabulary became a 30-line wall
 * inside a schema form. This renders one button that names the selection and
 * opens a scrollable, resizable panel with Select All / Select None.
 *
 * Accepts the same `options` shape CheckboxField did (`{ key, text, color }`)
 * and `getValue()` likewise returns a plain array of keys, so it is a drop-in
 * swap at both ends.
 */
export class MultiSelectField extends Field {
  /**
   * @param {{
   *   options?: Array<string|{key: string, text?: string, color?: string}>,
   *   value?: Array<string|{key: string}>,
   *   label?: string,
   *   emptyText?: string,
   *   emptySummary?: string,
   *   inline?: boolean,
   *   onChange?: ((keys: string[]) => void)|null,
   * } & Record<string, any>} [params]
   */
  constructor({ label = '', emptyText, emptySummary = 'None selected', inline = false, ...rest } = {}) {
    super(rest);
    this._label = label;
    this._emptyText = emptyText;
    this._emptySummary = emptySummary;
    this._inline = inline;
    /** @type {ReturnType<typeof createMultiSelectDropdown>|null} */
    this._dropdown = null;
    /** @type {Array<{uuid: string, name: string, color?: string}>} */
    this._items = (this._options || []).map((opt) => MultiSelectField._toItem(opt));
    this._selected = MultiSelectField._toKeys(this._value);
  }

  /** Normalise a CheckboxField-style option into a dropdown item. */
  static _toItem(opt) {
    if (typeof opt === 'string') return { uuid: opt, name: opt };
    return { uuid: opt.key ?? opt.text, name: opt.text ?? opt.key, color: opt.color };
  }

  /** Values may arrive as bare keys or as `{ key, ... }` objects. */
  static _toKeys(value) {
    if (!Array.isArray(value)) return [];
    return value.map((v) => (v && typeof v === 'object' ? v.key : v)).filter((k) => k != null);
  }

  async renderEdit() {
    // Rebuild on every render so the widget is seeded from the current
    // selection — callers re-render in place (e.g. after adding a tag).
    this._dropdown?.destroy();
    this._dropdown = createMultiSelectDropdown({
      label: this._label,
      items: this._items,
      defaultSelected: this._selected,
      showModeToggle: false,
      allowExclude: false,
      emptyText: this._emptyText,
      emptySummary: this._emptySummary,
      inline: this._inline,
      onChange: (selected) => {
        this._selected = selected;
        this._onChange?.(selected);
      },
    });
    return this._dropdown.element;
  }

  async renderView() {
    const span = document.createElement('span');
    const names = this._selected.map((key) => this._items.find((i) => i.uuid === key)?.name ?? key);
    span.textContent = names.length ? names.join(', ') : '(none)';
    return span;
  }

  /**
   * Append an option created after construction (the "+ New tag" flow) and,
   * by default, select it. Keeps `_items` and the live widget in step so a
   * later re-render doesn't drop the new option.
   * @param {string|{key: string, text?: string, color?: string}} opt
   * @param {{ select?: boolean }} [options]
   */
  addOption(opt, { select = true } = {}) {
    const item = MultiSelectField._toItem(opt);
    if (this._items.some((i) => i.uuid === item.uuid)) return;
    this._items.push(item);
    if (select && !this._selected.includes(item.uuid)) this._selected = [...this._selected, item.uuid];
    this._dropdown?.addItem(item, { select });
  }

  /** @returns {string[]} selected keys */
  getValue() {
    return [...this._selected];
  }

  destroy() {
    this._dropdown?.destroy();
    this._dropdown = null;
  }
}
