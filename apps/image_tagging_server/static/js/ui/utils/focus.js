/**
 * Focus helpers shared by the hotkey dispatcher and the canvas stage.
 *
 * Lives here rather than in app/hotkeys.js so canvas/main.js can use it without
 * importing back into the module that already imports the canvas modules.
 */

/** <input> types that consume typed characters, so a hotkey must not steal them. */
const TEXT_INPUT_TYPES = new Set([
  'text', 'search', 'email', 'url', 'tel', 'password', 'number',
  'date', 'datetime-local', 'month', 'week', 'time',
]);

/**
 * True when the element is a text-entry field — the only kind of focus that
 * should swallow canvas hotkeys.
 *
 * Deliberately narrower than `tagName === 'INPUT'`: the inference panel's model
 * checkboxes and confidence slider are inputs too, and treating them as text
 * entry killed every hotkey for the rest of the canvas's life, because Konva
 * calls preventDefault() on a pointerdown that hits a shape — so clicking a
 * bbox to "focus the canvas" never moved focus off the input.
 *
 * @param {Element|null|undefined} el
 * @returns {boolean}
 */
export function isTextEntryElement(el) {
  if (!el) return false;
  if (el.isContentEditable) return true;
  const tag = el.tagName;
  if (tag === 'TEXTAREA' || tag === 'SELECT') return true;
  if (tag !== 'INPUT') return false;
  // .type is normalized, lowercase, and reports 'text' for unknown types.
  return TEXT_INPUT_TYPES.has(el.type);
}

/** Drop focus from a focused text-entry field so canvas hotkeys resume. No-op otherwise. */
export function blurTextEntryFocus() {
  const el = document.activeElement;
  if (isTextEntryElement(el)) el.blur();
}
