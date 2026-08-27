/**
 * Render a plugin's `params_schema()` into a form, and read it back.
 *
 * One renderer for every registry that declares a schema — trackers, actions, and
 * whatever comes next. The alternative, which this replaces for actions, was a
 * hand-written form per kind: two lists of parameters with nothing comparing them, so a
 * parameter added to the Python side simply never appeared in the UI, and one removed
 * from it carried on being written into saved config by a form that still offered it.
 * Both of those actually happened.
 *
 * ### The escape hatches, and why they exist
 *
 * A pure schema renderer would have been a regression, because the hand-written forms
 * earned their keep in four ways a flat schema cannot express. Rather than give that up,
 * a kind may pass a `decorator`:
 *
 * - **`hidden`** — a parameter the action supports but this form must not offer.
 *   `record_event_timelapse.output_dir` is the case: typing a path there puts the mp4
 *   outside the tree the retention sweep and the file endpoint know about.
 * - **`widgets`** — a type needing more than an input, e.g. `directory`, which wants a
 *   browse button and a folder picker.
 * - **`after`** — live previews computed from *several* fields at once ("→ 120× — a 30
 *   min visit plays in 15 s"). These are the difference between a form people configure
 *   and one they leave at its defaults, and no per-field declaration can produce them.
 * - **`labels` / `widths` / `order` / `testids`** — presentation, plus stable test ids so
 *   the existing browser tests keep testing behaviour rather than being rewritten to
 *   match a generated naming scheme.
 *
 * Everything not decorated still renders from the schema alone, which is the point: a
 * new action kind gets a working form with no change to this file or its caller.
 */

/** Widget per declared `type`. Keep in step with the renderable set asserted in
 *  `tests/db_server/test_watch_profiles.py::TestActionListing`.
 *
 *  `directory` and `training_collector_uuid` are listed but not built here — this
 *  module has no file browser and does no fetching. They reach the caller's
 *  `makeWidget`, which is the only reason they must NOT fall through to the `str`
 *  default: an unlisted type is coerced to `str` before `makeWidget` is consulted. */
const KNOWN_TYPES = new Set(['bool', 'int', 'float', 'str', 'directory', 'training_collector_uuid']);

/**
 * @param {object} schema The plugin's `paramsSchema`, keyed by snake_case param name.
 * @param {object} stored Saved params. May hold keys from a *different* kind — the user
 *   switched the picker — so the schema is the authority on what is read back.
 * @param {() => void} onAnyChange
 * @param {object} [opts]
 * @param {string} [opts.testidPrefix] Prefix for generated `data-testid`s.
 * @param {string} [opts.emptyText] Shown when the kind declares no parameters at all.
 * @param {object} [opts.decorator] See the module comment.
 * @param {(field: object, value: any, onChange: () => void) => {element: HTMLElement, read: () => any}} [opts.makeWidget]
 *   Builds a control for a type this module does not handle itself (currently
 *   `directory`, which needs the caller's picker).
 * @returns {{element: HTMLElement, getParams: () => object, controls: object}}
 */
export function buildSchemaParams(schema, stored, onAnyChange, opts = {}) {
  const {
    testidPrefix = 'schema-param',
    emptyText = 'This has no options.',
    decorator = {},
    makeWidget = null,
    fieldWrapper,
    numberInput,
    el,
    num,
  } = opts;

  const hidden = decorator.hidden instanceof Set ? decorator.hidden : new Set(decorator.hidden || []);
  // Renders *into* the caller's element when given one, rather than into a wrapper of
  // its own. The action rows are a flex line of fields, and an intervening div makes
  // every field a child of a block instead — which stacks the row and costs ~80px of
  // editor scroll per action. There is a density test that catches exactly this.
  const box = opts.container || el('div', decorator.className || 'schema-params');
  const readers = {};
  const controls = {};

  // Declared order by default — a schema lists its fields in the order the author
  // thought about them, which is nearly always the order to read them in. `order` only
  // has to name the keys a kind wants moved to the front.
  const keys = Object.keys(schema || {}).filter(k => !hidden.has(k));
  const ordered = [...(decorator.order || []).filter(k => keys.includes(k)), ...keys.filter(k => !(decorator.order || []).includes(k))];

  for (const key of ordered) {
    const field = schema[key] || {};
    const type = KNOWN_TYPES.has(field.type) ? field.type : 'str';
    const saved = stored?.[key];
    // A saved value only counts if this kind declares the key; switching kinds leaves
    // foreign keys behind, and rendering them would resurrect a setting the user
    // cannot see and did not choose.
    const value = saved !== undefined ? saved : field.default;

    let control;
    let read;
    let element;

    if (Array.isArray(field.options) && field.options.length) {
      // A declared option list is a closed set — MQTT QoS is 0, 1 or 2 and nothing else.
      // Rendering it as a number box invites a 3 that the broker will reject at publish
      // time, somewhere far from this form.
      control = document.createElement('select');
      for (const option of field.options) {
        const node = document.createElement('option');
        node.value = String(option);
        node.textContent = String(option);
        control.appendChild(node);
      }
      control.value = String(value ?? field.default ?? field.options[0]);
      control.addEventListener('change', onAnyChange);
      const cast = type === 'int' || type === 'float' ? Number : String;
      read = () => cast(control.value);
      element = control;
    } else if (type === 'bool') {
      control = document.createElement('input');
      control.type = 'checkbox';
      // `!== false` rather than `=== true`: a schema default of true must survive a
      // stored value that has never been written, and `undefined` must take the default
      // rather than reading as unchecked.
      control.checked = value !== undefined ? value === true : field.default === true;
      control.addEventListener('change', onAnyChange);
      read = () => control.checked;
      element = control;
    } else if (type === 'int' || type === 'float') {
      const step = decorator.steps?.[key] ?? (type === 'int' ? '1' : '0.05');
      control = numberInput(value ?? '', { step, min: decorator.mins?.[key] ?? 0 });
      control.addEventListener('input', onAnyChange);
      // Number('') === 0, which has already caused a silent-stop bug in this codebase —
      // an empty box must mean "the default", not "zero".
      read = () => (type === 'int' ? Math.round(num(control, field.default ?? 0)) : num(control, field.default ?? 0));
      element = control;
    } else if (makeWidget) {
      const widget = makeWidget(type, field, value, onAnyChange);
      if (widget) {
        control = widget.control ?? widget.element;
        read = widget.read;
        element = widget.element;
      }
    }

    if (!element) {
      control = document.createElement('input');
      control.type = 'text';
      control.value = value ?? '';
      control.addEventListener('input', onAnyChange);
      read = () => control.value.trim();
      element = control;
    }

    control.dataset.testid = decorator.testids?.[key] || `${testidPrefix}-${key}`;
    if (field.help) control.title = field.help;

    readers[key] = read;
    controls[key] = { control, element, field, read };

    const wrapped = fieldWrapper(decorator.labels?.[key] || key.replace(/_/g, ' '), element, {
      width: type === 'bool' ? null : (decorator.widths?.[key] || '6rem'),
      // The schema's own help, so the caption explains itself. A kind that declares none
      // says so rather than hovering blank — a bare `track_buffer` explains nothing.
      help: field.help || `${key} — this declares no description for it.`,
    });

    // Previews go directly after the field they are about, so the number and its
    // consequence are read together. A list is allowed because one field can need more
    // than one: the directory fields want their placeholder help *and* a rendered
    // example of the path those placeholders produce.
    const after = decorator.after?.[key];
    const extras = after ? [after(controls, box)].flat().filter(Boolean) : [];

    if (extras.length) {
      // Field and hints in one group, so the row wraps *between* groups and never
      // between a number and the sentence explaining what it means. As separate flex
      // items they were free to land on different lines, which put "→ the 30 s before it
      // opened" underneath some unrelated field.
      //
      // Only when there is something to group — an extra element around every plain
      // field would be DOM for nothing, and the row's own flex layout handles those.
      const group = el('div', 'schema-field-group');
      group.appendChild(wrapped);
      for (const node of extras) group.appendChild(node);
      box.appendChild(group);
    } else {
      box.appendChild(wrapped);
    }
  }

  if (!box.children.length) box.appendChild(el('span', 'watch-hint', emptyText));

  return {
    element: box,
    controls,
    getParams: () => Object.fromEntries(Object.entries(readers).map(([k, r]) => [k, r()])),
  };
}
