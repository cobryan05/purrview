import { fetchLabels } from '/app-static/js/shared/api/labels.js';
import { fetchSourceIngestTiers, fetchSourceList } from '/app-static/js/shared/api/sources.js';
import { fetchModelGroups } from '/app-static/js/shared/api/modelGroups.js';
import { fetchModelsList } from '/app-static/js/shared/api/models.js';
import {
  fetchWatchProfiles, createWatchProfile, updateWatchProfile, deleteWatchProfile,
  validateWatchProfile, fetchTrackers,
} from '/app-static/js/shared/api/watchProfiles.js';
import { toast, buildPanelEditorBar } from '/app-static/js/ui/utils/index.js';
import { createListEditorScreen } from '/app-static/js/ui/utils/listEditorScreen.js';
import { promptRestartAffectedTasks } from '/app-static/js/ui/utils/restartAffectedTasks.js';
import { openSourceRegionsModal } from '/app-static/js/ui/sourceRegionsModal.js';
import { fetchSourceRegions, REGION_ROLE_MASK, REGION_ROLE_ZONE } from '/app-static/js/shared/api/sourceRegions.js';
import { createMultiSelectDropdown } from '/app-static/js/ui/utils/createMultiSelectDropdown.js';
import { openFileBrowserModal } from '/app-static/js/ui/utils/openFileBrowserModal.js';
import { openAnchoredPopup } from '/app-static/js/ui/utils/anchoredPopup.js';
import { flattenHierarchy } from '/app-static/js/ui/labelList.js';
import { takePendingSelection } from '/app-static/js/app/utils/tabs.js';

let _rendering = false;

// Relations, worded as a sentence fragment so a rule row reads as English. This is
// the only layout that makes "IoU vs containment" comprehensible to a non-expert.
// `iou` used to be worded "overlaps", which is the exact confusion it causes: IoU only
// reads as overlap for same-size boxes, and a cat lapping at a water bowl scores near
// zero on it. The honest wording sends that case to `overlaps` instead.
const RELATIONS = [
  { key: 'contains', text: 'is inside' },
  { key: 'overlaps', text: 'overlaps' },
  { key: 'centroid_in', text: 'has its centre in' },
  { key: 'iou', text: 'closely matches (same size)' },
  { key: 'distance', text: 'is near' },
  { key: 'present', text: 'is present' },
];

const ACTION_KINDS = [
  { key: 'publish_mqtt', text: 'Publish MQTT' },
  { key: 'publish_mqtt_image', text: 'Publish MQTT image' },
  { key: 'record_clip', text: 'Record clip' },
  { key: 'save_snapshot', text: 'Save snapshot' },
  { key: 'save_timelapse', text: 'Save timelapse' },
];

const TRIGGERS = [
  { key: 'event_started', text: 'event start' },
  { key: 'event_ended', text: 'event end' },
  { key: 'frame_analysed', text: 'every analysed frame' },
];

/**
 * The trigger that fires per frame rather than per event, mirroring
 * `FRAME_ANALYSED` in `apps/helpers/watch/scoping.py`. A frame belongs to no rule, so
 * an action on it has no rule scope to show.
 */
const FRAME_ANALYSED_TRIGGER = 'frame_analysed';

/**
 * Triggers each action kind can actually do something on. Absent = every trigger.
 *
 * Only `record_clip` is constrained, and the constraint is real rather than a policy:
 * `RecordClipAction.run` returns early unless the phase is END, because the window to
 * cut is not known until the event closes — that is the whole point of recording into
 * a rewindable buffer. Offering the other two triggers let a watcher be saved with a
 * clip action that could never fire, with no error at save time and no clip at run
 * time. Keyed by kind so a second constrained action is one line here.
 */
const ACTION_TRIGGERS = {
  record_clip: ['event_ended'],
  save_timelapse: ['frame_analysed'],
};

/** Why an action kind's trigger list is shorter than the full set. */
const TRIGGER_LIMIT_HELP = {
  record_clip:
    'A clip can only be cut when the event ends. Its length is not known until then — '
    + 'the pre-roll is taken retroactively out of the rolling buffer, which is what lets '
    + 'you record the seconds before the rule fired. Use the pre-roll and post-roll '
    + 'settings to cover what happened either side.',
  save_timelapse:
    'A timelapse has to keep shooting through the hours when nothing happens, so it '
    + 'runs on the frame the watcher is already pulling from the camera rather than on '
    + 'an event. Its own interval decides how often a frame is kept — everything in '
    + 'between is discarded without being written. For a picture of one event, use '
    + 'Save snapshot instead.',
};

/** The triggers offered for one action kind. */
function triggersFor(kind) {
  const allowed = ACTION_TRIGGERS[kind];
  return allowed ? TRIGGERS.filter((t) => allowed.includes(t.key)) : TRIGGERS;
}

/**
 * Default image topic template, mirroring `DEFAULT_IMAGE_TOPIC_TEMPLATE` in
 * `apps/helpers/actions/mqttPublishImageAction.py`. Keyed by source rather than by
 * rule: the frame belongs to the camera, whatever happened to fire on it.
 */
const DEFAULT_IMAGE_TOPIC_TEMPLATE = '{watcher}/{source}/image';

/**
 * Default timelapse directory, relative to the image dataset root. Mirrors
 * `DEFAULT_TIMELAPSE_DIR` in `apps/helpers/actions/saveTimelapseAction.py`.
 */
const DEFAULT_TIMELAPSE_DIR = 'timelapse';

/**
 * Default snapshot folder, relative to the image dataset root. Mirrors
 * `DEFAULT_SNAPSHOT_DIR` in `apps/helpers/actions/saveSnapshotAction.py`.
 */
const DEFAULT_SNAPSHOT_DIR = 'watch_snapshots';

/**
 * Default MQTT topic template — reproduces the historical hard-coded scheme exactly, so
 * an action saved before templates existed publishes where it always did.
 * Mirrors `DEFAULT_TOPIC_TEMPLATE` in `apps/helpers/actions/mqttPublishAction.py`.
 */
const DEFAULT_TOPIC_TEMPLATE = '{watcher}/{source}/event/{rule}';

/**
 * The placeholder vocabulary, mirroring `apps/helpers/watch/templating.py`.
 * Kept in sync by hand — the backend is the authority; this list is for help only.
 *
 * Each entry carries its own one-line meaning. A bare list of names told you a token
 * existed but not what it resolved to, which for pairs like `{rule}` / `{rule_name}` or
 * `{source}` / `{source_config_name}` is the only thing worth knowing.
 */
const TEMPLATE_PLACEHOLDER_HELP = [
  { token: '{rule}', help: 'The rule\'s event label, or its name when it has none.' },
  { token: '{rule_name}', help: 'The rule\'s name, always — never the event label.' },
  { token: '{event_label}', help: 'The rule\'s event label. Empty when it has none.' },
  { token: '{rule_uuid}', help: 'Stable id of the rule. Survives a rename.' },
  { token: '{watcher}', help: 'This watcher\'s name.' },
  { token: '{watcher_uuid}', help: 'Stable id of the watcher.' },
  { token: '{source}', help: 'The source\'s MQTT name for this watcher, falling back to its own name.' },
  { token: '{source_config_name}', help: 'The source\'s name as the Sources pane shows it, ignoring any MQTT name.' },
  { token: '{source_uuid}', help: 'Stable id of the source. Survives a rename.' },
  { token: '{subject}', help: 'Which label actually matched — the cat, not the whole list.' },
  { token: '{subject_uuid}', help: 'Stable id of the matched subject.' },
  { token: '{target}', help: 'Which zone or label it matched against — the litter box that fired.' },
  { token: '{target_uuid}', help: 'Stable id of the matched target.' },
  { token: '{phase}', help: 'start, update or end.' },
  { token: '{event_uuid}', help: 'Stable id of this one event, for correlating topics.' },
  // Local time, not UTC: these name folders a human browses, and a UTC day boundary
  // files an evening under tomorrow. Dashes in {time} because it goes into filenames.
  { token: '{date}', help: 'The local date, as 2026-08-18.' },
  { token: '{time}', help: 'The local time, as 14-30-00. Dashes, so it is safe in a filename.' },
  { token: '{year}', help: 'Local year, as 2026.' },
  { token: '{month}', help: 'Local month, zero-padded, as 08.' },
  { token: '{day}', help: 'Local day of the month, zero-padded, as 18.' },
  { token: '{hour}', help: 'Local hour, 24-hour and zero-padded, as 14.' },
];

/** Names only, for the places that just need to list them. */
const TEMPLATE_PLACEHOLDERS = TEMPLATE_PLACEHOLDER_HELP.map(p => p.token);

/** Stand-in values for the topic preview, so the shape of the result is visible. */
const TEMPLATE_EXAMPLE = {
  rule: 'cat-using-litter-box',
  rule_name: 'cat-using-litter-box',
  event_label: 'litterbox',
  rule_uuid: 'rule-uuid',
  watcher: 'house-cams',
  watcher_uuid: 'watcher-uuid',
  // The watcher's MQTT name for the source when one is set, which is the whole reason
  // the override exists — a short, stable topic level rather than the descriptive name
  // the Sources list shows.
  source: 'back-room',
  source_config_name: 'back-room-reolink-upstairs',
  source_uuid: 'source-uuid',
  subject: 'tabby',
  subject_uuid: 'label-uuid',
  target: 'carpeted-litter-box',
  target_uuid: 'zone-uuid',
  phase: 'start',
  event_uuid: 'event-uuid',
  date: '2026-08-18',
  time: '14-30-00',
  year: '2026',
  month: '08',
  day: '18',
  hour: '14',
};

/**
 * Render a topic template against example values, for the live hint under the field.
 *
 * Mirrors `render_topic`: unknown placeholders are left verbatim so a typo is visible,
 * and empty levels collapse rather than emitting `a//b`.
 */
function renderTopicExample(template) {
  const rendered = template.replace(/\{([a-zA-Z][a-zA-Z0-9_]*)\}/g, (match, key) => (
    Object.prototype.hasOwnProperty.call(TEMPLATE_EXAMPLE, key) ? TEMPLATE_EXAMPLE[key] : match
  ));
  return rendered.split('/').filter(Boolean).join('/');
}

/**
 * Read a numeric input, falling back when it is blank.
 *
 * `Number('')` is 0, which has already caused a silent-stop bug in this codebase —
 * an empty "Max duration" field would become a 0-second cap that closes every event
 * immediately.
 */
function num(input, fallback) {
  const raw = input?.value;
  if (raw === '' || raw == null) return fallback;
  const parsed = Number(raw);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text != null) node.textContent = text;
  return node;
}

function select(options, value, onChange) {
  const sel = document.createElement('select');
  for (const opt of options) {
    const o = document.createElement('option');
    o.value = opt.key;
    o.textContent = opt.text;
    sel.appendChild(o);
  }
  sel.value = value ?? options[0]?.key;
  if (onChange) sel.addEventListener('change', onChange);
  return sel;
}

// ── Model selection ─────────────────────────────────────────────────────────
// A watcher picks *either* a Model Stack or one bare model. Two dropdowns would need a
// rule about which wins; one dropdown with two optgroups makes them exclusive by
// construction, which is exactly what the backend enforces. The option values are
// prefixed because a stack uuid and a model name share no namespace and a bare value
// could not say which it was.
const MODEL_CHOICE_GROUP = 'g:';
const MODEL_CHOICE_MODEL = 'm:';

/**
 * Encode a stored (groupUuid, modelName) pair as an option value.
 * @returns {string} '' when neither is set
 */
function modelChoiceValue(groupUuid, modelName) {
  if (groupUuid) return `${MODEL_CHOICE_GROUP}${groupUuid}`;
  if (modelName) return `${MODEL_CHOICE_MODEL}${modelName}`;
  return '';
}

/**
 * Decode an option value back into the two fields the API expects.
 *
 * Both are returned every time, one of them null — the payload has to *clear* the
 * other field, not omit it, or switching a source from a stack to a single model
 * would leave the old stack in place and the backend would reject the pair.
 *
 * @returns {{modelGroupUuid: ?string, modelName: ?string}}
 */
function readModelChoice(value) {
  if (value?.startsWith(MODEL_CHOICE_GROUP)) {
    return { modelGroupUuid: value.slice(MODEL_CHOICE_GROUP.length), modelName: null };
  }
  if (value?.startsWith(MODEL_CHOICE_MODEL)) {
    return { modelGroupUuid: null, modelName: value.slice(MODEL_CHOICE_MODEL.length) };
  }
  return { modelGroupUuid: null, modelName: null };
}

/**
 * A `<select>` of "(fallback) | stacks… | models…".
 *
 * @param {{groupOptions: Array, modelNames: string[]}} ctx
 * @param {string} emptyText  what "neither" means here — inherit, or nothing at all
 * @param {string} value      from modelChoiceValue()
 */
function modelChoiceSelect(ctx, emptyText, value, onChange, { testid = 'watch-model-choice' } = {}) {
  const sel = document.createElement('select');
  // The profile default and the per-source override are the same widget in different
  // places; distinct testids because the default renders first and would otherwise
  // capture every `.first` locator aimed at a source row.
  sel.dataset.testid = testid;

  const none = document.createElement('option');
  none.value = '';
  none.textContent = emptyText;
  sel.appendChild(none);

  const addGroup = (label, options) => {
    if (!options.length) return;
    const group = document.createElement('optgroup');
    group.label = label;
    for (const opt of options) {
      const o = document.createElement('option');
      o.value = opt.key;
      o.textContent = opt.text;
      group.appendChild(o);
    }
    sel.appendChild(group);
  };

  addGroup('Model stacks', ctx.groupOptions.map(g => ({ key: `${MODEL_CHOICE_GROUP}${g.key}`, text: g.text })));
  addGroup('Models', ctx.modelNames.map(name => ({ key: `${MODEL_CHOICE_MODEL}${name}`, text: name })));

  // A stack or model that has since been deleted would otherwise silently fall back to
  // the first option and be saved as that on the next edit. Keep it, marked.
  if (value && !sel.querySelector(`option[value="${CSS.escape(value)}"]`)) {
    const orphan = document.createElement('option');
    orphan.value = value;
    orphan.textContent = `${value.slice(2)} (missing)`;
    sel.appendChild(orphan);
  }

  sel.value = value || '';
  if (onChange) sel.addEventListener('change', onChange);
  return sel;
}

function numberInput(value, { step = '1', min = null, placeholder = '' } = {}) {
  const input = document.createElement('input');
  input.type = 'number';
  input.step = step;
  if (min != null) input.min = String(min);
  input.placeholder = placeholder;
  input.value = value ?? '';
  return input;
}

/**
 * A text input plus a `Browse…` button that opens the shared directory picker.
 *
 * Reuses `openFileBrowserModal` rather than growing a picker of its own, so creating a
 * directory that does not exist yet — which is the normal case for a new timelapse — is
 * the same "New Folder" affordance as everywhere else in the app.
 *
 * The input stays editable, unlike `DirBrowserField`'s read-only one: an action row is
 * a dense inline form, and typing `timelapse/front-door` is faster than three clicks
 * for a user who already knows where it goes.
 *
 * Paths are **dataset-root relative**, which is what the picker deals in and what the
 * `/api/images/list` and `/api/images/mkdir` endpoints behind it accept. Leading slashes
 * are cosmetic here; the action's `resolve_output_dir` accepts either spelling.
 */
function dirPickerField(value, onAnyChange, { placeholder = 'dataset root' } = {}) {
  const wrap = el('div', 'watch-dir-picker');
  const input = document.createElement('input');
  input.type = 'text';
  input.value = value ?? '';
  input.placeholder = placeholder;
  input.dataset.testid = 'watch-action-dir';
  const listeners = [];
  const changed = () => { listeners.forEach(fn => fn()); onAnyChange?.(); };
  input.addEventListener('input', changed);

  const browse = document.createElement('button');
  browse.type = 'button';
  browse.className = 'outline secondary watch-dir-picker__browse';
  browse.textContent = 'Browse…';
  browse.dataset.testid = 'watch-action-dir-browse';
  browse.addEventListener('click', async () => {
    const picked = await openFileBrowserModal({
      initialPath: input.value.trim() ? `/${input.value.trim().replace(/^\/+/, '')}` : '/',
      allowCreateDir: true,
    });
    // null is "cancelled", which must leave the typed value alone.
    if (picked === null) return;
    input.value = picked.replace(/^\/+/, '');
    changed();
  });

  wrap.append(input, browse);
  return {
    element: wrap,
    // Exposed so a `(?)` can insert a token at the caret, which needs the input itself
    // rather than its value — see `templateHelp`.
    input,
    getValue: () => input.value,
    onChange: (fn) => listeners.push(fn),
  };
}

/**
 * A compact inline `label: control` pair.
 *
 * Deliberately not `makeFieldRow`: that helper pins its label to a fixed 100px column
 * and gives the control `flex: 1`, which is right for a stacked settings panel and
 * wrong here — it turned a four-field row into four full-width lines. These pairs size
 * to their content and flow along the row instead.
 */
function wfield(labelText, control, { width = null } = {}) {
  const wrap = el('label', 'watch-field');
  wrap.appendChild(el('span', 'watch-field__label', labelText));
  if (width) control.style.width = width;
  wrap.appendChild(control);
  return wrap;
}

/**
 * A `(?)` beside a topic field, opening the placeholder vocabulary.
 *
 * The tokens used to be reachable only as the topic input's `title`, which is not
 * discoverable at all: a hover tooltip is invisible until you already suspect it is
 * there, so the templating feature read as a free-text box you had to guess at.
 *
 * Clicking a row inserts that token at the cursor rather than only describing it —
 * the list is then the fastest way to *write* a template, not merely documentation
 * about one, and it removes any chance of a typo in a token name.
 *
 * @param {HTMLInputElement} input the topic field tokens are inserted into
 */
function templateHelp(input, { intro, testid = 'watch-template-help', label } = {}) {
  return helpPopover({
    testid,
    label: label || 'Placeholders you can use here',
    build: (panel, close) => {
      panel.appendChild(el(
        'p',
        'watch-template-help__intro',
        intro
        || 'Substituted per topic level and slugified. Click one to insert it. An unknown '
        + 'placeholder is left as-is, so a typo shows up on the broker rather than '
        + 'silently collapsing a level.',
      ));

      for (const { token, help } of TEMPLATE_PLACEHOLDER_HELP) {
        const item = el('button', 'watch-template-help__item');
        item.type = 'button';
        item.appendChild(el('code', 'watch-template-help__token', token));
        item.appendChild(el('span', 'watch-template-help__desc', help));
        item.addEventListener('click', () => {
          // At the cursor, not appended: a template is usually edited in the middle, and
          // an insert that always lands at the end would have to be cut and re-pasted.
          const start = input.selectionStart ?? input.value.length;
          const end = input.selectionEnd ?? start;
          input.value = input.value.slice(0, start) + token + input.value.slice(end);
          const caret = start + token.length;
          input.setSelectionRange(caret, caret);
          input.focus();
          // The field's own listeners drive the live example and the dirty flag, and a
          // programmatic value change fires neither on its own.
          input.dispatchEvent(new Event('input', { bubbles: true }));
        });
        panel.appendChild(item);
      }
    },
  });
}

/**
 * A `(?)` that opens a short static explanation.
 *
 * Shares `templateHelp`'s classes and open/close behaviour deliberately: the two read
 * as the same affordance because they are, and a second set of styles for a popover
 * that differs only in having nothing clickable inside it would drift apart.
 *
 * Exists because a *removed* option explains itself least of all — a trigger list that
 * silently has one entry looks like the feature was never built, so the reason it is
 * short has to be reachable from where the choice would have been.
 *
 * @param {string} text the explanation
 * @param {{testid?: string, label?: string}} [opts]
 */
function noteHelp(text, { testid = 'watch-note-help', label = 'Show more information' } = {}) {
  return helpPopover({
    testid,
    label,
    build: (panel) => panel.appendChild(el('p', 'watch-template-help__intro', text)),
  });
}

/**
 * The `(?)` affordance both help flavours are built from.
 *
 * Positioning is delegated to `openAnchoredPopup` rather than done in CSS, which is the
 * whole reason this exists as one function. The panel used to be `position: absolute;
 * right: 0` inside the 1.35em button's wrapper, so a 24rem panel hung ~23rem to the left
 * of the button — fine for a field near the right of a wide window, and most of the way
 * off the left edge of the screen for every other case. `openAnchoredPopup` measures the
 * panel, clamps it into the viewport on both axes, flips it above or below depending on
 * the room available, and brings the outside-click and ESC dismissal with it.
 *
 * The panel therefore lives on `document.body` while open and is removed on close, which
 * also frees it from the action row's `overflow` — the other reason a popover anchored
 * inside a dense row is hard to see.
 *
 * @param {{testid: string, label: string, build: (panel: HTMLElement, close: () => void) => void}} opts
 */
function helpPopover({ testid, label, build }) {
  const wrap = el('span', 'watch-template-help');

  const btn = el('button', 'watch-template-help__btn', '?');
  btn.type = 'button';
  btn.dataset.testid = `${testid}-btn`;
  btn.title = label;
  btn.setAttribute('aria-label', label);
  btn.setAttribute('aria-expanded', 'false');

  let popup = null;
  function close() {
    popup?.close();
    popup = null;
    btn.setAttribute('aria-expanded', 'false');
  }
  btn.addEventListener('click', () => {
    if (popup) {
      close();
      return;
    }
    const panel = el('div', 'watch-template-help__panel');
    panel.dataset.testid = `${testid}-panel`;
    build(panel, close);
    popup = openAnchoredPopup(btn, panel);
    btn.setAttribute('aria-expanded', 'true');
  });

  wrap.appendChild(btn);
  return wrap;
}

/** The small ✕ that ends every row. Full-width "Remove" buttons cost a line each. */
function removeButton(onDelete, title) {
  const btn = el('button', 'watch-row-remove', '✕');
  btn.type = 'button';
  btn.title = title;
  btn.setAttribute('aria-label', title);
  btn.addEventListener('click', onDelete);
  return btn;
}

/**
 * A section heading with its "+ Add" button on the same line.
 *
 * The add button used to sit on its own line below the list, so every section spent
 * two lines on chrome before showing any content.
 */
function sectionHeader(text, addButton) {
  const head = el('div', 'watch-section-head');
  head.appendChild(el('span', 'panel-section-label', text));
  if (addButton) head.appendChild(addButton);
  return head;
}

// ── Tracker params ──────────────────────────────────────────────────────────
/**
 * Renders one tracker's `params_schema()` into a form, and reads it back.
 *
 * Schema-driven rather than hand-written per tracker: that is the whole point of the
 * registry, and it means adding a tracker needs no change in this file. Field *keys*
 * arrive snake_case (they are data, not model fields, so they are not camelised) and
 * are stored back exactly as declared — the Python side looks them up by those names.
 *
 * @param {object} schema `paramsSchema` for the selected tracker.
 * @param {object} stored Saved `trackerParamsJson`, which may be from another tracker.
 * @param {() => void} onAnyChange
 */
function buildTrackerParams(schema, stored, onAnyChange) {
  const box = el('div', 'watch-tracker-params');
  box.dataset.testid = 'watch-tracker-params';
  const readers = {};

  for (const [key, field] of Object.entries(schema || {})) {
    const type = field.type || 'str';
    const saved = stored?.[key];
    // A saved value only applies if it came from this tracker; switching trackers
    // leaves foreign keys in the stored params, and the schema is the authority.
    const value = saved !== undefined ? saved : field.default;
    let control;

    if (type === 'bool') {
      control = document.createElement('input');
      control.type = 'checkbox';
      control.checked = value === true;
      control.addEventListener('change', onAnyChange);
      readers[key] = () => control.checked;
    } else if (type === 'int' || type === 'float') {
      control = numberInput(value ?? '', { step: type === 'int' ? '1' : '0.05', min: 0 });
      control.addEventListener('input', onAnyChange);
      // Number('') === 0, which has already caused a silent-stop bug in this codebase —
      // an empty box must mean "the default", not "zero".
      readers[key] = () => (type === 'int' ? Math.round(num(control, field.default)) : num(control, field.default));
    } else {
      control = document.createElement('input');
      control.type = 'text';
      control.value = value ?? '';
      control.addEventListener('input', onAnyChange);
      readers[key] = () => control.value.trim();
    }

    control.dataset.testid = `watch-tracker-param-${key}`;
    if (field.help) control.title = field.help;
    box.appendChild(wfield(key.replace(/_/g, ' '), control, { width: type === 'bool' ? null : '6rem' }));
  }

  if (!box.children.length) {
    box.appendChild(el('span', 'watch-hint', 'This tracker has no options.'));
  }

  return {
    element: box,
    getParams: () => Object.fromEntries(Object.entries(readers).map(([k, read]) => [k, read()])),
  };
}

// ── Label lifetime row ──────────────────────────────────────────────────────
/**
 * One per-label tracking lifetime: how long that label's track survives without a
 * detection, and whether it latches.
 *
 * In **seconds, not frames**, matching the tracker: a frame count means 1 s on a 5 fps
 * camera and 0.2 s on a 25 fps one. This is what makes "a litter box survives 30 s of
 * misses while a cat survives 1 s" expressible.
 */
function buildLifetimeRow(lifetime, ctx) {
  const row = el('div', 'watch-lifetime-row watch-row');
  row.dataset.testid = 'watch-lifetime-row';

  const label = select(ctx.labelOptions, lifetime.labelUuid || ctx.labelOptions[0]?.key, ctx.onAnyChange);
  label.dataset.testid = 'watch-lifetime-label';
  row.appendChild(wfield('Label', label));

  const ttl = numberInput(lifetime.ttlS ?? 2.0, { step: '0.5', min: 0 });
  ttl.dataset.testid = 'watch-lifetime-ttl';
  ttl.title = 'Seconds the track survives with no detection before it is assumed gone.';
  ttl.addEventListener('input', ctx.onAnyChange);
  row.appendChild(wfield('TTL s', ttl, { width: '5rem' }));

  const anchor = document.createElement('input');
  anchor.type = 'checkbox';
  anchor.checked = lifetime.isAnchor === true;
  anchor.dataset.testid = 'watch-lifetime-anchor';
  anchor.title =
    'Latch this label’s box once seen confidently, instead of trusting a detector that '
    + 'intermittently misses a stationary object. Appropriate for furniture, not for animals.';
  anchor.addEventListener('change', ctx.onAnyChange);
  row.appendChild(wfield('Anchor', anchor));

  row.appendChild(removeButton(ctx.onDelete, 'Remove lifetime'));

  return {
    element: row,
    getLifetime: () => ({
      labelUuid: label.value || null,
      // ttl_s is rejected server-side at <= 0, so an emptied box must fall back rather
      // than send 0 and fail the whole save.
      ttlS: num(ttl, 2.0) || 2.0,
      isAnchor: anchor.checked,
    }),
  };
}

/**
 * The four-state buffer badge, from a source's ingest tier.
 *
 * Wording and glyphs match `WatchPipeline._buffer_label()`, which renders the same
 * four states on the running task's status card. Two places saying different things
 * about the same source is how a user learns to trust neither.
 *
 * @param {?{tier: string, reason: string, buffered: boolean, transcoding: boolean}} plan
 * @returns {{text: string, title: string, buffered: boolean}}
 */
function bufferBadgeFor(plan) {
  if (!plan) return { text: '', title: '', buffered: true };
  const suffix = plan.transcoding ? ' — transcoding (per-camera CPU)' : '';
  switch (plan.tier) {
    case 'native':
      return { text: '● rewind (native)', title: plan.reason + suffix, buffered: true };
    case 'ffmpeg':
      return { text: '● rewind (ffmpeg ingest)', title: plan.reason + suffix, buffered: true };
    case 'ring':
      return { text: '◐ frame ring (no rewind)', title: plan.reason, buffered: false };
    default:
      return { text: '⚠ no buffer', title: plan.reason, buffered: false };
  }
}

// ── Source row ──────────────────────────────────────────────────────────────
function buildSourceRow(src, ctx) {
  const row = el('div', 'watch-source-row watch-row');
  row.dataset.testid = 'watch-source-row';

  // Repointing a row at a different camera changes the uuid the actions' source scopes
  // are written against, so they have to be rebuilt — not just marked dirty.
  const sourceSel = select(ctx.sourceOptions, src.sourceUuid, () => {
    ctx.onSourceChanged?.();
    ctx.onAnyChange();
  });
  sourceSel.dataset.testid = 'watch-source-select';
  row.appendChild(wfield('Source', sourceSel));

  // What `{source}` publishes as, for this watcher only. The placeholder tracks the
  // selected camera so "blank means the source's own name" needs no help text — and so
  // it keeps reading correctly after the row is repointed at a different camera.
  const mqttName = document.createElement('input');
  mqttName.type = 'text';
  mqttName.value = src.mqttName || '';
  mqttName.dataset.testid = 'watch-source-mqtt-name';
  mqttName.title =
    'Name this source publishes under in MQTT topics — what {source} renders as. '
    + 'Leave blank to use the source\'s own name. Set it when the name that reads well '
    + 'in the Sources list is not the short, stable topic level you want to subscribe to.';
  function syncMqttPlaceholder() {
    mqttName.placeholder = ctx.sourceNameFor(sourceSel.value) || 'source name';
  }
  syncMqttPlaceholder();
  sourceSel.addEventListener('change', syncMqttPlaceholder);
  mqttName.addEventListener('input', ctx.onAnyChange);
  row.appendChild(wfield('MQTT as', mqttName, { width: '8rem' }));

  const modelSel = modelChoiceSelect(
    ctx,
    '(profile default)',
    modelChoiceValue(src.modelGroupUuid, src.modelName),
    ctx.onAnyChange,
  );
  row.appendChild(wfield('Models', modelSel));

  // The three rate fields are numeric and short; the units live in the label so the
  // inputs can stay narrow enough to sit on the same line as the selects.
  const rates = el('div', 'watch-row-rates');
  const fps = numberInput(src.analysisFps ?? '', { step: '0.5', min: 0, placeholder: 'auto' });
  fps.dataset.testid = 'watch-analysis-fps';
  const width = numberInput(src.analysisWidth ?? '', { step: '16', min: 0, placeholder: 'auto' });
  const retention = numberInput(src.recordRetentionS ?? '', { step: '5', min: 0, placeholder: 'auto' });
  fps.addEventListener('input', ctx.onAnyChange);
  width.addEventListener('input', ctx.onAnyChange);
  retention.addEventListener('input', ctx.onAnyChange);
  rates.appendChild(wfield('fps', fps, { width: '4.5rem' }));
  rates.appendChild(wfield('width px', width, { width: '5rem' }));
  rates.appendChild(wfield('buffer s', retention, { width: '4.5rem' }));

  // The honesty affordance: "buffer s" above is a request for pre-roll, and a snapshot
  // camera cannot honour it at any value. Without this the mistake is invisible until
  // an event fires and the clip that was supposed to appear simply doesn't.
  const badge = el('span', 'watch-buffer-badge');
  badge.dataset.testid = 'watch-buffer-badge';
  rates.appendChild(badge);

  // Frame-ring bounds. Shown only for a source that has no rewind buffer, because that
  // is the only case where a ring is kept — but always *sent*, see getSource below.
  const ringFields = el('span', 'watch-ring-fields');
  ringFields.dataset.testid = 'watch-ring-fields';
  const ringSeconds = numberInput(src.ringSeconds ?? '', { step: '5', min: 0, placeholder: '20' });
  ringSeconds.dataset.testid = 'watch-ring-seconds';
  const ringMaxMb = numberInput(src.ringMaxMb ?? '', { step: '8', min: 0, placeholder: '32' });
  ringMaxMb.dataset.testid = 'watch-ring-max-mb';
  ringSeconds.addEventListener('input', ctx.onAnyChange);
  ringMaxMb.addEventListener('input', ctx.onAnyChange);
  ringFields.appendChild(wfield('ring s', ringSeconds, { width: '4.5rem' }));
  ringFields.appendChild(wfield('ring MB', ringMaxMb, { width: '4.5rem' }));
  rates.appendChild(ringFields);

  function refreshBadge() {
    const info = bufferBadgeFor(ctx.ingestTierFor?.(sourceSel.value));
    badge.textContent = info.text;
    badge.title = info.title;
    badge.classList.toggle('watch-buffer-badge--unbuffered', !info.buffered);
    // Not disabled — the value is still stored and still meaningful if the source is
    // later repointed at a stream. Marked instead, so the field explains itself.
    retention.classList.toggle('watch-input--inert', !info.buffered);
    retention.title = info.buffered ? '' : `Pre-roll cannot be honoured: ${info.title}`;
    // The ring is what gives an unbuffered source any pre-roll at all, so its bounds
    // are only worth showing there; on a rewindable source they would be dead controls.
    ringFields.hidden = info.buffered;
  }
  refreshBadge();
  sourceSel.addEventListener('change', refreshBadge);

  row.appendChild(rates);

  // Regions live behind a modal because they need a video frame as a backdrop, which is
  // far too heavy to inline in a list row. They belong to the *source*, so editing them
  // saves immediately rather than waiting for this watcher to be saved — the camera's
  // geometry is not this watcher's to hold hostage.
  const actions = el('div', 'watch-row-actions');
  const regionsBtn = el('button', 'panel-btn', 'Regions…');
  regionsBtn.type = 'button';
  regionsBtn.dataset.testid = 'watch-zones-btn';
  regionsBtn.addEventListener('click', async () => {
    const result = await openSourceRegionsModal({
      sourceUuid: sourceSel.value,
      sourceName: ctx.sourceNameFor(sourceSel.value),
      // Opened from a watcher, so each mask also offers "applies to this watcher".
      // Opened from the Sources pane there is no watcher to scope it to, and the modal
      // renders the geometry alone.
      maskOptOut: {
        disabled: disabledMasks,
        onChange: (next) => { disabledMasks = next; ctx.onAnyChange(); },
      },
    });
    if (result) {
      // Rules offer zones from every source, and the mask checkboxes below are derived
      // from this source's masks, so both have to be rebuilt. Marking the editor dirty
      // is not enough — nothing re-reads regions until the pane is reopened.
      await ctx.onRegionsChanged?.();
      ctx.onAnyChange();
    }
  });
  actions.appendChild(regionsBtn);
  actions.appendChild(removeButton(() => ctx.onDelete(), 'Remove source'));
  row.appendChild(actions);

  // ── Mask opt-out ──
  // Masks act on every detection, so a source owning them is right for the common case
  // ("ignore the driveway, always") and wrong for the rare one (a people-watcher masking
  // a street a car-watcher needs). All ticked by default: a mask added to a camera
  // tomorrow must protect the watchers written today without anyone revisiting them.
  // The per-watcher mask opt-out is edited inside the Regions modal, beside the masks it
  // refers to, rather than as another control in this row. Two reasons: the row is
  // already a full line and a permanent control here costs everyone a line, and this is
  // a *rare* setting — the whole point of the default is that it needs no configuration.
  let disabledMasks = new Set((src.disabledMaskUuidsJson || []).map(String));
  function renderMasks() {
    // Masks the camera no longer has cannot stay opted out: the uuid is rejected on save
    // and the checkbox it came from no longer exists to clear it.
    const live = new Set(ctx.masksForSource(sourceSel.value).map(m => m.uuid));
    disabledMasks = new Set([...disabledMasks].filter(u => live.has(u)));
  }
  renderMasks();
  sourceSel.addEventListener('change', () => {
    // A different camera has different masks; carrying the old source's opt-outs across
    // would disable masks by uuid that this source has never heard of, which the backend
    // rejects on save.
    disabledMasks = new Set();
    renderMasks();
  });

  return {
    element: row,
    getSourceUuid: () => sourceSel.value,
    refreshMasks: renderMasks,
    getSource: () => ({
      sourceUuid: sourceSel.value,
      order: 0,
      enabled: true,
      // null, not '', so "cleared" and "never set" are one state. The backend
      // normalises blanks too; doing it here as well keeps the payload honest.
      mqttName: mqttName.value.trim() || null,
      ...readModelChoice(modelSel.value),
      analysisFps: fps.value === '' ? null : num(fps, null),
      analysisWidth: width.value === '' ? null : num(width, null),
      recordRetentionS: retention.value === '' ? null : num(retention, null),
      // Always sent, including while hidden, and `bufferMode` even though nothing
      // renders it: a watch profile is saved as a whole-list replace, so a field the
      // editor omits is a field the save silently resets to its default. These three
      // were harmless to drop while nothing read them — the watcher reads all three
      // now, so dropping them would quietly undo a configured ring on every save.
      bufferMode: src.bufferMode ?? 'auto',
      ringSeconds: ringSeconds.value === '' ? 20.0 : num(ringSeconds, 20.0),
      ringMaxMb: ringMaxMb.value === '' ? 32.0 : num(ringMaxMb, 32.0),
      // Only masks that still exist on this source — a stale uuid is rejected on save,
      // and a mask deleted in the editor must not keep a dead opt-out alive.
      disabledMaskUuidsJson: ctx
        .masksForSource(sourceSel.value)
        .filter(m => disabledMasks.has(m.uuid))
        .map(m => m.uuid),
    }),
  };
}

// ── Rule row ────────────────────────────────────────────────────────────────
function buildRuleRow(rule, ctx) {
  const row = el('div', 'watch-rule-row watch-row');
  row.dataset.testid = 'watch-rule-row';

  const nameInput = document.createElement('input');
  nameInput.type = 'text';
  nameInput.value = rule.name || '';
  nameInput.placeholder = 'e.g. cat using litter box';
  nameInput.dataset.testid = 'watch-rule-name';
  nameInput.addEventListener('input', ctx.onAnyChange);
  // `change` fires on commit (blur / Enter), not per keystroke — the actions' scope
  // pickers name this rule, and rebuilding them mid-word would steal focus.
  nameInput.addEventListener('change', () => ctx.onNameCommitted?.());
  // Explicitly narrow so the name and the sentence share a line. At its natural width
  // the pair overflowed the row by a handful of pixels and cost a whole extra line.
  row.appendChild(wfield('Rule', nameInput, { width: '10rem' }));

  // The sentence: [subject A] [relation] [subject B]. Each subject is a *list* — "any of
  // these three cats is inside any of these two litter boxes" is one rule, not six, and
  // the engine reports which members actually matched so an action can name them.
  const sentence = el('div', 'watch-rule-sentence');

  let subjectARefs = [...(rule.subjectARefsJson || [])];
  let subjectBRefs = [...(rule.subjectBRefsJson || [])];
  // "…and everything under it in the label tree". Stored as a flag and expanded when the
  // watcher starts, so a label added to the tree later is covered without editing the
  // rule — which also means it takes effect on the watcher's next restart, not instantly.
  let subjectADescendants = Boolean(rule.subjectAIncludeDescendants);
  let subjectBDescendants = Boolean(rule.subjectBIncludeDescendants);

  const subjectAHost = el('span', 'watch-rule-slot');
  subjectAHost.dataset.testid = 'watch-rule-subject-a';
  let subjectADropdown = null;

  function buildSubjectA() {
    subjectADropdown?.destroy();
    subjectAHost.innerHTML = '';
    subjectADropdown = createMultiSelectDropdown({
      // No "Labels:" prefix — the sentence around it already says what this slot is, and
      // the prefix is wide enough to push the tuning fields onto another line.
      label: '',
      items: ctx.labelItems,
      defaultSelected: subjectARefs,
      // A rule slot picks things; it does not build a query. Mode and exclusion are
      // filter concepts and would read here as though a rule could match "all of" three
      // cats at once, which is not what the engine does.
      showModeToggle: false,
      allowExclude: false,
      searchable: true,
      emptySummary: 'None selected',
      emptyText: '(no labels defined)',
      filterKey: 'watch-rule-subject-a',
      hierarchyToggle: {
        label: 'Include child labels',
        default: subjectADescendants,
        onChange: (value) => {
          subjectADescendants = value;
          ctx.onAnyChange();
        },
      },
      onChange: (selected) => {
        subjectARefs = selected;
        ctx.onAnyChange();
      },
    });
    subjectAHost.appendChild(subjectADropdown.element);
  }
  buildSubjectA();

  const relation = select(RELATIONS, rule.relation || 'contains', () => {
    syncArity();
    ctx.onAnyChange();
  });
  relation.dataset.testid = 'watch-rule-relation';

  const subjectBKind = select(
    [{ key: 'zone', text: 'Zone' }, { key: 'label', text: 'Label' }],
    rule.subjectBKind === 'label' ? 'label' : 'zone',
    () => {
      // Switching kind changes what the uuids *mean*, so keeping them would leave label
      // uuids selected in a zone picker — the shape of the bug that used to silently
      // rewrite "is inside Litter Box" as "is inside Cat" on the next save.
      subjectBRefs = [];
      // Zones have no hierarchy, and the backend rejects the flag on a zone slot, so a
      // leftover true would make the whole save fail with a message about a control the
      // user can no longer see.
      subjectBDescendants = false;
      buildSubjectB();
      ctx.onAnyChange();
    },
  );
  subjectBKind.dataset.testid = 'watch-rule-subject-b-kind';

  const subjectBHost = el('span', 'watch-rule-slot');
  subjectBHost.dataset.testid = 'watch-rule-subject-b';
  let subjectBDropdown = null;

  function buildSubjectB() {
    subjectBDropdown?.destroy();
    subjectBHost.innerHTML = '';
    const isLabel = subjectBKind.value === 'label';
    const items = isLabel ? ctx.labelItems : ctx.zoneItems();
    // Drop refs that are no longer on offer — a zone deleted in the region editor must
    // not linger as an invisible selection that fails validation on save.
    const available = new Set(items.map(i => i.uuid));
    subjectBRefs = subjectBRefs.filter(r => available.has(r));
    subjectBDropdown = createMultiSelectDropdown({
      // Unprefixed for the same reason as subject A: the `Zone`/`Label` select sitting
      // immediately to its left already names the kind.
      label: '',
      items,
      defaultSelected: subjectBRefs,
      showModeToggle: false,
      allowExclude: false,
      searchable: isLabel,
      emptySummary: 'None selected',
      emptyText: isLabel ? '(no labels defined)' : '(no zones drawn on this watcher’s sources)',
      filterKey: 'watch-rule-subject-b',
      // Zones have no hierarchy, so the toggle only exists for a label-kind subject B —
      // and the backend rejects the flag on a zone slot rather than ignoring it.
      hierarchyToggle: isLabel
        ? {
          label: 'Include child labels',
          default: subjectBDescendants,
          onChange: (value) => {
            subjectBDescendants = value;
            ctx.onAnyChange();
          },
        }
        : null,
      onChange: (selected) => {
        subjectBRefs = selected;
        ctx.onAnyChange();
      },
    });
    subjectBHost.appendChild(subjectBDropdown.element);
  }
  buildSubjectB();

  sentence.append(el('span', null, 'When'), subjectAHost, relation, subjectBKind, subjectBHost);
  row.appendChild(sentence);

  function syncArity() {
    // `present` is arity 1; showing a second subject for it would imply a
    // comparison that never happens.
    const isUnary = relation.value === 'present';
    subjectBKind.hidden = isUnary;
    subjectBHost.hidden = isUnary;
  }

  // Tuning keeps its four inputs in threshold / start / clear / max order — the
  // Playwright defaults test indexes `.watch-rule-tuning input` positionally.
  // The long labels ("Ends after clear (s)") became units-in-the-label plus a
  // tooltip: spelled out, the four of them alone filled a line.
  const tuning = el('div', 'watch-rule-tuning');
  const threshold = numberInput(rule.threshold ?? 0.6, { step: '0.05', min: 0 });
  const minDuration = numberInput(rule.minDurationS ?? 2, { step: '0.5', min: 0 });
  const rearm = numberInput(rule.rearmDelayS ?? 10, { step: '1', min: 0 });
  const maxDuration = numberInput(rule.maxDurationS ?? 300, { step: '10', min: 1 });
  for (const input of [threshold, minDuration, rearm, maxDuration]) {
    input.addEventListener('input', ctx.onAnyChange);
  }
  const tuningFields = [
    ['match', threshold, 'Match threshold — how strongly the relation must hold'],
    ['starts after', minDuration, 'Seconds the relation must hold before the event starts'],
    ['ends after', rearm, 'Seconds the relation must stay clear before the event ends'],
    ['max', maxDuration, 'Hard cap on event length, in seconds'],
  ];
  for (const [label, input, title] of tuningFields) {
    const field = wfield(label, input, { width: '4.5rem' });
    field.title = title;
    tuning.appendChild(field);
  }
  row.appendChild(tuning);

  row.appendChild(removeButton(() => ctx.onDelete(), 'Remove rule'));

  syncArity();

  return {
    element: row,
    /**
     * Adopt a newly-drawn set of zones.
     *
     * Called after the region editor saves. Rebuilding preserves any selection still on
     * offer and drops the rest, so a rule already pointing at a zone keeps it and only a
     * rule whose zone genuinely went away loses its reference.
     */
    refreshZones: () => {
      if (subjectBKind.value !== 'label') buildSubjectB();
    },
    destroy: () => {
      subjectADropdown?.destroy();
      subjectBDropdown?.destroy();
    },
    getRule: () => {
      const isUnary = relation.value === 'present';
      return {
        uuid: rule.uuid || null,
        order: 0,
        name: nameInput.value.trim() || 'Unnamed rule',
        enabled: true,
        subjectAKind: 'label',
        subjectARefsJson: subjectARefs,
        subjectAIncludeDescendants: subjectADescendants,
        relation: relation.value,
        subjectBKind: isUnary ? 'none' : subjectBKind.value,
        subjectBRefsJson: isUnary ? [] : subjectBRefs,
        // Never sent true for a unary or zone-kind slot — the backend rejects the flag
        // where there is no hierarchy to descend.
        subjectBIncludeDescendants:
          !isUnary && subjectBKind.value === 'label' && subjectBDescendants,
        threshold: num(threshold, 0.6),
        // These three defaults are not arbitrary: without them a raw threshold flaps
        // into an MQTT storm and a dozen clips for one visit.
        minDurationS: num(minDuration, 2),
        rearmDelayS: num(rearm, 10),
        maxDurationS: num(maxDuration, 300),
      };
    },
  };
}

// ── Action row ──────────────────────────────────────────────────────────────
function buildActionRow(action, ctx) {
  const row = el('div', 'watch-action-row watch-row');
  row.dataset.testid = 'watch-action-row';

  const kind = select(ACTION_KINDS, action.kind || 'publish_mqtt', () => {
    syncTriggers();
    syncParams();
    ctx.onAnyChange();
  });
  kind.dataset.testid = 'watch-action-kind';
  const trigger = select(TRIGGERS, action.trigger || 'event_ended', () => { syncScopeVisibility(); ctx.onAnyChange(); });
  trigger.dataset.testid = 'watch-action-trigger';

  // Sits beside the trigger, and only when that kind's list is short — a `(?)` next to
  // an unconstrained dropdown is a promise of information that isn't there.
  const triggerHelp = el('span', 'watch-action-trigger-help');
  triggerHelp.dataset.testid = 'watch-action-trigger-help';

  /**
   * Rebuild the trigger options for the current action kind.
   *
   * Rebuilt rather than disabled: a greyed-out option still reads as "this ought to
   * work", and the honest thing for a trigger the action can never fire on is not to
   * offer it. A stored value that is no longer offered falls back to the first allowed
   * one, so switching an existing per-frame action to "Record clip" leaves a
   * configuration that actually runs instead of one that silently does nothing.
   */
  function syncTriggers() {
    const allowed = triggersFor(kind.value);
    const previous = trigger.value;
    trigger.innerHTML = '';
    for (const opt of allowed) {
      const o = document.createElement('option');
      o.value = opt.key;
      o.textContent = opt.text;
      trigger.appendChild(o);
    }
    trigger.value = allowed.some((t) => t.key === previous) ? previous : allowed[0].key;

    triggerHelp.innerHTML = '';
    const help = TRIGGER_LIMIT_HELP[kind.value];
    if (help && allowed.length < TRIGGERS.length) {
      triggerHelp.appendChild(noteHelp(help, {
        testid: 'watch-action-trigger-help',
        label: 'Why this action only has one trigger',
      }));
    }
    // The trigger drives whether a rule scope makes sense, so a coerced value has to
    // take the scope picker with it.
    syncScopeVisibility();
  }

  // Scope: two lists, both empty-means-all, mirroring `helpers/watch/scoping.py`.
  // Deliberately *not* a rules × sources grid — that grid is almost entirely empty and
  // stops fitting on screen at about four cameras, while two "All" chips keep the
  // common case a thing you never open.
  let ruleScope = [...(action.ruleUuidsJson || [])];
  let sourceScope = [...(action.sourceUuidsJson || [])];
  let ruleDropdown = null;
  let sourceDropdown = null;

  const ruleScopeField = el('div', 'watch-action-scope');
  ruleScopeField.dataset.testid = 'watch-action-rule-scope';
  const sourceScopeField = el('div', 'watch-action-scope');
  sourceScopeField.dataset.testid = 'watch-action-source-scope';

  /**
   * Rebuild both scope pickers from the current rules and sources.
   *
   * Rebuilt rather than mutated for the same reason `buildRuleRow` rebuilds its zone
   * dropdown in `refreshZones`: the multi-select takes its items at construction, and
   * a watcher's rules and sources are edited in the same form, right above this row.
   * Selections survive, minus anything that no longer exists — keeping a uuid whose
   * rule was just deleted would fail validation on save with no visible cause.
   */
  function buildScopes() {
    const ruleItems = ctx.ruleScopeItems();
    const sourceItems = ctx.sourceScopeItems();
    ruleScope = ruleScope.filter(uuid => ruleItems.some(i => i.uuid === uuid));
    sourceScope = sourceScope.filter(uuid => sourceItems.some(i => i.uuid === uuid));

    ruleDropdown?.destroy();
    sourceDropdown?.destroy();
    ruleScopeField.innerHTML = '';
    sourceScopeField.innerHTML = '';

    ruleDropdown = createMultiSelectDropdown({
      label: 'Rules',
      items: ruleItems,
      defaultSelected: ruleScope,
      showModeToggle: false,
      allowExclude: false,
      // "All" is the honest summary of an empty selection here: an unscoped action
      // really does fire for every rule, so "None selected" would read as the opposite
      // of what it does.
      emptySummary: 'All',
      // A rule has no uuid until the watcher has been saved once, so it cannot be
      // referenced yet. Saying so beats an empty panel that looks broken.
      emptyText: 'No saved rules yet — save the watcher first',
      onChange: (selected) => { ruleScope = [...selected]; ctx.onAnyChange(); },
    });
    sourceDropdown = createMultiSelectDropdown({
      label: 'Sources',
      items: sourceItems,
      defaultSelected: sourceScope,
      showModeToggle: false,
      allowExclude: false,
      emptySummary: 'All',
      emptyText: 'This watcher has no sources yet',
      onChange: (selected) => { sourceScope = [...selected]; ctx.onAnyChange(); },
    });
    ruleScopeField.appendChild(ruleDropdown.element);
    sourceScopeField.appendChild(sourceDropdown.element);
  }

  /** A per-frame action has no rule to be scoped against, so the picker is hidden. */
  function syncScopeVisibility() {
    ruleScopeField.hidden = trigger.value === FRAME_ANALYSED_TRIGGER;
  }

  // A label for humans, never read by the engine. Three "Save snapshot" rows scoped to
  // different cameras are otherwise indistinguishable in the list, and the parameters
  // that tell them apart are spread across a dense row. Placeholdered rather than
  // required: an unnamed action still says what kind it is.
  //
  // Deliberately label-less and narrow. The row's single-line density is an invariant
  // with a test behind it, and a "Name" caption plus a comfortable input was a third
  // line on its own — the placeholder carries the same information for a field that is
  // empty exactly when it needs explaining.
  const name = document.createElement('input');
  name.type = 'text';
  name.value = action.name || '';
  name.placeholder = 'note…';
  name.dataset.testid = 'watch-action-name';
  name.className = 'watch-action-name';
  name.title = 'A note for you — e.g. "timelapse of the back garden every 30s". Not used by the watcher.';
  name.addEventListener('input', ctx.onAnyChange);
  row.appendChild(name);

  row.appendChild(wfield('Action', kind));
  row.appendChild(wfield('on', trigger));
  row.appendChild(triggerHelp);
  // Appended bare, not through `wfield`: the dropdown's trigger already reads "Rules" /
  // "Sources", so a field label repeats the word and spends the horizontal budget that
  // keeps this row on one line. `wfield` also wraps its control in a `<label>`, which
  // around a panel of checkboxes means clicking the caption toggles the first one.
  row.appendChild(ruleScopeField);
  row.appendChild(sourceScopeField);

  buildScopes();
  // Before syncScopeVisibility's own call site: syncTriggers may coerce the trigger,
  // and the scope picker follows from whatever value survives that.
  syncTriggers();

  const params = el('div', 'watch-action-params');
  row.appendChild(params);

  const stored = { ...(action.paramsJson || {}) };
  let readParams = () => ({});

  function syncParams() {
    params.innerHTML = '';
    if (kind.value === 'record_clip') {
      const preroll = numberInput(stored.preroll_s ?? 10, { step: '1', min: 0 });
      const postroll = numberInput(stored.postroll_s ?? 5, { step: '1', min: 0 });
      const maxLength = numberInput(stored.max_length_s ?? 180, { step: '10', min: 1 });
      for (const i of [preroll, postroll, maxLength]) i.addEventListener('input', ctx.onAnyChange);
      params.appendChild(wfield('Pre-roll s', preroll, { width: '4.5rem' }));
      params.appendChild(wfield('post-roll s', postroll, { width: '4.5rem' }));
      params.appendChild(wfield('max s', maxLength, { width: '4.5rem' }));
      readParams = () => ({
        preroll_s: num(preroll, 10),
        postroll_s: num(postroll, 5),
        max_length_s: num(maxLength, 180),
      });
    } else if (kind.value === 'publish_mqtt') {
      const retain = document.createElement('input');
      retain.type = 'checkbox';
      retain.checked = Boolean(stored.retain);
      retain.addEventListener('change', ctx.onAnyChange);
      params.appendChild(wfield('Retain', retain));

      // The topic is a template now, not a fixed path: a rule matching several cats and
      // several litter boxes knows which ones fired, and routing on them is the whole
      // point of exposing those values.
      const topic = document.createElement('input');
      topic.type = 'text';
      topic.value = stored.topic_template || DEFAULT_TOPIC_TEMPLATE;
      topic.placeholder = DEFAULT_TOPIC_TEMPLATE;
      topic.dataset.testid = 'watch-action-topic';
      topic.title = 'Topic to publish to. Broker is configured in Settings.';
      topic.addEventListener('input', ctx.onAnyChange);
      params.appendChild(wfield('Topic', topic, { width: '13rem' }));
      // The placeholder vocabulary lives behind this, not in the input's tooltip —
      // see `templateHelp`.
      params.appendChild(templateHelp(topic));

      // Ellipsised rather than wrapped: a resolved topic is long, and letting it push
      // the action row onto a second line is what made a small watcher scroll. The full
      // value stays available as a tooltip, and textContent is complete either way.
      const example = el('span', 'watch-hint watch-hint--truncate', '');
      example.dataset.testid = 'watch-action-topic-example';
      function syncExample() {
        example.textContent = `e.g. ${renderTopicExample(topic.value || DEFAULT_TOPIC_TEMPLATE)}`;
        example.title = example.textContent;
      }
      topic.addEventListener('input', syncExample);
      syncExample();
      params.appendChild(example);

      readParams = () => ({
        retain: retain.checked,
        qos: 0,
        // Blank means "unset", not "publish to the empty topic".
        topic_template: topic.value.trim() || DEFAULT_TOPIC_TEMPLATE,
      });
    } else if (kind.value === 'publish_mqtt_image') {
      // Its own topic, defaulting to a source-keyed one: the frame belongs to the
      // camera, not to whichever rule happened to fire on it.
      const topic = document.createElement('input');
      topic.type = 'text';
      topic.value = stored.topic_template || DEFAULT_IMAGE_TOPIC_TEMPLATE;
      topic.placeholder = DEFAULT_IMAGE_TOPIC_TEMPLATE;
      topic.dataset.testid = 'watch-action-image-topic';
      topic.title = 'Topic for the JPEG. Broker is configured in Settings.';
      topic.addEventListener('input', ctx.onAnyChange);
      params.appendChild(wfield('Topic', topic, { width: '13rem' }));
      params.appendChild(templateHelp(topic));

      const example = el('span', 'watch-hint watch-hint--truncate', '');
      example.dataset.testid = 'watch-action-image-topic-example';
      function syncExample() {
        example.textContent = `e.g. ${renderTopicExample(topic.value || DEFAULT_IMAGE_TOPIC_TEMPLATE)}`;
        example.title = example.textContent;
      }
      topic.addEventListener('input', syncExample);
      syncExample();
      params.appendChild(example);

      const annotate = document.createElement('input');
      annotate.type = 'checkbox';
      annotate.checked = stored.annotate !== false;
      annotate.dataset.testid = 'watch-action-image-annotate';
      annotate.addEventListener('change', ctx.onAnyChange);
      params.appendChild(wfield('Draw boxes', annotate));

      // On by default, unlike the event action's: an image is state, and retaining it
      // is what makes a subscriber show a picture at once instead of waiting for the
      // next frame. A retained event edge would replay as though it were live.
      const retain = document.createElement('input');
      retain.type = 'checkbox';
      retain.checked = stored.retain !== false;
      retain.addEventListener('change', ctx.onAnyChange);
      params.appendChild(wfield('Retain', retain));

      const quality = numberInput(stored.quality ?? 80, { step: '5', min: 1, max: 100 });
      const maxWidth = numberInput(stored.max_width ?? 0, { step: '80', min: 0 });
      const maxFps = numberInput(stored.max_fps ?? 0, { step: '0.5', min: 0 });
      maxFps.dataset.testid = 'watch-action-image-max-fps';
      maxFps.title =
        'Ceiling on publishes per second per source. 0 publishes every analysed frame.';
      maxWidth.title = 'Shrink to at most this width before encoding. 0 keeps the analysis size.';
      for (const i of [quality, maxWidth, maxFps]) i.addEventListener('input', ctx.onAnyChange);
      params.appendChild(wfield('Quality', quality, { width: '4.5rem' }));
      params.appendChild(wfield('max width', maxWidth, { width: '5rem' }));
      params.appendChild(wfield('max fps', maxFps, { width: '4.5rem' }));

      readParams = () => ({
        topic_template: topic.value.trim() || DEFAULT_IMAGE_TOPIC_TEMPLATE,
        annotate: annotate.checked,
        retain: retain.checked,
        qos: 0,
        quality: num(quality, 80),
        max_width: num(maxWidth, 0),
        max_fps: num(maxFps, 0),
      });
    } else if (kind.value === 'save_timelapse') {
      // Browsed rather than typed, unlike `save_snapshot`'s: a timelapse directory is
      // one the user makes for the purpose (and the picker can make it), where a
      // snapshot directory is normally left at its default. The path is dataset-root
      // relative — the only space the browser and /api/images/mkdir speak — and
      // `resolve_output_dir` in the action accepts that spelling.
      const dir = dirPickerField(stored.output_dir ?? DEFAULT_TIMELAPSE_DIR, ctx.onAnyChange);
      params.appendChild(wfield('Output dir', dir.element, { width: '15rem' }));

      // The per-day subdirectory is created by the action, not picked here, so the
      // hint is the only place the user learns the frames are not written flat.
      const layout = el('span', 'watch-hint', '');
      layout.dataset.testid = 'watch-action-timelapse-layout';
      function syncLayout() {
        const root = (dir.getValue() || DEFAULT_TIMELAPSE_DIR).replace(/^\/+|\/+$/g, '') || DEFAULT_TIMELAPSE_DIR;
        layout.textContent = `→ ${root}/2026-08-18/back-room_14-30-00.jpg`;
        layout.title = 'One directory per day, named for the camera and the local time.';
      }
      dir.onChange(syncLayout);
      syncLayout();
      params.appendChild(layout);

      const every = numberInput(stored.interval_s ?? 30, { step: '5', min: 1 });
      every.dataset.testid = 'watch-action-timelapse-interval';
      every.title =
        'Seconds between kept frames, per camera. Aligned to the clock, so 30 lands on '
        + ':00 and :30 and stays aligned across a restart.';
      const quality = numberInput(stored.quality ?? 90, { step: '5', min: 1, max: 100 });
      const maxWidth = numberInput(stored.max_width ?? 0, { step: '80', min: 0 });
      maxWidth.title = 'Shrink to at most this width before encoding. 0 keeps the frame\'s own size.';
      for (const i of [every, quality, maxWidth]) i.addEventListener('input', ctx.onAnyChange);
      params.appendChild(wfield('Every s', every, { width: '4.5rem' }));
      params.appendChild(wfield('quality', quality, { width: '4.5rem' }));
      params.appendChild(wfield('max width', maxWidth, { width: '5rem' }));

      // On by default, like the MQTT image action: the analysis downscale exists to
      // make inference cheap, and a timelapse is watched by a human.
      const fullRes = document.createElement('input');
      fullRes.type = 'checkbox';
      fullRes.checked = stored.full_resolution !== false;
      fullRes.dataset.testid = 'watch-action-timelapse-full-res';
      fullRes.addEventListener('change', ctx.onAnyChange);
      params.appendChild(wfield('Full res', fullRes));

      // Off by default: burnt-in boxes make the frames useless as training data and
      // flicker distractingly at playback speed.
      const annotate = document.createElement('input');
      annotate.type = 'checkbox';
      annotate.checked = Boolean(stored.annotate);
      annotate.dataset.testid = 'watch-action-timelapse-annotate';
      annotate.addEventListener('change', ctx.onAnyChange);
      params.appendChild(wfield('Draw boxes', annotate));

      readParams = () => ({
        output_dir: dir.getValue().trim() || DEFAULT_TIMELAPSE_DIR,
        interval_s: num(every, 30),
        quality: num(quality, 90),
        max_width: num(maxWidth, 0),
        full_resolution: fullRes.checked,
        annotate: annotate.checked,
      });
    } else {
      // A template, not a fixed path — the same vocabulary the topics use, so "a folder
      // per camera" or "a folder per rule" is a thing you can write rather than a
      // feature request. Browsable for the same reason the timelapse field is: the
      // folder usually does not exist yet, and the picker can make it.
      const dir = dirPickerField(stored.output_dir ?? DEFAULT_SNAPSHOT_DIR, ctx.onAnyChange, {
        placeholder: DEFAULT_SNAPSHOT_DIR,
      });
      params.appendChild(wfield('Output dir', dir.element, { width: '15rem' }));
      params.appendChild(templateHelp(dir.input, {
        label: 'Placeholders you can use in this folder',
        intro: 'Substituted per path segment and slugified, so a camera named "Kitchen / Cam" '
          + 'cannot turn one folder into two. Click one to insert it. A YYYY-MM-DD folder is '
          + 'added automatically unless you place {date} yourself.',
      }));

      const example = el('span', 'watch-hint watch-hint--truncate', '');
      example.dataset.testid = 'watch-action-snapshot-example';
      function syncExample() {
        const root = (dir.getValue().trim() || DEFAULT_SNAPSHOT_DIR).replace(/^\/+|\/+$/g, '');
        // `renderTopicExample` substitutes and collapses empty segments, which is exactly
        // what `render_path` does to a folder template — the two differ only in what the
        // result is fed to, so there is no second renderer to keep in step.
        const rendered = renderTopicExample(root);
        // The automatic day level, shown only when it is actually added — otherwise the
        // preview claims a folder the action will not create.
        const day = root.includes('{date}') ? '' : `${TEMPLATE_EXAMPLE.date}/`;
        example.textContent = `→ ${rendered}/${day}14-30-00_back-room_cat-using-litter-box.jpg`;
        example.title = example.textContent;
      }
      dir.onChange(syncExample);
      syncExample();
      params.appendChild(example);

      readParams = () => ({ output_dir: dir.getValue().trim() || DEFAULT_SNAPSHOT_DIR });
    }
  }

  row.appendChild(removeButton(() => ctx.onDelete(), 'Remove action'));

  syncParams();

  return {
    element: row,
    refreshScopes: buildScopes,
    destroy: () => {
      ruleDropdown?.destroy();
      sourceDropdown?.destroy();
    },
    getAction: () => ({
      uuid: action.uuid || null,
      // A per-frame action has no rule, so its rule scope is not sent even if one was
      // picked before the trigger was switched — the backend ignores it, and saving it
      // anyway would leave a stale scope to reappear if the trigger changed back.
      ruleUuidsJson: trigger.value === FRAME_ANALYSED_TRIGGER ? [] : [...ruleScope],
      sourceUuidsJson: [...sourceScope],
      order: 0,
      enabled: true,
      // Blank means unnamed, not an empty label — the backend stores null either way.
      name: name.value.trim() || null,
      kind: kind.value,
      trigger: trigger.value,
      paramsJson: readParams(),
    }),
  };
}

/**
 * Renders the Configuration → Watchers pane.
 *
 * Same list-beside-editor shape as Datasets / Filter Chains / Model Stacks, because a
 * Watch Profile is the same kind of thing: a curated definition that a task merely
 * executes.
 */
export async function renderWatchProfileManager({ target }) {
  if (_rendering) return;
  _rendering = true;
  try {
    const container = typeof target === 'string' ? document.getElementById(target) : target;
    if (!container) return;
    container.innerHTML = '';

    const screen = createListEditorScreen({
      container,
      prefix: 'wp',
      title: 'Watchers',
      itemUuidAttr: 'watchUuid',
      searchable: true,
      searchPlaceholder: 'Filter watchers…',
      newLabel: '+ New watcher',
      onNew: () => openEditor(null),
    });

    const [profilesMap, sourcesMap, labelsMap, modelGroups, modelNames, trackers, ingestTiers] = await Promise.all([
      fetchWatchProfiles().catch(() => new Map()),
      fetchSourceList().catch(() => new Map()),
      // fetchLabels() returns a Map, not an array — the fallback must match.
      fetchLabels().catch(() => new Map()),
      fetchModelGroups().catch(() => new Map()),
      // fetchModelsList() is the array of names, unlike its Map-returning neighbours.
      fetchModelsList().catch(() => []),
      fetchTrackers().catch(() => []),
      // Its own fetch rather than a field on the source list: the tier is derived, not
      // stored, and the sources list is served from a cache this must not inherit.
      fetchSourceIngestTiers().catch(() => new Map()),
    ]);

    const sourceOptions = [...sourcesMap.values()].map(s => ({ key: s.uuid, text: s.name || s.uuid }));
    // Indented tree rather than a flat list: a vocabulary of cats under "Cat" under
    // "Animal" is unnavigable alphabetically, and the rule slots are multi-select so the
    // user needs to *find* three siblings, not just one label.
    const labelItems = flattenHierarchy(labelsMap).map(l => ({
      uuid: l.uuid,
      name: l.name || l.uuid,
      color: l.color || null,
      depth: l.depth,
      parentUuid: l.parentUuid ?? null,
    }));
    // Flat `<select>` options for the label-lifetime rows, which pick exactly one label.
    // Indented so the same hierarchy is legible there too — a plain alphabetical list of
    // every cat, breed and species is not.
    const labelOptions = labelItems.map(l => ({
      key: l.uuid,
      text: `${'  '.repeat(l.depth || 0)}${l.name}`,
    }));

    // Regions are keyed by source and fetched in one call rather than per source: a
    // watcher with eight cameras would otherwise open the editor with eight requests.
    let regionsBySource = new Map();
    async function loadRegions() {
      try {
        const all = await fetchSourceRegions(null);
        const next = new Map();
        for (const region of all) {
          if (!next.has(region.sourceUuid)) next.set(region.sourceUuid, []);
          next.get(region.sourceUuid).push(region);
        }
        regionsBySource = next;
      } catch {
        // An empty map degrades to "no zones on offer", which the pickers already say
        // out loud. Failing the whole pane over it would be worse.
        regionsBySource = new Map();
      }
    }
    await loadRegions();

    const regionsFor = (sourceUuid, role) =>
      (regionsBySource.get(sourceUuid) || []).filter(r => r.role === role);
    const modelGroupOptions = [...modelGroups.values()].map(g => ({ key: g.uuid, text: g.name || g.uuid }));
    const sourceNameFor = uuid => sourcesMap.get(uuid)?.name || uuid;
    // Undefined (not null) for an unknown source, so the badge renders nothing rather
    // than asserting "no buffer" about a source it simply has no answer for.
    const ingestTierFor = uuid => ingestTiers.get(uuid);

    function renderList(selectedUuid) {
      screen.listBody.innerHTML = '';
      const profiles = [...profilesMap.values()].sort((a, b) => a.name.localeCompare(b.name));
      screen.setCount(profiles.length);
      if (!profiles.length) {
        // createListItem/createListEmpty build a node; appending is the caller's job.
        screen.listBody.appendChild(screen.createListEmpty('No watchers yet.'));
        return;
      }
      for (const profile of profiles) {
        const item = screen.createListItem({
          uuid: profile.uuid,
          name: profile.name,
          meta: `${profile.sources?.length || 0} source(s), ${profile.rules?.length || 0} rule(s)`,
          onClick: () => openEditor(profile),
        });
        item.dataset.testid = 'watcher-list-item';
        screen.listBody.appendChild(item);
      }
      screen.updateListSelection(selectedUuid);
    }

    function openEditor(profile) {
      screen.editorBody.innerHTML = '';
      const sourceRows = [];
      const ruleRows = [];
      const actionRows = [];
      const lifetimeRows = [];

      const bar = buildPanelEditorBar({
        container: screen.editorBar,
        item: profile,
        newTitle: 'New watcher',
        deleteTitle: 'Delete watcher',
        onSave: () => save(),
        onDelete: profile ? () => remove() : null,
      });
      const onAnyChange = () => bar.setDirty(true);

      // ── Identity ──
      // Four stacked full-width fields became one wrapping strip: the whole identity
      // block is now roughly the height a single one of them used to occupy.
      const identity = el('div', 'watch-identity');
      const nameInput = document.createElement('input');
      nameInput.type = 'text';
      nameInput.value = profile?.name || '';
      nameInput.placeholder = 'e.g. House cams';
      nameInput.dataset.testid = 'watcher-name';
      nameInput.addEventListener('input', onAnyChange);
      identity.appendChild(wfield('Name', nameInput));

      const enabled = document.createElement('input');
      enabled.type = 'checkbox';
      enabled.checked = profile ? profile.enabled !== false : true;
      enabled.addEventListener('change', onAnyChange);
      identity.appendChild(wfield('Enabled', enabled));

      const description = document.createElement('input');
      description.type = 'text';
      description.value = profile?.description || '';
      description.placeholder = 'optional';
      description.addEventListener('input', onAnyChange);
      identity.appendChild(wfield('Description', description));

      const defaultModels = modelChoiceSelect(
        { groupOptions: modelGroupOptions, modelNames },
        '(none)',
        modelChoiceValue(profile?.defaultModelGroupUuid, profile?.defaultModelName),
        onAnyChange,
        { testid: 'watch-default-model-choice' },
      );
      identity.appendChild(wfield('Default models', defaultModels));
      screen.editorBody.appendChild(identity);

      // ── Sources ──
      const addSourceBtn = el('button', 'panel-btn', '+ Add source');
      addSourceBtn.type = 'button';
      addSourceBtn.dataset.testid = 'watch-add-source';
      screen.editorBody.appendChild(sectionHeader('Sources', addSourceBtn));
      const sourcesBox = el('div', 'watch-rows');
      sourcesBox.dataset.testid = 'watch-sources-box';
      screen.editorBody.appendChild(sourcesBox);

      function addSource(src) {
        const entry = buildSourceRow(src, {
          sourceOptions,
          groupOptions: modelGroupOptions,
          modelNames,
          sourceNameFor,
          ingestTierFor,
          masksForSource: uuid => regionsFor(uuid, REGION_ROLE_MASK),
          onRegionsChanged: () => refreshRegions(),
          onSourceChanged: () => { refreshActionScopes(); for (const rule of ruleRows) rule.refreshZones?.(); },
          onAnyChange,
          onDelete: () => {
            entry.element.remove();
            sourceRows.splice(sourceRows.indexOf(entry), 1);
            // Removing a source removes its zones from the rules' options.
            for (const rule of ruleRows) rule.refreshZones?.();
            // ...and removes it from every action that was scoped to it, which would
            // otherwise be saved as a scope pointing at a source the watcher no
            // longer has and rejected by the backend with no visible cause.
            refreshActionScopes();
            onAnyChange();
          },
        });
        sourceRows.push(entry);
        sourcesBox.appendChild(entry.element);
        for (const rule of ruleRows) rule.refreshZones?.();
        refreshActionScopes();
      }

      addSourceBtn.addEventListener('click', () => {
        addSource({ sourceUuid: sourceOptions[0]?.key });
        onAnyChange();
      });

      // ── Tracking ──
      // The tracker turns per-frame detections into objects with identity over time,
      // which is what every rule below is written against — so it belongs above Rules.
      const addLifetimeBtn = el('button', 'panel-btn', '+ Add label lifetime');
      addLifetimeBtn.type = 'button';
      addLifetimeBtn.dataset.testid = 'watch-add-lifetime';

      const trackerOptions = trackers.map(t => ({ key: t.typename, text: t.displayName || t.typename }));
      const trackerSel = select(
        trackerOptions.length ? trackerOptions : [{ key: 'IouTracker', text: 'IoU / SORT-lite' }],
        profile?.trackerTypename || 'IouTracker',
        () => { renderTrackerParams(); onAnyChange(); },
      );
      trackerSel.dataset.testid = 'watch-tracker';

      // The picker rides the section's own heading line and the params collapse behind a
      // closed twisty, so the whole section costs no height beyond the heading it needed
      // anyway. A profile with one source, rule and action must still fit without the
      // editor scrolling — there is a Playwright test that says so.
      const trackingHead = sectionHeader('Tracking', addLifetimeBtn);
      trackingHead.insertBefore(wfield('Tracker', trackerSel), addLifetimeBtn);

      const trackerParamsHost = el('div', 'watch-tracker-params-host');
      const tuning = document.createElement('details');
      tuning.className = 'watch-tracker-tuning';
      tuning.dataset.testid = 'watch-tracker-tuning';
      const tuningSummary = document.createElement('summary');
      tuningSummary.textContent = 'Tuning';
      tuning.append(tuningSummary, trackerParamsHost);
      trackingHead.insertBefore(tuning, addLifetimeBtn);

      screen.editorBody.appendChild(trackingHead);

      let trackerParams = null;
      function renderTrackerParams() {
        const chosen = trackers.find(t => t.typename === trackerSel.value);
        trackerParamsHost.innerHTML = '';
        trackerParams = buildTrackerParams(
          chosen?.paramsSchema || {},
          // Only reuse saved params while the selected tracker is the saved one; another
          // tracker's keys are meaningless to this one.
          trackerSel.value === profile?.trackerTypename ? profile?.trackerParamsJson : null,
          onAnyChange,
        );
        trackerParamsHost.appendChild(trackerParams.element);
      }
      renderTrackerParams();

      const lifetimesBox = el('div', 'watch-rows');
      lifetimesBox.dataset.testid = 'watch-lifetimes-box';
      screen.editorBody.appendChild(lifetimesBox);

      function addLifetime(lifetime) {
        const entry = buildLifetimeRow(lifetime, {
          labelOptions: labelOptions.length ? labelOptions : [{ key: '', text: '(no labels)' }],
          onAnyChange,
          onDelete: () => {
            entry.element.remove();
            lifetimeRows.splice(lifetimeRows.indexOf(entry), 1);
            onAnyChange();
          },
        });
        lifetimeRows.push(entry);
        lifetimesBox.appendChild(entry.element);
      }

      addLifetimeBtn.addEventListener('click', () => { addLifetime({}); onAnyChange(); });

      // ── Rules ──
      const addRuleBtn = el('button', 'panel-btn', '+ Add rule');
      addRuleBtn.type = 'button';
      addRuleBtn.dataset.testid = 'watch-add-rule';
      screen.editorBody.appendChild(sectionHeader('Rules', addRuleBtn));
      const rulesBox = el('div', 'watch-rows');
      rulesBox.dataset.testid = 'watch-rules-box';
      screen.editorBody.appendChild(rulesBox);

      // Zones on offer come from the sources this watcher actually uses, so a rule can
      // only ever reference a zone the watcher will have loaded at runtime. Prefixed
      // with the camera name because two cameras may each have a "Doorway".
      function zoneItems() {
        const items = [];
        const multipleSources = sourceRows.length > 1;
        for (const entry of sourceRows) {
          const sourceUuid = entry.getSourceUuid();
          for (const zone of regionsFor(sourceUuid, REGION_ROLE_ZONE)) {
            items.push({
              uuid: zone.uuid,
              name: multipleSources ? `${sourceNameFor(sourceUuid)} — ${zone.name}` : zone.name,
            });
          }
        }
        return items;
      }

      /**
       * The rules an action can be scoped to.
       *
       * Only saved rules: a rule gets its uuid from the backend on first save, and
       * offering an unsaved one would let the user pick a scope that cannot be stored.
       */
      function ruleScopeItems() {
        return ruleRows
          .map(r => r.getRule())
          .filter(rule => rule.uuid)
          .map(rule => ({ uuid: rule.uuid, name: rule.name }));
      }

      /** The sources an action can be scoped to — the ones this watcher actually uses. */
      function sourceScopeItems() {
        return sourceRows
          .map(entry => entry.getSourceUuid())
          .filter(Boolean)
          .map(uuid => ({ uuid, name: sourceNameFor(uuid) }));
      }

      /**
       * Push the current rules and sources into every action's scope pickers.
       *
       * Called on structural changes only (a rule or source added, removed or
       * repointed), never from `onAnyChange` — that fires on every keystroke, and
       * rebuilding a dropdown under the cursor would drop focus mid-word.
       */
      function refreshActionScopes() {
        for (const entry of actionRows) entry.refreshScopes?.();
      }

      /** Re-read regions from the server and push them into every row that shows them. */
      async function refreshRegions() {
        await loadRegions();
        for (const entry of ruleRows) entry.refreshZones?.();
        for (const entry of sourceRows) entry.refreshMasks?.();
      }

      function addRule(rule) {
        const entry = buildRuleRow(rule, {
          labelItems,
          zoneItems,
          onAnyChange,
          onDelete: () => {
            entry.destroy?.();
            entry.element.remove();
            ruleRows.splice(ruleRows.indexOf(entry), 1);
            // A deleted rule must leave the scope of every action that named it.
            refreshActionScopes();
            onAnyChange();
          },
          // A renamed rule should read by its new name in the scope pickers. On
          // `change` (commit), not `input` — rebuilding a dropdown on every keystroke
          // would fight the user for focus.
          onNameCommitted: refreshActionScopes,
        });
        ruleRows.push(entry);
        rulesBox.appendChild(entry.element);
        refreshActionScopes();
      }

      addRuleBtn.addEventListener('click', () => { addRule({}); onAnyChange(); });

      // ── Actions ──
      const addActionBtn = el('button', 'panel-btn', '+ Add action');
      addActionBtn.type = 'button';
      addActionBtn.dataset.testid = 'watch-add-action';
      screen.editorBody.appendChild(sectionHeader('Actions', addActionBtn));
      const actionsBox = el('div', 'watch-rows');
      actionsBox.dataset.testid = 'watch-actions-box';
      screen.editorBody.appendChild(actionsBox);

      function addAction(action) {
        const entry = buildActionRow(action, {
          ruleScopeItems,
          sourceScopeItems,
          onAnyChange,
          onDelete: () => {
            entry.destroy?.();
            entry.element.remove();
            actionRows.splice(actionRows.indexOf(entry), 1);
            onAnyChange();
          },
        });
        actionRows.push(entry);
        actionsBox.appendChild(entry.element);
      }

      addActionBtn.addEventListener('click', () => { addAction({}); onAnyChange(); });

      // ── Validation report ──
      const report = el('div', 'watch-validation');
      report.dataset.testid = 'watch-validation';
      screen.editorBody.appendChild(report);

      for (const lifetime of profile?.labelLifetimes || []) addLifetime(lifetime);
      for (const src of profile?.sources || []) addSource(src);
      for (const rule of profile?.rules || []) addRule(rule);
      for (const action of profile?.actions || []) addAction(action);

      async function showValidation(uuid) {
        try {
          const result = await validateWatchProfile(uuid);
          report.innerHTML = '';
          for (const err of result.errors) {
            report.appendChild(el('div', 'notices-bar__item--error', `⚠ ${err}`));
          }
          for (const warn of result.warnings) {
            report.appendChild(el('div', 'notices-bar__item--warning', `• ${warn}`));
          }
          if (result.valid && !result.warnings.length) {
            report.appendChild(el('div', 'notices-bar__item--info', '✓ This watcher looks ready to run.'));
          }
        } catch (e) {
          report.textContent = '';
        }
      }

      function collect() {
        return {
          name: nameInput.value.trim(),
          description: description.value.trim(),
          enabled: enabled.checked,
          // Both keys, always: switching away from a stack has to null it out, or the
          // backend sees a stack *and* a model and rejects the pair.
          defaultModelGroupUuid: readModelChoice(defaultModels.value).modelGroupUuid,
          defaultModelName: readModelChoice(defaultModels.value).modelName,
          trackerTypename: trackerSel.value,
          trackerParamsJson: trackerParams?.getParams() || {},
          // A row with no label selected (no labels exist yet) would fail server-side
          // validation and take the whole save with it.
          labelLifetimes: lifetimeRows.map(entry => entry.getLifetime()).filter(l => l.labelUuid),
          sources: sourceRows.map((entry, i) => ({ ...entry.getSource(), order: i })),
          rules: ruleRows.map((entry, i) => ({ ...entry.getRule(), order: i })),
          actions: actionRows.map((entry, i) => ({ ...entry.getAction(), order: i })),
        };
      }

      async function save() {
        const payload = collect();
        if (!payload.name) {
          toast('A watcher needs a name', 5000, 'error');
          return;
        }
        try {
          const saved = profile
            ? await updateWatchProfile(profile.uuid, payload)
            : await createWatchProfile(payload);
          profilesMap.set(saved.uuid, saved);
          toast(`Saved watcher "${saved.name}"`, 5000, 'success');
          bar.setDirty(false);
          renderList(saved.uuid);
          // Await the re-render: an unawaited one silently swallows its errors.
          await showValidation(saved.uuid);
          openEditor(saved);
          // A running watcher loaded its rules at start and will not see these
          // edits until it restarts. No-op when nothing is running on it.
          await promptRestartAffectedTasks(saved.uuid, { label: 'watcher' });
        } catch (e) {
          toast(e.message || 'Failed to save watcher', 5000, 'error');
        }
      }

      async function remove() {
        try {
          await deleteWatchProfile(profile.uuid);
          profilesMap.delete(profile.uuid);
          toast(`Deleted watcher "${profile.name}"`, 5000, 'success');
          screen.editorBody.innerHTML = '';
          screen.editorBar.innerHTML = '';
          renderList(null);
        } catch (e) {
          toast(e.message || 'Failed to delete watcher', 5000, 'error');
        }
      }

      screen.setDoSave(save);
      if (profile) showValidation(profile.uuid);
    }

    // Arriving from a Runs launch card's ⚙ means "show me *this* watcher".
    const wantedUuid = takePendingSelection('tab-watcher-management');
    const wanted = wantedUuid ? profilesMap.get(wantedUuid) ?? null : null;
    renderList(wanted?.uuid ?? null);
    if (wanted) openEditor(wanted);
  } finally {
    _rendering = false;
  }
}
