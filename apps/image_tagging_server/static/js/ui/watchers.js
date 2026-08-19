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
import { openOverlapPreviewModal } from '/app-static/js/ui/utils/overlapPreviewModal.js';
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

/**
 * What a rule's `threshold` actually means, per relation — the caption and the tooltip
 * for the same number.
 *
 * One field, six meanings. It was captioned `match` for all of them, which is a
 * geometric word for a number that under `present` is nothing of the kind: there is no
 * second box, and `evaluate_predicate` compares it against the *detection confidence*
 * of the best track. Captioning that `conf` is the difference between a number you can
 * reason about and one you can only fiddle with.
 *
 * `distance` gets its own caption for the opposite reason: its threshold is an upper
 * bound (a distance you must be *within*), so raising it loosens the rule while raising
 * every other relation's tightens it. `centroid_in` says out loud that it ignores the
 * number entirely — a centre is inside or it is not.
 *
 * Mirrors the branches in `evaluate_predicate` (`apps/helpers/watch/rules.py`).
 */
const THRESHOLD_HELP = {
  present: {
    label: 'conf',
    title: 'Minimum detection confidence, 0–1. The label must be detected at least this '
      + 'strongly to count as present — no geometry is measured for this relation. Raise it '
      + 'to ignore shaky detections, lower it to catch a half-hidden subject.',
  },
  contains: {
    label: 'match',
    title: 'How much of the subject must lie inside the target, as a fraction of the '
      + 'subject’s own area. 1 demands the whole box; 0.5 fires when half of it is in.',
  },
  overlaps: {
    label: 'match',
    // No markdown emphasis in a tooltip: a `title` renders as plain text, so asterisks
    // reach the user as asterisks. The wording carries the emphasis instead.
    title: 'How much the two must overlap, as a fraction of the smaller box’s area rather '
      + 'than of the pair — so a cat lapping at a bowl scores on the cat, not on the bowl.',
  },
  iou: {
    label: 'match',
    title: 'Intersection over union the pair must reach. Only reads as “overlap” for boxes '
      + 'of similar size; a small box inside a large one scores near zero.',
  },
  distance: {
    label: 'within',
    title: 'Greatest centre-to-centre distance that still counts, as a fraction of the '
      + 'frame diagonal. Higher is looser here — the opposite of every other relation.',
  },
  centroid_in: {
    label: 'match',
    title: 'Not used by this relation: the subject’s centre is either inside the target or '
      + 'it is not. Use the nearby margin to widen the target instead.',
  },
};

const ACTION_KINDS = [
  { key: 'publish_mqtt', text: 'Publish MQTT' },
  { key: 'publish_mqtt_image', text: 'Publish MQTT image' },
  { key: 'record_clip', text: 'Record clip' },
  // "Event timelapse", not "Record event timelapse": the kind dropdown sizes itself to
  // its widest option, and the longer label pushed the whole action row onto a second
  // line. It also reads better beside "Save timelapse" — the distinction between them is
  // per-event versus all-day, which is exactly what the first word says.
  { key: 'record_event_timelapse', text: 'Event timelapse' },
  { key: 'save_snapshot', text: 'Save snapshot' },
  { key: 'save_timelapse', text: 'Save timelapse' },
  { key: 'save_event', text: 'Save event' },
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
  record_event_timelapse: ['event_ended'],
  save_timelapse: ['frame_analysed'],
  save_event: ['event_started'],
};

/** Why an action kind's trigger list is shorter than the full set. */
const TRIGGER_LIMIT_HELP = {
  record_clip:
    'A clip can only be cut when the event ends. Its length is not known until then — '
    + 'the pre-roll is taken retroactively out of the rolling buffer, which is what lets '
    + 'you record the seconds before the rule fired. Use the pre-roll and post-roll '
    + 'settings to cover what happened either side.',
  record_event_timelapse:
    'The video is compiled when the event ends, because that is when its last frame is '
    + 'known. It does not only run then: while the event is open it quietly keeps a frame '
    + 'every few seconds, and those are what it compiles. Nothing is kept once the event '
    + 'is over, so an event that never opens costs nothing. For a timelapse that runs all '
    + 'day whether or not anything is happening, use Save timelapse instead.',
  save_timelapse:
    'A timelapse has to keep shooting through the hours when nothing happens, so it '
    + 'runs on the frame the watcher is already pulling from the camera rather than on '
    + 'an event. Its own interval decides how often a frame is kept — everything in '
    + 'between is discarded without being written. For a picture of one event, use '
    + 'Save snapshot instead.',
  save_event:
    'The event is filed when it opens, so it is visible in Events while it is still '
    + 'happening rather than only once it is over. There is nothing left to do at the '
    + 'end — the row is closed out with its duration either way.',
};

/**
 * Action kinds that put the event in the Events view. Mirrors `Action.logs_event()` in
 * `apps/helpers/actions/action.py`, which is what the watcher actually reads — this
 * copy only drives the hint below the action list.
 *
 * An event with none of these in scope still fires its actions and still publishes; it
 * simply leaves no row. That is the point: a rule like "a cat is present" matches for
 * hours a day, and filing every match buried the events that had a clip to watch.
 */
const EVENT_LOGGING_KINDS = new Set(['record_clip', 'record_event_timelapse', 'save_event']);

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
  browse.title = 'Pick a folder under the image dataset root, or create one. You can also type '
    + 'the path directly, placeholders included.';
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
 *
 * `help` goes on the wrapping `<label>`, not the control: a `title` is inherited by
 * descendants, so one assignment covers hovering the caption *and* the input. Putting
 * it here rather than at each call site is what makes "every settable thing explains
 * itself" a property of the form instead of a list of places to remember.
 */
function wfield(labelText, control, { width = null, help = null } = {}) {
  const wrap = el('label', 'watch-field');
  wrap.appendChild(el('span', 'watch-field__label', labelText));
  if (width) control.style.width = width;
  if (help) wrap.title = help;
  wrap.appendChild(control);
  return wrap;
}

/**
 * The on/off box that leads a rule or action row.
 *
 * A real switch, not a note to yourself: `StreamWatcherTask` builds its rule list from
 * `profile.rules if r.enabled` and skips any action row whose flag is false, so an
 * unticked row is never evaluated. That is the opposite of the watcher-level "Enabled"
 * box in the identity strip, which nothing reads at run time and says so.
 *
 * It exists because the alternative was deleting a rule to silence it for an evening and
 * rebuilding it from memory afterwards — losing the thresholds, the hysteresis and the
 * scope along with it.
 *
 * Dims the row rather than hiding or collapsing anything. A disabled rule still has to be
 * readable and editable (turning it off without losing how it was configured is the whole
 * point), and a row that changed size on toggle would reflow the editor under the cursor.
 */
function enabledToggle(row, current, onChange, { testid, title }) {
  const box = document.createElement('input');
  box.type = 'checkbox';
  // `!== false` rather than truthiness: a row being built from scratch has no flag at
  // all, and a new rule defaults to enabled.
  box.checked = current !== false;
  box.className = 'watch-row-enabled';
  box.dataset.testid = testid;
  box.title = title;
  const sync = () => row.classList.toggle('watch-row--disabled', !box.checked);
  box.addEventListener('change', () => { sync(); onChange(); });
  sync();
  return box;
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

/**
 * Wrap a section's heading and its list in one bordered panel.
 *
 * The editor was a flat run of headings and bordered rows in a single column, and
 * Sources, Rules and Actions all render as near-identical rows — so the only thing
 * marking the boundary between two sections was a half-rem of margin, and the three
 * read as one long list. The panel gives each section an edge and a surface of its own.
 *
 * @param {HTMLElement} head A `sectionHeader`, already built (Tracking inserts controls
 *   into its own before wrapping).
 * @param {...HTMLElement} bodies
 */
function sectionPanel(head, ...bodies) {
  const section = el('div', 'watch-section');
  section.dataset.testid = 'watch-section';
  section.append(head, ...bodies);
  return section;
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
    box.appendChild(wfield(key.replace(/_/g, ' '), control, {
      width: type === 'bool' ? null : '6rem',
      // The schema's own help, so the caption explains itself too. A tracker that
      // declares none says so rather than hovering blank — the fields are named for the
      // paper they come from, and a bare `track_buffer` explains nothing on its own.
      help: field.help || `${key} — this tracker declares no description for it.`,
    }));
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
  row.appendChild(wfield('Label', label, {
    help: 'The label this lifetime applies to. Labels with no row here use the tracker’s own '
      + 'defaults.',
  }));

  const ttl = numberInput(lifetime.ttlS ?? 2.0, { step: '0.5', min: 0 });
  ttl.dataset.testid = 'watch-lifetime-ttl';
  ttl.title = 'Seconds the track survives with no detection before it is assumed gone.';
  ttl.addEventListener('input', ctx.onAnyChange);
  row.appendChild(wfield('TTL s', ttl, { width: '5rem', help: ttl.title }));

  const anchor = document.createElement('input');
  anchor.type = 'checkbox';
  anchor.checked = lifetime.isAnchor === true;
  anchor.dataset.testid = 'watch-lifetime-anchor';
  anchor.title =
    'Latch this label’s box once seen confidently, instead of trusting a detector that '
    + 'intermittently misses a stationary object. Appropriate for furniture, not for animals.';
  anchor.addEventListener('change', ctx.onAnyChange);
  row.appendChild(wfield('Anchor', anchor, { help: anchor.title }));

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

/**
 * The `buffer s` field's help. A constant because `refreshBadge` rewrites the input's
 * own title to explain an unbuffered source, and has to be able to put this back.
 */
// Deliberately opens by saying what this is *not*. The previous wording — "so a clip can
// start before the rule fired" — described this field by naming the pre-roll setting's
// job, and the reasonable conclusion was that this is where you choose your lead-in.
// Shortening it to a preferred pre-roll then quietly starves the clip drain, and the only
// symptom is that long events lose their opening.
const BUFFER_SECONDS_HELP =
  'How much video this camera keeps rewindable, in seconds. Blank uses the default, which is '
  + 'almost always right. This is NOT the pre-roll setting: how much lead-in a clip gets is set '
  + 'on the Record clip action, and it can be shorter than this. Two things need the room — '
  + 'pre-roll is cut from it, and an event that outlasts it is copied out to disk from it as it '
  + 'runs, which needs several seconds of headroom to keep up. Making it smaller buys nothing '
  + 'and risks both. The badge beside it says whether this camera can keep a rewindable buffer '
  + 'at all.';

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
  row.appendChild(wfield('Source', sourceSel, {
    help: 'The camera this row configures. Repointing it moves the row’s settings to the new '
      + 'camera and drops anything that belonged to the old one — its mask opt-outs, and any '
      + 'action or rule scoped to it.',
  }));

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
  row.appendChild(wfield('MQTT as', mqttName, { width: '8rem', help: mqttName.title }));

  const modelSel = modelChoiceSelect(
    ctx,
    '(profile default)',
    modelChoiceValue(src.modelGroupUuid, src.modelName),
    ctx.onAnyChange,
  );
  row.appendChild(wfield('Models', modelSel, {
    help: 'Which model or model stack runs on this camera. Leave at “(profile default)” to '
      + 'use the watcher’s Default models — set it here only when one camera needs something '
      + 'the others do not.',
  }));

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
  rates.appendChild(wfield('fps', fps, {
    width: '4.5rem',
    help: 'Frames per second pulled from this camera for analysis. Blank uses the watcher’s '
      + 'own default. This is the main cost knob: every frame is a full inference pass, and '
      + 'the tracker only needs enough of them to follow an object between detections.',
  }));
  rates.appendChild(wfield('width px', width, {
    width: '5rem',
    help: 'Downscale each frame to this width before inference. Blank uses the default. '
      + 'Smaller is faster and loses small or distant objects first; it does not affect what a '
      + 'recorded clip looks like.',
  }));
  // "rewind s", not "buffer s": the old caption named the mechanism, which left the
  // field looking like the generic place to configure anything clip-shaped. "Rewind" is
  // what it actually gives you, and it does not read as a synonym for pre-roll.
  rates.appendChild(wfield('rewind s', retention, {
    width: '4.5rem',
    help: BUFFER_SECONDS_HELP,
  }));

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
  ringFields.appendChild(wfield('ring s', ringSeconds, {
    width: '4.5rem',
    help: 'Seconds of analysed frames kept in memory for a camera that cannot rewind. It is '
      + 'what gives such a source any pre-roll at all — a clip cut from it is built from these '
      + 'frames, not from video.',
  }));
  ringFields.appendChild(wfield('ring MB', ringMaxMb, {
    width: '4.5rem',
    help: 'Memory ceiling for that ring, per camera. Whichever limit is hit first wins, so a '
      + 'high-resolution camera keeps fewer seconds than the number beside this asks for.',
  }));
  rates.appendChild(ringFields);

  function refreshBadge() {
    const info = bufferBadgeFor(ctx.ingestTierFor?.(sourceSel.value));
    badge.textContent = info.text;
    badge.title = info.title;
    badge.classList.toggle('watch-buffer-badge--unbuffered', !info.buffered);
    // Not disabled — the value is still stored and still meaningful if the source is
    // later repointed at a stream. Marked instead, so the field explains itself.
    retention.classList.toggle('watch-input--inert', !info.buffered);
    // The base help when it works, the reason it cannot when it does not — never `''`,
    // which means "no tooltip" and blocks the wrapper's from showing through.
    retention.title = info.buffered
      ? BUFFER_SECONDS_HELP
      : `Pre-roll cannot be honoured: ${info.title}`;
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
  regionsBtn.title =
    'Draw zones and masks on this camera’s picture. Zones are what a rule matches against; '
    + 'masks hide part of the frame from every detection. They belong to the camera, so '
    + 'editing them saves at once rather than waiting for this watcher to be saved.';
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

  const enabledBox = enabledToggle(row, rule.enabled, ctx.onAnyChange, {
    testid: 'watch-rule-enabled',
    title: 'Whether the watcher evaluates this rule at all. Unticked it is skipped entirely — '
      + 'no events, no actions, nothing recorded — but everything you configured here is kept, '
      + 'so this is the way to silence a rule for an evening without rebuilding it afterwards. '
      + 'Takes effect on the watcher’s next start.',
  });
  row.appendChild(enabledBox);

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
  row.appendChild(wfield('Rule', nameInput, {
    width: '10rem',
    help: 'What this rule is called. Actions are scoped to rules by this name, and it is '
      + 'what {rule_name} renders as in a topic or folder template.',
  }));

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
  subjectAHost.title =
    'The labels this rule watches for. Picking several means “any one of these”, not all '
    + 'of them together — one rule for three cats rather than three rules.';
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
  relation.title =
    'How the subject has to relate to the target for the rule to hold. “is inside” asks how '
    + 'much of the subject is within it; “overlaps” measures against the smaller of the two; '
    + '“closely matches” is IoU and only suits boxes of similar size; “has its centre in” is '
    + 'the cheapest robust test; “is near” compares centre points; “is present” needs no '
    + 'target at all. Use Preview… to see any of them on boxes you can drag.';

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
  subjectBKind.title =
    'Whether the subject is compared against a zone you have drawn on a camera, or against '
    + 'another detected label. Switching this clears the picker beside it — the two kinds '
    + 'do not share ids.';

  const subjectBHost = el('span', 'watch-rule-slot');
  subjectBHost.dataset.testid = 'watch-rule-subject-b';
  subjectBHost.title =
    'What the subject is compared against. Several entries mean “any one of these” — one '
    + 'rule can span two litter boxes. Zones come from the sources this watcher uses; draw '
    + 'them with Regions… on a source row.';
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

  // Which cameras this rule is evaluated on. Empty means all of them, exactly as an
  // action's scope does — and it is the *sharper* of the two: scoping the action leaves
  // the event raised on every camera and only silences the side effect, while scoping
  // the rule stops the event existing. "A cat is present" is the case that needs this
  // one, being true in every room all day.
  let sourceScope = [...(rule.sourceUuidsJson || [])];
  let sourceScopeDropdown = null;
  const sourceScopeHost = el('span', 'watch-rule-slot');
  sourceScopeHost.dataset.testid = 'watch-rule-source-scope';
  sourceScopeHost.title =
    'Which cameras this rule is evaluated on. Empty means all of them. Sharper than scoping '
    + 'an action: an unscoped rule still raises its event everywhere and only the side effect '
    + 'is silenced, while scoping the rule stops the event existing.';

  function buildSourceScope() {
    const items = ctx.sourceScopeItems();
    // A source removed or repointed in the rows above must not survive here as a uuid
    // this watcher no longer has — the backend rejects it, from a control the user can
    // no longer see.
    sourceScope = sourceScope.filter(uuid => items.some(i => i.uuid === uuid));
    sourceScopeDropdown?.destroy();
    sourceScopeHost.innerHTML = '';
    sourceScopeDropdown = createMultiSelectDropdown({
      label: '',
      items,
      defaultSelected: sourceScope,
      showModeToggle: false,
      allowExclude: false,
      // "All sources" rather than "None selected": an unscoped rule really does run
      // everywhere, so the empty state must not read as the opposite of what it does.
      emptySummary: 'All sources',
      emptyText: 'This watcher has no sources yet',
      onChange: (selected) => { sourceScope = [...selected]; ctx.onAnyChange(); },
    });
    sourceScopeHost.appendChild(sourceScopeDropdown.element);
  }
  buildSourceScope();

  sentence.append(el('span', null, 'When'), subjectAHost, relation, subjectBKind, subjectBHost);
  sentence.append(el('span', null, 'on'), sourceScopeHost);
  row.appendChild(sentence);

  function syncArity() {
    // `present` is arity 1; showing a second subject for it would imply a
    // comparison that never happens.
    const isUnary = relation.value === 'present';
    subjectBKind.hidden = isUnary;
    subjectBHost.hidden = isUnary;
    // The margin widens boxes before they are compared, so it means nothing to a
    // relation that compares no boxes (`present`) or only their centres (`distance`) —
    // and the backend rejects a nonzero one there rather than ignoring it. Hidden
    // rather than disabled: `getRule` sends 0 for both, so a value typed under an
    // earlier relation cannot survive as a payload the save would bounce.
    const usesMargin = !isUnary && relation.value !== 'distance';
    for (const field of marginField) field.hidden = !usesMargin;
    // Nothing to drag for `present`: there is no second box.
    previewBtn.hidden = isUnary;
    // The threshold field is renamed, not just re-explained: under `present` it is a
    // detection confidence and calling it `match` was simply wrong. See THRESHOLD_HELP.
    const help = THRESHOLD_HELP[relation.value] || THRESHOLD_HELP.contains;
    thresholdLabel.textContent = help.label;
    thresholdField.title = help.title;
  }

  // Tuning keeps its four inputs in threshold / start / clear / max order — the
  // Playwright defaults test indexes `.watch-rule-tuning input` positionally.
  // The long labels ("Ends after clear (s)") became units-in-the-label plus a
  // tooltip: spelled out, the four of them alone filled a line.
  const tuning = el('div', 'watch-rule-tuning');
  const threshold = numberInput(rule.threshold ?? 0.6, { step: '0.05', min: 0 });
  threshold.dataset.testid = 'watch-rule-threshold';
  // Bounds mirror `InteractionRuleWrite.margin`: past -1 a box erodes to nothing and
  // the rule can never fire, and +2 already reaches two subject-widths away.
  const margin = numberInput(rule.margin ?? 0, { step: '0.1', min: -1 });
  margin.max = '2';
  const minDuration = numberInput(rule.minDurationS ?? 2, { step: '0.5', min: 0 });
  const rearm = numberInput(rule.rearmDelayS ?? 10, { step: '1', min: 0 });
  // min 0, not 1: zero is the "never expire" value, for a rule whose event should last
  // exactly as long as its subject is there. See the caption's tooltip.
  const maxDuration = numberInput(rule.maxDurationS ?? 300, { step: '10', min: 0 });
  for (const input of [threshold, margin, minDuration, rearm, maxDuration]) {
    input.addEventListener('input', ctx.onAnyChange);
  }
  const tuningFields = [
    // Caption and tooltip are placeholders: `syncArity` rewrites both from
    // THRESHOLD_HELP for the relation in force, and runs before this row is shown.
    ['match', threshold, THRESHOLD_HELP.contains.title],
    [
      'nearby',
      margin,
      'Widens both boxes before matching, in units of the smaller box’s width: 1 fires '
      + 'when they are within about one subject-width. Sized to the object, not the frame, '
      + 'so it means the same thing near the camera and far from it. Negative demands a '
      + 'deeper overlap. 0 matches the boxes exactly as detected.',
    ],
    // 'starts'/'ends' rather than 'starts after'/'ends after': the row gained a fifth
    // field and the preview button, and the two long labels alone were what pushed the
    // button onto a line of its own. Both keep the sentence-length tooltip.
    ['starts', minDuration, 'Seconds the relation must hold before the event starts'],
    ['ends', rearm, 'Seconds the relation must stay clear before the event ends'],
    [
      'max',
      maxDuration,
      'Hard cap on event length, in seconds, so a stuck detection cannot hold an event open '
      + 'forever. Reaching it closes the event and immediately starts a new one while the '
      + 'subject is still there, which chops a long visit into a chain of events — and a '
      + 'chain of separate timelapses. 0 disables the cap, for an event that should last '
      + 'exactly as long as its subject does.',
    ],
  ];
  const marginField = [];
  let thresholdField = null;
  let thresholdLabel = null;
  for (const [label, input, title] of tuningFields) {
    // 3.8rem, not the 4.5 these fields used to get: there are five of them plus the
    // preview button now, and at the old width the button wrapped onto a second line —
    // which costs the whole row 33px and pushes a 1/1/1 profile into a scrollbar.
    const field = wfield(label, input, { width: '3.8rem' });
    field.title = title;
    if (input === margin) {
      field.dataset.testid = 'watch-rule-margin-field';
      marginField.push(field);
    }
    if (input === threshold) {
      field.dataset.testid = 'watch-rule-threshold-field';
      thresholdField = field;
      thresholdLabel = field.querySelector('.watch-field__label');
    }
    tuning.appendChild(field);
  }

  // The threshold is the one number in this editor with no intuition behind it — a
  // fraction of an area nobody sees. The preview is where it becomes a thing you drag.
  const previewBtn = el('button', 'panel-btn watch-rule-preview', 'Preview…');
  previewBtn.type = 'button';
  previewBtn.dataset.testid = 'watch-rule-preview';
  previewBtn.title = 'Try this relation on two boxes you can drag, and see the score it '
    + 'produces. Apply sends the relation, match and nearby values back into this row.';
  previewBtn.addEventListener('click', () => {
    const labelName = uuid => ctx.labelItems.find(i => i.uuid === uuid)?.name || null;
    const subjectAUuid = subjectARefs[0] || null;
    const subjectBUuid = subjectBRefs[0] || null;
    const isZoneTarget = subjectBKind.value === 'zone';
    openOverlapPreviewModal({
      relation: relation.value,
      threshold: num(threshold, 0.6),
      margin: num(margin, 0),
      // Only the first ref of each slot: a rule spanning three cats still has one
      // geometry, and picking a sample per member would say nothing extra.
      subjectA: subjectAUuid ? { uuid: subjectAUuid, name: labelName(subjectAUuid) || 'Subject' } : null,
      subjectB: !isZoneTarget && subjectBUuid ? { uuid: subjectBUuid, name: labelName(subjectBUuid) || 'Target' } : null,
      subjectBName: isZoneTarget
        ? (ctx.zoneItems().find(z => z.uuid === subjectBUuid)?.name || 'Zone')
        : 'Target',
      onApply: ({ relation: appliedRelation, threshold: appliedThreshold, margin: appliedMargin }) => {
        relation.value = appliedRelation;
        threshold.value = String(Number(appliedThreshold.toFixed(2)));
        margin.value = String(Number(appliedMargin.toFixed(2)));
        syncArity();
        ctx.onAnyChange();
      },
    });
  });
  tuning.appendChild(previewBtn);
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
    /** Adopt the watcher's current source list — see `buildScopes` on the action row. */
    refreshScopes: () => buildSourceScope(),
    destroy: () => {
      subjectADropdown?.destroy();
      subjectBDropdown?.destroy();
      sourceScopeDropdown?.destroy();
    },
    getRule: () => {
      const isUnary = relation.value === 'present';
      return {
        uuid: rule.uuid || null,
        order: 0,
        name: nameInput.value.trim() || 'Unnamed rule',
        enabled: enabledBox.checked,
        // Always sent: a watch profile saves as a whole-list replace, so a field the
        // editor omits is a field the save resets to its default.
        sourceUuidsJson: [...sourceScope],
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
        // Zero unless the relation actually measures boxes — see `syncArity`.
        margin: isUnary || relation.value === 'distance' ? 0 : num(margin, 0),
        // These three defaults are not arbitrary: without them a raw threshold flaps
        // into an MQTT storm and a dozen clips for one visit.
        minDurationS: num(minDuration, 2),
        rearmDelayS: num(rearm, 10),
        // 0 is a real value here ("never expire"), so it must not be coerced to the
        // default — `num` only falls back on a blank or unparseable field.
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
  kind.title =
    'What this action does. Changing it replaces the settings to its right, and may narrow '
    + 'the trigger beside it — a clip can only be cut when an event ends.';
  const trigger = select(TRIGGERS, action.trigger || 'event_ended', () => { syncScopeVisibility(); ctx.onAnyChange(); });
  trigger.dataset.testid = 'watch-action-trigger';
  trigger.title =
    'When this action runs. “event start” and “event end” fire once each per event; “every '
    + 'analysed frame” fires per frame regardless of whether any rule is holding, which is why '
    + 'it has no rule scope.';

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
  ruleScopeField.title =
    'Which rules’ events this action responds to. Empty means all of them. Only rules that '
    + 'have been saved can be picked — a new rule has no id until then.';
  const sourceScopeField = el('div', 'watch-action-scope');
  sourceScopeField.dataset.testid = 'watch-action-source-scope';
  sourceScopeField.title =
    'Which cameras this action runs for. Empty means all of them. The event is still raised '
    + 'on the others — only this side effect is skipped. To stop the event itself, scope the '
    + 'rule instead.';

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
  const enabledBox = enabledToggle(row, action.enabled, ctx.onAnyChange, {
    testid: 'watch-action-enabled',
    title: 'Whether this action runs. Unticked the watcher never constructs it, so it costs '
      + 'nothing and publishes nothing, but its parameters and scope are kept — the way to stop '
      + 'recording for a week without losing the setup. Takes effect on the watcher’s next start.',
  });
  row.appendChild(enabledBox);

  const name = document.createElement('input');
  name.type = 'text';
  name.value = action.name || '';
  name.placeholder = 'note…';
  name.dataset.testid = 'watch-action-name';
  name.className = 'watch-action-name';
  name.title = 'A note for you — e.g. "timelapse of the back garden every 30s". Not used by the watcher.';
  name.addEventListener('input', ctx.onAnyChange);
  row.appendChild(name);

  row.appendChild(wfield('Action', kind, { help: kind.title }));
  row.appendChild(wfield('on', trigger, { help: trigger.title }));
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
      params.appendChild(wfield('Pre-roll s', preroll, {
        width: '4.5rem',
        help: 'Seconds of video kept from before the event started. Taken retroactively out of '
          + 'the source’s rolling buffer, so it cannot exceed that camera’s “buffer s”.',
      }));
      params.appendChild(wfield('post-roll s', postroll, {
        width: '4.5rem',
        help: 'Seconds kept after the event ended — long enough to see the subject leave.',
      }));
      params.appendChild(wfield('max s', maxLength, {
        width: '4.5rem',
        help: 'Hard ceiling on the finished clip, pre-roll and post-roll included. A longer '
          + 'event is cut short at this length — the event itself is unaffected.',
      }));
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
      params.appendChild(wfield('Retain', retain, {
        help: 'Ask the broker to keep the last message on this topic and hand it to new '
          + 'subscribers. Off by default here: an event edge is a moment, and a retained one '
          + 'replays to every subscriber that connects as though it were happening now.',
      }));

      // The topic is a template now, not a fixed path: a rule matching several cats and
      // several litter boxes knows which ones fired, and routing on them is the whole
      // point of exposing those values.
      const topic = document.createElement('input');
      topic.type = 'text';
      topic.value = stored.topic_template || DEFAULT_TOPIC_TEMPLATE;
      topic.placeholder = DEFAULT_TOPIC_TEMPLATE;
      topic.dataset.testid = 'watch-action-topic';
      topic.title = 'Topic to publish to. Broker is configured in Settings. Placeholders like '
        + '{source} and {rule} are substituted — see the ? beside this field.';
      topic.addEventListener('input', ctx.onAnyChange);
      params.appendChild(wfield('Topic', topic, { width: '13rem', help: topic.title }));
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
      topic.title = 'Topic the JPEG is published on. Broker is configured in Settings. Defaults '
        + 'to a source-keyed topic: the frame belongs to the camera, not to whichever rule fired.';
      topic.addEventListener('input', ctx.onAnyChange);
      params.appendChild(wfield('Topic', topic, { width: '13rem', help: topic.title }));
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
      params.appendChild(wfield('Draw boxes', annotate, {
        help: 'Burn the tracked boxes and labels into the published JPEG. On by default — this '
          + 'image is for a dashboard to show a human, not for training data.',
      }));

      // On by default, unlike the event action's: an image is state, and retaining it
      // is what makes a subscriber show a picture at once instead of waiting for the
      // next frame. A retained event edge would replay as though it were live.
      const retain = document.createElement('input');
      retain.type = 'checkbox';
      retain.checked = stored.retain !== false;
      retain.addEventListener('change', ctx.onAnyChange);
      params.appendChild(wfield('Retain', retain, {
        help: 'Ask the broker to keep the last image on this topic. On by default, unlike the '
          + 'event action’s: a picture is state, and retaining it is what makes a dashboard show '
          + 'something the moment it connects instead of waiting for the next frame.',
      }));

      const quality = numberInput(stored.quality ?? 80, { step: '5', min: 1, max: 100 });
      const maxWidth = numberInput(stored.max_width ?? 0, { step: '80', min: 0 });
      const maxFps = numberInput(stored.max_fps ?? 0, { step: '0.5', min: 0 });
      maxFps.dataset.testid = 'watch-action-image-max-fps';
      maxFps.title =
        'Ceiling on publishes per second per source. 0 publishes every analysed frame.';
      maxWidth.title = 'Shrink to at most this width before encoding. 0 keeps the analysis size.';
      for (const i of [quality, maxWidth, maxFps]) i.addEventListener('input', ctx.onAnyChange);
      params.appendChild(wfield('Quality', quality, {
        width: '4.5rem',
        help: 'JPEG quality, 1–100. 80 is a good balance; above 90 the file grows fast for '
          + 'little visible gain, and every frame goes over the broker.',
      }));
      params.appendChild(wfield('max width', maxWidth, { width: '5rem', help: maxWidth.title }));
      params.appendChild(wfield('max fps', maxFps, { width: '4.5rem', help: maxFps.title }));

      readParams = () => ({
        topic_template: topic.value.trim() || DEFAULT_IMAGE_TOPIC_TEMPLATE,
        annotate: annotate.checked,
        retain: retain.checked,
        qos: 0,
        quality: num(quality, 80),
        max_width: num(maxWidth, 0),
        max_fps: num(maxFps, 0),
      });
    } else if (kind.value === 'record_event_timelapse') {
      // No output directory field, unlike `save_timelapse`. The action defaults to the
      // recordings root, which is where the retention sweep looks — a typed path puts the
      // mp4 somewhere nothing will ever delete it, and the row it hangs off the event
      // would point outside the tree the file endpoint serves. Not sent at all rather
      // than sent as the default, so the action keeps owning the value.
      const every = numberInput(stored.interval_s ?? 10, { step: '5', min: 1 });
      every.dataset.testid = 'watch-action-event-timelapse-interval';
      every.title = 'Seconds of real time between captured frames. Frames in between are discarded '
        + 'without being written. Measured from the moment the event opened, so the first frame is '
        + 'always the start of the visit.';
      const fps = numberInput(stored.output_fps ?? 12, { step: '1', min: 1 });
      fps.dataset.testid = 'watch-action-event-timelapse-fps';
      fps.title = 'Frames per second of the finished video. Raising it shortens the result rather '
        + 'than smoothing it — the frames are seconds apart either way. Capped at 60.';
      for (const i of [every, fps]) i.addEventListener('input', ctx.onAnyChange);
      params.appendChild(wfield('Every s', every, { width: '4.5rem', help: every.title }));
      params.appendChild(wfield('play fps', fps, { width: '4.5rem', help: fps.title }));

      // The speed-up is the product of the two fields above, and it is the number the
      // user actually has in mind ("I want the two hours in ten seconds"). Neither field
      // means much alone, and working it out by hand is exactly the arithmetic that makes
      // people leave a timelapse at its defaults and never find out what it does.
      const speed = el('span', 'watch-hint watch-hint--truncate', '');
      speed.dataset.testid = 'watch-action-event-timelapse-speed';
      function syncSpeed() {
        const interval = Math.max(1, num(every, 10));
        const rate = Math.max(1, Math.min(60, num(fps, 12)));
        const factor = interval * rate;
        // A half-hour visit: long enough to be the case worth previewing, short enough
        // that the answer stays in seconds at every sane setting.
        const plays = (30 * 60) / factor;
        const shown = plays >= 10 ? Math.round(plays) : Math.round(plays * 10) / 10;
        speed.textContent = `→ ${Math.round(factor)}× — a 30 min visit plays in ${shown} s`;
        speed.title = speed.textContent;
      }
      for (const i of [every, fps]) i.addEventListener('input', syncSpeed);
      syncSpeed();
      params.appendChild(speed);

      const minFrames = numberInput(stored.min_frames ?? 10, { step: '1', min: 1 });
      minFrames.dataset.testid = 'watch-action-event-timelapse-min-frames';
      minFrames.title = 'Events with fewer captured frames than this produce no video at all. The '
        + 'event, its snapshot and any clip are unaffected. Stops a brief visit from filling the '
        + 'events list with half-second videos.';
      const maxFrames = numberInput(stored.max_frames ?? 3600, { step: '100', min: 1 });
      maxFrames.dataset.testid = 'watch-action-event-timelapse-max-frames';
      maxFrames.title = 'Hard ceiling per event, so a stuck detection cannot fill the disk. Frames '
        + 'past it are dropped and the video ends early; the event is unaffected.';
      const maxWidth = numberInput(stored.max_width ?? 1280, { step: '80', min: 0 });
      maxWidth.title = 'Shrink frames to at most this width before encoding. 0 keeps their own size.';
      const quality = numberInput(stored.quality ?? 85, { step: '5', min: 1 });
      for (const i of [minFrames, maxFrames, maxWidth, quality]) i.addEventListener('input', ctx.onAnyChange);
      params.appendChild(wfield('min frames', minFrames, { width: '5rem', help: minFrames.title }));
      params.appendChild(wfield('max frames', maxFrames, { width: '5rem', help: maxFrames.title }));
      params.appendChild(wfield('max width', maxWidth, { width: '5rem', help: maxWidth.title }));
      params.appendChild(wfield('quality', quality, {
        width: '4.5rem',
        help: 'JPEG quality of each captured frame, 1–100, before it is encoded into the video.',
      }));

      // Off by default here, unlike `save_timelapse`'s: that one writes keepable frames,
      // this one writes a summary watched at 120×, where the extra pixels cost encode
      // time and disk for very little.
      const fullRes = document.createElement('input');
      fullRes.type = 'checkbox';
      fullRes.checked = Boolean(stored.full_resolution);
      fullRes.dataset.testid = 'watch-action-event-timelapse-full-res';
      fullRes.addEventListener('change', ctx.onAnyChange);
      params.appendChild(wfield('Full res', fullRes, {
        help: 'Capture the camera’s own frame rather than the downscaled one inference ran on. Off '
          + 'by default: this is a summary watched at speed, so the extra pixels cost encode time '
          + 'and disk for very little.',
      }));

      const annotate = document.createElement('input');
      annotate.type = 'checkbox';
      annotate.checked = Boolean(stored.annotate);
      annotate.dataset.testid = 'watch-action-event-timelapse-annotate';
      annotate.addEventListener('change', ctx.onAnyChange);
      params.appendChild(wfield('Draw boxes', annotate, {
        help: 'Burn the detection boxes into each frame. Unlike a clip, whose boxes are drawn over '
          + 'the video at play time, these are permanent — a timelapse’s frames are seconds apart, '
          + 'so there is no continuous track to draw from.',
      }));

      readParams = () => ({
        interval_s: num(every, 10),
        output_fps: num(fps, 12),
        min_frames: num(minFrames, 10),
        max_frames: num(maxFrames, 3600),
        max_width: num(maxWidth, 1280),
        quality: num(quality, 85),
        full_resolution: fullRes.checked,
        annotate: annotate.checked,
      });
    } else if (kind.value === 'save_timelapse') {
      // Browsed rather than typed, unlike `save_snapshot`'s: a timelapse directory is
      // one the user makes for the purpose (and the picker can make it), where a
      // snapshot directory is normally left at its default. The path is dataset-root
      // relative — the only space the browser and /api/images/mkdir speak — and
      // `resolve_output_dir` in the action accepts that spelling.
      const dir = dirPickerField(stored.output_dir ?? DEFAULT_TIMELAPSE_DIR, ctx.onAnyChange);
      params.appendChild(wfield('Output dir', dir.element, {
        width: '15rem',
        help: 'Where the frames are written, relative to the image dataset root. Placeholders '
          + 'are allowed — see the ? beside it — and a per-day folder is added for you. Browse… '
          + 'can create the folder if it does not exist yet.',
      }));
      params.appendChild(templateHelp(dir.input, {
        label: 'Placeholders you can use in this folder',
        intro: 'Substituted per path segment and slugified. One action covers every camera in '
          + 'the watcher, so use {source} to give each its own folder — without it they all '
          + 'share one. Must start with a real folder name before any placeholder. A YYYY-MM-DD '
          + 'folder is added automatically unless you place {date} yourself.',
      }));

      // The per-day subdirectory is created by the action, not picked here, so the
      // hint is the only place the user learns the frames are not written flat.
      const layout = el('span', 'watch-hint watch-hint--truncate', '');
      layout.dataset.testid = 'watch-action-timelapse-layout';
      function syncLayout() {
        const root = (dir.getValue() || DEFAULT_TIMELAPSE_DIR).replace(/^\/+|\/+$/g, '') || DEFAULT_TIMELAPSE_DIR;
        const rendered = renderTopicExample(root);
        const day = root.includes('{date}') ? '' : `${TEMPLATE_EXAMPLE.date}/`;
        layout.textContent = `→ ${rendered}/${day}back-room_14-30-00.jpg`;
        layout.title = root.includes('{source}')
          ? 'A folder per camera, then one per day.'
          : 'Every camera in this watcher writes into this one folder — add {source} to split them.';
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
      params.appendChild(wfield('Every s', every, { width: '4.5rem', help: every.title }));
      params.appendChild(wfield('quality', quality, {
        width: '4.5rem',
        help: 'JPEG quality, 1–100. Higher than the MQTT image default because these frames are '
          + 'kept on disk and watched back as a sequence, where compression artefacts crawl.',
      }));
      params.appendChild(wfield('max width', maxWidth, { width: '5rem', help: maxWidth.title }));

      // On by default, like the MQTT image action: the analysis downscale exists to
      // make inference cheap, and a timelapse is watched by a human.
      const fullRes = document.createElement('input');
      fullRes.type = 'checkbox';
      fullRes.checked = stored.full_resolution !== false;
      fullRes.dataset.testid = 'watch-action-timelapse-full-res';
      fullRes.addEventListener('change', ctx.onAnyChange);
      params.appendChild(wfield('Full res', fullRes, {
        help: 'Save the camera’s own frame rather than the downscaled one inference ran on. On '
          + 'by default: the analysis downscale exists to make inference cheap, and a timelapse '
          + 'is watched by a human.',
      }));

      // Off by default: burnt-in boxes make the frames useless as training data and
      // flicker distractingly at playback speed.
      const annotate = document.createElement('input');
      annotate.type = 'checkbox';
      annotate.checked = Boolean(stored.annotate);
      annotate.dataset.testid = 'watch-action-timelapse-annotate';
      annotate.addEventListener('change', ctx.onAnyChange);
      params.appendChild(wfield('Draw boxes', annotate, {
        help: 'Burn the tracked boxes into each saved frame. Off by default: burnt-in boxes make '
          + 'the frames useless as training data and flicker distractingly at playback speed.',
      }));

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
      params.appendChild(wfield('Output dir', dir.element, {
        width: '15rem',
        help: 'Where snapshots are written, relative to the image dataset root. Placeholders are '
          + 'allowed — see the ? beside it — so “a folder per camera” or “per rule” is something '
          + 'you can write. A per-day folder is added for you.',
      }));
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
      enabled: enabledBox.checked,
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
      // Assigned once the Actions section exists. Declared here because every edit goes
      // through `onAnyChange`, including edits made before that point — the hint is a
      // no-op until there is something to sync.
      let syncEventHint = () => {};
      const onAnyChange = () => { bar.setDirty(true); syncEventHint(); };

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
      identity.appendChild(wfield('Name', nameInput, {
        help: 'What this watcher is called, in the list on the left and in the Runs tab. It is '
          + 'also what {watcher} renders as in a topic or folder template, so a rename changes '
          + 'where things are published — use {watcher_uuid} where that would matter.',
      }));

      const enabled = document.createElement('input');
      enabled.type = 'checkbox';
      enabled.checked = profile ? profile.enabled !== false : true;
      enabled.addEventListener('change', onAnyChange);
      // Deliberately modest wording: nothing reads this flag at run time today — the task
      // filters on each *row's* enabled flag, not the profile's — so promising that
      // unticking it stops the watcher would be a lie the user would find out the hard way.
      identity.appendChild(wfield('Enabled', enabled, {
        help: 'Your own on/off marker for this watcher. It does not stop a run that is already '
          + 'going — stop that from the Runs tab.',
      }));

      const description = document.createElement('input');
      description.type = 'text';
      description.value = profile?.description || '';
      description.placeholder = 'optional';
      description.addEventListener('input', onAnyChange);
      identity.appendChild(wfield('Description', description, {
        help: 'A note for you. Never read by the watcher and never used in a template.',
      }));

      const defaultModels = modelChoiceSelect(
        { groupOptions: modelGroupOptions, modelNames },
        '(none)',
        modelChoiceValue(profile?.defaultModelGroupUuid, profile?.defaultModelName),
        onAnyChange,
        { testid: 'watch-default-model-choice' },
      );
      identity.appendChild(wfield('Default models', defaultModels, {
        help: 'The model or model stack every source in this watcher uses unless its own row '
          + 'overrides it. A stack and a single model are exclusive — picking one clears the '
          + 'other.',
      }));
      screen.editorBody.appendChild(identity);

      // ── Sources ──
      const addSourceBtn = el('button', 'panel-btn', '+ Add source');
      addSourceBtn.type = 'button';
      addSourceBtn.dataset.testid = 'watch-add-source';
      addSourceBtn.title = 'Add a camera for this watcher to analyse. Every rule and action below '
        + 'applies to all of them unless you scope it to some.';
      const sourcesBox = el('div', 'watch-rows');
      sourcesBox.dataset.testid = 'watch-sources-box';
      screen.editorBody.appendChild(sectionPanel(sectionHeader('Sources', addSourceBtn), sourcesBox));

      function addSource(src) {
        const entry = buildSourceRow(src, {
          sourceOptions,
          groupOptions: modelGroupOptions,
          modelNames,
          sourceNameFor,
          ingestTierFor,
          masksForSource: uuid => regionsFor(uuid, REGION_ROLE_MASK),
          onRegionsChanged: () => refreshRegions(),
          onSourceChanged: () => { refreshScopes(); for (const rule of ruleRows) rule.refreshZones?.(); },
          onAnyChange,
          onDelete: () => {
            entry.element.remove();
            sourceRows.splice(sourceRows.indexOf(entry), 1);
            // Removing a source removes its zones from the rules' options.
            for (const rule of ruleRows) rule.refreshZones?.();
            // ...and removes it from every action that was scoped to it, which would
            // otherwise be saved as a scope pointing at a source the watcher no
            // longer has and rejected by the backend with no visible cause.
            refreshScopes();
            onAnyChange();
          },
        });
        sourceRows.push(entry);
        sourcesBox.appendChild(entry.element);
        for (const rule of ruleRows) rule.refreshZones?.();
        refreshScopes();
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
      addLifetimeBtn.title = 'Override how long one label’s track survives without a detection — '
        + 'a litter box can outlast 30s of misses where a cat should not.';

      const trackerOptions = trackers.map(t => ({ key: t.typename, text: t.displayName || t.typename }));
      const trackerSel = select(
        trackerOptions.length ? trackerOptions : [{ key: 'IouTracker', text: 'IoU / SORT-lite' }],
        profile?.trackerTypename || 'IouTracker',
        () => { renderTrackerParams(); onAnyChange(); },
      );
      trackerSel.dataset.testid = 'watch-tracker';
      trackerSel.title =
        'How per-frame detections are joined into objects with an identity over time. Every '
        + 'rule below is written against those objects, not against raw detections — which is '
        + 'what lets a rule say "for 2 seconds". Its options are under Tuning.';

      // The picker rides the section's own heading line and the params collapse behind a
      // closed twisty, so the whole section costs no height beyond the heading it needed
      // anyway. A profile with one source, rule and action must still fit without the
      // editor scrolling — there is a Playwright test that says so.
      const trackingHead = sectionHeader('Tracking', addLifetimeBtn);
      trackingHead.insertBefore(wfield('Tracker', trackerSel, { help: trackerSel.title }), addLifetimeBtn);

      const trackerParamsHost = el('div', 'watch-tracker-params-host');
      const tuning = document.createElement('details');
      tuning.className = 'watch-tracker-tuning';
      tuning.dataset.testid = 'watch-tracker-tuning';
      const tuningSummary = document.createElement('summary');
      tuningSummary.textContent = 'Tuning';
      tuningSummary.title = 'The selected tracker’s own options. Collapsed by default — the '
        + 'defaults are sane, and each field explains itself on hover.';
      tuning.append(tuningSummary, trackerParamsHost);
      trackingHead.insertBefore(tuning, addLifetimeBtn);

      const lifetimesBox = el('div', 'watch-rows');
      lifetimesBox.dataset.testid = 'watch-lifetimes-box';
      screen.editorBody.appendChild(sectionPanel(trackingHead, lifetimesBox));

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
      addRuleBtn.title = 'Add a condition that raises an event while it holds. Rules decide when '
        + 'something happened; the actions below decide what is done about it.';
      const rulesBox = el('div', 'watch-rows');
      rulesBox.dataset.testid = 'watch-rules-box';
      screen.editorBody.appendChild(sectionPanel(sectionHeader('Rules', addRuleBtn), rulesBox));

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
       * Push the current rules and sources into every scope picker — the actions' pair
       * and each rule's source scope.
       *
       * Called on structural changes only (a rule or source added, removed or
       * repointed), never from `onAnyChange` — that fires on every keystroke, and
       * rebuilding a dropdown under the cursor would drop focus mid-word.
       */
      function refreshScopes() {
        for (const entry of ruleRows) entry.refreshScopes?.();
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
          sourceScopeItems,
          onAnyChange,
          onDelete: () => {
            entry.destroy?.();
            entry.element.remove();
            ruleRows.splice(ruleRows.indexOf(entry), 1);
            // A deleted rule must leave the scope of every action that named it.
            refreshScopes();
            onAnyChange();
          },
          // A renamed rule should read by its new name in the scope pickers. On
          // `change` (commit), not `input` — rebuilding a dropdown on every keystroke
          // would fight the user for focus.
          onNameCommitted: refreshScopes,
        });
        ruleRows.push(entry);
        rulesBox.appendChild(entry.element);
        refreshScopes();
      }

      addRuleBtn.addEventListener('click', () => { addRule({}); onAnyChange(); });

      // ── Actions ──
      const addActionBtn = el('button', 'panel-btn', '+ Add action');
      addActionBtn.type = 'button';
      addActionBtn.dataset.testid = 'watch-add-action';
      addActionBtn.title = 'Add something to do when an event starts or ends, or on every analysed '
        + 'frame — publish MQTT, record a clip, save a snapshot or a timelapse frame.';
      const actionsBox = el('div', 'watch-rows');
      actionsBox.dataset.testid = 'watch-actions-box';
      // The immediate counterpart of the backend's "Nothing files events" warning, which
      // only appears once the watcher has been saved and validated. Worth saying at the
      // moment the actions are being chosen: a watcher that publishes but files nothing
      // is a legitimate setup, and an empty Events list is otherwise indistinguishable
      // from a camera that saw nothing all day.
      //
      // In the section *header*, not above the rows, and terse with the explanation on
      // the tooltip. The editor's "a 1/1/1 profile does not scroll" budget is 30px and
      // spent; a full sentence on its own line costs ~20 of them for a watcher that is
      // configured perfectly correctly, which is the wrong thing to charge for a hint.
      const eventHint = el('span', 'watch-hint', '⚠ files no events');
      eventHint.dataset.testid = 'watch-actions-event-hint';
      eventHint.title =
        'None of these actions records the event, so these rules will not appear in Events. '
        + 'They still publish, and still show in the watcher’s log — only the Events list is '
        + 'empty. Add a Save event action to list them there.';
      const actionsHead = sectionHeader('Actions', addActionBtn);
      actionsHead.appendChild(eventHint);
      screen.editorBody.appendChild(sectionPanel(actionsHead, actionsBox));

      syncEventHint = () => {
        eventHint.hidden = actionRows.some((entry) => {
          const a = entry.getAction();
          return a.enabled && EVENT_LOGGING_KINDS.has(a.kind) && a.trigger !== FRAME_ANALYSED_TRIGGER;
        });
      };

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
      syncEventHint();

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
