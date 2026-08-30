/**
 * What makes two events "the same kind of thing", and the cross-view highlight built on it.
 *
 * The events page shows 1277 rows for what a user thinks of as 22 things — measured, not
 * estimated: the live history holds 22 distinct `(rule, subject, target)` identities, and
 * the top three of them account for 935 events. Every view here is a different projection
 * of that same grouping, so the key has to be computed in exactly one place.
 *
 * **The grain is chosen by the user, not fixed.** The finest grouping answers "Luna at the
 * Tupperware", which is the right unit for triage but the wrong one for a question like
 * "how much is each cat drinking?" — there, `Cat Drinking Water` alone splits into eight
 * rows across four vessels, and no row sums them. So the key is built from a *selected
 * set of dimensions*, and every view re-folds against it.
 *
 * The backend always groups at the finest grain and the client sums up to whatever is
 * asked for, so changing the grain costs no request. Summing here also guarantees a
 * coarse total is exactly the sum of its parts — the same reason `watch_event_counts`
 * derives its three tiers from one query rather than three.
 */

/** The separator. A literal that cannot appear in a uuid, so keys cannot collide. */
const SEP = '|';

/** Dimension name → the field carrying it on both a WatchEventRead and a group row. */
export const GROUP_DIMENSION_FIELDS = {
  rule: 'ruleUuid',
  subject: 'subjectRef',
  target: 'targetRef',
  source: 'sourceUuid',
};

/**
 * The grouping grains offered, as `[id, dimensions, label, description]`.
 *
 * `detail` is the default because it is the unit an event actually happened at, and it
 * is what the flat grid has always shown. The coarser ones exist because a summary is a
 * different question from a triage queue: `Cat Drinking Water` is one thing that keeps
 * happening, and per-cat is how you compare them side by side.
 */
export const GROUP_BY_MODES = [
  ['detail', ['rule', 'subject', 'target'], 'Actor & target', 'One row per actor, target and event type'],
  ['actor', ['rule', 'subject'], 'Actor', 'One row per actor per event type — cats side by side'],
  ['event', ['rule'], 'Event type', 'One row per event type, every actor summed'],
  ['subject', ['subject'], 'Actor only', 'One row per actor, across every kind of event'],
];

/** @param {string} id @returns {string[]} the dimensions for a mode id, defaulting to detail */
export function dimensionsFor(id) {
  return (GROUP_BY_MODES.find(([modeId]) => modeId === id) ?? GROUP_BY_MODES[0])[1];
}

/** The orders the grouped card grid and the timeline lanes can be read in. */
export const GROUP_SORTS = [
  ['count', 'Most events'],
  ['recent', 'Most recent'],
  ['name', 'Name (A–Z)'],
  ['rule', 'Event type (A–Z)'],
];

/**
 * Order folded groups for display.
 *
 * "Biggest first" answers "what dominates my day", which is why it is the default, but
 * it is useless for finding a particular cat in a list of twenty — an alphabetical order
 * is the only one where you can look something up rather than scan for it.
 *
 * ⚠️ Sorts a **copy**. `foldGroups` returns rows in a total, deterministic order and the
 * timeline and the card grid are handed the same array; sorting in place would let the
 * order depend on which view rendered first.
 *
 * Every comparator falls back to the incoming order, which is already total, so ties are
 * broken identically on every render rather than by whatever the engine's sort happens
 * to do with equal keys.
 *
 * @param {Array<Object>} groups - as returned by `foldGroups`
 * @param {string} sortBy - a `GROUP_SORTS` id
 * @param {string[]} dimensions
 * @returns {Array<Object>}
 */
export function sortGroups(groups, sortBy, dimensions) {
  const byName = (a, b) =>
    groupHeadline(a, dimensions).primary.localeCompare(groupHeadline(b, dimensions).primary, undefined, {
      sensitivity: 'base',
      numeric: true,
    });
  const comparators = {
    count: () => 0,
    // Empty `lastSeen` sorts last rather than first: a row that has never fired is not
    // the most recent thing that happened.
    recent: (a, b) => (b.lastSeen || '').localeCompare(a.lastSeen || ''),
    name: byName,
    rule: (a, b) => (a.ruleName || '').localeCompare(b.ruleName || '', undefined, { sensitivity: 'base' }) || byName(a, b),
  };
  const compare = comparators[sortBy] ?? comparators.count;
  return [...groups].sort(compare);
}

/**
 * @param {Object} source - a WatchEventRead or a WatchEventGroup; both carry these fields
 * @param {Object} [opts]
 * @param {string[]} [opts.dimensions] - which parts of the identity to key on
 * @param {boolean} [opts.splitByCamera] - add the camera, whatever the grain
 * @returns {string}
 */
export function identityKey(source, { dimensions = ['rule', 'subject', 'target'], splitByCamera = false } = {}) {
  const parts = dimensions.map(d => source?.[GROUP_DIMENSION_FIELDS[d]] ?? '');
  // Appended rather than folded into `dimensions` so the camera stays orthogonal: it
  // subdivides whichever grain is chosen instead of being a fifth grain of its own.
  //
  // A null subject or target is a real identity, not a missing one — the `Motion Test`
  // rule has neither. Normalising to '' keeps them groupable instead of dropping them,
  // the same care `_exclude` takes over NULL on the backend.
  parts.push(splitByCamera ? (source?.sourceUuid ?? '') : '');
  return parts.join(SEP);
}

/**
 * The two lines a row is captioned with, for the grain it is actually grouped at.
 *
 * A caption has to name what the row *is*, and at a coarser grain the finer fields are
 * no longer true of it: labelling a per-cat row "Luna → Tupperware" would be a lie about
 * a row that also counts her visits to the fountain. So only the grouped dimensions are
 * named, and the rule drops to a second line whenever it is not already the headline —
 * without which two rules acting on the same cat would produce two lanes both reading
 * "Luna", the duplicate-headline problem the camera toggle exists to avoid.
 *
 * @param {Object} group
 * @param {string[]} dimensions
 * @returns {{primary: string, secondary: string}}
 */
export function groupHeadline(group, dimensions) {
  const parts = [];
  if (dimensions.includes('subject') && group.subjectName) parts.push(group.subjectName);
  if (dimensions.includes('target') && group.targetName) parts.push(group.targetName);
  const primary = parts.join(' → ') || (dimensions.includes('rule') ? group.ruleName : '') || 'Event';
  const secondary = dimensions.includes('rule') && group.ruleName && group.ruleName !== primary
    ? group.ruleName
    : '';
  return { primary, secondary };
}

/**
 * Sum the backend's finest-grain groups up to the grain the UI is showing.
 *
 * @param {Array<Object>} groups - WatchEventGroup rows, always at the finest grain
 * @param {Object} [opts]
 * @param {string[]} [opts.dimensions]
 * @param {boolean} [opts.splitByCamera]
 * @returns {Array<Object>} merged groups, biggest first
 */
export function foldGroups(groups, { dimensions = ['rule', 'subject', 'target'], splitByCamera = false } = {}) {
  const byKey = new Map();
  for (const group of groups) {
    const key = identityKey(group, { dimensions, splitByCamera });
    const existing = byKey.get(key);
    if (!existing) {
      byKey.set(key, {
        ...group,
        key,
        dimensions,
        // Fields outside the grain are cleared, not carried. A per-actor row holding the
        // `targetRef` of whichever sub-group happened to be first would caption itself
        // "Luna → Tupperware" while counting her visits to the fountain too, and its
        // drill-down would then filter to a target that is not what the row means.
        ruleUuid: dimensions.includes('rule') ? group.ruleUuid : null,
        ruleName: dimensions.includes('rule') ? group.ruleName : null,
        subjectRef: dimensions.includes('subject') ? group.subjectRef : null,
        subjectName: dimensions.includes('subject') ? group.subjectName : null,
        targetRef: dimensions.includes('target') ? group.targetRef : null,
        targetName: dimensions.includes('target') ? group.targetName : null,
        // Which cameras this row covers, so a folded row can say "2 cameras" rather
        // than silently picking one of them.
        sourceUuids: group.sourceUuid ? [group.sourceUuid] : [],
        sourceUuid: splitByCamera ? group.sourceUuid : null,
      });
      continue;
    }
    existing.count += group.count;
    existing.totalDurationS += group.totalDurationS || 0;
    existing.withRecording += group.withRecording || 0;
    existing.withBlips += group.withBlips || 0;
    if (group.firstSeen && (!existing.firstSeen || group.firstSeen < existing.firstSeen)) {
      existing.firstSeen = group.firstSeen;
    }
    // The representative follows the newest event, so the folded card's art is the most
    // recent picture across every camera rather than whichever group happened to be first.
    if (group.lastSeen && (!existing.lastSeen || group.lastSeen > existing.lastSeen)) {
      existing.lastSeen = group.lastSeen;
      if (group.representativeUuid) existing.representativeUuid = group.representativeUuid;
    }
    if (group.sourceUuid && !existing.sourceUuids.includes(group.sourceUuid)) {
      existing.sourceUuids.push(group.sourceUuid);
    }
  }
  // Same ordering the backend applies, reapplied because folding changes the counts it
  // was based on. Ties broken by recency then key so the order is total, and therefore
  // stable between two renders of an unchanged set.
  return [...byKey.values()].sort(
    (a, b) => b.count - a.count || (b.lastSeen || '').localeCompare(a.lastSeen || '') || a.key.localeCompare(b.key),
  );
}

/**
 * The filter state that isolates one identity — what a group card drills into.
 *
 * Returns the *singular* filter parameters rather than the list ones, because they mean
 * the same thing to the backend and leaving the user's list selections untouched lets
 * "Clear" put the page back exactly as it was.
 *
 * Only the dimensions the row is actually grouped at become filters. A per-actor row
 * that also filtered by target would open onto a fraction of the events it counted, and
 * the card's number would visibly disagree with the grid it led to.
 *
 * @param {Object} group
 * @param {Object} [opts]
 * @param {string[]} [opts.dimensions] - defaults to the row's own, as recorded by `foldGroups`
 * @param {boolean} [opts.splitByCamera]
 * @returns {Object} a partial EventFilterState
 */
export function groupDrilldownFilters(group, { dimensions = null, splitByCamera = false } = {}) {
  const dims = dimensions ?? group.dimensions ?? ['rule', 'subject', 'target'];
  const pick = (dim, value) => (dims.includes(dim) && value ? [value] : []);
  return {
    ruleUuids: pick('rule', group.ruleUuid),
    subjectRefs: pick('subject', group.subjectRef),
    targetRefs: pick('target', group.targetRef),
    sourceUuids: splitByCamera && group.sourceUuid ? [group.sourceUuid] : [],
  };
}

/**
 * The facet options a drill-down needs the filter bar to already know about.
 *
 * The filter panel drops selections its facet list has never heard of — the guard that
 * stops a filter surviving the deletion of the last event holding its value. A group is
 * the one caller that can out-run that list: the facets refresh on load and on delete,
 * so an identity that first fired ten minutes ago is real, groupable, and absent from
 * the dropdown. Passing its name and ref through is what keeps the drill-down from
 * silently clearing itself and showing every event instead.
 *
 * @param {Object} group
 * @returns {Object} keyed by facet, matching `createEventFilters`'s `seed` option
 */
export function groupFacetSeeds(group) {
  return {
    rules: group.ruleUuid ? [{ ref: group.ruleUuid, name: group.ruleName || group.ruleUuid }] : [],
    subjects: group.subjectRef ? [{ ref: group.subjectRef, name: group.subjectName || group.subjectRef }] : [],
    targets: group.targetRef ? [{ ref: group.targetRef, name: group.targetName || group.targetRef }] : [],
    sources: group.sourceUuid ? [{ ref: group.sourceUuid, name: group.sourceUuid }] : [],
  };
}

/**
 * Page-scoped highlight, keyed on identity rather than on the element.
 *
 * Hovering one "Maple → Tupperware" anywhere lights every other one everywhere — a
 * timeline lane, a group card and every matching tile in the flat grid, because all
 * three register under the same key. That cross-view link is the entire reason this is
 * a shared controller rather than a `mouseenter` handler per view.
 *
 * **Deliberately not on the app-wide EventBus.** This fires on every `mouseenter` over a
 * strip that can hold hundreds of marks; a global pub/sub for a page-local,
 * high-frequency, purely visual concern is the wrong layer, and `EventTypes` should not
 * grow a constant only two sibling modules could ever mean anything by.
 *
 * Focus is applied by flipping **one** class on the root and one per member, never by
 * writing inline styles — the dimming is CSS's job, and a style write per element per
 * mousemove is how a hover becomes janky.
 *
 * @param {HTMLElement} root - gets `event-focus-active` while anything is focused
 * @returns {{register: Function, clear: Function, setFocus: Function, getFocus: Function,
 *            reset: Function}}
 */
export function createFocusController(root) {
  /** @type {Map<string, Set<HTMLElement>>} */
  const members = new Map();
  let active = null;
  // A focus that survives the pointer leaving. Hover is the fast path, but it is a
  // pointer-only gesture: on a touch screen there is no hover at all, so without a
  // sticky mode the highlight would simply not exist there. Pinning is worth having on
  // a desktop too — it is how you compare one identity's marks against the clock
  // without holding the mouse still.
  let pinned = null;

  function apply(key, on) {
    for (const el of members.get(key) ?? []) el.classList.toggle('is-focused', on);
  }

  function setFocus(key) {
    // A pin outranks hover, so moving the pointer across the page cannot silently
    // dismiss a highlight the user deliberately fixed.
    const next = pinned || key || null;
    if (next === active) return;
    if (active) apply(active, false);
    active = next;
    if (active) apply(active, true);
    root.classList.toggle('event-focus-active', Boolean(active));
    root.classList.toggle('event-focus-pinned', Boolean(pinned));
  }

  /** Toggle the sticky focus. Pinning the already-pinned key releases it. */
  function togglePin(key) {
    pinned = pinned === key ? null : key || null;
    // Recompute from scratch: `setFocus` short-circuits when the key is unchanged, and
    // releasing a pin leaves exactly that case with a different meaning.
    const next = pinned || null;
    if (active) apply(active, false);
    active = next;
    if (active) apply(active, true);
    root.classList.toggle('event-focus-active', Boolean(active));
    root.classList.toggle('event-focus-pinned', Boolean(pinned));
    return pinned;
  }

  /**
   * Register an element as a member of an identity, and wire its hover.
   *
   * Returns an unregister function, matching the EventBus's `subscribe` convention so
   * callers do not have to remember which of the two shapes this one is.
   */
  function register(key, element, { hover = true } = {}) {
    let set = members.get(key);
    if (!set) members.set(key, (set = new Set()));
    set.add(element);
    if (active === key) element.classList.add('is-focused');
    if (hover) {
      element.addEventListener('mouseenter', () => setFocus(key));
      element.addEventListener('mouseleave', () => setFocus(null));
      // Keyboard and screen-reader users reach the highlight too; focusin/out rather
      // than focus/blur so it works when the focus lands on a child control.
      element.addEventListener('focusin', () => setFocus(key));
      element.addEventListener('focusout', () => setFocus(null));
    }
    return () => {
      set.delete(element);
      if (!set.size) members.delete(key);
    };
  }

  /**
   * Forget every registration, without touching the active key.
   *
   * Called at the top of a re-render: the elements are about to be discarded, and a
   * Map still holding them would keep the whole previous page alive.
   */
  function reset() {
    members.clear();
  }

  return {
    register,
    setFocus,
    togglePin,
    getFocus: () => active,
    getPinned: () => pinned,
    clear: () => {
      pinned = null;
      setFocus(null);
    },
    reset,
  };
}
