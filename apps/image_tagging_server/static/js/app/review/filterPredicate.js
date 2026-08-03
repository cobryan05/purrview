/**
 * Shared client-side filter predicate — F1 of the "unify Review filters" plan.
 *
 * This is the curated filter contract that runs in-memory over already-fetched
 * `ImageRecordRead`-shaped objects, so the same filters that drive the main
 * Review page's server-side search (as of PR 1.4b, entirely the discriminated-
 * union `_apply_filter_terms` in apps/helpers/db/db_client.py — the old flat-
 * field `_apply_image_search_filters` logic it originally mirrored was
 * deleted in that PR) can also run inside similarity/dissimilarity modals,
 * which have no server-side search of their own. This module's own state
 * shape (`defaultCuratedFilterState()`) is unchanged by 1.4b — it's fed
 * directly from widget selections (label/tag pickers, verification control),
 * never serialised to the wire, so there was nothing here that needed to
 * change to track the server's flat-field -> terms cutover; only the
 * doc comments below (which named the now-deleted flat-field code) needed
 * updating.
 *
 * Shapes (already on the objects returned by fetchImagesByUuids/fetchReviewQueue):
 *   bbox = { uuid, labelUuid, label, tags: [{uuid, icon}] }
 *   img  = { uuid, filename, bboxes: [bbox], tags, createdAt, updatedAt }
 *
 * `labels` is the standard Map<uuid, labelRecord> (records carry `parentUuid`)
 * used throughout the review UI (see ui/labelList.js).
 */
import { VERIFICATION_TAG_BY_STATE } from '/app-static/js/shared/api/review.js';
import { expandWithDescendants } from '/app-static/js/ui/labelList.js';

/**
 * Maps a `bboxVerification` tri-state value ('any' has no entry — it's a
 * no-op) to the tag UUID a matching bbox must carry. Defined once in
 * shared/api/review.js (re-exported here for existing importers, e.g.
 * `makeBboxFilterPredicate` in runModelOnResults.js) so it can't drift
 * from the server-side filter-body builder's copy.
 */
export { VERIFICATION_TAG_BY_STATE };

/**
 * @returns {object} a fresh curated filter state, all fields at their
 *   "no-op" default — `imageMatchesFilter`/`bboxMatchesFilter` match everything.
 */
export function defaultCuratedFilterState() {
  return {
    bboxVerification: 'any', // 'any' | 'verified' | 'unverified' | 'false_pos'
    labelUuids: [],
    labelExcludeUuids: [],
    labelMatchMode: 'any', // 'any' | 'all'
    labelIncludeChildren: false,
    tagUuids: [],
    tagExcludeUuids: [],
    tagMatchMode: 'any', // 'any' | 'all'
  };
}

/**
 * True when `state` carries at least one active bbox-scoped constraint
 * (verification tri-state, label include/exclude, or tag include/exclude).
 * When false the curated filter is a no-op and matches everything — used by
 * `imageMatchesFilter` to let bbox-free images through under the default state.
 * @param {ReturnType<typeof defaultCuratedFilterState>} state
 * @returns {boolean}
 */
export function hasActiveBboxFilter(state) {
  return !!(
    (state.bboxVerification && state.bboxVerification !== 'any') ||
    state.labelUuids?.length > 0 ||
    state.labelExcludeUuids?.length > 0 ||
    state.tagUuids?.length > 0 ||
    state.tagExcludeUuids?.length > 0
  );
}

/**
 * Expand a label-uuid list to a membership Set, optionally including
 * descendants — the one place both this module and `makeBboxFilterPredicate`
 * (runModelOnResults.js) need "resolve a label filter to a Set" so the
 * expansion policy can't drift between the two call sites.
 *
 * @param {string[]} labelUuids
 * @param {Map<string,object>} labels
 * @param {boolean} includeChildren
 * @returns {Set<string>}
 */
export function expandLabelFilterSet(labelUuids, labels, includeChildren) {
  return includeChildren ? expandWithDescendants(labelUuids, labels) : new Set(labelUuids);
}

/**
 * True if `bbox.tags` contains at least one of `tagUuidSet`.
 * @param {{tags?: Array<{uuid: string}>}} bbox
 * @param {Set<string>} tagUuidSet
 */
export function bboxHasAnyTag(bbox, tagUuidSet) {
  return (bbox.tags || []).some(t => tagUuidSet.has(t.uuid));
}

/**
 * True if `bbox.tags` contains every one of `tagUuidSet`.
 * @param {{tags?: Array<{uuid: string}>}} bbox
 * @param {Set<string>} tagUuidSet
 */
export function bboxHasAllTags(bbox, tagUuidSet) {
  const have = new Set((bbox.tags || []).map(t => t.uuid));
  for (const uuid of tagUuidSet) {
    if (!have.has(uuid)) return false;
  }
  return true;
}

/**
 * Per-bbox curated-filter predicate. Mirrors the server's
 * `_apply_image_search_filters` semantics (apps/helpers/db/db_client.py:187),
 * with one deliberate deviation — see the label-exclude block below.
 *
 * @param {{uuid: string, labelUuid: ?string, label?: object, tags?: Array<{uuid:string}>}} bbox
 * @param {ReturnType<typeof defaultCuratedFilterState>} state
 * @param {Map<string, object>} labels
 * @returns {boolean}
 */
export function bboxMatchesFilter(bbox, state, labels) {
  // ── Bbox verification (tri-state, exclusive) ──────────────────────────────
  if (state.bboxVerification && state.bboxVerification !== 'any') {
    const wantTagUuid = VERIFICATION_TAG_BY_STATE[state.bboxVerification];
    if (!wantTagUuid || !bboxHasAnyTag(bbox, new Set([wantTagUuid]))) return false;
  }

  // ── Label include ──────────────────────────────────────────────────────────
  if (state.labelUuids?.length > 0) {
    const includeSet = expandLabelFilterSet(state.labelUuids, labels, state.labelIncludeChildren);
    if (!includeSet.has(bbox.labelUuid)) return false;
  }

  // ── Label exclude ────────────────────────────────────────────────────────
  // Deviation from the plan doc's proposed spec: the plan suggested expanding
  // exclusions with descendants under the same `labelIncludeChildren` flag.
  // The server does NOT do this. Pre-1.4b, `_apply_image_search_filters`'s
  // "Label exclusions" block was a plain `BBox.label_uuid == label_uuid`
  // check with no descendant expansion. As of 1.4b it's an ObjectLabelTerm
  // with `negate: true` — and callers (db_client.build_dataset_search_params,
  // shared/api/review.js's buildReviewFilterBody) explicitly pass
  // `rollUp: false` for it, since ObjectLabelTerm.rollUp defaults true and
  // would otherwise silently change this behaviour. Either way,
  // `label_include_children`/`labelIncludeChildren` only ever affects the
  // include-side label_uuids/label_match_mode predicate. We mirror the
  // server here: exclusion is an exact-UUID match, never child-expanded.
  if (state.labelExcludeUuids?.length > 0) {
    if (state.labelExcludeUuids.includes(bbox.labelUuid)) return false;
  }

  // ── Tag include ──────────────────────────────────────────────────────────
  if (state.tagUuids?.length > 0) {
    const tagSet = new Set(state.tagUuids);
    const included = state.tagMatchMode === 'all'
      ? bboxHasAllTags(bbox, tagSet)
      : bboxHasAnyTag(bbox, tagSet);
    if (!included) return false;
  }

  // ── Tag exclude ──────────────────────────────────────────────────────────
  if (state.tagExcludeUuids?.length > 0) {
    if (bboxHasAnyTag(bbox, new Set(state.tagExcludeUuids))) return false;
  }

  return true;
}

/**
 * Per-image curated-filter predicate: true if any bbox on `img` matches
 * `bboxMatchesFilter`, mirroring how `render()` builds one entry per matching
 * bbox and shows the image if any exist. When `labelMatchMode === 'all'` and
 * labels are selected, additionally requires that every selected label is
 * "covered" by some matching bbox (the image-level AND-across-labels gate —
 * matches the server's per-label-uuid EXISTS loop for `label_match_mode="all"`
 * in `_apply_image_search_filters`).
 *
 * @param {{uuid: string, bboxes?: Array<object>}} img
 * @param {ReturnType<typeof defaultCuratedFilterState>} state
 * @param {Map<string, object>} labels
 * @returns {boolean}
 */
export function imageMatchesFilter(img, state, labels) {
  const matching = (img.bboxes || []).filter(b => bboxMatchesFilter(b, state, labels));
  if (matching.length === 0) {
    // No matching bboxes. Only reject when there is an active bbox-scoped
    // constraint the image failed — with the no-op default state there is
    // nothing to filter on, so a bbox-free (or otherwise unconstrained) image
    // must pass, matching the pre-curated behavior where every fetched image
    // was shown. (Image-oriented modals — similar/dissimilar/diversity — return
    // whole images that frequently have zero bboxes.)
    return !hasActiveBboxFilter(state);
  }

  if (state.labelMatchMode === 'all' && state.labelUuids?.length > 0) {
    // For each selected label, expand to itself (+descendants if requested)
    // and require some matching bbox's label to fall in that expanded set —
    // i.e. every selected label is "covered" by at least one matching bbox.
    for (const selectedUuid of state.labelUuids) {
      const coverSet = expandLabelFilterSet([selectedUuid], labels, state.labelIncludeChildren);
      const covered = matching.some(b => coverSet.has(b.labelUuid));
      if (!covered) return false;
    }
  }

  return true;
}
