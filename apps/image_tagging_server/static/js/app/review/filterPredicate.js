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
import { readConfidence } from '/app-static/js/shared/domain/image/confidence.js';
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
    bboxVerification: 'any', // 'any' | 'verified' | 'unverified'
    labelUuids: [],
    labelExcludeUuids: [],
    labelMatchMode: 'any', // 'any' | 'all'
    labelIncludeChildren: false,
    tagUuids: [],
    tagExcludeUuids: [],
    tagMatchMode: 'any', // 'any' | 'all'
    // Model-confidence band, in the 0–1 unit the metadata stores (the UI control
    // talks percent). Bounds are inclusive and independently optional; null is
    // the no-op. Mirrors the server's BBoxConfidenceTerm.
    confidenceMin: null,
    confidenceMax: null,
    // Whether a bbox carrying no confidence at all satisfies the band. Default
    // false: an absent confidence is not 0.
    confidenceIncludeMissing: false,
    // Image-scoped (NOT bbox-scoped — see `imageMatchesFilename`): a
    // case-insensitive substring test over `img.filename`. '' is the no-op.
    filenameSearch: '',
    // Image-scoped date range over `img.createdAt`, mirroring the main page's
    // DateTerm (`field: 'created'`). ISO strings, as produced by
    // `new Date(input.value).toISOString()`; null is the no-op.
    dateFrom: null,
    dateTo: null,
  };
}

/**
 * Parse a datetime string off an image record into epoch millis.
 *
 * The API hands back **naive** timestamps — `"2026-08-08T15:59:08.408748"`,
 * no offset — even though `created_at`/`updated_at` are declared
 * `DateTime(timezone=True)`: SQLite has no native datetime type and drops the
 * zone, so the value round-trips as a bare string and Pydantic serialises it
 * without one. `new Date()` reads an offset-less ISO *date-time* as **local**
 * time, which would silently shift every comparison by the viewer's UTC
 * offset (and shift it differently for two users looking at the same image).
 * The stored instants are UTC, so an offset-less string is pinned to UTC here.
 *
 * @param {?string} value
 * @returns {?number} epoch millis, or null if absent/unparseable
 */
export function parseRecordDate(value) {
  if (!value) return null;
  const s = String(value);
  const iso = /(?:Z|[+-]\d{2}:?\d{2})$/.test(s) ? s : `${s}Z`;
  const t = Date.parse(iso);
  return Number.isNaN(t) ? null : t;
}

/**
 * Image-scoped created-date range test — the client-side counterpart of the
 * server's `DateTerm` with `field: 'created'` (the only field the review UI
 * ever sends; see buildReviewFilterBody in shared/api/review.js).
 *
 * Bounds are inclusive on both ends, matching the server's `col >= from` /
 * `col <= to`. An image with no `createdAt` fails any active bound, mirroring
 * SQL's NULL comparison never being true.
 *
 * @param {{createdAt?: string}} img
 * @param {ReturnType<typeof defaultCuratedFilterState>} state
 * @returns {boolean}
 */
export function imageMatchesDate(img, state) {
  const from = parseRecordDate(state.dateFrom);
  const to = parseRecordDate(state.dateTo);
  if (from === null && to === null) return true;

  const created = parseRecordDate(img.createdAt);
  if (created === null) return false;
  if (from !== null && created < from) return false;
  if (to !== null && created > to) return false;
  return true;
}

/**
 * Image-scoped filename substring test — the client-side counterpart of the
 * server's `FilenameTerm` (`ImageRecord.filename.like('%pat%')`,
 * db_client._apply_filter_terms).
 *
 * Two deliberate differences from the server, both consequences of where this
 * runs rather than choices:
 *  - **Relative, not absolute path.** The DB stores absolute filenames and the
 *    server LIKEs against those; every object that reaches this predicate has
 *    already been rewritten to a FILES_ROOT-relative path by
 *    `Manager._image_rel_path` (manager.py, both `get_images_by_uuids` and the
 *    review-queue path). So a pattern matching only the FILES_ROOT prefix
 *    ("image_datasets") matches everything server-side and nothing here.
 *  - **Explicitly case-insensitive.** SQLite's LIKE is case-insensitive for
 *    ASCII by default; `String.includes` is not, so we lowercase both sides to
 *    keep the main page and the modals agreeing on the common (ASCII) case.
 *
 * @param {{filename?: string}} img
 * @param {ReturnType<typeof defaultCuratedFilterState>} state
 * @returns {boolean}
 */
export function imageMatchesFilename(img, state) {
  const needle = (state.filenameSearch || '').trim().toLowerCase();
  if (!needle) return true;
  return (img.filename || '').toLowerCase().includes(needle);
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
    hasActiveConfidenceFilter(state) ||
    state.labelUuids?.length > 0 ||
    state.labelExcludeUuids?.length > 0 ||
    state.tagUuids?.length > 0 ||
    state.tagExcludeUuids?.length > 0
  );
}

/**
 * True when `state` carries an active confidence constraint. `includeMissing`
 * alone is NOT active: with no bounds it would match every bbox, which is what
 * the no-op default already does (and matches the server, where a bounds-less
 * term with include_missing is `true()`).
 *
 * @param {ReturnType<typeof defaultCuratedFilterState>} state
 * @returns {boolean}
 */
export function hasActiveConfidenceFilter(state) {
  return Number.isFinite(state?.confidenceMin) || Number.isFinite(state?.confidenceMax);
}

/**
 * Per-bbox confidence-band test — the client-side counterpart of the server's
 * `BBoxConfidenceTerm` (db_client `_bbox_confidence_expr`).
 *
 * `readConfidence` handles the same two things the SQL expression does: the two
 * metadata keys a confidence may live under, and its being a string on some rows
 * and a number on others. A bbox with no confidence has `undefined` here, which
 * mirrors SQL NULL — it fails any active bound unless `confidenceIncludeMissing`
 * opts it back in.
 *
 * @param {{metadataJson?: object}} bbox
 * @param {ReturnType<typeof defaultCuratedFilterState>} state
 * @returns {boolean}
 */
export function bboxMatchesConfidence(bbox, state) {
  if (!hasActiveConfidenceFilter(state)) return true;

  const confidence = readConfidence(bbox?.metadataJson);
  if (confidence === undefined) return !!state.confidenceIncludeMissing;
  if (Number.isFinite(state.confidenceMin) && confidence < state.confidenceMin) return false;
  if (Number.isFinite(state.confidenceMax) && confidence > state.confidenceMax) return false;
  return true;
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

  // ── Model confidence band ────────────────────────────────────────────────
  if (!bboxMatchesConfidence(bbox, state)) return false;

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
 * The filename and date filters are applied first and independently: both are
 * image-scoped, so they must reject the image outright rather than participate
 * in the "any bbox matches" logic below (a bbox-free image with a non-matching
 * filename has to fail even though `hasActiveBboxFilter` is false).
 *
 * @param {{uuid: string, filename?: string, createdAt?: string, bboxes?: Array<object>}} img
 * @param {ReturnType<typeof defaultCuratedFilterState>} state
 * @param {Map<string, object>} labels
 * @returns {boolean}
 */
export function imageMatchesFilter(img, state, labels) {
  if (!imageMatchesFilename(img, state)) return false;
  if (!imageMatchesDate(img, state)) return false;

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
