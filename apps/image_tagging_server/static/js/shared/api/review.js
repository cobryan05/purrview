import { apiFetch } from './apiUtils.js';
// System tag UUIDs (must match constants in apps/helpers/db/types.py)
export const UNVERIFIED_TAG_UUID = '00000000-0000-0000-0000-000000000001';
// Keep in sync with apps/helpers/db/types.py
export const REJECTION_TAG_UUID = '00000000-0000-0000-0000-000000000004';
export const VERIFIED_TAG_UUID = '00000000-0000-0000-0000-000000000005';

// Both are in the verification exclusive group — mutually exclusive per bbox
export const VERIFICATION_TAG_UUIDS = new Set([UNVERIFIED_TAG_UUID, VERIFIED_TAG_UUID]);

// Maps the bbox-verification tri-state filter's state values ('any' is a
// no-op with no entry here) to the tag UUID that must be present on the bbox.
// This is the single canonical copy — app/review/filterPredicate.js imports
// and re-exports it (that module already imports the UUID constants above
// from here, so this is the established import direction; filterPredicate.js
// is the higher-level module and review.js must not import back from it).
export const VERIFICATION_TAG_BY_STATE = {
  verified: VERIFIED_TAG_UUID,
  unverified: UNVERIFIED_TAG_UUID,
};

/**
 * Build the JSON body sent to the review-queue / bulk-apply-filter endpoints.
 * The single source of truth for filter-body assembly — fetchReviewQueue,
 * bulkApplyAll and bulkApplyImages each call this with the option subset they
 * support (pagination, limit and bboxVerification only apply to some of them)
 * rather than each re-deriving the same term-assembly logic independently,
 * which had drifted out of sync between them.
 *
 * As of PR 1.4b the request body carries a single `terms` list (the
 * discriminated-union filter-term model) instead of the old flat fields
 * (tagUuids, labelUuids, imageLabelFilters, filenameSearch, ...) — those
 * flat ImageSearchParams fields were deleted server-side. This function's
 * *option* shape (the caller-facing opts object) is unchanged; only what it
 * serialises onto the wire changed, so every call site below keeps working
 * unmodified.
 *
 * @param {object} opts - same shape as fetchReviewQueue's opts, plus `limit`
 *   left `undefined` to omit the field entirely (bulkApplyImages paginates
 *   server-side across all pages and never sends a limit).
 * @returns {object} the request body
 */
function buildReviewFilterBody({
  tagUuids,
  tagExcludeUuids = [],
  showAllBboxes = false,
  tagMatchMode = 'any',
  labelUuids = [],
  labelExcludeUuids = [],
  labelMatchMode = 'any',
  labelIncludeChildren = false,
  labelUuid,
  sortBy,
  sortOrder,
  limit,
  requireBboxes = false,
  filenameSearch,
  dateFrom = null,
  dateTo = null,
  afterUuid = null,
  afterSortValue = null,
  imageLabelFilters = [],
  imageLabelConditions = [],
  verificationCheck = null,
  verificationLabelUuids = [],
  bboxVerification = null,
} = {}) {
  const body = {
    showAllBboxes,
    sortBy,
    sortOrder,
  };
  if (limit !== undefined) body.limit = limit;
  if (requireBboxes) body.requireBboxes = true;

  const terms = [];

  if (!showAllBboxes && tagUuids && tagUuids.length > 0) {
    terms.push({ kind: 'bboxTag', tagUuids, matchMode: tagMatchMode });
  }
  for (const uuid of tagExcludeUuids) {
    terms.push({ kind: 'bboxTag', tagUuids: [uuid], negate: true });
  }
  if (bboxVerification && bboxVerification !== 'any') {
    // 'verified'/'unverified' are the BBoxVerificationTerm.state literals verbatim.
    terms.push({ kind: 'bboxVerification', state: bboxVerification });
  }

  if (labelUuids.length > 0) {
    terms.push({ kind: 'objectLabel', labelUuids, matchMode: labelMatchMode, rollUp: labelIncludeChildren });
  }
  if (labelUuid && !labelUuids.includes(labelUuid)) {
    terms.push({ kind: 'objectLabel', labelUuids: [labelUuid] });
  }
  for (const uuid of labelExcludeUuids) {
    // rollUp: false — the pre-1.4b flat labelExclusions field was a plain
    // equality check, never child-expanded (see filterPredicate.js's
    // matching client-side divergence comment). ObjectLabelTerm.rollUp
    // defaults true server-side, so this must be explicit.
    terms.push({ kind: 'objectLabel', labelUuids: [uuid], negate: true, rollUp: false });
  }

  if (filenameSearch) terms.push({ kind: 'filename', patterns: [filenameSearch] });
  if (dateFrom || dateTo) {
    const term = { kind: 'date', field: 'created' };
    if (dateFrom) term.from = dateFrom;
    if (dateTo) term.to = dateTo;
    terms.push(term);
  }

  if (verificationCheck) {
    const targets = verificationLabelUuids.length > 0 ? verificationLabelUuids : labelUuids;
    terms.push({ kind: 'labelVerification', check: verificationCheck, labelUuids: targets });
  }

  // Explicit imageLabelFilters (direct format) take precedence over conditions.
  // imageLabelConditions (from the filter panel) are mapped to the same term kind.
  for (const f of imageLabelFilters) {
    // Same two corrections as the imageLabelConditions loop below, for the same
    // reasons: 'uncertain' is the term model's 'unreviewed' (passing the raw
    // ImageItemStatus value into an ImageLabelState slot 422s), and rollUp must
    // be explicit — the pre-1.4b flat ImageLabelFilter was an exact-label
    // predicate, whereas ImageLabelTerm.roll_up defaults true server-side.
    const state = f.status === 'uncertain' ? 'unreviewed' : f.status;
    terms.push({
      kind: 'imageLabel',
      labelUuid: f.labelUuid,
      state,
      rollUp: f.rollUp ?? false,
      negate: f.exclude ?? false,
    });
  }
  for (const c of imageLabelConditions) {
    // 'uncertain' (the condition-panel's "never assessed" state) maps to the
    // term model's 'unreviewed' (NOT EXISTS any row) — the correct semantic
    // equivalent; the pre-1.4b flat ImageLabelFilter.status='uncertain' path
    // queried image_labels.status == 'uncertain', which per the tri-state
    // invariant (a row only ever exists for present/absent) always matched
    // zero rows. See PR 1.4b report for detail.
    const state = c.state === 'uncertain' ? 'unreviewed' : c.state;
    // rollUp/negate (slice 3, PR 3.2): default to the server's own defaults
    // (roll_up: true, negate: false — apps/helpers/db/types.py's
    // ImageLabelTerm) when the condition predates the UI control that sets
    // them, so existing saved presets/datasets keep behaving as before.
    terms.push({ kind: 'imageLabel', labelUuid: c.labelUuid, state, rollUp: c.rollUp ?? true, negate: c.negate ?? false });
  }

  if (afterUuid) body.afterUuid = afterUuid;
  if (afterSortValue) body.afterSortValue = afterSortValue;
  if (terms.length > 0) body.terms = terms;

  return body;
}

/**
 * Search the review queue with flexible filters and cursor-based pagination.
 *
 * @param {{
 *   tagUuids?: string[],          // bbox tags to include; default [UNVERIFIED_TAG_UUID]
 *   showAllBboxes?: boolean,       // ignore tagUuids, show all bboxes
 *   tagMatchMode?: 'any'|'all',   // OR (default) or AND across tagUuids
 *   labelUuids?: string[],         // bbox label filter (multi-select)
 *   labelMatchMode?: 'any'|'all', // OR (default) or AND across labelUuids
 *   labelIncludeChildren?: boolean, // also match child labels in the hierarchy
 *   labelUuid?: string,            // legacy single-label filter (still accepted by backend)
 *   sortBy?: 'created_at'|'updated_at'|'uuid',
 *   sortOrder?: 'asc'|'desc',
 *   filenameSearch?: string,
 *   dateFrom?: string|null,        // ISO datetime (filters on images.created_at)
 *   dateTo?: string|null,
 *   afterUuid?: string|null,       // keyset cursor: UUID of last image on previous page
 *   afterSortValue?: string|null,  // ISO datetime of last image's sort field (when sortBy != 'uuid')
 *   imageLabelFilters?: Array<{labelUuid: string, status: 'present'|'absent'|'uncertain',
 *                              exclude?: boolean, rollUp?: boolean}>,  // rollUp defaults false
 *   imageLabelConditions?: {labelUuid: string, state: 'present'|'absent'|'uncertain',
 *                           rollUp?: boolean, negate?: boolean}[],  // rollUp defaults true
 *   bboxVerification?: 'any'|'verified'|'unverified',  // 'any' is a no-op
 *   verificationLabelUuids?: string[],  // V2: active training config's target labels,
 *                                        // overrides labelUuids as the verificationCheck target
 *   limit?: number,
 * }} opts
 */
export async function fetchReviewQueue({
  tagUuids = [UNVERIFIED_TAG_UUID],
  tagExcludeUuids = [],
  showAllBboxes = false,
  tagMatchMode = 'any',
  labelUuids = [],
  labelExcludeUuids = [],
  labelMatchMode = 'any',
  labelIncludeChildren = false,
  labelUuid,
  sortBy = 'created_at',
  sortOrder = 'desc',
  filenameSearch,
  dateFrom = null,
  dateTo = null,
  afterUuid = null,
  afterSortValue = null,
  imageLabelFilters = [],
  imageLabelConditions = [],
  verificationCheck = null,
  verificationLabelUuids = [],
  bboxVerification = null,
  requireBboxes = false,
  limit = 50,
} = {}) {
  const body = buildReviewFilterBody({
    tagUuids,
    tagExcludeUuids,
    showAllBboxes,
    tagMatchMode,
    labelUuids,
    labelExcludeUuids,
    labelMatchMode,
    labelIncludeChildren,
    labelUuid,
    sortBy,
    sortOrder,
    limit,
    requireBboxes,
    filenameSearch,
    dateFrom,
    dateTo,
    afterUuid,
    afterSortValue,
    imageLabelFilters,
    imageLabelConditions,
    verificationCheck,
    verificationLabelUuids,
    bboxVerification,
  });

  const res = await apiFetch('/api/images/review', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  const data = await res.json();
  if (!res.ok || data.status !== 'success') throw new Error(data.message || 'Failed to fetch review queue');
  return {
    images: data.images ?? [],
    hasMore: data.hasMore ?? false,
    totalCount: data.totalCount ?? 0,
  };
}

/**
 * Replace all bboxes on an image. Each bbox in the list is a plain object:
 * { uuid, labelUuid, x, y, width, height, tagUuids }
 * @param {string} imagePath
 * @param {Array<object>} bboxes
 */
export async function updateImageBboxes(imagePath, bboxes) {
  const payload = { imagePath, boxes: bboxes };
  const res = await apiFetch('/api/images/metadata/update', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  const data = await res.json();
  if (!res.ok || data.status !== 'success') throw new Error(data.message || 'Failed to update image');
}



/**
 * Return a copy of tagUuids with the verification tag replaced by newTagUuid.
 * @param {string[]} tagUuids
 * @param {string} newTagUuid
 * @returns {string[]}
 */
/**
 * Generic exclusive-group swap: remove all tags that belong to the same exclusive
 * group (identified by groupUuids) and append newTagUuid.
 *
 * @param {string[]} tagUuids      - current tag UUID list
 * @param {Set<string>} groupUuids - the full set of UUIDs in the exclusive group
 * @param {string} newTagUuid      - the new group member to set
 * @returns {string[]}
 */
export function swapExclusiveGroupTag(tagUuids, groupUuids, newTagUuid) {
  return [...tagUuids.filter(t => !groupUuids.has(t)), newTagUuid];
}

/**
 * Return a copy of tagUuids with the verification tag replaced by newTagUuid.
 * @param {string[]} tagUuids
 * @param {string} newTagUuid
 */
export function swapVerificationTag(tagUuids, newTagUuid) {
  return swapExclusiveGroupTag(tagUuids, VERIFICATION_TAG_UUIDS, newTagUuid);
}

/**
 * Return a URL suitable for use as <img src> for a bbox crop thumbnail.
 * @param {string} imagePath
 * @param {string} bboxUuid
 * @param {number} [width=200]
 * @returns {string}
 */
export function getBboxThumbnailUrl(imagePath, bboxUuid, width = 200) {
  const encodedPath = imagePath.split('/').map(encodeURIComponent).join('/');
  return `/api/images/${encodedPath}/bbox/${bboxUuid}/thumbnail?width=${width}`;
}

export function getImageThumbnailUrl(imagePath, width = 200) {
  const encodedPath = imagePath.split('/').map(encodeURIComponent).join('/');
  return `/api/images/${encodedPath}/thumbnail?width=${width}`;
}

/**
 * Update a single bbox's label and tags.
 * @param {string} bboxUuid
 * @param {{ labelUuid: string, tagUuids: string[] }} opts
 */
export async function updateBbox(bboxUuid, { labelUuid, tagUuids }) {
  const res = await apiFetch(`/api/images/bbox/${bboxUuid}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ labelUuid, tagUuids }),
  });
  if (!res.ok) throw new Error(`updateBbox failed: ${res.status}`);
}

/**
 * Delete a single bbox.
 * @param {string} bboxUuid
 */
export async function deleteBbox(bboxUuid) {
  const res = await apiFetch(`/api/images/bbox/${bboxUuid}`, { method: 'DELETE' });
  if (!res.ok) throw new Error(`deleteBbox failed: ${res.status}`);
}

/**
 * Insert a single new bbox on an image (targeted insert — leaves siblings
 * untouched). Used by "Run Model on Results" to materialize a detection that
 * overlapped no existing box.
 * @param {{ imagePath: string, labelUuid: string, x: number, y: number,
 *   width: number, height: number, tagUuids?: string[], metadataJson?: object }} opts
 * @returns {Promise<string>} the new bbox uuid
 */
export async function addBbox({ imagePath, labelUuid, x, y, width, height, tagUuids = [], metadataJson = null }) {
  const res = await apiFetch('/api/images/bbox', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ imagePath, labelUuid, x, y, width, height, tagUuids, metadataJson }),
  });
  if (!res.ok) throw new Error(`addBbox failed: ${res.status}`);
  const data = await res.json();
  return data.bboxUuid;
}

/**
 * Apply an action to all bboxes matching the current filter, across all pages.
 * The filter params mirror fetchReviewQueue; bboxTagFilter / bboxLabelFilter
 * are the client-computed bbox-level subset filters.
 */
export async function bulkApplyAll({
  // Filter params (same shape as fetchReviewQueue, no pagination)
  tagUuids = [UNVERIFIED_TAG_UUID],
  tagExcludeUuids = [],
  showAllBboxes = false,
  tagMatchMode = 'any',
  labelUuids = [],
  labelExcludeUuids = [],
  labelMatchMode = 'any',
  labelIncludeChildren = false,
  sortBy = 'created_at',
  sortOrder = 'desc',
  filenameSearch,
  dateFrom = null,
  dateTo = null,
  imageLabelFilters = [],
  imageLabelConditions = [],
  verificationCheck = null,
  verificationLabelUuids = [],
  bboxVerification = null,
  requireBboxes = false,
  // Bbox-level subset filters (client-computed)
  bboxTagFilter = [],
  bboxLabelFilter = [],
  // Action params
  action,
  labelUuid = null,
  selectedTagUuids = [],
} = {}) {
  const filterBody = buildReviewFilterBody({
    tagUuids,
    tagExcludeUuids,
    showAllBboxes,
    tagMatchMode,
    labelUuids,
    labelExcludeUuids,
    labelMatchMode,
    labelIncludeChildren,
    sortBy,
    sortOrder,
    limit: 200,
    requireBboxes,
    filenameSearch,
    dateFrom,
    dateTo,
    imageLabelFilters,
    imageLabelConditions,
    verificationCheck,
    verificationLabelUuids,
    bboxVerification,
  });

  const res = await apiFetch('/api/images/bbox/bulk-apply', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      filter: filterBody,
      action,
      labelUuid: labelUuid ?? null,
      tagUuids: selectedTagUuids,
      bboxTagFilter,
      bboxLabelFilter,
      showAllBboxes,
      // Sent as its own bbox-level field, not only as a filter term: the term
      // selects images that HAVE a bbox in this state, which would still let
      // the action hit the other-state bboxes sharing those images — exactly
      // what the grid hides. The server ANDs this per bbox.
      bboxVerification: bboxVerification && bboxVerification !== 'any' ? bboxVerification : null,
    }),
  });
  const data = await res.json();
  if (!res.ok || data.status !== 'success') throw new Error(data.message || 'Bulk apply failed');
  return { modifiedCount: data.modifiedCount ?? 0, failCount: data.failCount ?? 0 };
}

/**
 * Count the bboxes a bulkApplyAll with this same filter would modify.
 *
 * Takes the identical option shape as bulkApplyAll minus the action fields, so
 * the count shown on the button and in the modal can't drift from the scope of
 * the action itself. NOT interchangeable with the review queue's totalCount —
 * that one counts matching *images*.
 *
 * @returns {Promise<number>}
 */
export async function countMatchingBboxes({
  tagUuids = [UNVERIFIED_TAG_UUID],
  tagExcludeUuids = [],
  showAllBboxes = false,
  tagMatchMode = 'any',
  labelUuids = [],
  labelExcludeUuids = [],
  labelMatchMode = 'any',
  labelIncludeChildren = false,
  filenameSearch,
  dateFrom = null,
  dateTo = null,
  imageLabelFilters = [],
  imageLabelConditions = [],
  verificationCheck = null,
  verificationLabelUuids = [],
  bboxVerification = null,
  requireBboxes = false,
  bboxTagFilter = [],
  bboxLabelFilter = [],
} = {}) {
  const filterBody = buildReviewFilterBody({
    tagUuids,
    tagExcludeUuids,
    showAllBboxes,
    tagMatchMode,
    labelUuids,
    labelExcludeUuids,
    labelMatchMode,
    labelIncludeChildren,
    sortBy: 'created_at',
    sortOrder: 'desc',
    requireBboxes,
    filenameSearch,
    dateFrom,
    dateTo,
    imageLabelFilters,
    imageLabelConditions,
    verificationCheck,
    verificationLabelUuids,
    bboxVerification,
  });

  const res = await apiFetch('/api/images/bbox/count', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      filter: filterBody,
      bboxTagFilter,
      bboxLabelFilter,
      showAllBboxes,
      bboxVerification: bboxVerification && bboxVerification !== 'any' ? bboxVerification : null,
    }),
  });
  const data = await res.json();
  if (!res.ok || data.status !== 'success') throw new Error(data.message || 'Bbox count failed');
  return data.count ?? 0;
}

/**
 * Apply an action to an explicit list of bboxes in one server-side pass.
 * Used by candidate-list-driven modals (similarity/diversity search) that have
 * no server-side filter equivalent for their already-known result set.
 *
 * @param {Object} opts
 * @param {?Record<string, {x: number, y: number, width: number, height: number}>}
 *   [opts.bboxGeometry] - bboxUuid -> replacement coordinates, applied on relabel
 *   (e.g. "Run Model on Results" swapping in a refinement model's suggested box).
 * @param {boolean} [opts.markVerified=true] - whether a relabel promotes the box to
 *   VERIFIED. True for human relabels (the default); "Run Model on Results" passes
 *   false so accepting a *model's* suggestion leaves the box unverified/reviewable.
 */
export async function bulkApplyBboxesByUuids({ bboxUuids, action, labelUuid = null, tagUuids = [], bboxGeometry = null, markVerified = true }) {
  const res = await apiFetch('/api/images/bbox/bulk-apply-by-uuids', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ bboxUuids, action, labelUuid, tagUuids, bboxGeometry, markVerified }),
  });
  const data = await res.json();
  if (!res.ok || data.status !== 'success') throw new Error(data.message || 'Bulk apply failed');
  return { modifiedCount: data.modifiedCount ?? 0, failCount: data.failCount ?? 0 };
}

/**
 * Apply an image-level action to an explicit list of images in one server-side
 * pass — the image-kind counterpart of bulkApplyBboxesByUuids.
 */
export async function bulkApplyImagesByUuids({ imageUuids, action, labelUuid = null, tagUuids = [] }) {
  const res = await apiFetch('/api/images/bulk-apply-by-uuids', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ imageUuids, action, labelUuid, tagUuids }),
  });
  const data = await res.json();
  if (!res.ok || data.status !== 'success') throw new Error(data.message || 'Bulk apply failed');
  return {
    modifiedCount: data.modifiedCount ?? 0,
    failCount: data.failCount ?? 0,
    skippedCount: data.skippedCount ?? 0,
  };
}

/**
 * Apply an image-level action to every image matching a server-side filter,
 * across all pages — the image-kind counterpart of bulkApplyAll. Unlike
 * bulkApplyImagesByUuids (explicit uuid list, capped by whatever the caller
 * already fetched), this paginates server-side so the caller never needs to
 * download the full candidate set (e.g. Number.MAX_SAFE_INTEGER) just to
 * apply an action to it.
 *
 * Filter params mirror fetchReviewQueue/bulkApplyAll's shape exactly — pass
 * the review filter panel's live getState() output. Action params use
 * `selectedTagUuids` (not `tagUuids`) to avoid colliding with the filter's
 * own `tagUuids`, same convention as bulkApplyAll.
 */
export async function bulkApplyImages({
  // Filter params (same shape as fetchReviewQueue, no pagination)
  tagUuids = [UNVERIFIED_TAG_UUID],
  tagExcludeUuids = [],
  showAllBboxes = false,
  tagMatchMode = 'any',
  labelUuids = [],
  labelExcludeUuids = [],
  labelMatchMode = 'any',
  labelIncludeChildren = false,
  sortBy = 'created_at',
  sortOrder = 'desc',
  filenameSearch,
  dateFrom = null,
  dateTo = null,
  imageLabelFilters = [],
  imageLabelConditions = [],
  verificationCheck = null,
  verificationLabelUuids = [],
  bboxVerification = null,
  requireBboxes = false,
  // Action params
  action,
  labelUuid = null,
  selectedTagUuids = [],
} = {}) {
  // The action here is image-level, so bboxVerification needs no bbox-level
  // counterpart (unlike bulkApplyAll) — as a filter term it selects exactly
  // the images the grid shows, which is the whole scope.
  // imageLabelConditions is new here (additive): callers used to have to
  // pre-map fs.imageLabelConditions to imageLabelFilters themselves before
  // this shared builder existed; now the builder does that merge for them,
  // same as fetchReviewQueue already did.
  const filterBody = buildReviewFilterBody({
    tagUuids,
    tagExcludeUuids,
    showAllBboxes,
    tagMatchMode,
    labelUuids,
    labelExcludeUuids,
    labelMatchMode,
    labelIncludeChildren,
    sortBy,
    sortOrder,
    requireBboxes,
    filenameSearch,
    dateFrom,
    dateTo,
    imageLabelFilters,
    imageLabelConditions,
    verificationCheck,
    verificationLabelUuids,
    bboxVerification,
  });

  const res = await apiFetch('/api/images/bulk-apply', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      filter: filterBody,
      action,
      labelUuid: labelUuid ?? null,
      tagUuids: selectedTagUuids,
    }),
  });
  const data = await res.json();
  if (!res.ok || data.status !== 'success') throw new Error(data.message || 'Bulk apply failed');
  return {
    modifiedCount: data.modifiedCount ?? 0,
    failCount: data.failCount ?? 0,
    skippedCount: data.skippedCount ?? 0,
  };
}

/**
 * Invalidate cached thumbnails for a list of bbox UUIDs.
 * @param {string[]} bboxUuids
 */
export async function invalidateBboxThumbnails(bboxUuids) {
  if (!bboxUuids.length) return;
  const res = await apiFetch('/api/images/bbox/thumbnails/invalidate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ bboxUuids }),
  });
  if (!res.ok) console.warn(`invalidateBboxThumbnails failed: ${res.status}`);
}
