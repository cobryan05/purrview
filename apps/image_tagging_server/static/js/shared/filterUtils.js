/**
 * Bidirectional conversion between Review filter state and Dataset filter criteria.
 *
 * As of PR 1.4b, DatasetFilterCriteria.filterJson carries a `terms` list (the
 * same discriminated-union filter-term shape as ImageSearchParams.terms)
 * instead of individual flat fields (tagUuids, dateFrom, filenameSearch,
 * imageLabelConditions, ...). Only labelExcludeUuids/labelMatchMode/
 * labelIncludeChildren/verificationCheck remain flat — they're modifiers of
 * Dataset.label_uuids, a structurally separate column, not self-contained
 * filter criteria (see apps/helpers/db/types.py DatasetFilterCriteria
 * docstring). The review filter panel's own state shape (reviewState) is
 * unchanged here — this PR does not touch the panel UI, only what gets
 * serialised when saving/loading a dataset.
 */

import { UNVERIFIED_TAG_UUID } from '/app-static/js/shared/api/review.js';

/**
 * Convert a review filter state to { filterJson, labelUuids } for saving as a dataset.
 *
 * @param {object} reviewState  - object from filterPanel.getState()
 * @returns {{ filterJson: object, labelUuids: string[] }}
 */
export function reviewStateToDatasetForm(reviewState) {
  const tagUuids = (reviewState.tagUuids ?? []).filter(u => u !== UNVERIFIED_TAG_UUID);
  const tagExcludeUuids = reviewState.tagExcludeUuids ?? [];
  const tagMatchMode = reviewState.tagMatchMode ?? 'any';
  const dateFrom = reviewState.dateFrom ?? null;
  const dateTo = reviewState.dateTo ?? null;
  const dateNegate = reviewState.dateNegate ?? false;
  const filenameSearch = reviewState.filenameSearch ?? '';
  const filenameNegate = reviewState.filenameNegate ?? false;
  const imageLabelConditions = reviewState.imageLabelConditions ?? [];
  const imageTagConditions = reviewState.imageTagConditions ?? [];
  // 'any' (and a missing key) is the no-op default — see review/filters.js's
  // DEFAULT_STATE — so it must not produce a term.
  const bboxVerification = reviewState.bboxVerification ?? 'any';

  const terms = [];
  if (bboxVerification !== 'any') terms.push({ kind: 'bboxVerification', state: bboxVerification });
  if (tagUuids.length > 0) terms.push({ kind: 'bboxTag', tagUuids, matchMode: tagMatchMode });
  for (const uuid of tagExcludeUuids) terms.push({ kind: 'bboxTag', tagUuids: [uuid], negate: true });
  if (dateFrom || dateTo) {
    const term = { kind: 'date', field: 'created' };
    if (dateFrom) term.from = dateFrom;
    if (dateTo) term.to = dateTo;
    if (dateNegate) term.negate = true;
    terms.push(term);
  }
  if (filenameSearch) terms.push({ kind: 'filename', patterns: [filenameSearch], negate: filenameNegate });
  for (const c of imageLabelConditions) {
    // 'uncertain' (never assessed) is the term model's 'unreviewed' — see
    // shared/api/review.js's buildReviewFilterBody for the same mapping.
    const state = c.state === 'uncertain' ? 'unreviewed' : c.state;
    // rollUp/negate must be written explicitly. The server-side defaults
    // (roll_up: true, negate: false — ImageLabelTerm in apps/helpers/db/types.py)
    // are not the panel's values, so omitting them doesn't "keep the default",
    // it *inverts* a negated condition and re-widens an exact-label one. Fall
    // back to those same server defaults for conditions that predate the UI
    // controls, so older saved Views keep behaving as before.
    terms.push({
      kind: 'imageLabel',
      labelUuid: c.labelUuid,
      state,
      rollUp: c.rollUp ?? true,
      negate: c.negate ?? false,
    });
  }
  for (const c of imageTagConditions) {
    // The picker's 'unset' (no image_tags row at all) is the term model's
    // 'unreviewed' — same mapping as the imageLabel conditions above.
    terms.push({
      kind: 'imageTag',
      tagUuid: c.tagUuid,
      state: c.state === 'unset' ? 'unreviewed' : c.state,
    });
  }

  return {
    filterJson: {
      labelExcludeUuids: reviewState.labelExcludeUuids ?? [],
      labelMatchMode: reviewState.labelMatchMode ?? 'any',
      labelIncludeChildren: reviewState.labelIncludeChildren ?? false,
      verificationCheck: reviewState.verificationCheck ?? null,
      terms,
    },
    labelUuids: reviewState.labelUuids ?? [],
  };
}

/**
 * Convert a saved dataset back to a partial review filter state.
 * Used to pre-populate the review filter panel from a saved dataset.
 *
 * @param {object} filterJson  - dataset.filterJson
 * @param {string[]} labelUuids - dataset.labelUuids
 * @returns {Partial<object>}  - suitable for filterPanel.setState()
 */
export function datasetFormToReviewState(filterJson, labelUuids) {
  const f = filterJson ?? {};
  const terms = f.terms ?? [];

  const tagTerm = terms.find(t => t.kind === 'bboxTag' && !t.negate);
  const tagExcludeUuids = terms
    .filter(t => t.kind === 'bboxTag' && t.negate)
    .flatMap(t => t.tagUuids ?? []);
  const dateTerm = terms.find(t => t.kind === 'date' && t.field === 'created');
  const filenameTerm = terms.find(t => t.kind === 'filename');
  const bboxVerificationTerm = terms.find(t => t.kind === 'bboxVerification' && !t.negate);
  const imageLabelConditions = terms
    .filter(t => t.kind === 'imageLabel')
    .map(t => ({
      labelUuid: t.labelUuid,
      state: t.state === 'unreviewed' ? 'uncertain' : t.state,
      // Mirror of the write side above: imageTagConditions.setState() re-defaults
      // any key it isn't given, so dropping these here loses them on load even
      // when they were stored correctly.
      rollUp: t.rollUp ?? true,
      negate: t.negate ?? false,
    }));
  const imageTagConditions = terms
    .filter(t => t.kind === 'imageTag')
    .map(t => ({
      tagUuid: t.tagUuid,
      state: t.state === 'unreviewed' ? 'unset' : t.state,
    }));

  return {
    tagUuids: tagTerm?.tagUuids ?? [],
    tagExcludeUuids,
    tagMatchMode: tagTerm?.matchMode ?? 'any',
    labelUuids: labelUuids ?? [],
    labelExcludeUuids: f.labelExcludeUuids ?? [],
    labelMatchMode: f.labelMatchMode ?? 'any',
    labelIncludeChildren: f.labelIncludeChildren ?? false,
    imageLabelConditions,
    imageTagConditions,
    filenameSearch: filenameTerm?.patterns?.[0] ?? '',
    filenameNegate: filenameTerm?.negate ?? false,
    dateFrom: dateTerm?.from ?? null,
    dateTo: dateTerm?.to ?? null,
    dateNegate: dateTerm?.negate ?? false,
    verificationCheck: f.verificationCheck ?? null,
    // Always present: filterPanel.setState is partial, so omitting the key on
    // a filter that carries no term would leave the previous value in place
    // and silently apply a filter the saved View never had.
    bboxVerification: bboxVerificationTerm?.state ?? 'any',
  };
}
