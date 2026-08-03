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
  const filenameSearch = reviewState.filenameSearch ?? '';
  const imageLabelConditions = reviewState.imageLabelConditions ?? [];

  const terms = [];
  if (tagUuids.length > 0) terms.push({ kind: 'bboxTag', tagUuids, matchMode: tagMatchMode });
  for (const uuid of tagExcludeUuids) terms.push({ kind: 'bboxTag', tagUuids: [uuid], negate: true });
  if (dateFrom || dateTo) {
    const term = { kind: 'date', field: 'created' };
    if (dateFrom) term.from = dateFrom;
    if (dateTo) term.to = dateTo;
    terms.push(term);
  }
  if (filenameSearch) terms.push({ kind: 'filename', patterns: [filenameSearch] });
  for (const c of imageLabelConditions) {
    // 'uncertain' (never assessed) is the term model's 'unreviewed' — see
    // shared/api/review.js's buildReviewFilterBody for the same mapping.
    const state = c.state === 'uncertain' ? 'unreviewed' : c.state;
    terms.push({ kind: 'imageLabel', labelUuid: c.labelUuid, state });
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
  const imageLabelConditions = terms
    .filter(t => t.kind === 'imageLabel')
    .map(t => ({ labelUuid: t.labelUuid, state: t.state === 'unreviewed' ? 'uncertain' : t.state }));

  return {
    tagUuids: tagTerm?.tagUuids ?? [],
    tagExcludeUuids,
    tagMatchMode: tagTerm?.matchMode ?? 'any',
    labelUuids: labelUuids ?? [],
    labelExcludeUuids: f.labelExcludeUuids ?? [],
    labelMatchMode: f.labelMatchMode ?? 'any',
    labelIncludeChildren: f.labelIncludeChildren ?? false,
    imageLabelConditions,
    filenameSearch: filenameTerm?.patterns?.[0] ?? '',
    dateFrom: dateTerm?.from ?? null,
    dateTo: dateTerm?.to ?? null,
    verificationCheck: f.verificationCheck ?? null,
  };
}
