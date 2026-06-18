/**
 * Bidirectional conversion between Review filter state and Dataset filter criteria.
 *
 * The dataset filter is now a flat object that mirrors the review filter panel state
 * exactly, so round-trips are lossless.
 */

import { UNVERIFIED_TAG_UUID } from '/app-static/js/shared/api/review.js';

/**
 * Convert a review filter state to { filterJson, labelUuids } for saving as a dataset.
 *
 * @param {object} reviewState  - object from filterPanel.getState()
 * @returns {{ filterJson: object, labelUuids: string[] }}
 */
export function reviewStateToDatasetForm(reviewState) {
  return {
    filterJson: {
      tagUuids: (reviewState.tagUuids ?? []).filter(u => u !== UNVERIFIED_TAG_UUID),
      tagExcludeUuids: reviewState.tagExcludeUuids ?? [],
      tagMatchMode: reviewState.tagMatchMode ?? 'any',
      labelExcludeUuids: reviewState.labelExcludeUuids ?? [],
      labelMatchMode: reviewState.labelMatchMode ?? 'any',
      labelIncludeChildren: reviewState.labelIncludeChildren ?? false,
      dateFrom: reviewState.dateFrom ?? null,
      dateTo: reviewState.dateTo ?? null,
      filenameSearch: reviewState.filenameSearch ?? '',
      verificationCheck: reviewState.verificationCheck ?? null,
      imageLabelConditions: reviewState.imageLabelConditions ?? [],
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
  return {
    tagUuids: f.tagUuids ?? [],
    tagExcludeUuids: f.tagExcludeUuids ?? [],
    tagMatchMode: f.tagMatchMode ?? 'any',
    labelUuids: labelUuids ?? [],
    labelExcludeUuids: f.labelExcludeUuids ?? [],
    labelMatchMode: f.labelMatchMode ?? 'any',
    labelIncludeChildren: f.labelIncludeChildren ?? false,
    imageLabelConditions: f.imageLabelConditions ?? [],
    filenameSearch: f.filenameSearch ?? '',
    dateFrom: f.dateFrom ?? null,
    dateTo: f.dateTo ?? null,
    verificationCheck: f.verificationCheck ?? null,
  };
}
