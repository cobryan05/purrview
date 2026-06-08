/**
 * Bidirectional conversion between Review filter state and Dataset filter conditions.
 *
 * Limitations (documented):
 * - tagMatchMode ('any'|'all') is lost on round-trip: review→dataset always emits
 *   individual TagConditions (equivalent to 'any'); dataset→review always restores 'any'.
 * - labelMatchMode and labelIncludeChildren have no dataset equivalent.
 * - imageLabelFilters map approximately to DatasetLabelStatusCondition.
 */

/**
 * Convert a review filter state to the dataset filter shape.
 *
 * @param {object} reviewState  - object from filterPanel.getState()
 * @returns {{ conditions: object[], labelUuids: string[] }}
 */
export function reviewStateToDatasetForm(reviewState) {
  const conditions = [];

  // Tags → TagCondition (one per tag)
  for (const tagUuid of reviewState.tagUuids ?? []) {
    // Skip the built-in UNVERIFIED_TAG_UUID — it's a UI default, not a meaningful dataset filter
    conditions.push({ type: 'tag', tagUuid, exclude: false });
  }
  for (const tagUuid of reviewState.tagExcludeUuids ?? []) {
    conditions.push({ type: 'tag', tagUuid, exclude: true });
  }

  // Date range → BboxAgeCondition
  if (reviewState.dateFrom || reviewState.dateTo) {
    const cond = { type: 'bbox_age' };
    if (reviewState.dateFrom) cond.after = reviewState.dateFrom;
    if (reviewState.dateTo) cond.before = reviewState.dateTo;
    conditions.push(cond);
  }

  // Filename search → FilenameCondition
  if (reviewState.filenameSearch) {
    conditions.push({ type: 'filename', pattern: reviewState.filenameSearch });
  }

  // imageLabelFilters → DatasetLabelStatusCondition (best-effort)
  // If any filter excludes "uncertain", add exclude_uncertain condition.
  const excludesUncertain = (reviewState.imageLabelFilters ?? []).some(
    f => f.status === 'uncertain' && f.exclude,
  );
  if (excludesUncertain) {
    conditions.push({ type: 'dataset_label_status', check: 'exclude_uncertain' });
  }

  return {
    conditions,
    labelUuids: reviewState.labelUuids ?? [],
  };
}

/**
 * Convert dataset filter conditions back to a partial review filter state.
 * Used to pre-populate the review filter panel from a saved dataset.
 *
 * @param {{ conditions: object[] }} filterJson
 * @param {string[]} labelUuids
 * @returns {Partial<object>}  - suitable for filterPanel.setState()
 */
export function datasetFormToReviewState(filterJson, labelUuids) {
  const conditions = filterJson?.conditions ?? [];
  const state = {
    tagUuids: [],
    tagExcludeUuids: [],
    tagMatchMode: 'any',
    labelUuids: labelUuids ?? [],
    labelExcludeUuids: [],
    labelMatchMode: 'any',
    labelIncludeChildren: false,
    filenameSearch: '',
    dateFrom: null,
    dateTo: null,
    imageLabelFilters: [],
  };

  for (const cond of conditions) {
    switch (cond.type) {
      case 'tag':
        if (!cond.exclude) state.tagUuids.push(cond.tagUuid);
        else state.tagExcludeUuids.push(cond.tagUuid);
        break;

      case 'bbox_age':
      case 'image_age':
        if (cond.after) state.dateFrom = cond.after;
        if (cond.before) state.dateTo = cond.before;
        break;

      case 'filename':
        state.filenameSearch = cond.pattern ?? '';
        break;

      case 'dataset_label_status':
        // No clean mapping to review imageLabelFilters — skip silently.
        // The condition remains in the dataset but won't appear in the review filter UI.
        break;
    }
  }

  return state;
}
