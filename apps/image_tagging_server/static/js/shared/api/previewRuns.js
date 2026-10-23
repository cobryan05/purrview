/**
 * Persisted "Run Model on Results" previews.
 *
 * A preview run is scratch state: candidate relabels a detector proposed over a
 * review scope, held server-side so closing the triage modal (backdrop click,
 * ESC, an accidental click-away) no longer throws away hours of inference, and
 * so a several-thousand-candidate run can be triaged across several sittings.
 *
 * ⚠️ Deliberately **not** cached through createCachedFetcher. Every other
 * shared/api module fetches reference data that changes rarely and is read from
 * many screens; these endpoints are per-run working state that the triage panel
 * mutates on every decision. A cache here would only ever be stale.
 */
import { apiFetch } from './apiUtils.js';

async function _json(res, what) {
  const data = await res.json().catch(() => ({}));
  if (!res.ok || data.status !== 'success') {
    throw new Error(data.message || `Failed to ${what}`);
  }
  return data;
}

/** Decision states a candidate can be in — mirrors PreviewDecision in apps/helpers/db/types.py. */
export const PreviewDecision = {
  PENDING: 'pending',
  ACCEPTED: 'accepted',
  REJECTED: 'rejected',
  APPLIED: 'applied',
  STALE: 'stale',
};

/**
 * `params.report` marks a run that carries no candidates at all and whose whole
 * result is its `stats` — a Verify Model accuracy report. Mirrors
 * PREVIEW_REPORT_PARAM in apps/helpers/db/types.py, which the server's TTL sweep
 * reads to keep such a run from being swept as "fully triaged".
 */
export const REPORT_PARAM = 'report';

/** True for a saved accuracy report rather than a triagable candidate run. */
export function isReportRun(run) {
  return Boolean(run?.params?.[REPORT_PARAM]);
}

/**
 * The task behind a classifier run. Its candidates are whole-image proposals
 * (no bbox), triaged in the classifier grid rather than the bbox triage panel.
 */
export const CLASSIFIER_RUN_TASK = 'PreviewClassifierMatchTask';

export function isClassifierRun(run) {
  return run?.taskTypename === CLASSIFIER_RUN_TASK;
}

/** What accepting a candidate does — mirrors PreviewDecisionAction. */
export const PreviewDecisionAction = {
  RELABEL: 'relabel',
  DELETE: 'delete',
};

/**
 * Create the (empty) run before the task starts, so the task can write its
 * candidates straight into it instead of shipping thousands of them through the
 * browser and back.
 */
export async function createPreviewRun({ name = null, modelName = '', taskTypename = '', params = {}, scope = {} }) {
  const res = await apiFetch('/api/preview-runs', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name, modelName, taskTypename, params, scope }),
  });
  return (await _json(res, 'create preview run')).run;
}

/** Every saved run, newest first. Expired unpinned runs are swept server-side by this call. */
export async function fetchPreviewRuns() {
  const res = await apiFetch('/api/preview-runs');
  return (await _json(res, 'list preview runs')).runs ?? [];
}

export async function fetchPreviewRun(uuid) {
  const res = await apiFetch(`/api/preview-runs/${uuid}`);
  return (await _json(res, 'fetch preview run')).run;
}

export async function updatePreviewRun(uuid, { name, pinned, stats } = {}) {
  const body = {};
  if (name !== undefined) body.name = name;
  if (pinned !== undefined) body.pinned = pinned;
  if (stats !== undefined) body.stats = stats;
  const res = await apiFetch(`/api/preview-runs/${uuid}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  return (await _json(res, 'update preview run')).run;
}

export async function deletePreviewRun(uuid) {
  const res = await apiFetch(`/api/preview-runs/${uuid}`, { method: 'DELETE' });
  await _json(res, 'delete preview run');
}

/**
 * Filtered, sorted, paginated candidates plus the *filtered* total.
 *
 * POST rather than GET because the filter set is a body (label multi-select,
 * confidence band, decision set) — the same shape as /api/images/review.
 * Pagination is offset-based; see PreviewCandidateQuery for why this one does
 * not use the review queue's keyset cursor.
 *
 * @returns {Promise<{candidates: object[], totalCount: number}>}
 */
export async function queryPreviewCandidates(runUuid, {
  decisions = [],
  proposedLabelUuids = [],
  // The "≠" half of the same two filters. Include and exclude travel together rather
  // than as a mode, because "every cat except Willow" is one filter — see
  // `PreviewCandidateQuery` for the server's side of the same contract.
  excludeDecisions = [],
  excludeProposedLabelUuids = [],
  confidenceMin = null,
  confidenceMax = null,
  isNew = null,
  sourceModels = [],
  sortBy = 'image_confidence',
  sortOrder = 'desc',
  limit = 100,
  offset = 0,
} = {}) {
  const res = await apiFetch(`/api/preview-runs/${runUuid}/candidates/query`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      decisions, proposedLabelUuids, excludeDecisions, excludeProposedLabelUuids,
      confidenceMin, confidenceMax,
      isNew, sourceModels, sortBy, sortOrder, limit, offset,
    }),
  });
  const data = await _json(res, 'fetch preview candidates');
  return { candidates: data.candidates ?? [], totalCount: data.totalCount ?? 0 };
}

/**
 * Bulk-set fields on a list of candidates. Every field is optional — the triage
 * panel sends only what changed, so a decision edit is one small debounced PATCH
 * rather than a re-post of the whole run.
 */
export async function updatePreviewCandidates(runUuid, candidateUuids, { decision, decisionAction, proposedLabelUuid, geometry } = {}) {
  if (!candidateUuids?.length) return 0;
  const body = { candidateUuids };
  if (decision !== undefined) body.decision = decision;
  if (decisionAction !== undefined) body.decisionAction = decisionAction;
  if (proposedLabelUuid !== undefined) body.proposedLabelUuid = proposedLabelUuid;
  if (geometry !== undefined) body.geometry = geometry;
  const res = await apiFetch(`/api/preview-runs/${runUuid}/candidates`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  return (await _json(res, 'update preview candidates')).updated ?? 0;
}

/**
 * Apply every `accepted` candidate, server-side and in one call.
 *
 * The browser used to loop per proposed-label group, which meant closing the
 * laptop halfway through a 3,000-candidate run left it half-applied with no
 * record of where it stopped. Each candidate is now marked `applied` as it
 * lands, so a re-run resumes rather than re-applies.
 *
 * @returns {Promise<{modified: number, added: number, deleted: number, failed: number, stale: number}>}
 */
export async function applyPreviewRun(runUuid, { markVerified = false } = {}) {
  const res = await apiFetch(`/api/preview-runs/${runUuid}/apply`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ markVerified }),
  });
  return (await _json(res, 'apply preview run')).result;
}
