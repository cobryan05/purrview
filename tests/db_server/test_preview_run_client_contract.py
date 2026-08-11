"""The generated db_client's wire format must match the preview-run endpoints.

PreviewModelRelabelTask writes its candidates through the **generated**
`db_client` (`PreviewRunsApi.add_preview_candidates`), not through hand-written
JSON. That is a seam no other test covers: the endpoint tests build their bodies
by hand, and the task tests mock the API away, so a field rename on either side
would leave both suites green and only fail at runtime, inside a background task,
with a 422 nobody sees.

These tests serialize with the generated model and post that exact payload to the
in-process app.
"""

import json

from db_client.models.add_preview_candidates_payload import AddPreviewCandidatesPayload
from db_client.models.preview_candidate_create import PreviewCandidateCreate


async def _make_run(http_client) -> str:
    resp = await http_client.post(
        "/api/preview-runs",
        json={"modelName": "det-a", "taskTypename": "PreviewModelRelabelTask"},
    )
    assert resp.status_code == 200, resp.text
    return resp.json()["run"]["uuid"]


class TestGeneratedClientPayloadIsAccepted:
    async def test_add_candidates_payload_round_trips(self, http_client):
        run_uuid = await _make_run(http_client)

        payload = AddPreviewCandidatesPayload(
            candidates=[
                PreviewCandidateCreate(
                    image_uuid="img-1",
                    bbox_uuid="bbox-1",
                    is_new=False,
                    current_label_uuid="lbl-cat",
                    proposed_label_uuid="lbl-dog",
                    confidence=0.87,
                    geometry={"x": 0.1, "y": 0.2, "width": 0.3, "height": 0.4},
                    source_model="det-a",
                )
            ],
            conflict_policy="last",
        )

        resp = await http_client.post(
            f"/api/preview-runs/{run_uuid}/candidates",
            json=json.loads(payload.to_json()),
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()["status"] == "success"
        assert resp.json()["updated"] == 1

        # Every field survived the alias round-trip — a silently-dropped one
        # would leave a candidate that looks fine until apply does nothing.
        query = await http_client.post(f"/api/preview-runs/{run_uuid}/candidates/query", json={})
        assert query.status_code == 200, query.text
        (candidate,) = query.json()["candidates"]
        assert candidate["imageUuid"] == "img-1"
        assert candidate["bboxUuid"] == "bbox-1"
        assert candidate["isNew"] is False
        assert candidate["currentLabelUuid"] == "lbl-cat"
        assert candidate["proposedLabelUuid"] == "lbl-dog"
        assert candidate["confidence"] == 0.87
        assert candidate["geometry"] == {"x": 0.1, "y": 0.2, "width": 0.3, "height": 0.4}
        assert candidate["sourceModel"] == "det-a"
        # `stale`, not `pending`: this test posts uuids for rows that were never
        # created, and read-time staleness reconciliation correctly says so. The
        # decision lifecycle itself is covered in test_preview_runs.py against
        # real images and bboxes; what matters here is only the wire format.
        assert candidate["decision"] == "stale"

    async def test_conflict_policy_survives_the_alias(self, http_client):
        """`conflict_policy` -> `conflictPolicy` is the field most likely to be
        silently dropped: dropping it just falls back to the default, so the run
        would quietly ignore the user's choice rather than error."""
        run_uuid = await _make_run(http_client)

        def _candidate(label_uuid: str, model: str) -> PreviewCandidateCreate:
            return PreviewCandidateCreate(
                image_uuid="img-1",
                bbox_uuid="bbox-1",
                proposed_label_uuid=label_uuid,
                confidence=0.5,
                source_model=model,
            )

        first = AddPreviewCandidatesPayload(candidates=[_candidate("lbl-dog", "det-a")], conflict_policy="first")
        await http_client.post(f"/api/preview-runs/{run_uuid}/candidates", json=json.loads(first.to_json()))

        second = AddPreviewCandidatesPayload(candidates=[_candidate("lbl-cat", "det-b")], conflict_policy="first")
        resp = await http_client.post(f"/api/preview-runs/{run_uuid}/candidates", json=json.loads(second.to_json()))
        assert resp.status_code == 200, resp.text

        query = await http_client.post(f"/api/preview-runs/{run_uuid}/candidates/query", json={})
        (candidate,) = query.json()["candidates"]
        assert candidate["proposedLabelUuid"] == "lbl-dog", "first-wins was ignored — check the conflictPolicy alias"
        assert candidate["sourceModel"] == "det-a"
