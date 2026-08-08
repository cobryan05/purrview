"""
Tests for fetchModelTrainingImages (shared/api/models.js).

Regression cover for a silent wire-format mismatch: ModelTrainingImagesResponse
inherits StatusResponse's `to_camel` alias generator, so the server sends
`trainImageUuids`, but this wrapper read `train_image_uuids`. Nothing errored —
it just resolved to empty arrays, so the "Browse training data" modal reported
"No training image UUIDs recorded in this model's sidecar" for every model, and
the training page's previous-run comparison would have found no run to compare.

Driven through page.evaluate rather than the models pane: the wrapper's job is
the mapping, and going through three levels of UI to reach it would test the UI
instead.
"""

import json

from playwright.sync_api import Page

_FETCH = """async (modelName) => {
    const { fetchModelTrainingImages } = await import('/app-static/js/shared/api/models.js');
    return await fetchModelTrainingImages(modelName);
}"""


def _route_training_images(page: Page, body: dict):
    page.route(
        "**/api/models/*/training-images",
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps(body),
        ),
    )


class TestFetchModelTrainingImages:
    def test_reads_camel_case_wire_fields(self, page: Page, base_url: str):
        page.goto(base_url)
        _route_training_images(
            page,
            {
                "status": "success",
                "trainImageUuids": ["a", "b"],
                "valImageUuids": ["c"],
                "trainedAt": "2026-07-01T09:00:00+00:00",
                "modelConfigNodeUuid": "node-1",
            },
        )

        result = page.evaluate(_FETCH, "some-model")

        assert result["train"] == ["a", "b"]
        assert result["val"] == ["c"]
        assert result["trainedAt"] == "2026-07-01T09:00:00+00:00"
        assert result["modelConfigNodeUuid"] == "node-1"

    def test_snake_case_still_works(self, page: Page, base_url: str):
        """Kept as a fallback so a by-name-serialised response isn't broken by the fix."""
        page.goto(base_url)
        _route_training_images(
            page,
            {
                "status": "success",
                "train_image_uuids": ["a"],
                "val_image_uuids": [],
                "trained_at": "2026-07-01T09:00:00+00:00",
                "model_config_node_uuid": "node-1",
            },
        )

        result = page.evaluate(_FETCH, "some-model")

        assert result["train"] == ["a"]
        assert result["trainedAt"] == "2026-07-01T09:00:00+00:00"
        assert result["modelConfigNodeUuid"] == "node-1"

    def test_missing_provenance_resolves_to_null(self, page: Page, base_url: str):
        """Models trained before those sidecar fields existed."""
        page.goto(base_url)
        _route_training_images(page, {"status": "success", "trainImageUuids": ["a"], "valImageUuids": []})

        result = page.evaluate(_FETCH, "some-model")

        assert result["train"] == ["a"]
        assert result["trainedAt"] is None
        assert result["modelConfigNodeUuid"] is None

    def test_failure_status_raises(self, page: Page, base_url: str):
        page.goto(base_url)
        _route_training_images(page, {"status": "failure", "message": "Model 'nope' not found"})

        result = page.evaluate(
            """async (modelName) => {
                const { fetchModelTrainingImages } = await import('/app-static/js/shared/api/models.js');
                try {
                    await fetchModelTrainingImages(modelName);
                    return { threw: false, message: null };
                } catch (err) {
                    return { threw: true, message: err.message };
                }
            }""",
            "nope",
        )

        assert result["threw"] is True
        assert "not found" in result["message"]
