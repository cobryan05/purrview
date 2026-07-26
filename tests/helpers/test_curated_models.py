"""Unit tests for apps.helpers.curatedModels — the curated pretrained-model catalogue.

Cheap structural checks only (every entry has the right required keys given its
kind); the actual reachability/layout verification for each URL is done manually
(see curatedModels.py's module docstring and MORE_YOLO_PLAN.md Section 6) — this
suite guards against catalogue entries silently regressing into an inconsistent
shape (e.g. a "requires_conversion" entry missing its pt_url, or a direct-ONNX
entry missing model_family), not against upstream URLs going stale.
"""

import pytest

from apps.helpers.curatedModels import CURATED_BASE_MODELS, get_curated_model


class TestCatalogueShape:
    def test_every_entry_has_key_and_display_name(self):
        for entry in CURATED_BASE_MODELS:
            assert isinstance(entry.get("key"), str) and entry["key"]
            assert isinstance(entry.get("display_name"), str) and entry["display_name"]

    def test_keys_are_unique(self):
        keys = [e["key"] for e in CURATED_BASE_MODELS]
        assert len(keys) == len(set(keys))

    def test_direct_onnx_entries_have_onnx_url_and_model_family(self):
        direct_entries = [e for e in CURATED_BASE_MODELS if not e.get("requires_conversion")]
        assert direct_entries, "expected at least one direct-ONNX entry"
        for entry in direct_entries:
            assert isinstance(entry.get("onnx_url"), str) and entry["onnx_url"].startswith("https://")
            assert isinstance(entry.get("model_family"), str) and entry["model_family"]
            assert "pt_url" not in entry

    def test_conversion_entries_have_pt_url_and_no_model_family(self):
        conversion_entries = [e for e in CURATED_BASE_MODELS if e.get("requires_conversion")]
        assert conversion_entries, "expected at least one requires_conversion entry"
        for entry in conversion_entries:
            assert entry["requires_conversion"] is True
            assert isinstance(entry.get("pt_url"), str) and entry["pt_url"].startswith("https://")
            # model_family is decided by the training server from the actual
            # exported ONNX shape (model_converter.py), not guessed here.
            assert "model_family" not in entry
            assert "onnx_url" not in entry


class TestGetCuratedModel:
    def test_known_key_returns_entry(self):
        key = CURATED_BASE_MODELS[0]["key"]
        assert get_curated_model(key) == CURATED_BASE_MODELS[0]

    def test_unknown_key_raises_keyerror(self):
        with pytest.raises(KeyError):
            get_curated_model("definitely-not-a-real-key")
