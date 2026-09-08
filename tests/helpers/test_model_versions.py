"""Tests for `apps/helpers/modelVersions.py` — the rollback history for trained models.

The rules worth pinning are the ones that make a restore trustworthy: the archived
copy must not move when the live model is rewritten, a restore must archive what it
replaces, and a restored generation must not inherit files it never had.
"""

import json

import pytest

from apps.helpers.modelVersions import (
    delete_preview,
    delete_versions,
    install_with_history,
    is_version_preview,
    list_versions,
    materialize_version,
    prune_versions,
    restore_version,
    snapshot_current,
    sweep_stale_previews,
    versions_usage,
)


def _write_model(models_dir, name="m", *, onnx=b"weights", sidecar=None, data=None):
    (models_dir / f"{name}.onnx").write_bytes(onnx)
    (models_dir / f"{name}.json").write_text(json.dumps(sidecar or {"classes": ["cat"]}))
    if data is not None:
        (models_dir / f"{name}.onnx.data").write_bytes(data)


class TestSnapshot:
    def test_archived_copy_survives_an_in_place_rewrite_of_the_live_model(self, tmp_path):
        """The reason versions are copies and not hardlinks.

        A hardlinked archive shares an inode with the live file, so any writer that
        opens the model in place — the sidecar is rewritten when a class is mapped
        to a label — silently rewrites history too.
        """
        _write_model(tmp_path, onnx=b"v1")
        version_id = snapshot_current(tmp_path, "m")

        (tmp_path / "m.onnx").write_bytes(b"v2-in-place")
        (tmp_path / "m.json").write_text(json.dumps({"classes": ["dog"]}))

        archived = tmp_path / "_versions" / "m" / version_id
        assert (archived / "m.onnx").read_bytes() == b"v1"
        assert json.loads((archived / "m.json").read_text())["classes"] == ["cat"]

    def test_first_ever_install_has_nothing_to_archive(self, tmp_path):
        assert snapshot_current(tmp_path, "brand-new") is None
        assert list_versions(tmp_path, "brand-new") == []

    def test_external_tensor_file_is_part_of_the_version(self, tmp_path):
        _write_model(tmp_path, data=b"tensors")
        version_id = snapshot_current(tmp_path, "m")
        assert list_versions(tmp_path, "m")[0]["files"] == ["m.json", "m.onnx", "m.onnx.data"]
        assert (tmp_path / "_versions" / "m" / version_id / "m.onnx.data").read_bytes() == b"tensors"

    def test_two_snapshots_in_the_same_second_are_both_kept(self, tmp_path):
        """Version ids are timestamps; two runs a second apart must not collide."""
        _write_model(tmp_path, onnx=b"a")
        first = snapshot_current(tmp_path, "m")
        (tmp_path / "m.onnx").write_bytes(b"b")
        second = snapshot_current(tmp_path, "m")
        assert first != second
        assert len(list_versions(tmp_path, "m")) == 2


class TestPrune:
    def test_keeps_the_newest_and_drops_the_rest(self, tmp_path):
        _write_model(tmp_path)
        ids = []
        for i in range(4):
            (tmp_path / "m.onnx").write_bytes(f"v{i}".encode())
            ids.append(snapshot_current(tmp_path, "m"))

        removed = prune_versions(tmp_path, "m", keep=2)

        surviving = [v["version_id"] for v in list_versions(tmp_path, "m")]
        assert surviving == ids[-1:-3:-1]  # newest two, newest first
        assert set(removed) == set(ids[:2])

    def test_keep_zero_removes_every_version(self, tmp_path):
        _write_model(tmp_path)
        snapshot_current(tmp_path, "m")
        prune_versions(tmp_path, "m", keep=0)
        assert list_versions(tmp_path, "m") == []

    def test_install_with_history_archives_then_prunes_in_one_call(self, tmp_path):
        _write_model(tmp_path)
        for i in range(5):
            (tmp_path / "m.onnx").write_bytes(f"gen{i}".encode())
            install_with_history(tmp_path, ["m"], keep=2)
        assert len(list_versions(tmp_path, "m")) == 2


class TestRestore:
    def test_restore_brings_back_the_old_weights_and_archives_the_current_ones(self, tmp_path):
        _write_model(tmp_path, onnx=b"good")
        old = snapshot_current(tmp_path, "m")
        (tmp_path / "m.onnx").write_bytes(b"bad-retrain")

        result = restore_version(tmp_path, "m", old)

        assert (tmp_path / "m.onnx").read_bytes() == b"good"
        # The bad model is archived, not destroyed — a restore is undoable.
        assert result["archived"] is not None
        undo = tmp_path / "_versions" / "m" / result["archived"] / "m.onnx"
        assert undo.read_bytes() == b"bad-retrain"

    def test_restoring_a_version_without_external_tensors_removes_the_live_ones(self, tmp_path):
        """A restored graph must never be left beside another generation's tensors.

        onnxruntime would load the pair and fail somewhere unrelated, or worse,
        succeed against the wrong weights.
        """
        _write_model(tmp_path, onnx=b"small")
        old = snapshot_current(tmp_path, "m")

        _write_model(tmp_path, onnx=b"big", data=b"tensors-for-big")
        restore_version(tmp_path, "m", old)

        assert (tmp_path / "m.onnx").read_bytes() == b"small"
        assert not (tmp_path / "m.onnx.data").exists()

    def test_unknown_version_raises(self, tmp_path):
        _write_model(tmp_path)
        with pytest.raises(FileNotFoundError):
            restore_version(tmp_path, "m", "20200101T000000Z")


class TestDeleteAndUsage:
    def test_deleting_a_model_history_leaves_the_live_model_alone(self, tmp_path):
        _write_model(tmp_path)
        snapshot_current(tmp_path, "m")
        assert delete_versions(tmp_path, "m") == 1
        assert list_versions(tmp_path, "m") == []
        assert (tmp_path / "m.onnx").exists()

    def test_usage_grows_with_each_kept_version_and_ignores_the_live_files(self, tmp_path):
        _write_model(tmp_path, onnx=b"x" * 1000)
        assert versions_usage(tmp_path) == {"bytes": 0, "versions": 0, "models": 0}

        snapshot_current(tmp_path, "m")
        one = versions_usage(tmp_path)
        assert one["versions"] == 1 and one["models"] == 1 and one["bytes"] >= 1000

        (tmp_path / "m.onnx").write_bytes(b"y" * 1000)
        snapshot_current(tmp_path, "m")
        two = versions_usage(tmp_path)
        assert two["versions"] == 2
        assert two["bytes"] > one["bytes"]


class TestVersionPreviews:
    """A version is compared by briefly becoming a model — see the module docstring."""

    def test_materialized_version_is_a_complete_model_beside_the_live_one(self, tmp_path):
        _write_model(tmp_path, onnx=b"old", sidecar={"classes": ["cat"]}, data=b"old-tensors")
        version_id = snapshot_current(tmp_path, "m")
        _write_model(tmp_path, onnx=b"new", sidecar={"classes": ["cat", "dog"]})

        name = materialize_version(tmp_path, "m", version_id)

        # Weights, sidecar and external tensors all come from the archived
        # generation: a preview scored against a live model must be that
        # generation in full, not its weights beside the current class list.
        assert (tmp_path / f"{name}.onnx").read_bytes() == b"old"
        assert json.loads((tmp_path / f"{name}.json").read_text())["classes"] == ["cat"]
        assert (tmp_path / f"{name}.onnx.data").read_bytes() == b"old-tensors"
        # And the live model is untouched.
        assert (tmp_path / "m.onnx").read_bytes() == b"new"

    def test_materializing_twice_reuses_one_copy(self, tmp_path):
        """Both sides of a comparison can pick the same version."""
        _write_model(tmp_path)
        version_id = snapshot_current(tmp_path, "m")
        first = materialize_version(tmp_path, "m", version_id)
        (tmp_path / f"{first}.onnx").write_bytes(b"in-use")

        second = materialize_version(tmp_path, "m", version_id)

        assert second == first
        assert (tmp_path / f"{first}.onnx").read_bytes() == b"in-use"

    def test_delete_preview_refuses_a_real_model_name(self, tmp_path):
        """The cleanup path runs with whatever name it was handed; a bug there
        must not be able to delete the model itself."""
        _write_model(tmp_path)
        assert delete_preview(tmp_path, "m") is False
        assert (tmp_path / "m.onnx").exists()

    def test_preview_names_are_recognisable(self, tmp_path):
        _write_model(tmp_path)
        name = materialize_version(tmp_path, "m", snapshot_current(tmp_path, "m"))
        assert is_version_preview(name)
        assert not is_version_preview("m")

    def test_sweep_removes_abandoned_previews_but_not_fresh_ones(self, tmp_path):
        """A preview outlives its comparison only when the browser went away."""
        import os
        import time

        _write_model(tmp_path)
        version_id = snapshot_current(tmp_path, "m")
        stale = materialize_version(tmp_path, "m", version_id)
        old = time.time() - 10 * 3600
        for suffix in (".onnx", ".json"):
            os.utime(tmp_path / f"{stale}{suffix}", (old, old))

        assert sweep_stale_previews(tmp_path, max_age_s=6 * 3600) == [stale]
        assert not (tmp_path / f"{stale}.onnx").exists()
        assert (tmp_path / "m.onnx").exists()

        fresh = materialize_version(tmp_path, "m", version_id)
        assert sweep_stale_previews(tmp_path, max_age_s=6 * 3600) == []
        assert (tmp_path / f"{fresh}.onnx").exists()

    def test_unknown_version_cannot_be_materialized(self, tmp_path):
        _write_model(tmp_path)
        with pytest.raises(FileNotFoundError):
            materialize_version(tmp_path, "m", "20200101T000000Z")
