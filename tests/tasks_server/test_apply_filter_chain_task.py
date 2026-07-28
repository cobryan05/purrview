"""Tests for ApplyFilterChainTask.

Mocks the FilterChain/Images/Tags REST clients and stubs _read_model_kind +
the inference backend's classify/detect calls at the seam — no real ONNX
session or live server needed (mirrors RunClassifierTask's test pattern).
"""

from contextlib import ExitStack
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from apps.helpers.metadataUtils import validate_bare_model_name
from apps.tasks_server.tasks._filter_chain_runner import Detection
from apps.tasks_server.tasks.ApplyFilterChainTask import ApplyFilterChainTask

_MOCK_MANAGER = MagicMock()
_MOCK_MANAGER.update_task_heartbeat = AsyncMock()


@pytest.fixture(autouse=True)
def _files_root_is_tmp_path(tmp_path, monkeypatch):
    # _load_image_bytes resolves filesystem-mode filenames against FILES_ROOT
    # (get_safe_path) rather than opening them as-is; tests write real image
    # bytes under tmp_path and pass that absolute path as `filename`, so
    # FILES_ROOT must be tmp_path for get_safe_path to recognize it as in-root.
    monkeypatch.setattr("apps.tasks_server.tasks.ApplyFilterChainTask.FILES_ROOT", tmp_path)


def _make_task(params: dict) -> ApplyFilterChainTask:
    task = ApplyFilterChainTask(task_config_uuid="cfg-uuid", params=params, manager=_MOCK_MANAGER)
    task.set_instance_uuid("11111111-1111-1111-1111-111111111111")
    return task


def _step(order=0, model_name="m1", action="tag_image", confidence=0.5, iou_threshold=0.5, action_params=None):
    return MagicMock(
        order=order,
        model_name=model_name,
        action=action,
        confidence=confidence,
        iou_threshold=iou_threshold,
        action_params=action_params or {},
        enabled=True,
    )


def _bbox(uuid="bbox-1", label_uuid="cat", x=0.1, y=0.1, width=0.2, height=0.2, tags=None, metadata_json=None):
    tag_objs = [MagicMock(uuid=t) for t in (tags or [])]
    return MagicMock(
        uuid=uuid,
        label_uuid=label_uuid,
        x=x,
        y=y,
        width=width,
        height=height,
        tags=tag_objs,
        metadata_json=metadata_json or {},
    )


def _stub(uuid="img-1", filename="/tmp/x.jpg"):
    # Mirrors batch_images_by_uuids' real payload: it serializes with
    # by_alias=True, so BatchImagesResponse.images is a List[dict] of camelCase
    # records — NOT attribute objects. Using a dict here keeps the image_uuids
    # scope tests faithful (an object mock would mask the `stub.filename`
    # AttributeError that silently skipped every image).
    return {"uuid": uuid, "filename": filename}


class TestScopeValidation:
    async def test_no_scope_raises(self):
        task = _make_task({"filter_chain_uuid": "chain-1"})
        with pytest.raises(ValueError, match="scope filter is required"):
            await task._init(task._params, None)

    async def test_missing_filter_chain_uuid_raises(self):
        task = _make_task({"directory": "some/dir"})
        with pytest.raises(ValueError, match="filter_chain_uuid"):
            await task._init(task._params, None)

    async def test_directory_scope_is_sufficient(self):
        task = _make_task({"filter_chain_uuid": "chain-1", "directory": "some/dir"})
        await task._init(task._params, None)
        assert task._directory == "some/dir"

    async def test_max_images_capped_at_ceiling(self):
        task = _make_task({"filter_chain_uuid": "chain-1", "image_uuids": ["a"], "max_images": 999999})
        await task._init(task._params, None)
        assert task._max_images == 5000


class TestValidateModelName:
    def test_bare_name_ok(self):
        validate_bare_model_name("my_model")  # does not raise

    @pytest.mark.parametrize("bad_name", ["../etc/passwd", "a/b", "a\\b", "..", ".", "", "/etc/passwd"])
    def test_rejects_path_traversal(self, bad_name):
        with pytest.raises(ValueError, match="Invalid model_name"):
            validate_bare_model_name(bad_name)


class TestRunEndToEnd:
    def _patch_common(self, chain_steps, mock_images_api, mock_tags_api=None, mock_labels_api=None):
        mock_chain = MagicMock(uuid="chain-1", name="TestChain", steps=chain_steps)
        mock_chain.name = "TestChain"  # MagicMock(name=...) sets the mock's repr name, not an attribute
        mock_filter_chains_api = MagicMock()
        mock_filter_chains_api.get_filter_chain.return_value = MagicMock(chain=mock_chain)

        mock_tags_api = mock_tags_api or MagicMock()
        mock_tags_api.list_tags.return_value = MagicMock(tags=[])

        # Only add_bbox steps fetch labels (name->uuid fallback); default to empty
        # unless the caller supplied its own labels mock.
        if mock_labels_api is None:
            mock_labels_api = MagicMock()
            mock_labels_api.list_labels.return_value = MagicMock(labels=[])

        mock_models_api = MagicMock()
        mock_models_api.pin_model_api_api_models_model_name_pin_post.return_value = MagicMock(
            status="success", pin_id="pin-1"
        )

        stack = ExitStack()
        stack.enter_context(
            patch("apps.tasks_server.tasks.ApplyFilterChainTask.FilterChainsApi", return_value=mock_filter_chains_api)
        )
        stack.enter_context(
            patch("apps.tasks_server.tasks.ApplyFilterChainTask.ImagesApi", return_value=mock_images_api)
        )
        stack.enter_context(patch("apps.tasks_server.tasks.ApplyFilterChainTask.TagsApi", return_value=mock_tags_api))
        stack.enter_context(
            patch("apps.tasks_server.tasks.ApplyFilterChainTask.LabelsApi", return_value=mock_labels_api)
        )
        stack.enter_context(
            patch("apps.tasks_server.tasks.ApplyFilterChainTask.InferenceModelsApi", return_value=mock_models_api)
        )
        return stack

    async def test_tag_image_action_applied_on_classifier_match(self, tmp_path):
        img_path = tmp_path / "a.jpg"
        img_path.write_bytes(b"fake")

        mock_images_api = MagicMock()
        mock_images_api.batch_images_by_uuids.return_value = MagicMock(images=[_stub(filename=str(img_path))])
        mock_images_api.get_image_metadata.return_value = MagicMock(status="success", image=MagicMock(bboxes=[]))
        mock_images_api.set_image_tag.return_value = MagicMock()

        step = _step(model_name="corrupt", action="tag_image", confidence=0.5, action_params={"tag_uuid": "tag-x"})
        patches = self._patch_common([step], mock_images_api)

        task = _make_task({"filter_chain_uuid": "chain-1", "image_uuids": [str(img_path)], "batch_size": 10})
        with (
            patches,
            patch("apps.tasks_server.tasks.ApplyFilterChainTask._read_model_kind", return_value="image_classifier"),
            patch(
                "apps.tasks_server.tasks.ApplyFilterChainTask._HttpInferenceBackend.classify_whole_image",
                new=AsyncMock(return_value=0.9),
            ),
        ):
            await task._init(task._params, None)
            result = await task._run()
            await task._deinit()

        assert result["images_processed"] == 1
        assert result["images_changed"] == 1
        assert result["images_tagged"] == 1
        mock_images_api.set_image_tag.assert_called_once()
        call_payload = mock_images_api.set_image_tag.call_args.args[0]
        assert call_payload.tag_uuid == "tag-x"
        assert call_payload.action == "add"

    async def test_discard_image_is_retroactive_noop(self, tmp_path):
        """discard_image never deletes an existing image — only counted."""
        img_path = tmp_path / "a.jpg"
        img_path.write_bytes(b"fake")

        mock_images_api = MagicMock()
        mock_images_api.batch_images_by_uuids.return_value = MagicMock(images=[_stub(filename=str(img_path))])
        mock_images_api.get_image_metadata.return_value = MagicMock(status="success", image=MagicMock(bboxes=[]))

        step = _step(model_name="corrupt", action="discard_image", confidence=0.5)
        patches = self._patch_common([step], mock_images_api)

        task = _make_task({"filter_chain_uuid": "chain-1", "image_uuids": [str(img_path)]})
        with (
            patches,
            patch("apps.tasks_server.tasks.ApplyFilterChainTask._read_model_kind", return_value="image_classifier"),
            patch(
                "apps.tasks_server.tasks.ApplyFilterChainTask._HttpInferenceBackend.classify_whole_image",
                new=AsyncMock(return_value=0.9),
            ),
        ):
            await task._init(task._params, None)
            result = await task._run()

        assert result["images_discard_matched"] == 1
        assert result["images_changed"] == 0

    async def test_strip_unverified_bboxes_deletes_only_unverified(self, tmp_path):
        from apps.helpers.db.types import UNVERIFIED_TAG_UUID

        img_path = tmp_path / "a.jpg"
        img_path.write_bytes(b"fake")

        unverified_bbox = _bbox(uuid="bbox-unverified", tags=[UNVERIFIED_TAG_UUID])
        verified_bbox = _bbox(uuid="bbox-verified", tags=["some-other-tag"])

        mock_images_api = MagicMock()
        mock_images_api.batch_images_by_uuids.return_value = MagicMock(images=[_stub(filename=str(img_path))])
        mock_images_api.get_image_metadata.return_value = MagicMock(
            status="success", image=MagicMock(bboxes=[unverified_bbox, verified_bbox])
        )
        mock_images_api.delete_bbox.return_value = MagicMock()

        step = _step(model_name="blurry", action="strip_unverified_bboxes", confidence=0.5)
        patches = self._patch_common([step], mock_images_api)

        task = _make_task({"filter_chain_uuid": "chain-1", "image_uuids": [str(img_path)]})
        with (
            patches,
            patch("apps.tasks_server.tasks.ApplyFilterChainTask._read_model_kind", return_value="image_classifier"),
            patch(
                "apps.tasks_server.tasks.ApplyFilterChainTask._HttpInferenceBackend.classify_whole_image",
                new=AsyncMock(return_value=0.9),
            ),
        ):
            await task._init(task._params, None)
            result = await task._run()

        assert result["bboxes_stripped"] == 1
        mock_images_api.delete_bbox.assert_called_once_with("bbox-unverified")

    async def test_batch_prefetch_avoids_per_image_metadata_fetch(self, tmp_path):
        from apps.helpers.db.types import UNVERIFIED_TAG_UUID

        # When batch_images_by_uuids already returns an image's bboxes, the per-image
        # get_image_metadata round-trip (the old N+1) must be skipped entirely.
        img_path = tmp_path / "a.jpg"
        img_path.write_bytes(b"fake")

        image_dict = {
            "uuid": "img-1",
            "filename": str(img_path),
            "bboxes": [
                {
                    "uuid": "bbox-unverified",
                    "x": 0.1,
                    "y": 0.1,
                    "width": 0.2,
                    "height": 0.2,
                    "labelUuid": "cat",
                    "tags": [{"uuid": UNVERIFIED_TAG_UUID, "name": "unverified"}],
                },
            ],
        }

        mock_images_api = MagicMock()
        mock_images_api.batch_images_by_uuids.return_value = MagicMock(images=[image_dict])
        mock_images_api.delete_bbox.return_value = MagicMock()

        step = _step(model_name="blurry", action="strip_unverified_bboxes", confidence=0.5)
        patches = self._patch_common([step], mock_images_api)

        task = _make_task({"filter_chain_uuid": "chain-1", "image_uuids": ["img-1"]})
        with (
            patches,
            patch("apps.tasks_server.tasks.ApplyFilterChainTask._read_model_kind", return_value="image_classifier"),
            patch(
                "apps.tasks_server.tasks.ApplyFilterChainTask._HttpInferenceBackend.classify_whole_image",
                new=AsyncMock(return_value=0.9),
            ),
        ):
            await task._init(task._params, None)
            result = await task._run()

        assert result["bboxes_stripped"] == 1
        mock_images_api.delete_bbox.assert_called_once_with("bbox-unverified")
        mock_images_api.get_image_metadata.assert_not_called()

    async def test_relabel_bbox_stashes_prior_label_and_stamps_review_tag(self, tmp_path):
        img_path = tmp_path / "a.jpg"
        img_path.write_bytes(b"fake")

        # A second, unrelated bbox on the same image with no label at all —
        # regression guard for the bug where a full-image update_image_metadata
        # replace would silently delete unlabeled bboxes as a side effect of
        # patching a sibling bbox (the server drops boxes lacking label_uuid).
        unrelated_unlabeled_bbox = _bbox(uuid="bbox-unlabeled", label_uuid=None, x=0.6, y=0.6, width=0.1, height=0.1)
        bbox = _bbox(uuid="bbox-1", label_uuid="cat", x=0.1, y=0.1, width=0.2, height=0.2, tags=["preexisting-tag"])

        mock_images_api = MagicMock()
        mock_images_api.batch_images_by_uuids.return_value = MagicMock(images=[_stub(filename=str(img_path))])
        mock_images_api.get_image_metadata.return_value = MagicMock(
            status="success", image=MagicMock(bboxes=[bbox, unrelated_unlabeled_bbox])
        )
        mock_images_api.update_bbox.return_value = MagicMock()

        step = _step(
            model_name="valves",
            action="relabel_bbox",
            confidence=0.5,
            iou_threshold=0.5,
            action_params={"target_label_uuid": "water_valve_label"},
        )
        patches = self._patch_common([step], mock_images_api)

        det = Detection(x=0.1, y=0.1, width=0.2, height=0.2, confidence=0.9)
        task = _make_task({"filter_chain_uuid": "chain-1", "image_uuids": [str(img_path)]})
        with (
            patches,
            patch("apps.tasks_server.tasks.ApplyFilterChainTask._read_model_kind", return_value="detector"),
            patch(
                "apps.tasks_server.tasks.ApplyFilterChainTask._HttpInferenceBackend.detect",
                new=AsyncMock(return_value=[det]),
            ),
        ):
            await task._init(task._params, None)
            result = await task._run()

        assert result["bboxes_relabeled"] == 1
        # Only the matched bbox is patched — never a full-image replace, and the
        # unrelated unlabeled sibling bbox is never touched at all.
        mock_images_api.update_bbox.assert_called_once()
        assert mock_images_api.update_bbox.call_args.args[0] == "bbox-1"
        payload = mock_images_api.update_bbox.call_args.args[1]
        assert payload.label_uuid == "water_valve_label"
        assert payload.metadata_json["relabel_prev_label_uuid"] == "cat"
        assert "preexisting-tag" in payload.tag_uuids
        assert not hasattr(mock_images_api, "update_image_metadata") or not mock_images_api.update_image_metadata.called

    async def test_relabel_then_strip_on_same_bbox_does_not_crash_or_double_apply(self, tmp_path):
        """A two-step chain where step 1 relabels a bbox and step 2's
        strip_unverified_bboxes image-level match also targets it: the bbox
        must end up deleted, and the queued relabel patch must NOT also fire
        against the now-deleted bbox."""
        img_path = tmp_path / "a.jpg"
        img_path.write_bytes(b"fake")

        bbox = _bbox(
            uuid="bbox-1",
            label_uuid="cat",
            x=0.1,
            y=0.1,
            width=0.2,
            height=0.2,
            tags=["00000000-0000-0000-0000-000000000001"],  # UNVERIFIED_TAG_UUID
        )

        mock_images_api = MagicMock()
        mock_images_api.batch_images_by_uuids.return_value = MagicMock(images=[_stub(filename=str(img_path))])
        mock_images_api.get_image_metadata.return_value = MagicMock(status="success", image=MagicMock(bboxes=[bbox]))
        mock_images_api.delete_bbox.return_value = MagicMock()
        mock_images_api.update_bbox.return_value = MagicMock()

        relabel_step = _step(
            order=0,
            model_name="valves",
            action="relabel_bbox",
            confidence=0.5,
            iou_threshold=0.5,
            action_params={"target_label_uuid": "water_valve_label"},
        )
        strip_step = _step(order=1, model_name="blurry", action="strip_unverified_bboxes", confidence=0.5)
        patches = self._patch_common([relabel_step, strip_step], mock_images_api)

        det = Detection(x=0.1, y=0.1, width=0.2, height=0.2, confidence=0.9)

        def _kind_for(model_name):
            return "image_classifier" if model_name == "blurry" else "detector"

        task = _make_task({"filter_chain_uuid": "chain-1", "image_uuids": [str(img_path)]})
        with (
            patches,
            patch("apps.tasks_server.tasks.ApplyFilterChainTask._read_model_kind", side_effect=_kind_for),
            patch(
                "apps.tasks_server.tasks.ApplyFilterChainTask._HttpInferenceBackend.detect",
                new=AsyncMock(return_value=[det]),
            ),
            patch(
                "apps.tasks_server.tasks.ApplyFilterChainTask._HttpInferenceBackend.classify_whole_image",
                new=AsyncMock(return_value=0.9),
            ),
        ):
            await task._init(task._params, None)
            result = await task._run()

        assert result["images_processed"] == 1
        mock_images_api.delete_bbox.assert_called_once_with("bbox-1")
        mock_images_api.update_bbox.assert_not_called()

    async def _run_add_bbox(self, tmp_path, det, dry_run=False, labels=None):
        """Shared driver: one add_bbox detector step over an image with no
        existing bboxes, returning (result, mock_images_api)."""
        img_path = tmp_path / "a.jpg"
        img_path.write_bytes(b"fake")

        mock_images_api = MagicMock()
        mock_images_api.batch_images_by_uuids.return_value = MagicMock(images=[_stub(filename=str(img_path))])
        mock_images_api.get_image_metadata.return_value = MagicMock(status="success", image=MagicMock(bboxes=[]))
        mock_images_api.add_bbox.return_value = MagicMock(bbox_uuid="new-1")

        mock_labels_api = MagicMock()
        mock_labels_api.list_labels.return_value = MagicMock(labels=labels or [])

        step = _step(model_name="det", action="add_bbox", confidence=0.5, iou_threshold=0.5)
        patches = self._patch_common([step], mock_images_api, mock_labels_api=mock_labels_api)

        params = {"filter_chain_uuid": "chain-1", "image_uuids": [str(img_path)]}
        if dry_run:
            params["dry_run"] = True
        task = _make_task(params)
        with (
            patches,
            patch("apps.tasks_server.tasks.ApplyFilterChainTask._read_model_kind", return_value="detector"),
            patch(
                "apps.tasks_server.tasks.ApplyFilterChainTask._HttpInferenceBackend.detect",
                new=AsyncMock(return_value=[det]),
            ),
        ):
            await task._init(task._params, None)
            result = await task._run()
        return result, mock_images_api

    async def test_add_bbox_creates_box_with_unverified_and_add_tag(self, tmp_path):
        from apps.helpers.db.types import UNVERIFIED_TAG_UUID

        det = Detection(x=0.4, y=0.4, width=0.1, height=0.1, confidence=0.85, label_uuid="dog_label")
        result, mock_images_api = await self._run_add_bbox(tmp_path, det)

        assert result["bboxes_added"] == 1
        assert result["images_changed"] == 1
        mock_images_api.add_bbox.assert_called_once()
        payload = mock_images_api.add_bbox.call_args.args[0]
        assert payload.label_uuid == "dog_label"
        assert payload.x == 0.4 and payload.width == 0.1
        assert UNVERIFIED_TAG_UUID in payload.tag_uuids
        # UNVERIFIED + this run's per-instance add tag.
        assert len(payload.tag_uuids) == 2
        assert payload.metadata_json["filter_chain_added"] is True
        assert payload.metadata_json["source_model"] == "det"

    async def test_add_bbox_dry_run_counts_without_writing(self, tmp_path):
        det = Detection(x=0.4, y=0.4, width=0.1, height=0.1, confidence=0.85, label_uuid="dog_label")
        result, mock_images_api = await self._run_add_bbox(tmp_path, det, dry_run=True)

        assert result["bboxes_added"] == 1
        mock_images_api.add_bbox.assert_not_called()

    async def test_add_bbox_unmapped_class_resolves_by_name(self, tmp_path):
        dog_label = MagicMock(uuid="dog-uuid")
        dog_label.name = "dog"  # MagicMock(name=...) sets repr, not the attribute
        det = Detection(x=0.4, y=0.4, width=0.1, height=0.1, confidence=0.85, label_uuid=None, class_str="dog")
        result, mock_images_api = await self._run_add_bbox(tmp_path, det, labels=[dog_label])

        assert result["bboxes_added"] == 1
        assert mock_images_api.add_bbox.call_args.args[0].label_uuid == "dog-uuid"

    async def test_add_bbox_unresolved_class_is_skipped(self, tmp_path):
        det = Detection(x=0.4, y=0.4, width=0.1, height=0.1, confidence=0.85, label_uuid=None, class_str="ghost")
        result, mock_images_api = await self._run_add_bbox(tmp_path, det, labels=[])

        assert result["bboxes_added"] == 0
        assert result["images_changed"] == 0
        mock_images_api.add_bbox.assert_not_called()

    async def test_add_bbox_force_label_overrides_detection_label(self, tmp_path):
        img_path = tmp_path / "a.jpg"
        img_path.write_bytes(b"fake")

        mock_images_api = MagicMock()
        mock_images_api.batch_images_by_uuids.return_value = MagicMock(images=[_stub(filename=str(img_path))])
        mock_images_api.get_image_metadata.return_value = MagicMock(status="success", image=MagicMock(bboxes=[]))
        mock_images_api.add_bbox.return_value = MagicMock(bbox_uuid="new-1")

        step = _step(
            model_name="det",
            action="add_bbox",
            action_params={"target_label_uuid": "forced-label"},
        )
        patches = self._patch_common([step], mock_images_api)
        det = Detection(x=0.4, y=0.4, width=0.1, height=0.1, confidence=0.85, label_uuid="dog_label")
        task = _make_task({"filter_chain_uuid": "chain-1", "image_uuids": [str(img_path)]})
        with (
            patches,
            patch("apps.tasks_server.tasks.ApplyFilterChainTask._read_model_kind", return_value="detector"),
            patch(
                "apps.tasks_server.tasks.ApplyFilterChainTask._HttpInferenceBackend.detect",
                new=AsyncMock(return_value=[det]),
            ),
        ):
            await task._init(task._params, None)
            await task._run()

        assert mock_images_api.add_bbox.call_args.args[0].label_uuid == "forced-label"

    async def test_dry_run_makes_no_mutations_but_counts_matches(self, tmp_path):
        img_path = tmp_path / "a.jpg"
        img_path.write_bytes(b"fake")

        mock_images_api = MagicMock()
        mock_images_api.batch_images_by_uuids.return_value = MagicMock(images=[_stub(filename=str(img_path))])
        mock_images_api.get_image_metadata.return_value = MagicMock(status="success", image=MagicMock(bboxes=[]))

        step = _step(model_name="corrupt", action="tag_image", confidence=0.5, action_params={"tag_uuid": "tag-x"})
        patches = self._patch_common([step], mock_images_api)

        task = _make_task({"filter_chain_uuid": "chain-1", "image_uuids": [str(img_path)], "dry_run": True})
        with (
            patches,
            patch("apps.tasks_server.tasks.ApplyFilterChainTask._read_model_kind", return_value="image_classifier"),
            patch(
                "apps.tasks_server.tasks.ApplyFilterChainTask._HttpInferenceBackend.classify_whole_image",
                new=AsyncMock(return_value=0.9),
            ),
        ):
            await task._init(task._params, None)
            result = await task._run()

        assert result["dry_run"] is True
        assert result["images_changed"] == 1
        mock_images_api.set_image_tag.assert_not_called()

    async def test_max_images_caps_explicit_image_uuids_list(self, tmp_path):
        mock_images_api = MagicMock()
        mock_images_api.batch_images_by_uuids.return_value = MagicMock(images=[])

        step = _step(model_name="corrupt", action="tag_image", confidence=0.5, action_params={"tag_uuid": "tag-x"})
        patches = self._patch_common([step], mock_images_api)

        many_uuids = [f"uuid-{i}" for i in range(10)]
        task = _make_task({"filter_chain_uuid": "chain-1", "image_uuids": many_uuids, "max_images": 3})
        with (
            patches,
            patch("apps.tasks_server.tasks.ApplyFilterChainTask._read_model_kind", return_value="image_classifier"),
        ):
            await task._init(task._params, None)
            await task._run()

        called_payload = mock_images_api.batch_images_by_uuids.call_args.args[0]
        assert len(called_payload.image_uuids) == 3
