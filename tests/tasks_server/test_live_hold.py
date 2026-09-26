"""Holding a Ring camera on its live tier while someone is watching it.

A live tier lapses `live_hold_s` (15 s) after the last `request_live`, and the live
view renews it on a timer for as long as the tab is visible. The cost of that is the
reason this is tested at all: a Ring camera reports no motion while it is streaming,
so a hold that outlives the watching is a camera that has gone quietly blind. What
matters here is that the release actually reaches the provider, and that a request and
a release arriving together cannot leave the camera held.
"""

from types import SimpleNamespace

import pytest

from apps.tasks_server.tasks.StreamWatcherTask import StreamWatcherTask


class _Provider:
    """A provider with a live tier, recording what the watcher asked of it."""

    def __init__(self, fail_release: bool = False):
        self.requested = 0
        self.released = 0
        self._fail_release = fail_release

    async def request_live(self):
        self.requested += 1

    async def release_live(self):
        self.released += 1
        if self._fail_release:
            raise RuntimeError("mqtt down")


def _pipeline(source_uuid: str, provider) -> SimpleNamespace:
    return SimpleNamespace(source_uuid=source_uuid, source_name=f"cam-{source_uuid}", provider=provider)


def _task(*pipelines) -> StreamWatcherTask:
    """A task with only the fields the live-hold path reads — no DB, no inference."""
    task = StreamWatcherTask.__new__(StreamWatcherTask)
    task._pipelines = list(pipelines)
    task._live_requests = set()
    task._release_requests = set()
    task.log = lambda *a, **k: None
    return task


class TestTheUIMayHandTheCameraBack:
    async def test_a_release_reaches_the_provider(self):
        provider = _Provider()
        task = _task(_pipeline("src-1", provider))

        await task.handle_command("release_live", {"sourceUuid": "src-1"})
        await task._honour_live_requests()

        assert provider.released == 1
        assert provider.requested == 0

    async def test_a_release_after_a_request_leaves_the_camera_released(self):
        """The view renews on a timer and releases on close, so the two can land in
        one loop pass. Ending that pass still holding the camera is the failure this
        exists to catch: nothing errors, and the camera stops reporting motion."""
        provider = _Provider()
        task = _task(_pipeline("src-1", provider))

        await task.handle_command("request_live", {"sourceUuid": "src-1"})
        await task.handle_command("release_live", {"sourceUuid": "src-1"})
        await task._honour_live_requests()

        assert provider.released == 1
        assert provider.requested == 0

    async def test_a_request_after_a_release_leaves_the_camera_held(self):
        """The mirror image — coming back to the tab must win over the close that
        preceded it, or the renewal is silently dropped."""
        provider = _Provider()
        task = _task(_pipeline("src-1", provider))

        await task.handle_command("release_live", {"sourceUuid": "src-1"})
        await task.handle_command("request_live", {"sourceUuid": "src-1"})
        await task._honour_live_requests()

        assert provider.requested == 1
        assert provider.released == 0

    async def test_each_source_is_held_and_released_independently(self):
        held, freed = _Provider(), _Provider()
        task = _task(_pipeline("src-1", held), _pipeline("src-2", freed))

        await task.handle_command("request_live", {"sourceUuid": "src-1"})
        await task.handle_command("release_live", {"sourceUuid": "src-2"})
        await task._honour_live_requests()

        assert (held.requested, held.released) == (1, 0)
        assert (freed.requested, freed.released) == (0, 1)

    async def test_requests_are_drained_rather_than_repeated_every_pass(self):
        provider = _Provider()
        task = _task(_pipeline("src-1", provider))

        await task.handle_command("release_live", {"sourceUuid": "src-1"})
        await task._honour_live_requests()
        await task._honour_live_requests()

        assert provider.released == 1


class TestSourcesWithoutALiveTier:
    async def test_a_provider_that_cannot_be_released_is_passed_over(self):
        """An RTSP camera is already live and has neither method. The command is
        accepted — the UI hides the button, it does not validate it — and must not
        raise into the watcher loop."""
        task = _task(_pipeline("src-1", SimpleNamespace()))

        await task.handle_command("release_live", {"sourceUuid": "src-1"})
        await task._honour_live_requests()  # must not raise

    async def test_a_failing_release_does_not_stop_the_other_sources(self):
        """The hold expires on its own shortly after, so one camera's failed release
        costs seconds — but it must not take the rest of the pass down with it."""
        broken = _Provider(fail_release=True)
        ok = _Provider()
        task = _task(_pipeline("src-1", broken), _pipeline("src-2", ok))

        await task.handle_command("release_live", {"sourceUuid": "src-1"})
        await task.handle_command("release_live", {"sourceUuid": "src-2"})
        await task._honour_live_requests()

        assert broken.released == 1
        assert ok.released == 1


class TestTheCommandIsValidated:
    async def test_a_release_without_a_source_is_rejected(self):
        task = _task(_pipeline("src-1", _Provider()))

        with pytest.raises(ValueError, match="release_live requires a sourceUuid"):
            await task.handle_command("release_live", {})

    async def test_a_release_for_a_source_this_watcher_does_not_have_is_rejected(self):
        """Silently accepting it would queue a release nothing ever drains."""
        task = _task(_pipeline("src-1", _Provider()))

        with pytest.raises(ValueError, match="No source"):
            await task.handle_command("release_live", {"sourceUuid": "src-9"})

        assert task._release_requests == set()
