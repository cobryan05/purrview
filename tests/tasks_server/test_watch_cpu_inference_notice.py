"""The watcher has to say when its inference fell back to the CPU.

The inference server already detects this (`onnxSession.create_session` retries on the
CPU when CUDA cannot initialise and returns a user-facing reason) and already reports it
in a detect-unit response's `notes`. But a watcher analyses through
`apps/helpers/frameInference.py`, which logs every note at DEBUG — so the one caller
running inference around the clock was the one with no way to find out, and the symptom
(frames analysed many times slower than usual) reads as a slow camera.

Mirrors `_cpu_training_notices` in the training manager, where the same failure already
surfaces as a standing app-wide notice for training jobs.

Note there is deliberately no "did you expect a GPU" setting here, unlike training.
`device_fallback` is set *only* when CUDA was asked for and refused; a CPU-only build
never asks, so the flag being set is already the whole signal.
"""

from types import SimpleNamespace

from apps.tasks_server.tasks.StreamWatcherTask import StreamWatcherTask

_REASON = "GPU memory was exhausted, so this model loaded on the CPU and will run much slower."


def _task() -> StreamWatcherTask:
    """A watcher with only the state the device-fallback path touches."""
    task = StreamWatcherTask.__new__(StreamWatcherTask)
    task._device_fallbacks = {}
    return task


def _meta(fallback=None) -> SimpleNamespace:
    return SimpleNamespace(device_fallback=fallback)


class TestRecordingTheFallback:
    def test_a_model_on_the_cpu_is_recorded(self):
        task = _task()
        task._record_device_fallbacks({"primary_model": _meta(_REASON)})
        assert task._device_fallbacks == {"primary_model": _REASON}

    def test_a_model_on_the_gpu_records_nothing(self):
        """The negative case is the one that keeps this notice meaningful — a warning
        that is always present is a warning nobody reads."""
        task = _task()
        task._record_device_fallbacks({"primary_model": _meta(None)})
        assert task._device_fallbacks == {}

    def test_a_model_that_recovers_stops_being_recorded(self):
        """A fallback is cleared by a reload onto the GPU, so the notice must retract."""
        task = _task()
        task._record_device_fallbacks({"primary_model": _meta(_REASON)})
        task._record_device_fallbacks({"primary_model": _meta(None)})
        assert task._device_fallbacks == {}

    def test_an_unreadable_meta_does_not_retract_an_existing_warning(self):
        """A meta that could not be fetched is absent from the dict, which is not
        evidence the fallback was resolved — and the inference server likeliest to fail
        that read is the unhealthy one whose warning matters most."""
        task = _task()
        task._record_device_fallbacks({"primary_model": _meta(_REASON)})
        task._record_device_fallbacks({})
        assert task._device_fallbacks == {"primary_model": _REASON}

    def test_only_the_models_reported_on_are_touched(self):
        task = _task()
        task._record_device_fallbacks({"a": _meta(_REASON), "b": _meta(_REASON)})
        task._record_device_fallbacks({"a": _meta(None)})
        assert task._device_fallbacks == {"b": _REASON}


class TestTheNotice:
    def test_no_fallback_means_no_notice(self):
        assert _task()._cpu_inference_notices() == []

    def test_a_degraded_model_produces_one_warning_naming_it(self):
        task = _task()
        task._device_fallbacks = {"primary_model": _REASON}

        notices = task._cpu_inference_notices()

        assert len(notices) == 1
        assert notices[0]["level"] == "warning"
        assert "primary_model" in notices[0]["message"]
        assert _REASON in notices[0]["message"]

    def test_two_degraded_models_do_not_collapse_into_one_line(self):
        """Keyed per model, as the training notice is — otherwise the second is lost."""
        task = _task()
        task._device_fallbacks = {"a": _REASON, "b": _REASON}

        keys = {n["key"] for n in task._cpu_inference_notices()}

        assert len(keys) == 2

    def test_the_level_is_a_warning_not_info(self):
        """Degraded inference is a fault to act on, and info is the level that gets
        scrolled past."""
        task = _task()
        task._device_fallbacks = {"primary_model": _REASON}
        assert all(n["level"] == "warning" for n in task._cpu_inference_notices())
