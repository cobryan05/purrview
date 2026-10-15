"""Unit tests for apps/helpers/onnxSession.py.

No real ONNX session is built: onnxruntime.InferenceSession is replaced with a
recorder so each test can choose which attempt fails and assert what the second
attempt asked for.
"""

import onnxruntime as ort
import pytest

from apps.helpers.onnxSession import create_session, is_gpu_exhaustion

CUBLAS_OOM = (
    "[ONNXRuntimeError] : 1 : FAIL : Exception during initialization: "
    "cuda_call.cc:122 CudaCall CUBLAS failure 3: the resource allocation failed ; GPU=0"
)


class _Recorder:
    """Stands in for ort.InferenceSession, failing per-attempt on demand."""

    def __init__(self, fail_on_cuda: Exception | None = None, fail_on_cpu: Exception | None = None):
        self._fail_on_cuda = fail_on_cuda
        self._fail_on_cpu = fail_on_cpu
        self.attempts: list[list[str]] = []

    def __call__(self, model_path, providers=None):
        requested = list(providers or [])
        self.attempts.append(requested)
        if self._fail_on_cuda and "CUDAExecutionProvider" in requested:
            raise self._fail_on_cuda
        if self._fail_on_cpu and "CUDAExecutionProvider" not in requested:
            raise self._fail_on_cpu
        return self

    def get_providers(self):
        return self.attempts[-1]


@pytest.fixture
def session_factory(monkeypatch):
    def install(**kwargs):
        recorder = _Recorder(**kwargs)
        monkeypatch.setattr(ort, "InferenceSession", recorder)
        return recorder

    return install


class TestIsGpuExhaustion:
    def test_allocation_failure_is_exhaustion_but_a_missing_driver_is_not(self):
        # The pair matters more than either half: the whole point of the marker
        # list is to separate "no room" from "no CUDA", because only the first
        # may be answered by quietly running on the CPU.
        assert is_gpu_exhaustion(RuntimeError(CUBLAS_OOM))
        assert not is_gpu_exhaustion(RuntimeError("libcublas.so.12: cannot open shared object file"))


class TestCreateSession:
    def test_gpu_success_makes_no_second_attempt_and_reports_no_fallback(self, session_factory):
        recorder = session_factory()
        _, reason = create_session("m.onnx", ["CUDAExecutionProvider", "CPUExecutionProvider"])
        assert reason is None
        assert recorder.attempts == [["CUDAExecutionProvider", "CPUExecutionProvider"]]

    def test_exhausted_gpu_retries_on_cpu_alone_and_reports_why(self, session_factory):
        recorder = session_factory(fail_on_cuda=RuntimeError(CUBLAS_OOM))
        _, reason = create_session("m.onnx", ["CUDAExecutionProvider", "CPUExecutionProvider"])
        # CPU *alone* on the retry: leaving CUDA in the list reproduces the
        # failure, since onnxruntime raises on constructing the provider rather
        # than falling through to the next one.
        assert recorder.attempts == [
            ["CUDAExecutionProvider", "CPUExecutionProvider"],
            ["CPUExecutionProvider"],
        ]
        assert reason and "CPU" in reason

    def test_a_non_memory_cuda_failure_raises_instead_of_degrading(self, session_factory):
        recorder = session_factory(fail_on_cuda=RuntimeError("Failed to load library libonnxruntime_providers_cuda"))
        with pytest.raises(RuntimeError, match="libonnxruntime_providers_cuda"):
            create_session("m.onnx", ["CUDAExecutionProvider", "CPUExecutionProvider"])
        assert len(recorder.attempts) == 1

    def test_a_cpu_only_request_is_never_retried(self, session_factory):
        recorder = session_factory(fail_on_cpu=RuntimeError("corrupt graph"))
        with pytest.raises(RuntimeError, match="corrupt graph"):
            create_session("m.onnx", ["CPUExecutionProvider"])
        assert len(recorder.attempts) == 1

    def test_a_broken_model_surfaces_the_gpu_error_not_the_cpu_one(self, session_factory):
        # Both attempts fail. The caller is told the GPU error, because that is
        # the one that describes why the fast path was abandoned; the CPU error
        # only repeats that the model is unloadable.
        session_factory(
            fail_on_cuda=RuntimeError(CUBLAS_OOM),
            fail_on_cpu=RuntimeError("corrupt graph"),
        )
        with pytest.raises(RuntimeError, match="CUBLAS failure 3"):
            create_session("m.onnx", ["CUDAExecutionProvider", "CPUExecutionProvider"])
