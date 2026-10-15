"""Unit tests for apps/helpers/gpuDevices.py.

The physical-index-vs-CUDA-ordinal translation is the whole point of this module: a
config storing a physical id silently runs on a different card (or none), and nothing
about the failure says so. `nvidia-smi` is stubbed so the mapping can be exercised
without a GPU.
"""

import subprocess

import pytest

from apps.helpers import gpuDevices

# Two identical cards — the case that forced the bus id into the label, since name and
# memory cannot tell them apart.
TWO_IDENTICAL = (
    "0, NVIDIA GeForce RTX 3060, 00000000:01:00.0, 12288, GPU-aaaaaaaa\n"
    "1, NVIDIA GeForce RTX 3060, 00000000:04:00.0, 12288, GPU-bbbbbbbb\n"
)


@pytest.fixture
def smi(monkeypatch):
    """Stub nvidia-smi's stdout; pass None to make the binary unavailable."""

    def install(stdout):
        def fake_run(*args, **kwargs):
            if stdout is None:
                raise FileNotFoundError("nvidia-smi")
            return subprocess.CompletedProcess(args=args, returncode=0, stdout=stdout, stderr="")

        monkeypatch.setattr(gpuDevices.subprocess, "run", fake_run)

    return install


class TestCudaOrdinalMapping:
    def test_an_unset_variable_leaves_ordinals_equal_to_physical_ids(self, smi, monkeypatch):
        smi(TWO_IDENTICAL)
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        assert [(d.index, d.cuda_ordinal) for d in gpuDevices.list_gpus()] == [(0, 0), (1, 1)]

    def test_exposing_only_the_second_card_renumbers_it_to_ordinal_zero(self, smi, monkeypatch):
        """The bug this module exists to prevent: physical 1 is CUDA 0 here, so storing
        the physical id would name a device that does not exist."""
        smi(TWO_IDENTICAL)
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
        by_index = {d.index: d.cuda_ordinal for d in gpuDevices.list_gpus()}
        assert by_index == {0: None, 1: 0}

    def test_reordering_the_variable_reorders_the_ordinals(self, smi, monkeypatch):
        smi(TWO_IDENTICAL)
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1,0")
        assert {d.index: d.cuda_ordinal for d in gpuDevices.list_gpus()} == {0: 1, 1: 0}

    def test_an_empty_variable_exposes_nothing_while_unset_exposes_everything(self, smi, monkeypatch):
        """Empty and unset are opposite states, and collapsing them would read as a
        container that lost its GPU."""
        smi(TWO_IDENTICAL)
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
        assert gpuDevices.visible_gpus() == []
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        assert len(gpuDevices.visible_gpus()) == 2

    def test_a_uuid_entry_is_matched_by_prefix(self, smi, monkeypatch):
        smi(TWO_IDENTICAL)
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-bbbb")
        assert {d.index: d.cuda_ordinal for d in gpuDevices.list_gpus()} == {0: None, 1: 0}

    def test_visible_gpus_excludes_unexposed_cards_and_orders_by_ordinal(self, smi, monkeypatch):
        smi(TWO_IDENTICAL)
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
        visible = gpuDevices.visible_gpus()
        assert [d.index for d in visible] == [1]


class TestLabels:
    def test_identical_cards_get_distinguishable_labels(self, smi, monkeypatch):
        smi(TWO_IDENTICAL)
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        first, second = (d.label for d in gpuDevices.list_gpus())
        # The assertion that matters is that they DIFFER — same model, same memory, so
        # a label built from name alone would be useless for choosing between them.
        assert first != second
        assert "01:00.0" in first and "04:00.0" in second


class TestDegradation:
    def test_a_host_without_nvidia_smi_reports_no_gpus_rather_than_raising(self, smi):
        """ "Which GPUs are there" must never fail a request: a CPU-only host is a
        supported configuration, and the callers hide their controls on an empty list."""
        smi(None)
        assert gpuDevices.list_gpus() == []

    def test_unparseable_rows_are_skipped_not_fatal(self, smi, monkeypatch):
        smi("garbage\n0, NVIDIA GeForce RTX 3060, 00000000:01:00.0, 12288, GPU-aaaa\n")
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        assert [d.index for d in gpuDevices.list_gpus()] == [0]

    def test_a_non_numeric_memory_column_still_yields_the_card(self, smi, monkeypatch):
        smi("0, NVIDIA GeForce RTX 3060, 00000000:01:00.0, [N/A], GPU-aaaa\n")
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        (device,) = gpuDevices.list_gpus()
        assert device.memory_mb is None
        assert device.index == 0
