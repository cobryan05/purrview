"""Estimated time remaining on a training run.

An epoch count alone doesn't answer the question people actually ask of a
running training ("is this worth waiting for, or should I come back tomorrow?"),
which needs epoch *duration* — a thing only the task process observes.

Run with:
    pytest tests/tasks_server/test_training_eta.py -v
"""

from apps.tasks_server.tasks._training_job_runner import TrainingJobRunnerMixin, format_eta


class _Runner(TrainingJobRunnerMixin):
    """The mixin's ETA half, exercised without any of the HTTP wiring."""

    def __init__(self):
        self._total_epochs = None
        self._epoch_times = []


def _timed(epochs_and_times):
    runner = _Runner()
    runner._epoch_times = list(epochs_and_times)
    return runner


class TestEstimateEtaSeconds:
    def test_no_estimate_before_two_epochs_have_been_timed(self):
        """One boundary measures nothing — an epoch duration needs two."""
        runner = _timed([(1, 100.0)])
        runner._total_epochs = 10
        assert runner.estimate_eta_seconds() is None

    def test_no_estimate_without_a_known_epoch_total(self):
        """Some trainers never report total_epochs; there is no 'remaining'."""
        runner = _timed([(1, 100.0), (2, 130.0)])
        assert runner.estimate_eta_seconds() is None

    def test_remaining_epochs_times_the_average_epoch(self):
        runner = _timed([(1, 100.0), (2, 130.0), (3, 160.0)])
        runner._total_epochs = 10
        # 30s/epoch over the measured window, 7 epochs left.
        assert runner.estimate_eta_seconds() == 210.0

    def test_a_gap_between_measured_epochs_is_averaged_not_assumed(self):
        """Trainers emit progress events per epoch, but a dropped stream can
        skip one — the divisor is the epoch span, not the sample count."""
        runner = _timed([(1, 100.0), (5, 500.0)])
        runner._total_epochs = 10
        assert runner.estimate_eta_seconds() == 500.0

    def test_the_last_epoch_ends_the_estimate(self):
        """A finished run must not persist a measured-looking '0s' remaining."""
        runner = _timed([(9, 100.0), (10, 130.0)])
        runner._total_epochs = 10
        assert runner.estimate_eta_seconds() is None


class TestRecordEpochTime:
    def test_boundaries_are_timestamped_in_order(self):
        runner = _Runner()
        runner._record_epoch_time(1)
        runner._record_epoch_time(2)
        assert [e for e, _ in runner._epoch_times] == [1, 2]
        assert runner._epoch_times[1][1] >= runner._epoch_times[0][1]

    def test_a_resumed_run_drops_replayed_epochs(self):
        """A restart replays epoch numbers already seen. Keeping the stale pair
        either side of the restart would time the restart, not an epoch."""
        runner = _Runner()
        for epoch in (1, 2, 3):
            runner._record_epoch_time(epoch)
        runner._record_epoch_time(2)
        assert [e for e, _ in runner._epoch_times] == [1, 2]

    def test_only_a_short_window_is_kept(self):
        """The estimate follows a trainer that changes speed, so old samples
        are dropped rather than averaged in forever."""
        runner = _Runner()
        for epoch in range(1, 51):
            runner._record_epoch_time(epoch)
        assert len(runner._epoch_times) <= 6


class TestFormatEta:
    def test_seconds_below_a_minute(self):
        assert format_eta(42) == "42s"

    def test_minutes_and_seconds_below_an_hour(self):
        assert format_eta(125) == "2m 05s"

    def test_hours_and_minutes_above_an_hour(self):
        assert format_eta(3600 * 2 + 180) == "2h 03m"

    def test_no_false_precision_on_long_runs(self):
        """Averaged over a handful of epochs, seconds are noise at this scale."""
        assert format_eta(3600 * 5 + 61) == "5h 01m"

    def test_a_negative_estimate_never_renders(self):
        assert format_eta(-5) == "0s"
