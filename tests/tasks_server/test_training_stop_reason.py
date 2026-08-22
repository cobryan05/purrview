"""The last line a finished training run writes to its log.

A run that stopped at epoch 14 of 50 used to log exactly what a run that used its
whole budget logged, and the reason existed only in the training server's own
process log — which the task UI never shows. These pin the two facts that were
missing and the degradation path for a trainer that reports neither.

Run with:
    pytest tests/tasks_server/test_training_stop_reason.py -v
"""

from apps.tasks_server.tasks._training_job_runner import TrainingJobRunnerMixin


class _Runner(TrainingJobRunnerMixin):
    """The mixin's summary half, exercised without any of the HTTP wiring."""

    def __init__(self, model_name="cats-v3"):
        self._model_name = model_name


def _summary(**done_event):
    return _Runner()._completion_summary(done_event, done_event.get("val_accuracy"))


class TestCompletionSummary:
    def test_an_early_stop_is_distinguishable_from_a_full_run(self):
        """The whole point: these two used to produce identical log lines."""
        early = _summary(
            val_accuracy=0.81,
            stop_reason="early_stopping",
            stop_detail="early stopping at epoch 14 of 50: validation loss did not improve for 10 epoch(s)",
            epochs_run=14,
            max_epochs=50,
        )
        full = _summary(
            val_accuracy=0.83,
            stop_reason="completed",
            stop_detail="ran the full 50 epoch(s)",
            epochs_run=50,
            max_epochs=50,
        )
        assert early != full
        assert "STOPPED EARLY" in early
        assert "STOPPED EARLY" not in full

    def test_it_says_how_far_the_run_got(self):
        line = _summary(stop_reason="early_stopping", epochs_run=14, max_epochs=50)
        assert "14/50" in line

    def test_the_detail_sentence_survives_into_the_line(self):
        """The token alone never says WHAT stopped improving or for how long."""
        detail = "early stopping at epoch 14 of 50: validation loss did not improve for 10 epoch(s)"
        assert detail in _summary(stop_reason="early_stopping", stop_detail=detail, epochs_run=14, max_epochs=50)

    def test_a_cancelled_run_is_not_reported_as_a_completed_one(self):
        line = _summary(stop_reason="cancelled", epochs_run=3, max_epochs=50)
        assert "cancelled" in line
        assert "ran to completion" not in line

    def test_the_best_epoch_is_named_when_known(self):
        assert "best epoch: 4" in _summary(stop_reason="early_stopping", epochs_run=14, max_epochs=50, best_epoch=4)

    def test_a_trainer_reporting_nothing_still_produces_the_original_line(self):
        """An older training server that has not been redeployed sends no stop
        fields at all. That must degrade to the previous message, not to a line
        with empty clauses hanging off it."""
        line = _summary(val_accuracy=0.9)
        assert line == "Training complete — model: cats-v3, val_accuracy: 0.9"

    def test_an_unknown_reason_token_is_shown_rather_than_swallowed(self):
        """A future trainer inventing a token must not silently drop the reason."""
        assert "interrupted" in _summary(stop_reason="interrupted", epochs_run=2, max_epochs=50)
