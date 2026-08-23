"""A task type declares the colour of its card in the Jobs pane.

The Active list is a grid of identically shaped cards, so "is that the training
or the collection feeding it?" could only be answered by reading each card's
typename badge. `Task.CARD_ACCENT` lets a task type claim a tint, which travels
to the browser on the type list and lands as `data-task-accent` on the card.

The colour itself is *not* here — a hex chosen inside a task class cannot know
what surface it will sit on. The task names an accent; tokens.css owns the
value and app.css owns the rule. That split is silent when it breaks: a name
with no matching rule renders as a plain card and nothing errors, which is why
these tests pin the three sides to each other.
"""

import re
from pathlib import Path

import apps.tasks_server.tasks  # noqa: F401  — importing registers every task type
from apps.helpers.consts import TaskAccent
from apps.tasks_server.tasks.Registry import task_registry

_CSS_DIR = Path(__file__).resolve().parents[2] / "apps" / "image_tagging_server" / "static" / "css"


class TestTaskAccentDeclarations:
    def test_every_declared_accent_is_a_palette_member(self):
        """A raw string typo ("blu") is otherwise invisible: it serves happily
        over the API and renders as an uncoloured card."""
        for typename, cls in task_registry.items():
            accent = cls.CARD_ACCENT
            assert accent is None or isinstance(
                accent, TaskAccent
            ), f"{typename}.CARD_ACCENT must be a TaskAccent or None, got {accent!r}"

    def test_training_and_collection_are_visually_distinct(self):
        """The pair the pane exists to separate. Both may change colour; they
        must not converge on the same one."""
        from apps.tasks_server.tasks.CollectTrainingImagesTask import CollectTrainingImagesTask
        from apps.tasks_server.tasks.TrainModelTask import TrainModelTask

        assert TrainModelTask.CARD_ACCENT is not None
        assert CollectTrainingImagesTask.CARD_ACCENT is not None
        assert TrainModelTask.CARD_ACCENT != CollectTrainingImagesTask.CARD_ACCENT


class TestTaskAccentStylesheetMirror:
    """`TaskAccent` (Python) ↔ tokens.css ↔ app.css have no build step sharing
    them. Assert the sides equal each other; never pin a colour value."""

    def test_tokens_define_every_accent(self):
        tokens = (_CSS_DIR / "tokens.css").read_text()
        defined = set(re.findall(r"--task-accent-([a-z]+)\s*:", tokens))
        assert defined == {a.value for a in TaskAccent}

    def test_card_rules_cover_every_accent(self):
        app_css = (_CSS_DIR / "app.css").read_text()
        styled = set(re.findall(r'\.task-instance-card\[data-task-accent="([a-z]+)"\]', app_css))
        assert styled == {a.value for a in TaskAccent}


class TestTaskAccentsApi:
    async def test_type_list_carries_accents(self, http_client):
        resp = await http_client.get("/api/tasks/types/list")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"

        accents = data["accents"]
        # Only the types that claim one — a task with no accent must be absent
        # rather than mapped to a default, so the frontend can tell the two
        # apart and leave the attribute off entirely.
        for typename, accent in accents.items():
            assert task_registry[typename].CARD_ACCENT == accent
        for typename, cls in task_registry.items():
            if cls.CARD_ACCENT is None:
                assert typename not in accents
