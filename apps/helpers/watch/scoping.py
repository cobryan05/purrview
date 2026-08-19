"""Which actions fire for which rules and which sources.

Pure, like the rest of `apps.helpers.watch` — it takes uuids and returns a boolean, so
it is testable without a broker, a camera or a database.

The problem it solves: a watcher shares one set of rules across every camera it
watches, which is the whole point of a watcher — "cats using the litter box" should be
written once, not once per camera. But the *side effects* are not equally shareable.
Recording a clip every time a cat appears is right for the one camera pointed at the
litter box and wrong for the other four. Before this, the only way to express that was
a second watcher with the rule copied into it, and a copied rule is one that drifts.

So scope lives on the **action**, and it is a pair of lists:

* ``rule_uuids`` — which rules this action responds to.
* ``source_uuids`` — which cameras it fires on.

**Empty means all**, on both axes. That is what keeps the common case free: an action
written before either axis existed, or written by someone who never opens the scope
dropdowns, applies everywhere — and an action stays applying to a camera added to the
watcher later, rather than silently not covering it.

A rule carries a source scope too, on the same empty-means-all terms, and the two are
deliberately different instruments. Narrowing the **action** leaves the rule evaluating
everywhere, so the event is still raised and still dispatched on every camera — only
the side effect is narrowed. Narrowing the **rule** stops it being evaluated there at
all.

Note what narrowing an action now *also* does. An event is written to the Events view
only where an in-scope action writes something to it (`Action.logs_event`), so scoping
a clip action to one camera scopes the event history with it — the other cameras keep
evaluating the rule and keep publishing, and file nothing. That is the intended
reading: the event list is a list of artifacts, not of matches. Adding a `save_event`
action is how a rule that produces no media gets filed anyway.
"""

from typing import Any, Collection, Optional, Sequence

#: Mirrors `WatchActionTrigger.FRAME_ANALYSED`. Spelled out here so the watcher task can
#: name the trigger without importing the ORM's types — it reads its profile through the
#: generated db client, where every field is already a plain string.
FRAME_ANALYSED = "frame_analysed"

#: The two event edges, spelled out here for the same reason — an action that declares
#: `triggers()` needs to name them without reaching into the ORM's enums.
EVENT_STARTED = "event_started"
EVENT_ENDED = "event_ended"

#: Triggers that are not tied to any rule. A frame is analysed whether or not a rule is
#: open, so an action on this trigger has no rule to be scoped against and its rule
#: scope is ignored rather than treated as unmatched — which would silently never fire.
RULELESS_TRIGGERS = frozenset({FRAME_ANALYSED})


def in_scope(scope: Optional[Sequence[Any]], value: Optional[str]) -> bool:
    """Whether `value` is covered by a scope list, where empty means everything."""
    if not scope:
        return True
    return value in {str(entry) for entry in scope}


def rule_applies_to_source(scope: Optional[Sequence[Any]], source_uuid: Optional[str]) -> bool:
    """Whether a rule with this source scope should be evaluated on this source.

    Differs from `in_scope` in exactly one case, which is the one that matters: an engine
    with **no source identity** (`source_uuid=None` — a preview, or a test) evaluates
    every rule rather than none. Falling through to `in_scope` there would silently make
    every scoped rule dead, and a rule that never fires is indistinguishable from a rule
    whose predicate is simply never true.
    """
    if source_uuid is None:
        return True
    return in_scope(scope, source_uuid)


def action_applies(
    row: Any,
    trigger: str,
    rule_uuid: Optional[str] = None,
    source_uuid: Optional[str] = None,
    fires_on: Optional[Collection[str]] = None,
) -> bool:
    """Whether one `WatchAction` row should fire for this transition.

    :param row: the action row, carrying `enabled`, `trigger`, `rule_uuids_json` and
        `source_uuids_json`
    :param trigger: the trigger being dispatched
    :param rule_uuid: the rule that fired, or None for a ruleless trigger
    :param source_uuid: the source it fired on
    :param fires_on: every trigger this action wants, from `Action.triggers(row)`.
        Defaults to the single trigger on the row — which is what every action that
        does one thing at one moment wants, and what this took for granted before
        `record_event_timelapse` needed to span an event's whole life.
    """
    if not getattr(row, "enabled", True):
        return False
    if trigger not in (fires_on if fires_on is not None else {getattr(row, "trigger", None)}):
        return False
    if not in_scope(getattr(row, "source_uuids_json", None), source_uuid):
        return False
    if trigger in RULELESS_TRIGGERS:
        return True
    return in_scope(getattr(row, "rule_uuids_json", None), rule_uuid)
