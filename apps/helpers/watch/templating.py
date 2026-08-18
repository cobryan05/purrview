"""Template parameters derived from a fired rule, and the renderer that fills them in.

The point of this module is that a rule like "when any of these cats is inside any of
these litter boxes" fires with *slot values* — which cat, which litter box — and those
values are the interesting part. Without them every match publishes to the same topic
and a Home Assistant automation cannot tell one cat from another.

Placeholders are `{name}`. Rendering is deliberately **not** `str.format`: a template is
user input, and `str.format` on user input exposes attribute and index access
(`{x.__class__}`), which is a well-known way to walk an object graph out of a format
string. A plain regex substitution has no such surface.

Two rendering modes, because the same parameters serve two very different sinks:

* ``render_topic`` slugifies each substituted value, because an MQTT topic level cannot
  contain ``/``, ``+`` or ``#`` without silently reshaping the topic tree.
* ``render_text`` substitutes raw values, for anything that is prose rather than a path.

An unknown placeholder is left **verbatim** rather than blanked. A typo'd ``{sorce}``
that renders as an empty string collapses two topic levels into one and produces a
plausible-looking topic that is quietly wrong; left as ``{sorce}`` it is obviously
broken the first time anyone looks at the broker.
"""

import re
from typing import Any, Dict, Optional

_PLACEHOLDER = re.compile(r"\{([a-zA-Z][a-zA-Z0-9_]*)\}")
_TOPIC_SAFE = re.compile(r"[^a-zA-Z0-9_-]+")


def slugify(value: str) -> str:
    """Make a value safe for one MQTT topic level.

    `/` and `+`/`#` in a name would silently reshape the topic tree, so they are
    collapsed rather than escaped.
    """
    slug = _TOPIC_SAFE.sub("-", (value or "").strip().lower()).strip("-")
    return slug or "unnamed"


def build_template_params(signal, context) -> Dict[str, str]:
    """The placeholder vocabulary available to an action's templates.

    Every value is a string so a template never has to care about types. Names fall back
    to uuids, and uuids fall back to empty, so a template referencing a slot a given rule
    doesn't have degrades to something inert rather than raising mid-event.

    :param signal: the :class:`WatchEventSignal` being dispatched
    :param context: the :class:`ActionContext` the task built for it
    """
    event_label = signal.metadata.get("eventLabel") if signal.metadata else None
    rule_display = event_label or signal.rule_name or ""

    params: Dict[str, str] = {
        # The rule, under both the name it was given and the label it publishes as.
        "rule": rule_display,
        "rule_name": signal.rule_name or "",
        "event_label": event_label or "",
        "rule_uuid": signal.rule_uuid or "",
        # Where it happened.
        "watcher": context.profile_name or context.profile_uuid or "",
        "watcher_uuid": context.profile_uuid or "",
        "source": context.source_name or context.source_uuid or "",
        "source_uuid": context.source_uuid or "",
        # The slots — the reason this module exists.
        "subject": signal.subject_name or signal.subject_ref or "",
        "subject_uuid": signal.subject_ref or "",
        "target": signal.target_name or signal.target_ref or "",
        "target_uuid": signal.target_ref or "",
        # The transition itself.
        "phase": getattr(signal.phase, "value", str(signal.phase)),
        "event_uuid": str((context.extra or {}).get("event_uuid") or ""),
    }
    return params


def render(template: str, params: Dict[str, str], slug: bool = False) -> str:
    """Substitute `{name}` placeholders. Unknown names are left verbatim — see module docs."""
    if not template:
        return ""

    def _sub(match: "re.Match[str]") -> str:
        key = match.group(1)
        if key not in params:
            return match.group(0)
        value = params[key]
        if not slug:
            return value
        # An empty *known* slot renders empty so `render_topic` can drop the level.
        # Routing it through `slugify` instead would emit its "unnamed" fallback, which
        # is right for a name that slugs away to nothing but wrong for a slot the rule
        # simply doesn't have — it would publish a topic level that looks deliberate.
        return slugify(value) if value else ""

    return _PLACEHOLDER.sub(_sub, template)


def render_topic(template: str, params: Dict[str, str]) -> str:
    """Render a template into an MQTT topic, slugifying each substituted value.

    Empty levels are collapsed: a template with an optional slot (`{subject}`) that a
    given rule leaves empty would otherwise emit `a//b`, and a zero-length topic level is
    legal MQTT but matches nothing anyone would think to subscribe to.
    """
    rendered = render(template, params, slug=True)
    levels = [level for level in rendered.split("/") if level]
    return "/".join(levels)


def render_text(template: str, params: Dict[str, str]) -> str:
    """Render a template as prose, substituting raw (un-slugified) values."""
    return render(template, params, slug=False)


def describe_params(params: Optional[Dict[str, Any]] = None) -> str:
    """A one-line summary of the available placeholders, for UI help text."""
    keys = sorted((params or _EXAMPLE_PARAMS).keys())
    return ", ".join(f"{{{k}}}" for k in keys)


#: Shape-only example used by `describe_params` and by the UI's placeholder hint. Kept
#: here so the vocabulary is documented in exactly one place.
_EXAMPLE_PARAMS: Dict[str, str] = {
    "rule": "",
    "rule_name": "",
    "event_label": "",
    "rule_uuid": "",
    "watcher": "",
    "watcher_uuid": "",
    "source": "",
    "source_uuid": "",
    "subject": "",
    "subject_uuid": "",
    "target": "",
    "target_uuid": "",
    "phase": "",
    "event_uuid": "",
}
