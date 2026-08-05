"""migrate legacy dataset filter_json to the term list

PR 1.4b replaced DatasetFilterCriteria's flat fields (tag_uuids, date_from,
filename_search, image_*_conditions) with a discriminated-union `terms` list and
switched the model to extra="forbid". No migration converted the stored rows, so
every database written before that revision still holds the flat shape — and
because DatasetRead validates filter_json on read, ONE legacy row makes the
entire GET /api/datasets listing fail, taking every dataset picker in the app
down with it. That is reproducible today on any pre-1.4b database, including the
seeded "Whole Database" system row, which every install has.

This performs the "migrate best-effort" half of what PR 1.4 promised. The
conversion mirrors reviewStateToDatasetForm() in shared/filterUtils.js, which is
the canonical flat→terms mapping used by the live UI:

  tag_uuids            → one bboxTag term (carrying tag_match_mode)
  tag_exclude_uuids    → one negated bboxTag term per uuid
  date_from / date_to  → one date term on `created`
  filename_search      → one filename term
  image_tag_conditions   → one imageTag term each ('unset' → 'unreviewed')
  image_label_conditions → one imageLabel term each ('uncertain' → 'unreviewed')

label_exclude_uuids / label_match_mode / label_include_children /
verification_check are NOT legacy — they modify Dataset.label_uuids, a separate
column, and survive unchanged.

Anything unrecognised is dropped rather than carried, since extra="forbid" would
reject it on the next read anyway; dropping a filter the user can see and
re-add beats a listing that 500s.

Revision ID: b7d31c904ae2
Revises: 6a2885d4ed23
Create Date: 2026-08-05
"""

import json
from typing import Any, Dict

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "b7d31c904ae2"
down_revision = "6a2885d4ed23"
branch_labels = None
depends_on = None


# Flat keys that PR 1.4b folded into `terms`. Presence of any of these is what
# marks a row as legacy.
_LEGACY_KEYS = (
    "tag_uuids",
    "tag_exclude_uuids",
    "tag_match_mode",
    "date_from",
    "date_to",
    "filename_search",
    "image_tag_conditions",
    "image_label_conditions",
)

# Keys that are still part of the current criteria model and pass through.
_KEPT_KEYS = (
    "label_exclude_uuids",
    "label_match_mode",
    "label_include_children",
    "verification_check",
)


def _convert(old: Dict[str, Any]) -> Dict[str, Any]:
    """Flat legacy criteria → current {kept fields…, terms: [...]} shape."""
    terms = list(old.get("terms") or [])

    tag_uuids = old.get("tag_uuids") or []
    if tag_uuids:
        terms.append(
            {
                "kind": "bboxTag",
                "tag_uuids": tag_uuids,
                "match_mode": old.get("tag_match_mode") or "any",
                "negate": False,
            }
        )
    for uuid in old.get("tag_exclude_uuids") or []:
        terms.append({"kind": "bboxTag", "tag_uuids": [uuid], "match_mode": "any", "negate": True})

    date_from, date_to = old.get("date_from"), old.get("date_to")
    if date_from or date_to:
        term: Dict[str, Any] = {"kind": "date", "field": "created", "negate": False}
        # DateTerm names the field `from_` internally (aliased to "from"); the
        # stored JSON uses the internal name, matching model_dump(mode="json").
        if date_from:
            term["from_"] = date_from
        if date_to:
            term["to"] = date_to
        terms.append(term)

    filename_search = old.get("filename_search")
    if filename_search:
        terms.append({"kind": "filename", "patterns": [filename_search], "negate": False})

    for cond in old.get("image_tag_conditions") or []:
        tag_uuid = cond.get("tag_uuid") or cond.get("tagUuid")
        if not tag_uuid:
            continue
        state = cond.get("state")
        terms.append(
            {
                "kind": "imageTag",
                "tag_uuid": tag_uuid,
                # The old 'unset' (never assessed) is the term model's 'unreviewed'.
                "state": "unreviewed" if state == "unset" else state,
                "negate": False,
            }
        )

    for cond in old.get("image_label_conditions") or []:
        label_uuid = cond.get("label_uuid") or cond.get("labelUuid")
        if not label_uuid:
            continue
        state = cond.get("state")
        terms.append(
            {
                "kind": "imageLabel",
                "label_uuid": label_uuid,
                # Same rename as above — 'uncertain' was the old spelling.
                "state": "unreviewed" if state == "uncertain" else state,
                # roll_up defaults True server-side, but the pre-1.4b semantics
                # were "this exact label only". Pin it so a migrated filter
                # keeps matching the same images it matched before.
                "roll_up": False,
                "negate": False,
            }
        )

    new: Dict[str, Any] = {key: old[key] for key in _KEPT_KEYS if key in old}
    new["terms"] = terms
    return new


def upgrade() -> None:
    conn = op.get_bind()
    rows = conn.execute(sa.text("SELECT uuid, filter_json FROM datasets")).fetchall()

    for uuid, raw in rows:
        if not raw:
            continue
        try:
            old = json.loads(raw) if isinstance(raw, str) else raw
        except (TypeError, ValueError):
            continue
        if not isinstance(old, dict):
            continue
        if not any(key in old for key in _LEGACY_KEYS):
            continue  # already in the current shape

        conn.execute(
            sa.text("UPDATE datasets SET filter_json = :fj WHERE uuid = :uuid"),
            {"fj": json.dumps(_convert(old)), "uuid": uuid},
        )


def downgrade() -> None:
    # One-way: the flat shape cannot express the terms a user may have added
    # since, and re-flattening would silently drop them.
    pass
