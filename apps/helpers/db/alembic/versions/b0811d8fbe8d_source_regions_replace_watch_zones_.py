"""source regions replace watch zones, multi-ref rule subjects, per-watcher mask opt-out

Revision ID: b0811d8fbe8d
Revises: 293b86a081d3
Create Date: 2026-08-17 19:34:04.237047

"""

from typing import Sequence, Union

import sqlalchemy as sa
import sqlmodel
from alembic import op
from sqlalchemy.dialects import sqlite

# revision identifiers, used by Alembic.
revision: str = "b0811d8fbe8d"
down_revision: Union[str, None] = "293b86a081d3"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _copy_zones_to_regions() -> None:
    """Carry hand-drawn zones from `watch_zones` onto the sources they were drawn on.

    Zone geometry is human work, so dropping the table without this loses it for good.
    The owning source is one join away — `watch_zones.profile_source_uuid` →
    `watch_profile_sources.source_uuid` — which is precisely the attribution the
    downgrade cannot reconstruct and the upgrade can.

    **Uuids are carried across unchanged.** `interaction_rules` names zones by uuid from
    a different table, so regenerating them here would leave every zone rule pointing at
    nothing: no error, no integrity failure, just a rule that never fires again.

    **Names are disambiguated per source.** Zones used to hang off a watcher's *use* of
    a source, so two watchers on one camera each kept their own copy of the same tray —
    the redundancy this revision exists to remove. Collapsing them onto the source makes
    those copies siblings, and `replace_source_regions` rejects two regions sharing
    (role, name). Left alone, the migration would succeed and then make the region
    editor unsaveable for that camera, with an error naming a zone the user only drew
    once. Suffixing the later copies keeps every uuid alive and the invariant intact;
    the user can merge or delete the duplicates at their leisure.
    """
    conn = op.get_bind()
    zones = conn.execute(
        sa.text(
            "SELECT wz.uuid, wps.source_uuid, wz.name, wz.kind, wz.points_json, "
            "       wz.reference_width, wz.reference_height "
            "  FROM watch_zones wz "
            "  JOIN watch_profile_sources wps ON wps.uuid = wz.profile_source_uuid "
            # Deterministic, so a rerun on a copy of the same DB assigns the same
            # suffixes rather than shuffling which duplicate is "(2)".
            " ORDER BY wps.source_uuid, wz.name, wz.uuid"
        )
    ).fetchall()

    seen_names: dict = {}
    next_order: dict = {}
    for zone in zones:
        uuid, source_uuid, name, kind, points_json, width, height = zone
        key = (source_uuid, (name or "").strip().lower())
        occurrence = seen_names.get(key, 0) + 1
        seen_names[key] = occurrence
        order = next_order.get(source_uuid, 0)
        next_order[source_uuid] = order + 1

        conn.execute(
            sa.text(
                "INSERT INTO source_regions (uuid, source_uuid, name, role, kind, points_json, "
                '                            reference_width, reference_height, "order") '
                "VALUES (:uuid, :source_uuid, :name, 'zone', :kind, :points_json, "
                "        :width, :height, :order)"
            ),
            {
                "uuid": uuid,
                "source_uuid": source_uuid,
                "name": name if occurrence == 1 else f"{name} ({occurrence})",
                "kind": kind,
                "points_json": points_json,
                "width": width,
                "height": height,
                "order": order,
            },
        )


def upgrade() -> None:
    # Autogenerate also proposed a REAL -> Float retype of `bboxes.confidence`. That is
    # a no-op on SQLite (both are NUMERIC affinity) and was dropped by hand: applying it
    # rewrites the whole bboxes table just to restate a generated column's long
    # expression, which is real risk on a large table for zero schema change.
    op.create_table(
        "source_regions",
        sa.Column("uuid", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("source_uuid", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("name", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("role", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("kind", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("points_json", sa.JSON(), nullable=True),
        sa.Column("reference_width", sa.Integer(), nullable=False),
        sa.Column("reference_height", sa.Integer(), nullable=False),
        sa.Column("order", sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(["source_uuid"], ["sources.uuid"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("uuid"),
    )
    with op.batch_alter_table("source_regions", schema=None) as batch_op:
        batch_op.create_index(batch_op.f("ix_source_regions_source_uuid"), ["source_uuid"], unique=False)

    _copy_zones_to_regions()

    with op.batch_alter_table("watch_zones", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_watch_zones_profile_source_uuid"))

    op.drop_table("watch_zones")

    # Add the list columns first, carry the old single refs across, and only then drop
    # the singular columns. Doing it in one batch would drop the source of the data in
    # the same table rewrite that is meant to read it.
    with op.batch_alter_table("interaction_rules", schema=None) as batch_op:
        batch_op.add_column(sa.Column("subject_a_refs_json", sa.JSON(), nullable=True))
        batch_op.add_column(sa.Column("subject_b_refs_json", sa.JSON(), nullable=True))

    # A one-element list is exactly what the old single ref meant. Without this a rule
    # written before multi-select survives the migration with an empty subject, which
    # never fires and reads in the editor as merely unconfigured rather than broken.
    for column, source in (
        ("subject_a_refs_json", "subject_a_ref"),
        ("subject_b_refs_json", "subject_b_ref"),
    ):
        op.execute(
            f"UPDATE interaction_rules "
            f"SET {column} = CASE "
            f"  WHEN {source} IS NULL OR {source} = '' THEN '[]' "
            f"  ELSE json_array({source}) "
            f"END"
        )

    with op.batch_alter_table("interaction_rules", schema=None) as batch_op:
        batch_op.drop_column("subject_b_ref")
        batch_op.drop_column("subject_a_ref")

    with op.batch_alter_table("watch_profile_sources", schema=None) as batch_op:
        batch_op.add_column(sa.Column("disabled_mask_uuids_json", sa.JSON(), nullable=True))

    # An empty list, not NULL: the read models type these as lists, and a NULL would fail
    # validation on the first GET rather than at write time where it would be obvious.
    op.execute(
        "UPDATE watch_profile_sources SET disabled_mask_uuids_json = '[]' WHERE disabled_mask_uuids_json IS NULL"
    )


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    with op.batch_alter_table("watch_profile_sources", schema=None) as batch_op:
        batch_op.drop_column("disabled_mask_uuids_json")

    with op.batch_alter_table("interaction_rules", schema=None) as batch_op:
        batch_op.add_column(sa.Column("subject_a_ref", sa.VARCHAR(), nullable=True))
        batch_op.add_column(sa.Column("subject_b_ref", sa.VARCHAR(), nullable=True))

    # Lossy by nature — the old column holds one ref. Keep the first so a downgraded
    # multi-label rule still fires for one of its labels rather than for none.
    for target, source in (
        ("subject_a_ref", "subject_a_refs_json"),
        ("subject_b_ref", "subject_b_refs_json"),
    ):
        op.execute(f"UPDATE interaction_rules SET {target} = json_extract({source}, '$[0]')")

    with op.batch_alter_table("interaction_rules", schema=None) as batch_op:
        batch_op.drop_column("subject_b_refs_json")
        batch_op.drop_column("subject_a_refs_json")

    # `watch_zones` is recreated empty: regions moved to the source, and there is no
    # profile_source to attribute a source-owned region back to without inventing one.
    op.create_table(
        "watch_zones",
        sa.Column("uuid", sa.VARCHAR(), nullable=False),
        sa.Column("profile_source_uuid", sa.VARCHAR(), nullable=False),
        sa.Column("name", sa.VARCHAR(), nullable=False),
        sa.Column("kind", sa.VARCHAR(), nullable=False),
        sa.Column("points_json", sqlite.JSON(), nullable=True),
        sa.Column("reference_width", sa.INTEGER(), nullable=False),
        sa.Column("reference_height", sa.INTEGER(), nullable=False),
        sa.ForeignKeyConstraint(["profile_source_uuid"], ["watch_profile_sources.uuid"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("uuid"),
    )
    with op.batch_alter_table("watch_zones", schema=None) as batch_op:
        batch_op.create_index(batch_op.f("ix_watch_zones_profile_source_uuid"), ["profile_source_uuid"], unique=False)

    with op.batch_alter_table("source_regions", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_source_regions_source_uuid"))

    op.drop_table("source_regions")
    # ### end Alembic commands ###
