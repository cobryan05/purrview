"""drop model confidence from already-verified bboxes

A confidence describes what a MODEL guessed. Once a human has verified the box
that guess is settled history, so a verified bbox stores none — enforced going
forward by `db_client._apply_verified_confidence_rule` at every bbox write.

This clears the rows that were verified BEFORE that rule existed. Without it the
review grid keeps showing a stale confidence badge on verified boxes, and the
confidence-band filter keeps matching them, until each one happens to be
rewritten by some later edit.

`bboxes.confidence` is a generated column over these same JSON keys, so removing
the keys is what actually drops the row out of the band filter — there is no
separate column to update.

Revision ID: c4e7a1b93f52
Revises: b7f2c4a91d36
Create Date: 2026-08-12 00:00:00.000000

"""

from typing import Sequence, Union

from alembic import op

from apps.helpers.db.types import BBOX_CONFIDENCE_METADATA_KEYS, VERIFIED_TAG_UUID

# revision identifiers, used by Alembic.
revision: str = "c4e7a1b93f52"
down_revision: Union[str, None] = "b7f2c4a91d36"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    # json_remove drops only the named keys and leaves the rest of the blob
    # alone — every other metadata key (run_model_added, relabel_prev_label_uuid,
    # …) must survive this.
    paths = ", ".join(f"'$.{key}'" for key in BBOX_CONFIDENCE_METADATA_KEYS)
    key_predicate = " OR ".join(
        f"json_extract(metadata_json, '$.{key}') IS NOT NULL" for key in BBOX_CONFIDENCE_METADATA_KEYS
    )
    bind.exec_driver_sql(f"""
        UPDATE bboxes
           SET metadata_json = json_remove(metadata_json, {paths})
         WHERE metadata_json IS NOT NULL
           AND json_valid(metadata_json)
           AND ({key_predicate})
           AND uuid IN (
                 SELECT bbox_uuid FROM bbox_tags WHERE tag_uuid = '{VERIFIED_TAG_UUID}'
               )
        """)


def downgrade() -> None:
    pass  # the discarded confidences are not recoverable
