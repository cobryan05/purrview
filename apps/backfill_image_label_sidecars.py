"""One-shot backfill: write existing image_labels rows into JSON sidecars.

Sidecars only started carrying whole-image label assessments (present / absent)
after they had already been in use for a long time, so every assessment made
before that lives solely in the database — the one place the sidecars exist to
be independent of. This walks the assessments already in the DB and writes them
into the sidecar next to each image.

Idempotent: rerunning rewrites identical content, so an interrupted run can
simply be repeated. Images with no sidecar yet are skipped rather than given a
new one, because a sidecar with no `uuid`/`boxes` is invisible to reconcile.

Run inside the container:

    python -m apps.backfill_image_label_sidecars
"""

import asyncio
import logging
import os
from pathlib import Path

from apps import DB_DIR
from apps.db_server.manager import Manager
from apps.helpers.db.db_client import DbClient

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


async def main() -> None:
    files_root = Path(os.environ.get("FILES_ROOT", "/app/image_datasets"))
    db_client = DbClient(DB_DIR, files_root=files_root)
    # Deliberately not init_db(): this is a repair pass over an existing
    # database, and running migrations as a side effect of a backfill is a
    # surprise nobody asked for.
    manager = Manager(db_client=db_client, files_root=files_root)
    try:
        stats = await manager.backfill_image_label_sidecars()
    finally:
        await db_client.close()

    print(
        f"wrote {stats['written']} sidecars; "
        f"skipped {stats['skipped_no_sidecar']} with no sidecar, "
        f"{stats['skipped_no_path']} with no resolvable path"
    )


if __name__ == "__main__":
    asyncio.run(main())
