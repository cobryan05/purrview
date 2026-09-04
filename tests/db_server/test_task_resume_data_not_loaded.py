"""Tests that an omitted checkpoint is never *read*, not merely never *sent*.

Keeping resume_data out of the listing payload (see
test_task_instance_resume_data_payload.py) was originally done in the response model:
the ORM loaded every checkpoint, ran json.loads over it, and the endpoint then
discarded the result. That is invisible in the payload and ruinous in production —
~1 MB per finished TrainModelTask, a listing polled twice a second by the task SSE
stream, and the decode running on db_server's event loop. It was measured holding
that loop at ~98% CPU with essentially no traffic, which surfaced as every tab in
the UI taking a visible beat to load.

So the contract these tests pin is the one the payload tests cannot see: when a
caller does not ask for resume_data, the column must not appear in the SELECT at all.
"""

import json

from sqlalchemy import event

from apps.helpers.consts import TaskStatus

CHECKPOINT = {"epoch_history": [{"epoch": n, "loss": 0.5} for n in range(50)]}


def _capture_sql(db_client) -> list:
    """Record every statement the client emits, for the duration of the test."""
    statements: list = []

    def _before_cursor_execute(conn, cursor, statement, parameters, context, executemany):
        statements.append(statement)

    event.listen(db_client._engine.sync_engine, "before_cursor_execute", _before_cursor_execute)
    return statements


async def _make_instance(db_client, name: str) -> str:
    config = await db_client.add_task_config(name=name, typename="TestTask", params={})
    instance = await db_client.insert_new_active_task(config.uuid)
    await db_client.set_task_resume_data(instance.uuid, CHECKPOINT)
    return instance.uuid


class TestCheckpointIsNotLoaded:
    async def test_default_listing_does_not_select_the_checkpoint_column(self, db_client):
        """The regression itself: deferring the column is what stops the decode.

        Asserting on the response would pass even with the column loaded and thrown
        away, which is exactly the bug that shipped.
        """
        await _make_instance(db_client, "no-load")

        statements = _capture_sql(db_client)
        await db_client.get_tasks()

        selects = [s for s in statements if "task_instances" in s and s.lstrip().upper().startswith("SELECT")]
        assert selects, "expected a SELECT against task_instances"
        assert not any(
            "resume_data_json" in s for s in selects
        ), "resume_data_json was read back despite not being requested:\n" + "\n".join(selects)

    async def test_opting_in_does_select_the_checkpoint_column(self, db_client):
        """The negative above must not be satisfiable by never reading it at all."""
        await _make_instance(db_client, "opt-in-load")

        statements = _capture_sql(db_client)
        instances = await db_client.get_tasks(include_resume_data=True)

        selects = [s for s in statements if "task_instances" in s and s.lstrip().upper().startswith("SELECT")]
        assert any("resume_data_json" in s for s in selects)
        assert [i.resume_data_json for i in instances] == [CHECKPOINT]

    async def test_deferred_rows_survive_detachment(self, db_client):
        """Reading a deferred column on a detached row raises DetachedInstanceError.

        get_tasks closes its session before returning, and TaskInstanceRead reads
        resume_data by its resume_data_json alias — so the deferred attribute has to
        come back as a loaded None rather than an unloaded one.
        """
        await _make_instance(db_client, "detached")

        instances = await db_client.get_tasks()

        assert [i.resume_data_json for i in instances] == [None]

    async def test_the_filter_still_reads_the_column_in_sql(self, db_client):
        """resumable= filters on the column in SQLite, which deferring must not break."""
        uuid = await _make_instance(db_client, "filter")

        assert [i.uuid for i in await db_client.get_tasks(resumable=True)] == [uuid]
        assert await db_client.get_tasks(resumable=False) == []


class TestCheckpointClearedOnCompletion:
    async def test_completing_a_task_drops_its_checkpoint(self, db_client):
        """A completed run has nothing to resume, so the blob is dead weight."""
        uuid = await _make_instance(db_client, "completed")

        await db_client.set_task_status(uuid, TaskStatus.COMPLETED)

        assert [i.resume_data_json for i in await db_client.get_tasks(include_resume_data=True)] == [None]

    async def test_erroring_a_task_keeps_its_checkpoint(self, db_client):
        """Recovery reasons about whether an interrupted task had a checkpoint;
        clearing it on error would destroy that evidence."""
        uuid = await _make_instance(db_client, "errored")

        await db_client.set_task_status(uuid, TaskStatus.ERROR, error_message="boom")

        instances = await db_client.get_tasks(include_resume_data=True)
        assert [i.resume_data_json for i in instances] == [CHECKPOINT]

    async def test_pausing_a_task_keeps_its_checkpoint(self, db_client):
        """PAUSED is the status resume_tasks restarts from — the checkpoint is the
        entire reason the row is still there."""
        uuid = await _make_instance(db_client, "paused")

        await db_client.set_task_status(uuid, TaskStatus.PAUSED)

        instances = await db_client.get_tasks(include_resume_data=True)
        assert [i.resume_data_json for i in instances] == [CHECKPOINT]

    async def test_the_checkpoint_is_gone_from_the_row_not_just_the_read(self, db_client):
        """Guard against a future 'fix' that hides the blob instead of deleting it —
        the point of clearing is to stop paying for it on disk and on decode."""
        uuid = await _make_instance(db_client, "really-gone")
        await db_client.set_task_status(uuid, TaskStatus.COMPLETED)

        async with db_client._async_session_maker() as session:
            from sqlalchemy import text

            raw = (
                await session.exec(
                    text("SELECT resume_data_json FROM task_instances WHERE uuid = :u").bindparams(u=uuid)
                )
            ).one()[0]

        assert raw is None or json.loads(raw) is None
