"""Timestamps must survive the SQLite round-trip as *aware* UTC datetimes.

Every timestamp column is declared `timezone=True` and every writer uses
`datetime.now(timezone.utc)`, but SQLite has no offset-bearing storage type: it
persists the naive wall-clock digits and hands them back with `tzinfo=None`.

That silence is the whole bug. A naive datetime serialises to
`"2026-08-20T15:47:55.537329"`, and per ES2015 a browser reads an offset-less
*date-time* string as **local time** — so a UTC instant renders shifted by the
viewer's offset, with no error anywhere. The NVR and Review tabs each grew their
own `${value}Z` patch; the Tasks tab never did, and showed task start/end times
hours in the future.

Task instances are the subject here because they carry four separate timestamp
columns and are what the Tasks tab actually renders.
"""

from datetime import datetime, timezone

import pytest


@pytest.fixture
async def config_uuid(http_client) -> str:
    r = await http_client.post(
        "/api/task-configs",
        json={"name": "tz-probe", "typename": "CollectImages", "params": {}},
    )
    assert r.status_code == 200, r.text
    return r.json()["config"]["uuid"]


def _has_offset(value: str) -> bool:
    """Does this ISO string tell a browser which zone it is in?"""
    # The exact spelling is Pydantic's business ("+00:00" vs "Z"); what matters
    # is that *some* offset is present, which is what `new Date()` keys off.
    return value.endswith("Z") or (len(value) > 6 and value[-6] in "+-")


class TestTimestampTimezones:
    async def test_orm_read_returns_aware_utc(self, http_client, db_client, config_uuid):
        """A datetime read back out of SQLite still knows it is UTC."""
        r = await http_client.post("/api/task-instances", json={"configUuid": config_uuid})
        uuid = r.json()["instance"]["uuid"]

        [instance] = await db_client.get_tasks(task_uuid=[uuid])

        assert instance.created_at.tzinfo is not None, "tzinfo was dropped by the SQLite round-trip"
        assert instance.created_at.utcoffset().total_seconds() == 0

    async def test_stored_instant_is_not_shifted(self, http_client, db_client, config_uuid):
        """Stamping the offset must not also *move* the instant."""
        before = datetime.now(timezone.utc)
        r = await http_client.post("/api/task-instances", json={"configUuid": config_uuid})
        after = datetime.now(timezone.utc)
        uuid = r.json()["instance"]["uuid"]

        [instance] = await db_client.get_tasks(task_uuid=[uuid])

        # A wrong-direction fix (stamping local time instead of UTC) still yields
        # an aware datetime, so tzinfo alone proves nothing — the instant has to
        # land inside the window the write actually happened in.
        assert before <= instance.created_at <= after

    async def test_api_timestamps_carry_an_offset(self, http_client, config_uuid):
        """The JSON a browser receives is unambiguous about its zone."""
        r = await http_client.post("/api/task-instances", json={"configUuid": config_uuid})
        created = r.json()["instance"]["createdAt"]

        assert _has_offset(created), f"no UTC offset in {created!r}"
        assert datetime.fromisoformat(created.replace("Z", "+00:00")).tzinfo is not None

    async def test_every_timestamp_column_is_covered(self, http_client, config_uuid):
        """The fix is on the column type, so *all* four stamps carry an offset."""
        r = await http_client.post("/api/task-instances", json={"configUuid": config_uuid})
        uuid = r.json()["instance"]["uuid"]

        stamp = datetime.now(timezone.utc).isoformat()
        r = await http_client.patch(
            f"/api/task-instances/{uuid}/timestamps",
            json={"startedAt": stamp, "endedAt": stamp},
        )
        assert r.status_code == 200, r.text
        r = await http_client.patch(f"/api/task-instances/{uuid}/heartbeat")
        assert r.status_code == 200, r.text

        r = await http_client.get("/api/task-instances", params={"uuids": [uuid]})
        [instance] = r.json()["instances"]

        naive = {
            field: value
            for field in ("createdAt", "startedAt", "endedAt", "heartbeatAt")
            if (value := instance[field]) is not None and not _has_offset(value)
        }
        assert not naive, f"timestamp fields serialised without a UTC offset: {naive}"
