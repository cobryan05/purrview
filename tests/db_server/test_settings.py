"""Integration tests for the user-settings API (GET/PUT /api/settings/{key}).

Uses the shared http_client fixture (full Manager + WebApp + in-process ASGI client
against an in-memory SQLite DB) from conftest.py.
"""

import pytest


async def test_get_setting_returns_404_for_missing_key(http_client):
    """A key that was never stored must return HTTP 404."""
    r = await http_client.get("/api/settings/nonexistent-key")
    assert r.status_code == 404


async def test_put_then_get_setting_roundtrip(http_client):
    """PUT stores the value; subsequent GET returns it with status success."""
    put_r = await http_client.put("/api/settings/hotkey_bindings", json={"value": '{"a":"b"}'})
    assert put_r.status_code == 200
    put_body = put_r.json()
    assert put_body["status"] == "success"
    assert put_body["key"] == "hotkey_bindings"
    assert put_body["value"] == '{"a":"b"}'

    get_r = await http_client.get("/api/settings/hotkey_bindings")
    assert get_r.status_code == 200
    get_body = get_r.json()
    assert get_body["status"] == "success"
    assert get_body["key"] == "hotkey_bindings"
    assert get_body["value"] == '{"a":"b"}'


async def test_put_setting_overwrites_existing_value(http_client):
    """A second PUT on the same key replaces the value (upsert, not duplicate insert)."""
    await http_client.put("/api/settings/my_key", json={"value": "first"})
    await http_client.put("/api/settings/my_key", json={"value": "second"})

    r = await http_client.get("/api/settings/my_key")
    assert r.status_code == 200
    assert r.json()["value"] == "second"


async def test_put_setting_allows_large_json_value(http_client):
    """Values survive a round-trip as arbitrary JSON strings without truncation or escaping."""
    big_json = '{"ctrl+a":"action:select","ctrl+b":"action:bbox","ctrl+z":"action:undo"}'
    await http_client.put("/api/settings/hotkeys", json={"value": big_json})

    r = await http_client.get("/api/settings/hotkeys")
    assert r.status_code == 200
    assert r.json()["value"] == big_json


async def test_get_setting_response_body_contains_key(http_client):
    """The response body includes the `key` field matching the URL path segment."""
    await http_client.put("/api/settings/my_pref", json={"value": "42"})

    r = await http_client.get("/api/settings/my_pref")
    body = r.json()
    assert body["key"] == "my_pref"
    assert body["value"] == "42"
