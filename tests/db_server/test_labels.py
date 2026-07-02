import pytest


async def test_list_labels_empty(http_client):
    r = await http_client.get("/api/labels")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "success"
    assert body["labels"] == []


async def test_create_label(http_client):
    r = await http_client.post("/api/labels", json={"name": "cat", "color": "#FF0000"})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "success"
    assert body["label"]["name"] == "cat"
    assert body["label"]["color"] == "#FF0000"
    assert "uuid" in body["label"]


async def test_create_and_list_label(http_client):
    await http_client.post("/api/labels", json={"name": "dog", "color": "#00FF00"})
    r = await http_client.get("/api/labels")
    body = r.json()
    assert body["status"] == "success"
    names = [lbl["name"] for lbl in body["labels"]]
    assert "dog" in names


async def test_update_label(http_client):
    create_r = await http_client.post("/api/labels", json={"name": "bird", "color": "#0000FF"})
    uuid = create_r.json()["label"]["uuid"]

    r = await http_client.patch(f"/api/labels/{uuid}", json={"name": "parrot", "color": "#FFFF00"})
    assert r.status_code == 200
    assert r.json()["status"] == "success"

    list_r = await http_client.get("/api/labels")
    names = [lbl["name"] for lbl in list_r.json()["labels"]]
    assert "parrot" in names
    assert "bird" not in names


async def test_update_label_type_via_camel_case(http_client):
    """The UI sends {labelType: ...} (camelCase). LabelUpdate must map it to
    label_type; otherwise the key is silently dropped and the type never changes.
    """
    create_r = await http_client.post("/api/labels", json={"name": "cat", "color": "#FF0000"})
    uuid = create_r.json()["label"]["uuid"]

    patch_r = await http_client.patch(f"/api/labels/{uuid}", json={"labelType": "dynamic"})
    assert patch_r.status_code == 200
    assert patch_r.json()["status"] == "success"

    list_r = await http_client.get("/api/labels")
    lbl = next(l for l in list_r.json()["labels"] if l["uuid"] == uuid)
    assert lbl["labelType"] == "dynamic"


async def test_delete_label(http_client):
    create_r = await http_client.post("/api/labels", json={"name": "fish", "color": "#AABBCC"})
    uuid = create_r.json()["label"]["uuid"]

    r = await http_client.delete(f"/api/labels/{uuid}")
    assert r.status_code == 200
    assert r.json()["status"] == "success"

    list_r = await http_client.get("/api/labels")
    names = [lbl["name"] for lbl in list_r.json()["labels"]]
    assert "fish" not in names


async def test_delete_label_with_children_fails(http_client):
    parent_r = await http_client.post("/api/labels", json={"name": "animal", "color": "#111111"})
    parent_uuid = parent_r.json()["label"]["uuid"]

    await http_client.post(
        "/api/labels",
        json={"name": "dog", "color": "#222222", "parent_uuid": parent_uuid},
    )

    r = await http_client.delete(f"/api/labels/{parent_uuid}")
    assert r.status_code == 200
    assert r.json()["status"] == "failure"


async def test_create_child_label(http_client):
    parent_r = await http_client.post("/api/labels", json={"name": "vehicle", "color": "#333333"})
    parent_uuid = parent_r.json()["label"]["uuid"]

    child_r = await http_client.post(
        "/api/labels",
        json={"name": "car", "color": "#444444", "parent_uuid": parent_uuid},
    )
    assert child_r.status_code == 200
    body = child_r.json()
    assert body["status"] == "success"
    assert body["label"]["parentUuid"] == parent_uuid


async def test_label_type_roundtrip_from_migration_default(db_client, http_client):
    """
    Regression: the migration adds label_type with server_default='standard' (lowercase
    enum value).  sa.Enum(LabelType) without values_callable stores/expects names
    ('STANDARD'), so it rejects 'standard' on read.  The ORM column must be configured
    to use values, not names, so rows written by the migration default survive a round-trip.
    """
    import sqlalchemy as sa

    # Insert a row the way the migration's server_default does: lowercase value.
    async with db_client._engine.begin() as conn:
        await conn.execute(
            sa.text(
                "INSERT INTO labels (uuid, name, label_type, created_at, updated_at)"
                " VALUES (:uuid, :name, :lt, datetime('now'), datetime('now'))"
            ),
            {"uuid": "test-migration-default", "name": "cat", "lt": "standard"},
        )

    # Fetching labels must succeed and return the correct label_type value.
    r = await http_client.get("/api/labels")
    assert r.status_code == 200, r.text
    labels = r.json()["labels"]
    assert len(labels) == 1
    assert labels[0]["labelType"] == "standard"


class TestLabelScope:
    async def test_create_label_defaults_to_object_scope(self, http_client):
        r = await http_client.post("/api/labels", json={"name": "cat", "color": "#FF0000"})
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "success"
        assert body["label"]["scope"] == "object"

    async def test_create_image_scope_label(self, http_client):
        r = await http_client.post(
            "/api/labels",
            json={"name": "corrupt", "color": "#FF0000", "scope": "image"},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "success"
        assert body["label"]["scope"] == "image"

    async def test_scope_roundtrips_via_list(self, http_client):
        await http_client.post("/api/labels", json={"name": "blurry", "color": "#aabbcc", "scope": "image"})
        r = await http_client.get("/api/labels")
        labels = r.json()["labels"]
        blurry = next(l for l in labels if l["name"] == "blurry")
        assert blurry["scope"] == "image"

    async def test_update_label_scope(self, http_client):
        r = await http_client.post("/api/labels", json={"name": "dark", "color": "#111"})
        uuid = r.json()["label"]["uuid"]

        patch_r = await http_client.patch(f"/api/labels/{uuid}", json={"scope": "image"})
        assert patch_r.status_code == 200
        assert patch_r.json()["status"] == "success"

        list_r = await http_client.get("/api/labels")
        dark = next(l for l in list_r.json()["labels"] if l["name"] == "dark")
        assert dark["scope"] == "image"

    async def test_scope_server_default_is_object(self, db_client, http_client):
        """Rows inserted without a scope column (migration backfill path) read back as object."""
        import sqlalchemy as sa

        async with db_client._engine.begin() as conn:
            await conn.execute(
                sa.text(
                    "INSERT INTO labels (uuid, name, label_type, scope, created_at, updated_at)"
                    " VALUES (:uuid, :name, 'standard', 'object', datetime('now'), datetime('now'))"
                ),
                {"uuid": "scope-default-test", "name": "migrated_label"},
            )

        r = await http_client.get("/api/labels")
        labels = r.json()["labels"]
        ml = next(l for l in labels if l["name"] == "migrated_label")
        assert ml["scope"] == "object"
