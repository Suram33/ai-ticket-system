"""Dynamic fields: defined at runtime, validated, saved per ticket, no schema change."""


def field_names(client):
    return [f["name"] for f in client.get("/api/fields").json()]


def test_default_fields_are_loaded(client):
    fields = {f["name"]: f for f in client.get("/api/fields").json()}
    assert {f["field_type"] for f in fields.values()} == {"text", "dropdown", "number", "date", "checkbox"}
    assert fields["contact_channel"]["required"] is True
    assert fields["contact_channel"]["options"] == ["Email", "Phone", "Chat", "Web form"]


def test_add_new_field_and_use_it_without_code_change(client, ticket_payload):
    r = client.post("/api/fields", json={"name": "warranty_tier", "label": "Warranty tier", "field_type": "dropdown",
                                         "required": True, "options": ["Gold", "Silver"]})
    assert r.status_code == 201
    assert "warranty_tier" in field_names(client)

    # The new required field is now enforced...
    missing = client.post("/api/tickets", json=ticket_payload)
    assert missing.status_code == 422
    assert "warranty_tier" in missing.json()["errors"]

    # ...and saved when provided.
    payload = {**ticket_payload, "custom_fields": {"contact_channel": "Chat", "warranty_tier": "Gold"}}
    t = client.post("/api/tickets", json=payload).json()
    assert t["custom_fields"] == {"contact_channel": "Chat", "warranty_tier": "Gold"}


def test_all_field_types_are_validated_and_typed(client, ticket_payload):
    payload = {**ticket_payload, "custom_fields": {
        "contact_channel": "Phone", "order_number": " ORD-1001 ", "order_amount": "49.90",
        "purchase_date": "2026-10-01", "vip_customer": "true"}}
    t = client.post("/api/tickets", json=payload).json()
    assert t["custom_fields"] == {"contact_channel": "Phone", "order_number": "ORD-1001", "order_amount": 49.9,
                                  "purchase_date": "2026-10-01", "vip_customer": True}


def test_invalid_values_return_clear_errors(client, ticket_payload):
    payload = {**ticket_payload, "custom_fields": {
        "contact_channel": "Pigeon", "order_amount": "abc", "purchase_date": "01/10/2026", "vip_customer": "maybe", "nope": 1}}
    r = client.post("/api/tickets", json=payload)
    assert r.status_code == 422
    errors = r.json()["errors"]
    assert set(errors) == {"contact_channel", "order_amount", "purchase_date", "vip_customer", "nope"}
    assert "must be one of" in errors["contact_channel"]
    assert client.get("/api/tickets").json() == []  # nothing saved on validation failure


def test_required_field_missing(client, ticket_payload):
    r = client.post("/api/tickets", json={**ticket_payload, "custom_fields": {}})
    assert r.status_code == 422
    assert r.json()["errors"] == {"contact_channel": "Contact channel is required"}


def test_update_custom_field_values(client, ticket_payload):
    t = client.post("/api/tickets", json=ticket_payload).json()
    u = client.put(f"/api/tickets/{t['id']}", json={"custom_fields": {"order_number": "ORD-7", "vip_customer": False}}).json()
    assert u["custom_fields"] == {"contact_channel": "Email", "order_number": "ORD-7", "vip_customer": False}
    cleared = client.put(f"/api/tickets/{t['id']}", json={"custom_fields": {"order_number": ""}}).json()
    assert "order_number" not in cleared["custom_fields"]
    assert client.put(f"/api/tickets/{t['id']}", json={"custom_fields": {"contact_channel": ""}}).status_code == 422


def test_deactivated_field_disappears_from_form(client, ticket_payload):
    fid = next(f["id"] for f in client.get("/api/fields").json() if f["name"] == "contact_channel")
    client.delete(f"/api/fields/{fid}")
    assert "contact_channel" not in field_names(client)
    # no longer required once inactive
    assert client.post("/api/tickets", json={**ticket_payload, "custom_fields": {}}).status_code == 201


def test_field_definition_validation(client):
    assert client.post("/api/fields", json={"name": "Bad Name", "label": "x", "field_type": "text"}).status_code == 422
    assert client.post("/api/fields", json={"name": "colour", "label": "Colour", "field_type": "dropdown"}).status_code == 422
    assert client.post("/api/fields", json={"name": "x", "label": "X", "field_type": "rainbow"}).status_code == 422
    assert client.post("/api/fields", json={"name": "order_number", "label": "Dup", "field_type": "text"}).status_code == 422


def test_no_schema_change_needed(services, client):
    with services["db"].session() as conn:
        before = conn.execute("SELECT sql FROM sqlite_master ORDER BY name").fetchall()
    client.post("/api/fields", json={"name": "serial", "label": "Serial", "field_type": "text"})
    with services["db"].session() as conn:
        after = conn.execute("SELECT sql FROM sqlite_master ORDER BY name").fetchall()
    assert [tuple(r) for r in before] == [tuple(r) for r in after]
