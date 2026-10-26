"""Ticket CRUD through the REST API."""


def test_create_ticket_saves_and_runs_workflow(client, ticket_payload):
    r = client.post("/api/tickets", json=ticket_payload)
    assert r.status_code == 201
    t = r.json()
    assert t["id"] > 0
    assert t["customer_name"] == "Jane Doe"
    assert t["status"] == "Open"
    assert t["predicted_solution"]          # ML step ran and was saved
    assert t["ai_response"]                 # RAG/LLM step ran and was saved
    assert t["final_response"] == t["ai_response"]
    assert t["response_status"] == "draft"
    assert t["created_at"] and t["updated_at"]

    # persisted: fetch again
    again = client.get(f"/api/tickets/{t['id']}").json()
    assert again["predicted_solution"] == t["predicted_solution"]
    assert again["custom_fields"] == {"contact_channel": "Email"}


def test_list_and_filter_tickets(client, ticket_payload):
    client.post("/api/tickets", json=ticket_payload)
    other = {**ticket_payload, "subject": "Charged twice", "description": "Two charges on my card", "priority": "Low"}
    client.post("/api/tickets", json=other)

    assert len(client.get("/api/tickets").json()) == 2
    assert [t["subject"] for t in client.get("/api/tickets?priority=Low").json()] == ["Charged twice"]
    assert [t["subject"] for t in client.get("/api/tickets?q=log").json()] == ["Unable to log in"]


def test_update_ticket_fields_and_status(client, ticket_payload):
    t = client.post("/api/tickets", json=ticket_payload).json()
    r = client.put(f"/api/tickets/{t['id']}", json={"status": "In Progress", "priority": "Urgent", "subject": "Still cannot log in"})
    assert r.status_code == 200
    u = r.json()
    assert (u["status"], u["priority"], u["subject"]) == ("In Progress", "Urgent", "Still cannot log in")
    assert u["customer_name"] == "Jane Doe"  # untouched fields are kept

    assert client.patch(f"/api/tickets/{t['id']}", json={"status": "Resolved"}).json()["status"] == "Resolved"


def test_soft_delete_restore_and_hard_delete(client, ticket_payload):
    t = client.post("/api/tickets", json=ticket_payload).json()

    d = client.delete(f"/api/tickets/{t['id']}").json()
    assert d["is_active"] is False
    assert client.get("/api/tickets").json() == []                          # hidden from the list
    assert len(client.get("/api/tickets?include_inactive=true").json()) == 1  # but still stored

    assert client.post(f"/api/tickets/{t['id']}/restore").json()["is_active"] is True

    assert client.delete(f"/api/tickets/{t['id']}?hard=true").json() == {"deleted": True, "id": t["id"]}
    assert client.get(f"/api/tickets/{t['id']}").status_code == 404


def test_unknown_ticket_returns_404(client):
    assert client.get("/api/tickets/999").status_code == 404
    assert client.put("/api/tickets/999", json={"status": "Closed"}).status_code == 404
    assert client.delete("/api/tickets/999").status_code == 404


def test_invalid_status_and_priority_rejected(client, ticket_payload):
    assert client.post("/api/tickets", json={**ticket_payload, "priority": "Super"}).status_code == 422
    t = client.post("/api/tickets", json=ticket_payload).json()
    assert client.put(f"/api/tickets/{t['id']}", json={"status": "Done"}).status_code == 422


def test_category_predicted_when_missing(client, ticket_payload):
    t = client.post("/api/tickets", json={**ticket_payload, "category": None}).json()
    assert t["category"] == "Account"
    kept = client.post("/api/tickets", json={**ticket_payload, "category": "VIP"}).json()
    assert kept["category"] == "VIP"  # an agent-provided category is never overwritten
