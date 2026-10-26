"""Ticket storage and the end-to-end workflow (save -> ML -> RAG/LLM -> review)."""
from __future__ import annotations

import json
import logging
import sqlite3
from typing import Any

from . import dynamic_fields as df
from .database import Database, now_iso
from .ml.predictor import SolutionPredictor
from .rag.pipeline import ResponseGenerator

log = logging.getLogger(__name__)

JSON_COLUMNS = ("rag_sources", "grounding_report")
EDITABLE = ("customer_name", "subject", "description", "category", "priority", "status")


class NotFound(Exception):
    pass


def row_to_ticket(conn: sqlite3.Connection, row: sqlite3.Row, with_fields: bool = True) -> dict:
    t = dict(row)
    for col in JSON_COLUMNS:
        t[col] = json.loads(t[col]) if t.get(col) else None
    t["is_active"] = bool(t["is_active"])
    if with_fields:
        t["custom_fields"] = df.load_values(conn, t["id"])
    return t


class TicketService:
    def __init__(self, db: Database, predictor: SolutionPredictor, generator: ResponseGenerator):
        self.db = db
        self.predictor = predictor
        self.generator = generator

    # ---------- CRUD ----------
    def get(self, ticket_id: int, include_inactive: bool = True) -> dict:
        with self.db.session() as conn:
            row = conn.execute("SELECT * FROM tickets WHERE id = ?", (ticket_id,)).fetchone()
            if not row or (not include_inactive and not row["is_active"]):
                raise NotFound(f"Ticket {ticket_id} not found")
            return row_to_ticket(conn, row)

    def list(self, status: str | None = None, category: str | None = None, priority: str | None = None,
             q: str | None = None, include_inactive: bool = False, limit: int = 100, offset: int = 0) -> list[dict]:
        where, params = [], []
        if not include_inactive:
            where.append("is_active = 1")
        for col, val in (("status", status), ("category", category), ("priority", priority)):
            if val:
                where.append(f"{col} = ?")
                params.append(val)
        if q:
            where.append("(subject LIKE ? OR description LIKE ? OR customer_name LIKE ?)")
            params += [f"%{q}%"] * 3
        sql = "SELECT * FROM tickets" + (f" WHERE {' AND '.join(where)}" if where else "")
        sql += " ORDER BY id DESC LIMIT ? OFFSET ?"
        with self.db.session() as conn:
            return [row_to_ticket(conn, r) for r in conn.execute(sql, (*params, limit, offset))]

    def create(self, data: dict, generate: bool = True) -> dict:
        custom = data.pop("custom_fields", {}) or {}
        # 1-2. validate dynamic fields and save the ticket first, so it is never lost
        with self.db.session() as conn:
            fields = df.list_fields(conn)
            cleaned = df.validate_values(fields, custom)
            ts = now_iso()
            cur = conn.execute(
                "INSERT INTO tickets (customer_name, subject, description, category, priority, status, created_at, updated_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (data["customer_name"], data["subject"], data.get("description") or "", data.get("category") or None,
                 data.get("priority", "Medium"), data.get("status", "Open"), ts, ts),
            )
            ticket_id = cur.lastrowid
            df.save_values(conn, ticket_id, fields, cleaned)
        # 3-4. ML prediction
        self.predict(ticket_id)
        # 5-6. RAG + LLM draft
        if generate:
            self.generate(ticket_id)
        return self.get(ticket_id)

    def update(self, ticket_id: int, changes: dict) -> dict:
        custom = changes.pop("custom_fields", None)
        self.get(ticket_id)  # 404 check
        with self.db.session() as conn:
            sets = {k: v for k, v in changes.items() if k in EDITABLE and v is not None}
            if custom is not None:
                fields = df.list_fields(conn)
                df.save_values(conn, ticket_id, fields, df.validate_values(fields, custom, partial=True))
            if sets or custom is not None:
                sets["updated_at"] = now_iso()
                conn.execute(f"UPDATE tickets SET {', '.join(f'{k} = ?' for k in sets)} WHERE id = ?",
                             (*sets.values(), ticket_id))
        return self.get(ticket_id)

    def deactivate(self, ticket_id: int) -> dict:
        self.get(ticket_id)
        with self.db.session() as conn:
            conn.execute("UPDATE tickets SET is_active = 0, status = 'Closed', updated_at = ? WHERE id = ?", (now_iso(), ticket_id))
        return self.get(ticket_id)

    def restore(self, ticket_id: int) -> dict:
        self.get(ticket_id)
        with self.db.session() as conn:
            conn.execute("UPDATE tickets SET is_active = 1, status = 'Open', updated_at = ? WHERE id = ?", (now_iso(), ticket_id))
        return self.get(ticket_id)

    def delete(self, ticket_id: int) -> None:
        self.get(ticket_id)
        with self.db.session() as conn:
            conn.execute("DELETE FROM tickets WHERE id = ?", (ticket_id,))

    # ---------- AI steps ----------
    def predict(self, ticket_id: int) -> dict:
        t = self.get(ticket_id)
        try:
            p = self.predictor.predict(t["subject"], t["description"])
            values: dict[str, Any] = {
                "predicted_solution": p.solution, "prediction_confidence": p.confidence,
                "predicted_category": p.category, "prediction_error": None,
            }
            if not t["category"] and p.category:
                values["category"] = p.category  # fill a missing category from the model
        except Exception as exc:
            log.warning("Prediction failed for ticket %s: %s", ticket_id, exc)
            values = {"predicted_solution": None, "prediction_confidence": None, "prediction_error": str(exc)}
        values["updated_at"] = now_iso()
        with self.db.session() as conn:
            conn.execute(f"UPDATE tickets SET {', '.join(f'{k} = ?' for k in values)} WHERE id = ?", (*values.values(), ticket_id))
        return self.get(ticket_id)

    def generate(self, ticket_id: int) -> dict:
        t = self.get(ticket_id)
        try:
            result = self.generator.generate(t, t["predicted_solution"])
            values = {
                "ai_response": result.response, "response_source": result.source, "response_model": result.model,
                "rag_sources": json.dumps(result.sources), "grounding_report": json.dumps(result.grounding),
                "generation_error": result.error, "generated_at": now_iso(),
            }
            # A new draft only replaces the final response if the agent has not approved one yet.
            if t["response_status"] != "approved":
                values["final_response"] = result.response
                values["response_status"] = "draft"
        except Exception as exc:  # retrieval itself failed; keep the ticket usable
            log.exception("Response generation failed for ticket %s", ticket_id)
            values = {"generation_error": f"Response generation failed: {exc}"}
        values["updated_at"] = now_iso()
        with self.db.session() as conn:
            conn.execute(f"UPDATE tickets SET {', '.join(f'{k} = ?' for k in values)} WHERE id = ?", (*values.values(), ticket_id))
        return self.get(ticket_id)

    def save_response(self, ticket_id: int, final_response: str, approve: bool) -> dict:
        self.get(ticket_id)
        with self.db.session() as conn:
            conn.execute(
                "UPDATE tickets SET final_response = ?, response_status = ?, updated_at = ? WHERE id = ?",
                (final_response, "approved" if approve else "draft", now_iso(), ticket_id),
            )
        return self.get(ticket_id)
