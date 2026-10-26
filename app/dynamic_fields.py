"""Dynamic (custom) ticket fields.

Field definitions live in the `field_definitions` table and values in
`ticket_field_values` (one row per ticket + field). Adding a new field is an
INSERT through the API or UI; the form is rendered from the definitions, so no
code or schema change is needed.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import date
from typing import Any

from .database import now_iso


class FieldValidationError(ValueError):
    def __init__(self, errors: dict[str, str]):
        self.errors = errors
        super().__init__("; ".join(f"{k}: {v}" for k, v in errors.items()))


def row_to_field(row: sqlite3.Row) -> dict:
    d = dict(row)
    d["required"] = bool(d["required"])
    d["is_active"] = bool(d["is_active"])
    d["options"] = json.loads(d["options"]) if d["options"] else None
    return d


def list_fields(conn: sqlite3.Connection, include_inactive: bool = False) -> list[dict]:
    sql = "SELECT * FROM field_definitions"
    if not include_inactive:
        sql += " WHERE is_active = 1"
    return [row_to_field(r) for r in conn.execute(sql + " ORDER BY sort_order, id")]


def get_field(conn: sqlite3.Connection, field_id: int) -> dict | None:
    row = conn.execute("SELECT * FROM field_definitions WHERE id = ?", (field_id,)).fetchone()
    return row_to_field(row) if row else None


def create_field(conn: sqlite3.Connection, data: dict) -> dict:
    if data["field_type"] == "dropdown" and not data.get("options"):
        raise FieldValidationError({"options": "dropdown fields need at least one option"})
    if data["field_type"] != "dropdown":
        data["options"] = None
    if conn.execute("SELECT 1 FROM field_definitions WHERE name = ?", (data["name"],)).fetchone():
        raise FieldValidationError({"name": f"a field named '{data['name']}' already exists"})
    cur = conn.execute(
        "INSERT INTO field_definitions (name, label, field_type, required, options, help_text, sort_order, created_at)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (data["name"], data["label"], data["field_type"], int(data.get("required", False)),
         json.dumps(data["options"]) if data.get("options") else None, data.get("help_text"),
         data.get("sort_order", 100), now_iso()),
    )
    return get_field(conn, cur.lastrowid)


def update_field(conn: sqlite3.Connection, field_id: int, changes: dict) -> dict | None:
    current = get_field(conn, field_id)
    if not current:
        return None
    if "options" in changes and current["field_type"] == "dropdown" and not changes["options"]:
        raise FieldValidationError({"options": "dropdown fields need at least one option"})
    sets, params = [], []
    for key, value in changes.items():
        if key == "options":
            value = json.dumps(value) if value and current["field_type"] == "dropdown" else None
        elif key in ("required", "is_active"):
            value = int(value)
        sets.append(f"{key} = ?")
        params.append(value)
    if sets:
        conn.execute(f"UPDATE field_definitions SET {', '.join(sets)} WHERE id = ?", (*params, field_id))
    return get_field(conn, field_id)


def _is_empty(value: Any) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def coerce_value(field: dict, value: Any) -> Any:
    """Validate and normalise one value according to its field type. Raises ValueError."""
    ftype = field["field_type"]
    if ftype == "text":
        text = str(value).strip()
        if len(text) > 1000:
            raise ValueError("must be at most 1000 characters")
        return text
    if ftype == "number":
        if isinstance(value, bool):
            raise ValueError("must be a number")
        try:
            num = float(value)
        except (TypeError, ValueError):
            raise ValueError("must be a number") from None
        if num != num or num in (float("inf"), float("-inf")):
            raise ValueError("must be a finite number")
        return int(num) if num.is_integer() else num
    if ftype == "date":
        try:
            return date.fromisoformat(str(value).strip()).isoformat()
        except ValueError:
            raise ValueError("must be a date in YYYY-MM-DD format") from None
    if ftype == "checkbox":
        if isinstance(value, bool):
            return value
        text = str(value).strip().lower()
        if text in ("true", "1", "yes", "on"):
            return True
        if text in ("false", "0", "no", "off"):
            return False
        raise ValueError("must be true or false")
    if ftype == "dropdown":
        text = str(value).strip()
        if text not in (field["options"] or []):
            raise ValueError(f"must be one of: {', '.join(field['options'] or [])}")
        return text
    raise ValueError(f"unknown field type {ftype}")


def validate_values(fields: list[dict], values: dict[str, Any], partial: bool = False) -> dict[str, Any]:
    """Validate submitted custom field values against the active definitions.

    partial=True (updates): required fields are only checked if they are submitted.
    """
    by_name = {f["name"]: f for f in fields}
    errors: dict[str, str] = {}
    cleaned: dict[str, Any] = {}
    for name in values:
        if name not in by_name:
            errors[name] = "unknown field"
    for name, f in by_name.items():
        submitted = name in values
        raw = values.get(name)
        # Empty strings / None count as missing. An unticked checkbox (False) is a real value.
        if _is_empty(raw):
            if f["required"] and (submitted or not partial):
                errors[name] = f"{f['label']} is required"
            elif submitted:
                cleaned[name] = None  # clear the value
            continue
        try:
            cleaned[name] = coerce_value(f, raw)
        except ValueError as exc:
            errors[name] = f"{f['label']} {exc}"
    if errors:
        raise FieldValidationError(errors)
    return cleaned


def save_values(conn: sqlite3.Connection, ticket_id: int, fields: list[dict], cleaned: dict[str, Any]) -> None:
    by_name = {f["name"]: f for f in fields}
    for name, value in cleaned.items():
        fid = by_name[name]["id"]
        if value is None:
            conn.execute("DELETE FROM ticket_field_values WHERE ticket_id = ? AND field_id = ?", (ticket_id, fid))
        else:
            conn.execute(
                "INSERT INTO ticket_field_values (ticket_id, field_id, value) VALUES (?, ?, ?)"
                " ON CONFLICT (ticket_id, field_id) DO UPDATE SET value = excluded.value",
                (ticket_id, fid, json.dumps(value)),
            )


def load_values(conn: sqlite3.Connection, ticket_id: int) -> dict[str, Any]:
    rows = conn.execute(
        "SELECT f.name, v.value FROM ticket_field_values v JOIN field_definitions f ON f.id = v.field_id WHERE v.ticket_id = ?",
        (ticket_id,),
    )
    return {r["name"]: json.loads(r["value"]) for r in rows}
