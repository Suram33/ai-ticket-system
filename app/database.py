"""SQLite access. Plain sqlite3 keeps the data layer small and easy to follow."""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

SCHEMA_PATH = Path(__file__).resolve().parent.parent / "schema.sql"

DEFAULT_FIELDS = [
    {"name": "contact_channel", "label": "Contact channel", "field_type": "dropdown", "required": True,
     "options": ["Email", "Phone", "Chat", "Web form"], "help_text": "How the customer reached us", "sort_order": 1},
    {"name": "order_number", "label": "Order number", "field_type": "text", "required": False,
     "options": None, "help_text": "If the ticket is about an order", "sort_order": 2},
    {"name": "order_amount", "label": "Order amount", "field_type": "number", "required": False,
     "options": None, "help_text": "Amount in USD", "sort_order": 3},
    {"name": "purchase_date", "label": "Purchase date", "field_type": "date", "required": False,
     "options": None, "help_text": None, "sort_order": 4},
    {"name": "vip_customer", "label": "VIP customer", "field_type": "checkbox", "required": False,
     "options": None, "help_text": None, "sort_order": 5},
]


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


class Database:
    def __init__(self, path: Path):
        self.path = Path(path)

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    @contextmanager
    def session(self) -> Iterator[sqlite3.Connection]:
        conn = self.connect()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def init(self, seed_fields: bool = True) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.session() as conn:
            conn.executescript(SCHEMA_PATH.read_text())
            if seed_fields:
                count = conn.execute("SELECT COUNT(*) FROM field_definitions").fetchone()[0]
                if count == 0:
                    for f in DEFAULT_FIELDS:
                        conn.execute(
                            "INSERT INTO field_definitions (name, label, field_type, required, options, help_text, sort_order, created_at)"
                            " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                            (f["name"], f["label"], f["field_type"], int(f["required"]),
                             json.dumps(f["options"]) if f["options"] else None, f["help_text"], f["sort_order"], now_iso()),
                        )
