-- SQLite schema for the AI Customer Ticket Response System.
-- The app runs this automatically on startup (CREATE ... IF NOT EXISTS),
-- so no manual setup is needed. It is kept here as documentation and for
-- anyone who wants to create the database by hand:  sqlite3 tickets.db < schema.sql

PRAGMA foreign_keys = ON;

-- Core ticket record. The ML and LLM outputs are stored on the ticket itself.
CREATE TABLE IF NOT EXISTS tickets (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_name           TEXT    NOT NULL,
    subject                 TEXT    NOT NULL,
    description             TEXT    NOT NULL DEFAULT '',
    category                TEXT,
    priority                TEXT    NOT NULL DEFAULT 'Medium'
                                    CHECK (priority IN ('Low', 'Medium', 'High', 'Urgent')),
    status                  TEXT    NOT NULL DEFAULT 'Open'
                                    CHECK (status IN ('Open', 'In Progress', 'Resolved', 'Closed')),

    -- Step 3-4: ML prediction
    predicted_solution      TEXT,
    prediction_confidence   REAL,
    predicted_category      TEXT,
    prediction_error        TEXT,

    -- Step 5-6: RAG + LLM draft
    ai_response             TEXT,
    response_source         TEXT,     -- 'llm' (Ollama) or 'fallback' (grounded template)
    response_model          TEXT,
    rag_sources             TEXT,     -- JSON list of retrieved knowledge-base chunks with scores
    generation_error        TEXT,
    grounding_report        TEXT,     -- JSON: which reply sentences are not backed by the knowledge base
    generated_at            TEXT,

    -- Step 7-8: agent review
    final_response          TEXT,
    response_status         TEXT    NOT NULL DEFAULT 'none'
                                    CHECK (response_status IN ('none', 'draft', 'approved')),

    is_active               INTEGER NOT NULL DEFAULT 1,
    created_at              TEXT    NOT NULL,
    updated_at              TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_tickets_status ON tickets (status);
CREATE INDEX IF NOT EXISTS idx_tickets_active ON tickets (is_active);

-- Dynamic field definitions. Adding a field is an INSERT, not a schema change.
CREATE TABLE IF NOT EXISTS field_definitions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT    NOT NULL UNIQUE,      -- machine key, e.g. "order_number"
    label       TEXT    NOT NULL,             -- shown in the form
    field_type  TEXT    NOT NULL CHECK (field_type IN ('text', 'dropdown', 'number', 'date', 'checkbox')),
    required    INTEGER NOT NULL DEFAULT 0,
    options     TEXT,                         -- JSON list, used by dropdown fields
    help_text   TEXT,
    sort_order  INTEGER NOT NULL DEFAULT 0,
    is_active   INTEGER NOT NULL DEFAULT 1,
    created_at  TEXT    NOT NULL
);

-- Values of dynamic fields per ticket (entity-attribute-value table).
CREATE TABLE IF NOT EXISTS ticket_field_values (
    ticket_id   INTEGER NOT NULL REFERENCES tickets (id) ON DELETE CASCADE,
    field_id    INTEGER NOT NULL REFERENCES field_definitions (id) ON DELETE CASCADE,
    value       TEXT,                         -- stored as text, typed via field_definitions.field_type
    PRIMARY KEY (ticket_id, field_id)
);
