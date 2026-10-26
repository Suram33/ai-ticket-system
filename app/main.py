"""FastAPI application: REST API + the single-page web UI."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import dynamic_fields as df
from .config import Settings
from .database import Database
from .ml.predictor import SolutionPredictor
from .rag.embeddings import create_embedder
from .rag.knowledge_base import KnowledgeBase
from .rag.llm import OllamaLLM
from .rag.pipeline import ResponseGenerator
from .schemas import (FieldDefinitionCreate, FieldDefinitionUpdate, PredictRequest, Priority, ResponseUpdate, Status,
                      TicketCreate, TicketUpdate)
from .tickets import NotFound, TicketService

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
STATIC_DIR = Path(__file__).resolve().parent / "static"


def build_services(settings: Settings) -> dict:
    db = Database(settings.database_path)
    db.init(seed_fields=settings.seed_default_fields)
    predictor = SolutionPredictor(settings.model_path, settings.training_data_path, settings.ml_confidence_threshold).load()
    embedder = create_embedder(settings.embedding_backend, settings.embedding_model, settings.ollama_url, settings.ollama_embedding_model)
    kb = KnowledgeBase(settings.kb_dir, embedder).build()
    llm = OllamaLLM(settings.ollama_url, settings.ollama_model, settings.llm_timeout_seconds, settings.llm_temperature) if settings.llm_enabled else None
    generator = ResponseGenerator(kb, llm, settings.rag_top_k, settings.rag_min_score)
    return {"db": db, "predictor": predictor, "kb": kb, "llm": llm, "generator": generator,
            "tickets": TicketService(db, predictor, generator)}


def create_app(settings: Optional[Settings] = None, services: Optional[dict] = None) -> FastAPI:
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.services = services or build_services(settings)
        yield

    app = FastAPI(title="AI Customer Ticket Response System", version="1.0.0", lifespan=lifespan)

    def svc(request: Request) -> dict:
        return request.app.state.services

    @app.exception_handler(NotFound)
    async def not_found(_: Request, exc: NotFound):
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    @app.exception_handler(df.FieldValidationError)
    async def field_error(_: Request, exc: df.FieldValidationError):
        return JSONResponse(status_code=422, content={"detail": "Invalid custom field values", "errors": exc.errors})

    # ---------------- system ----------------
    @app.get("/api/health")
    def health(request: Request):
        s = svc(request)
        llm = s["llm"]
        return {
            "status": "ok",
            "ml": {"available": s["predictor"].available, "error": s["predictor"].error, "metadata": s["predictor"].metadata},
            "llm": {"enabled": llm is not None, "model": llm.name if llm else None, **(llm.status() if llm else {"reachable": False})},
            "rag": {"embedding_backend": s["kb"].embedder.name, "documents": s["kb"].documents, "chunks": len(s["kb"].chunks)},
        }

    # ---------------- tickets ----------------
    @app.get("/api/tickets")
    def list_tickets(request: Request, status: Optional[Status] = None, category: Optional[str] = None,
                     priority: Optional[Priority] = None, q: Optional[str] = None, include_inactive: bool = False,
                     limit: int = Query(100, ge=1, le=500), offset: int = Query(0, ge=0)):
        return svc(request)["tickets"].list(status, category, priority, q, include_inactive, limit, offset)

    @app.post("/api/tickets", status_code=201)
    def create_ticket(request: Request, body: TicketCreate, generate: Optional[bool] = None):
        do_generate = settings.auto_generate_on_create if generate is None else generate
        return svc(request)["tickets"].create(body.model_dump(), generate=do_generate)

    @app.get("/api/tickets/{ticket_id}")
    def get_ticket(request: Request, ticket_id: int):
        return svc(request)["tickets"].get(ticket_id)

    @app.api_route("/api/tickets/{ticket_id}", methods=["PUT", "PATCH"])
    def update_ticket(request: Request, ticket_id: int, body: TicketUpdate):
        return svc(request)["tickets"].update(ticket_id, body.model_dump(exclude_unset=True))

    @app.delete("/api/tickets/{ticket_id}")
    def delete_ticket(request: Request, ticket_id: int, hard: bool = False):
        """Default is a soft delete (deactivate). ?hard=true removes the row permanently."""
        if hard:
            svc(request)["tickets"].delete(ticket_id)
            return {"deleted": True, "id": ticket_id}
        return svc(request)["tickets"].deactivate(ticket_id)

    @app.post("/api/tickets/{ticket_id}/restore")
    def restore_ticket(request: Request, ticket_id: int):
        return svc(request)["tickets"].restore(ticket_id)

    @app.post("/api/tickets/{ticket_id}/predict")
    def repredict(request: Request, ticket_id: int):
        return svc(request)["tickets"].predict(ticket_id)

    @app.post("/api/tickets/{ticket_id}/generate")
    def regenerate(request: Request, ticket_id: int):
        return svc(request)["tickets"].generate(ticket_id)

    @app.put("/api/tickets/{ticket_id}/response")
    def save_response(request: Request, ticket_id: int, body: ResponseUpdate):
        return svc(request)["tickets"].save_response(ticket_id, body.final_response, body.approve)

    # ---------------- dynamic fields ----------------
    @app.get("/api/fields")
    def list_fields(request: Request, include_inactive: bool = False):
        with svc(request)["db"].session() as conn:
            return df.list_fields(conn, include_inactive)

    @app.post("/api/fields", status_code=201)
    def create_field(request: Request, body: FieldDefinitionCreate):
        with svc(request)["db"].session() as conn:
            return df.create_field(conn, body.model_dump())

    @app.put("/api/fields/{field_id}")
    def update_field(request: Request, field_id: int, body: FieldDefinitionUpdate):
        with svc(request)["db"].session() as conn:
            field = df.update_field(conn, field_id, body.model_dump(exclude_unset=True))
        if not field:
            raise HTTPException(404, f"Field {field_id} not found")
        return field

    @app.delete("/api/fields/{field_id}")
    def deactivate_field(request: Request, field_id: int):
        """Fields are deactivated, not deleted, so old tickets keep their values."""
        with svc(request)["db"].session() as conn:
            field = df.update_field(conn, field_id, {"is_active": False})
        if not field:
            raise HTTPException(404, f"Field {field_id} not found")
        return field

    # ---------------- AI helpers ----------------
    @app.post("/api/predict")
    def predict(request: Request, body: PredictRequest):
        predictor = svc(request)["predictor"]
        try:
            return predictor.predict(body.subject, body.description).to_dict()
        except ValueError as exc:
            raise HTTPException(422, str(exc))
        except RuntimeError as exc:
            raise HTTPException(503, str(exc))

    @app.post("/api/ml/retrain")
    def retrain(request: Request):
        try:
            return svc(request)["predictor"].retrain()
        except Exception as exc:
            raise HTTPException(500, f"Training failed: {exc}")

    @app.get("/api/kb/search")
    def kb_search(request: Request, q: str = Query(..., min_length=1), top_k: int = Query(3, ge=1, le=10)):
        return [r.to_dict() for r in svc(request)["kb"].search(q, top_k=top_k)]

    # ---------------- UI ----------------
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(STATIC_DIR / "index.html")

    return app


app = create_app()
