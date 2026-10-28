"""Agent-to-agent FastAPI server.

This optional server lets Agent 1 (company research) POST company records
directly to Agent 2 for discovery. It runs in a separate process (or
alongside the dashboard) on ``API_HOST:API_PORT``.

For a key-free demo set ``SEARCH_PROVIDER=mock`` (offline deterministic
extraction); otherwise the real Tavily + Claude providers are used.

Endpoints
---------
POST /discover      — trigger contact discovery for one or more companies
GET  /contacts      — query discovered contacts (filtered)
GET  /contacts/{id} — get a single contact record
GET  /companies     — list companies processed
GET  /runs          — list discovery runs
GET  /health        — health check
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Optional

import fastapi
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .config import Settings
from .discovery import run_discovery
from .inputs import parse_company_dict
from .models import Agent1Company, ContactReviewState
from .storage import Database
from .verification import verify_contact_details

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class DiscoverRequest(BaseModel):
    """POST /discover payload. At minimum ``company_name`` per record."""
    companies: list[dict[str, Any]]


class DiscoverResponse(BaseModel):
    run_id: str
    company_count: int
    status: str
    message: str


class ReviewRequest(BaseModel):
    state: str  # pending | approved | rejected


class ContactResponse(BaseModel):
    run_id: str
    company_id: str
    company_name: str
    contacts: list[dict[str, Any]]


class HealthResponse(BaseModel):
    status: str
    version: str


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------
def create_app(db: Optional[Database] = None, settings: Optional[Settings] = None) -> FastAPI:
    settings = settings or Settings.from_env()
    db = db or Database(settings.database_path)

    app = FastAPI(title="Agent 2 - B2B Contact Discovery API", version="1.0.0")

    # ---- health -----------------------------------------------------------
    @app.get("/health", response_model=HealthResponse)
    def health():
        return {"status": "ok", "version": __import__("agent2").__version__}

    # ---- discover ----------------------------------------------------------
    @app.post("/discover", response_model=DiscoverResponse)
    def discover(req: DiscoverRequest, background_tasks: fastapi.BackgroundTasks):
        # Mock mode is always offline — needs no API keys.
        offline = settings.search_provider == "mock"
        problems = [] if offline else settings.problems_for_discovery()
        if problems:
            raise HTTPException(400, detail="; ".join(problems))

        companies = []
        for i, rec in enumerate(req.companies):
            if not isinstance(rec, dict):
                continue
            comp = parse_company_dict(rec)
            if comp.company_name:
                companies.append(comp)

        if not companies:
            raise HTTPException(400, detail="No valid company records provided.")

        batch_id = f"api-{uuid.uuid4().hex[:12]}"
        db.start_run(batch_id, len(companies))

        background_tasks.add_task(_run_and_store, batch_id, companies, db, settings)
        return DiscoverResponse(
            run_id=batch_id,
            company_count=len(companies),
            status="running",
            message="Discovery started. Check /contacts and /runs for results.",
        )

    # ---- contacts ----------------------------------------------------------
    @app.get("/contacts")
    def list_contacts(
        run_id: Optional[str] = None,
        company_id: Optional[str] = None,
        review_state: Optional[str] = None,
        country: Optional[str] = None,
        title: Optional[str] = None,
    ):
        return {"contacts": db.list_contacts(run_id, company_id, review_state, country, title_contains=title)}

    @app.get("/contacts/{company_id}")
    def get_company_contacts(company_id: str, run_id: Optional[str] = None):
        contacts = db.list_contacts(run_id, company_id=company_id)
        if not contacts:
            raise HTTPException(404, detail=f"No contacts found for company_id={company_id}")
        company_name = contacts[0].get("company_name", "")
        return {"company_id": company_id, "company_name": company_name, "contacts": contacts}

    # ---- companies ---------------------------------------------------------
    @app.get("/companies")
    def list_companies(run_id: Optional[str] = None):
        return {"companies": db.list_companies(run_id)}

    # ---- runs --------------------------------------------------------------
    @app.get("/runs")
    def list_runs(limit: int = 50):
        return {"runs": db.list_runs(limit)}

    # ---- review (edit+approve) ---------------------------------------------
    @app.patch("/contacts/{company_id}/{contact_key}/review")
    def set_review(company_id: str, contact_key: str,
                   req: ReviewRequest, run_id: Optional[str] = None):
        if req.state not in ("pending", "approved", "rejected"):
            raise HTTPException(400, detail=f"Invalid state: {req.state}")
        run = run_id or _latest_run(db)
        if not run:
            raise HTTPException(404, detail="No runs found.")
        db.set_review_state(run, company_id, contact_key, req.state)
        return {"status": "ok"}

    # ---- PATCH contacts update (edit fields) --------------------------------
    @app.patch("/contacts/{company_id}/{contact_key}")
    def update_contact(company_id: str, contact_key: str, fields: dict[str, Any],
                       run_id: Optional[str] = None):
        run = run_id or _latest_run(db)
        if not run:
            raise HTTPException(404, detail="No runs found.")
        db.update_contact_fields(run, company_id, contact_key, fields)
        return {"status": "ok"}

    return app


def _latest_run(db: Database) -> Optional[str]:
    runs = db.list_runs(limit=1)
    return runs[0]["run_id"] if runs else None


def _run_and_store(batch_id: str, companies: list[Agent1Company],
                   db: Database, settings: Settings) -> None:
    """Background task: run discovery, verify, persist."""
    offline = settings.search_provider == "mock"
    try:
        if offline:
            from .discovery import run_discovery_offline
            results = run_discovery_offline(companies, settings)
        else:
            results = run_discovery(companies, settings)
        contacts_count = 0
        for result in results:
            db.upsert_company(batch_id, _find_company(companies, result.company_id))
            if result.error:
                continue
            db.save_company_contacts(batch_id, result.company_id,
                                     _find_company(companies, result.company_id),
                                     result, result.contacts, proofs=[])
            verify_contact_details(batch_id, result.company_id, result.contacts, settings)
            for contact in result.contacts:
                db.upsert_contact(batch_id, result.company_id, contact)
            contacts_count += len(result.contacts)
        db.finish_run(batch_id, contacts_count, status="completed")
    except Exception as exc:
        logger.exception("Background discovery run %s failed", batch_id)
        db.finish_run(batch_id, 0, status="error", notes=str(exc))


def _find_company(companies: list[Agent1Company], company_id: str) -> Agent1Company:
    for comp in companies:
        if comp.company_id == company_id:
            return comp
    return Agent1Company(company_id=company_id, company_name=company_id)


# ---------------------------------------------------------------------------
# Runnable entry point
# ---------------------------------------------------------------------------
def main() -> None:
    import uvicorn

    settings = Settings.from_env()
    app = create_app(settings=settings)
    uvicorn.run(app, host=settings.api_host, port=settings.api_port, log_level="info")


if __name__ == "__main__":
    main()