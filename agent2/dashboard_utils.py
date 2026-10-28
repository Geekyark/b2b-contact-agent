"""Helpers used by the Streamlit dashboard (kept out of dashboard.py for testability)."""

from __future__ import annotations

import logging
import uuid
from typing import Optional

from .config import Settings
from .discovery import run_discovery
from .models import Agent1Company
from .storage import Database
from .verification import verify_contact_details

logger = logging.getLogger(__name__)


def load_database(settings: Settings) -> Database:
    """Open (and create if needed) the SQLite database used by the dashboard."""
    return Database(settings.database_path)


def start_discovery(companies: list[Agent1Company], settings: Settings, db: Database) -> tuple[str, str]:
    """Synchronous wrapper: discover, verify, persist. Returns (run_id, error)."""
    run_id = f"ui-{uuid.uuid4().hex[:12]}"
    db.start_run(run_id, len(companies))
    for comp in companies:
        db.upsert_company(run_id, comp)
    try:
        if settings.search_provider == "mock":
            from .discovery import run_discovery_offline
            results = run_discovery_offline(companies, settings)
        else:
            results = run_discovery(companies, settings)
    except Exception as exc:
        logger.exception("UI discovery failed")
        db.finish_run(run_id, 0, status="error", notes=str(exc))
        return run_id, str(exc)
    total = 0
    for result in results:
        if result.error:
            continue
        company = next((c for c in companies if c.company_id == result.company_id), None)
        if company is None:
            continue
        verify_contact_details(run_id, result.company_id, result.contacts, settings)
        db.save_company_contacts(run_id, result.company_id, company, result,
                                 result.contacts, proofs=[])
        for contact in result.contacts:
            db.upsert_contact(run_id, result.company_id, contact)
        total += len(result.contacts)
    db.finish_run(run_id, total, status="completed")
    return run_id, ""