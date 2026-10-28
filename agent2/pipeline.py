"""Orchestration: CLI + programmatic pipeline for Agent 2.

A run is:

1. Load companies from Agent 1 (JSON/CSV file or an in-memory list).
2. Discover decision-makers per company (search + Claude extraction).
3. Persist companies, contacts and proofs to SQLite.
4. Run configured email/phone verification.
5. Export contacts to CSV / Excel / JSON.

De-dup: discovery of the same company twice stores a new run with its own
run_id; the previous run's rows remain untouched. Within a run, contacts are
deduplicated per company in ``discovery._dedupe_contacts``.

The database never auto-deletes; only the user's explicit actions in the
dashboard or test helpers remove rows.
"""

from __future__ import annotations

import argparse
import logging
import sys
import uuid
from pathlib import Path
from typing import Optional

from .config import Settings
from .discovery import run_discovery
from .exporters import export_all
from .inputs import load_companies
from .models import Agent1Company, CompanyContacts
from .storage import Database
from .verification import verify_contact_details

logger = logging.getLogger(__name__)


def process_batch(companies: list[Agent1Company], settings: Settings,
                  db: Database, run_id: Optional[str] = None,
                  do_verify: bool = True, do_export: bool = True,
                  anthropic_client=None, search_provider=None,
                  offline: bool = False,
                  stop_on_error: bool = False) -> tuple[list[CompanyContacts], str]:
    """Run discovery, persist, verify and export for a batch of companies.

    Returns ``(results, run_id)``. ``results`` are the ``CompanyContacts``
    records (already persisted); ``run_id`` identifies the run for lookups.
    """
    run_id = run_id or f"run-{uuid.uuid4().hex[:12]}"
    db.start_run(run_id, len(companies))
    for comp in companies:
        db.upsert_company(run_id, comp)

    if offline:
        from .discovery import run_discovery_offline
        results = run_discovery_offline(companies, settings)
    else:
        results = run_discovery(companies, settings, anthropic_client=anthropic_client,
                                search_provider=search_provider, stop_on_error=stop_on_error)

    contacts_total = 0
    for result in results:
        if result.error:
            continue
        db.save_company_contacts(
            run_id, result.company_id,
            _find_company(companies, result.company_id),
            result, result.contacts, proofs=[],
        )
        if do_verify:
            verify_contact_details(run_id, result.company_id, result.contacts, settings)
            for contact in result.contacts:
                db.upsert_contact(run_id, result.company_id, contact)
        contacts_total += len(result.contacts)

    if do_export:
        export_all(results, settings.output_dir, include_rejected=True)

    db.finish_run(run_id, contacts_total, status="completed")
    return results, run_id


def _find_company(companies: list[Agent1Company], company_id: str) -> Agent1Company:
    for comp in companies:
        if comp.company_id == company_id:
            return comp
    return Agent1Company(company_id=company_id, company_name=company_id)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agent2",
        description="Agent 2 - B2B Decision-Maker Contact Finder. "
                    "Reads company records from Agent 1, finds decision-makers and their "
                    "publicly available professional contact details, stores and exports them.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Run contact discovery for a batch")
    run.add_argument("input", help="Path to companies JSON or CSV (Agent 1 output)")
    run.add_argument("--client", "-c", nargs="?", const="a", default=None,
                     help="Optional company_id to re-run just one company (debug)")
    run.add_argument("--no-verify", action="store_true", help="Skip configured verification providers")
    run.add_argument("--no-export", action="store_true", help="Skip CSV/Excel/JSON exports")
    run.add_argument("--run-id", default=None, help="Explicit run id (default: generated)")
    run.add_argument("--db", default=None, help="Database path override")

    export = sub.add_parser("export", help="Export existing contacts from the database")
    export.add_argument("--run-id", default=None, help="Restrict to one run")
    export.add_argument("--db", default=None, help="Database path override")
    export.add_argument("--out", default=None, help="Output directory (default: settings.output_dir)")
    export.add_argument("--approved-only", action="store_true",
                        help="Export only approved contacts")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = build_parser().parse_args(argv)
    settings = Settings.from_env()
    if args.db:
        settings.database_path = args.db

    if args.command == "run":
        problems = settings.problems_for_discovery()
        # Fully-offline mode needs no keys: mock search + built-in rule-based
        # extraction (deterministic; no network; exercises the real grounding).
        if settings.search_provider == "mock":
            problems = []
            logger.info("Offline mode: mock search provider + built-in rule-based extraction.")
        if problems:
            for p in problems:
                logger.error("- %s", p)
            logger.error("Fix configuration in .env, or use SEARCH_PROVIDER=mock to run without a search API key.")
            return 2

        input_path = args.input
        if str(input_path) == "-":
            import json, sys as _sys
            text = _sys.stdin.read()
            from .inputs import load_companies_from_json_text
            result = load_companies_from_json_text(text)
            companies = result.companies
            if result.errors:
                logger.warning("Ignoring %d malformed records: %s", len(result.errors), "; ".join(result.errors[:3]))
        else:
            loaded = load_companies(Path(input_path))
            companies = loaded.companies
            if loaded.errors:
                logger.warning("Ignoring %d malformed records: %s", len(loaded.errors), "; ".join(loaded.errors[:3]))
            if not companies:
                logger.error("No valid companies loaded from %s", input_path)
                return 1

        logger.info("Loaded %d companies from %s", len(companies), input_path)

        if args.client:
            companies = [c for c in companies if c.company_id == args.client]
            if not companies:
                logger.error("No company with id %r in input", args.client)
                return 1

        from .discovery import run_discovery_offline
        offline = settings.search_provider == "mock"

        try:
            with Database(settings.database_path) as db:
                results, run_id = process_batch(
                    companies, settings, db,
                    run_id=args.run_id,
                    do_verify=not args.no_verify,
                    do_export=not args.no_export,
                    search_provider=None,
                    offline=offline,
                )
        except Exception as exc:
            logger.exception("Batch failed")
            return 1

        failed = [f"{r.company_id} ({r.company_name}): {r.error}" for r in results if r.error]
        total = sum(len(r.contacts) for r in results)
        logger.info("Run %s: %d companies, %d contacts found", run_id, len(results), total)
        for f in failed:
            logger.warning("  failed: %s", f)
        if not args.no_export:
            logger.info("Exports written to %s", settings.output_dir)
        return 0

    if args.command == "export":
        with Database(settings.database_path) as db:
            run_id = args.run_id or (db.list_runs(1)[0]["run_id"] if db.list_runs(1) else None)
            if not run_id:
                logger.error("No runs in the database - run discovery first.")
                return 1
            rows = db.list_contacts(run_id)
            # group rows back into results shape for the exporter
            by_company: dict[str, dict[str, object]] = {}
            for row in rows:
                by_company.setdefault(row["company_id"], {
                    "company_id": row["company_id"],
                    "company_name": row["company_name"],
                    "contacts": [],
                })
                contact = {
                    "company_id": row["company_id"],
                    "name": row.get("name", ""),
                    "title": row.get("title", ""),
                    "department": row.get("department", ""),
                    "country": row.get("country", ""),
                    "linkedin_url": row.get("linkedin_url", ""),
                    "work_email": row.get("work_email", ""),
                    "email_status": row.get("email_status", ""),
                    "business_phone": row.get("business_phone", ""),
                    "phone_status": row.get("phone_status", ""),
                    "official_company_contact_page": row.get("official_company_contact_page", ""),
                    "source_identity_url": row.get("source_identity_url", ""),
                    "source_contact_url": row.get("source_contact_url", ""),
                    "role_relevance": row.get("role_relevance", ""),
                    "source_urls": _parse_json_list(row.get("source_urls", "[]")),
                    "confidence_score": row.get("confidence_score", 0),
                    "verification_notes": row.get("verification_notes", ""),
                    "research_date": row.get("research_date", ""),
                    "review_state": row.get("review_state", "pending"),
                    "source_tags": _parse_json_list(row.get("source_tags", "[]")),
                }
                by_company[row["company_id"]]["contacts"].append(contact)
            from .models import CompanyContacts, Contact
            results = [
                CompanyContacts(company_id=g["company_id"], company_name=g["company_name"],
                                contacts=[Contact(**cont) for cont in g["contacts"]])
                for g in by_company.values()
            ]
            out = args.out or settings.output_dir
            paths = export_all(results, out, include_rejected=not args.approved_only)
            logger.info("Exporting run %s (%d rows)", run_id, len(rows))
            for fmt, p in paths.items():
                logger.info("Wrote %s -> %s", fmt.upper(), p)
        return 0

    return 0


def _parse_json_list(value) -> list:
    """Parse a JSON list stored as text (SQLite), tolerating empty/None."""
    import json as _json

    if not value:
        return []
    if isinstance(value, list):
        return value
    try:
        return _json.loads(value)
    except (ValueError, TypeError):
        return []


if __name__ == "__main__":
    sys.exit(main())