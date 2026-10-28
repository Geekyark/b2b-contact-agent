"""SQLite storage for Agent 2.

The database persists:

- companies: the Agent 1 company records that were ingested (with an
  Agent 2 run id so re-runs are traceable),
- companies_input: one marked-up copy per batch, preserving Agent 1's
  original records even when the company row is updated,
- contacts: discovered contacts, one row per contact per company,
- proofs: source-backed evidence rows for each contact (identity, role,
  email, phone, LinkedIn, relevance),
- runs: the discovery run metadata.

Everything here is plain SQL through the stdlib ``sqlite3``; no ORM.
"""

from __future__ import annotations

import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional

from .models import Agent1Company, CompanyContacts, Contact

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    company_count INTEGER NOT NULL DEFAULT 0,
    contact_count INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'running',
    notes TEXT
);

CREATE TABLE IF NOT EXISTS companies (
    run_id TEXT NOT NULL,
    company_id TEXT NOT NULL,
    company_name TEXT NOT NULL,
    website TEXT DEFAULT '',
    country TEXT DEFAULT '',
    city TEXT DEFAULT '',
    industry TEXT DEFAULT '',
    company_size TEXT DEFAULT '',
    business_model TEXT DEFAULT '',
    products_services TEXT DEFAULT '',
    technology_requirements TEXT DEFAULT '',
    business_challenges TEXT DEFAULT '',
    reasons_potential_client TEXT DEFAULT '',
    research_source_urls TEXT DEFAULT '[]',
    existing_identifiers TEXT DEFAULT '{}',
    extra_json TEXT DEFAULT '{}',
    PRIMARY KEY (run_id, company_id)
);

CREATE TABLE IF NOT EXISTS companies_input (
    run_id TEXT NOT NULL,
    company_id TEXT NOT NULL,
    payload TEXT NOT NULL,
    inserted_at TEXT NOT NULL,
    PRIMARY KEY (run_id, company_id)
);

CREATE TABLE IF NOT EXISTS contacts (
    run_id TEXT NOT NULL,
    company_id TEXT NOT NULL,
    contact_key TEXT NOT NULL,
    name TEXT,
    title TEXT,
    department TEXT,
    country TEXT,
    linkedin_url TEXT,
    work_email TEXT,
    email_status TEXT,
    business_phone TEXT,
    phone_status TEXT,
    official_company_contact_page TEXT,
    source_identity_url TEXT,
    source_contact_url TEXT,
    role_relevance TEXT,
    source_urls TEXT DEFAULT '[]',
    confidence_score REAL DEFAULT 0,
    verification_notes TEXT,
    research_date TEXT,
    review_state TEXT DEFAULT 'pending',
    source_tags TEXT DEFAULT '[]',
    PRIMARY KEY (run_id, company_id, contact_key)
);

CREATE TABLE IF NOT EXISTS proofs (
    run_id TEXT NOT NULL,
    company_id TEXT NOT NULL,
    contact_key TEXT NOT NULL,
    fact_type TEXT NOT NULL,
    value TEXT,
    source_url TEXT,
    snippet TEXT,
    PRIMARY KEY (run_id, company_id, contact_key, fact_type, value)
);

CREATE INDEX IF NOT EXISTS idx_contacts_email ON contacts (work_email);
CREATE INDEX IF NOT EXISTS idx_contacts_review ON contacts (review_state);
"""


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Database:
    """Thin wrapper around sqlite3 with a per-thread connection.

    SQLite connections are bound to the thread that created them. The wrapper
    keeps one connection per thread so the same ``Database`` object can be
    used from the Streamlit main thread and from FastAPI worker threads
    without "objects created in a thread" errors.
    """

    def __init__(self, path: str = "data/agent2.db"):
        self.path = str(path)
        parent = Path(self.path).parent
        if str(parent) not in (".", ""):
            parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        # force an early connection so schema is created and errors surface fast
        self._conn()

    # -- connection helpers ------------------------------------------------
    def _conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.path)
            conn.row_factory = sqlite3.Row
            conn.executescript(SCHEMA)
            conn.commit()
            self._local.conn = conn
        return conn

    @contextmanager
    def cursor(self) -> Iterator[sqlite3.Cursor]:
        cur = self._conn().cursor()
        try:
            yield cur
            self._conn().commit()
        except Exception:
            self._conn().rollback()
            raise

    def close(self) -> None:
        """Close the current thread's connection (other threads are untouched)."""
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None

    def __enter__(self) -> "Database":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # -- runs --------------------------------------------------------------
    def start_run(self, run_id: str, company_count: int) -> None:
        with self.cursor() as cur:
            cur.execute(
                "INSERT OR REPLACE INTO runs (run_id, started_at, company_count, status) "
                "VALUES (?, ?, ?, 'running')",
                (run_id, now_iso(), company_count),
            )

    def finish_run(self, run_id: str, contact_count: int, status: str = "completed", notes: str = "") -> None:
        with self.cursor() as cur:
            cur.execute(
                "UPDATE runs SET finished_at = ?, contact_count = ?, status = ?, notes = ? WHERE run_id = ?",
                (now_iso(), contact_count, status, notes, run_id),
            )

    def list_runs(self, limit: int = 50) -> list[dict[str, Any]]:
        with self.cursor() as cur:
            rows = cur.execute(
                "SELECT * FROM runs ORDER BY started_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    def get_run(self, run_id: str) -> Optional[dict[str, Any]]:
        with self.cursor() as cur:
            row = cur.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        return dict(row) if row else None

    # -- companies ---------------------------------------------------------
    def upsert_company(self, run_id: str, company: Agent1Company) -> None:
        import json as _json

        with self.cursor() as cur:
            cur.execute(
                """
                INSERT INTO companies (
                    run_id, company_id, company_name, website, country, city, industry,
                    company_size, business_model, products_services, technology_requirements,
                    business_challenges, reasons_potential_client, research_source_urls,
                    existing_identifiers, extra_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id, company_id) DO UPDATE SET
                    company_name = excluded.company_name,
                    website = excluded.website,
                    country = excluded.country,
                    city = excluded.city,
                    industry = excluded.industry,
                    company_size = excluded.company_size,
                    business_model = excluded.business_model,
                    products_services = excluded.products_services,
                    technology_requirements = excluded.technology_requirements,
                    business_challenges = excluded.business_challenges,
                    reasons_potential_client = excluded.reasons_potential_client,
                    research_source_urls = excluded.research_source_urls,
                    existing_identifiers = excluded.existing_identifiers,
                    extra_json = excluded.extra_json
                """,
                (
                    run_id, company.company_id, company.company_name, company.website,
                    company.country, company.city, company.industry, company.company_size,
                    company.business_model, company.products_services,
                    company.technology_requirements, company.business_challenges,
                    company.reasons_potential_client,
                    _json.dumps(company.company_research_source_urls),
                    _json.dumps(company.existing_company_identifiers),
                    _json.dumps(company.model_dump(exclude={"company_id", "company_name"})),
                ),
            )
            cur.execute(
                "INSERT OR REPLACE INTO companies_input (run_id, company_id, payload, inserted_at) "
                "VALUES (?, ?, ?, ?)",
                (run_id, company.company_id, _json.dumps(company.model_dump()), now_iso()),
            )

    # -- contacts + proofs -------------------------------------------------
    def upsert_contact(self, run_id: str, company_id: str, contact: Contact) -> None:
        import json as _json

        key = contact_key(contact.name, contact.work_email, contact.linkedin_url)
        with self.cursor() as cur:
            cur.execute(
                """
                INSERT INTO contacts (
                    run_id, company_id, contact_key, name, title, department, country,
                    linkedin_url, work_email, email_status, business_phone, phone_status,
                    official_company_contact_page, source_identity_url, source_contact_url,
                    role_relevance, source_urls, confidence_score, verification_notes,
                    research_date, review_state, source_tags
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id, company_id, contact_key) DO UPDATE SET
                    name = excluded.name,
                    title = excluded.title,
                    department = excluded.department,
                    country = excluded.country,
                    linkedin_url = excluded.linkedin_url,
                    work_email = excluded.work_email,
                    email_status = excluded.email_status,
                    business_phone = excluded.business_phone,
                    phone_status = excluded.phone_status,
                    official_company_contact_page = excluded.official_company_contact_page,
                    source_identity_url = excluded.source_identity_url,
                    source_contact_url = excluded.source_contact_url,
                    role_relevance = excluded.role_relevance,
                    source_urls = excluded.source_urls,
                    confidence_score = excluded.confidence_score,
                    verification_notes = excluded.verification_notes,
                    research_date = excluded.research_date,
                    review_state = excluded.review_state,
                    source_tags = excluded.source_tags
                """,
                (
                    run_id, company_id, key, contact.name, contact.title, contact.department,
                    contact.country, contact.linkedin_url, contact.work_email,
                    contact.email_status.value if hasattr(contact.email_status, "value") else contact.email_status,
                    contact.business_phone,
                    contact.phone_status.value if hasattr(contact.phone_status, "value") else contact.phone_status,
                    contact.official_company_contact_page, contact.source_identity_url,
                    contact.source_contact_url, contact.role_relevance,
                    _json.dumps(contact.source_urls), contact.confidence_score,
                    contact.verification_notes, contact.research_date,
                    contact.review_state.value if hasattr(contact.review_state, "value") else contact.review_state,
                    _json.dumps(getattr(contact, "source_tags", [])),
                ),
            )

    def save_company_contacts(self, run_id: str, company_id: str,
                              company: Agent1Company, result: CompanyContacts,
                              contacts: Iterable[Contact], proofs: list[dict[str, str]]) -> None:
        """Persist companies, contacts and proofs for one company's discovery."""
        self.upsert_company(run_id, company)
        for c in contacts:
            self.upsert_contact(run_id, company_id, c)
        for p in proofs:
            self.upsert_proof(run_id, company_id, p.get("contact_key", ""), p)

    def upsert_proof(self, run_id: str, company_id: str, contact_key_value: str | None,
                     proof: dict[str, str]) -> None:
        with self.cursor() as cur:
            cur.execute(
                """
                INSERT OR REPLACE INTO proofs
                    (run_id, company_id, contact_key, fact_type, value, source_url, snippet)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (run_id, company_id, contact_key_value or "", proof.get("fact_type", ""),
                 proof.get("value", ""), proof.get("source_url", ""), proof.get("snippet", "")),
            )

    def get_proofs(self, run_id: str, company_id: str, contact_key_value: str) -> list[dict[str, Any]]:
        with self.cursor() as cur:
            rows = cur.execute(
                "SELECT fact_type, value, source_url, snippet FROM proofs "
                "WHERE run_id = ? AND company_id = ? AND contact_key = ?",
                (run_id, company_id, contact_key_value),
            ).fetchall()
        return [dict(r) for r in rows]

    # -- queries ------------------------------------------------------------
    def list_companies(self, run_id: Optional[str] = None) -> list[dict[str, Any]]:
        with self.cursor() as cur:
            if run_id:
                rows = cur.execute(
                    "SELECT DISTINCT company_id, company_name, website, country, city, industry FROM companies WHERE run_id = ? ORDER BY company_name",
                    (run_id,),
                ).fetchall()
            else:
                rows = cur.execute(
                    "SELECT DISTINCT company_id, company_name, website, country, city, industry FROM companies ORDER BY company_name"
                ).fetchall()
        return [dict(r) for r in rows]

    def list_contacts(self, run_id: Optional[str] = None,
                      company_id: Optional[str] = None,
                      review_state: Optional[str] = None,
                      email_status: Optional[str] = None,
                      country: Optional[str] = None,
                      title_contains: Optional[str] = None) -> list[dict[str, Any]]:
        sql = [
            "SELECT c.*, co.company_name, co.website, co.country AS company_country FROM contacts c "
            "LEFT JOIN companies co ON co.run_id = c.run_id AND co.company_id = c.company_id WHERE 1=1"
        ]
        params: list[Any] = []
        if run_id:
            sql.append("AND c.run_id = ?")
            params.append(run_id)
        if company_id:
            sql.append("AND c.company_id = ?")
            params.append(company_id)
        if review_state:
            sql.append("AND c.review_state = ?")
            params.append(review_state)
        if email_status:
            sql.append("AND c.email_status = ?")
            params.append(email_status)
        if country:
            sql.append("AND (co.country LIKE ? OR c.country LIKE ?)")
            params += [f"%{country}%", f"%{country}%"]
        if title_contains:
            sql.append("AND c.title LIKE ?")
            params.append(f"%{title_contains}%")
        sql.append("ORDER BY co.company_name, c.name")
        with self.cursor() as cur:
            rows = cur.execute(" ".join(sql), params).fetchall()
        return [dict(r) for r in rows]

    def count_contacts(self, run_id: Optional[str] = None) -> int:
        with self.cursor() as cur:
            if run_id:
                row = cur.execute("SELECT COUNT(*) AS n FROM contacts WHERE run_id = ?", (run_id,)).fetchone()
            else:
                row = cur.execute("SELECT COUNT(*) AS n FROM contacts").fetchone()
        return int(row["n"])

    def set_review_state(self, run_id: str, company_id: str, contact_key_value: str,
                         review_state: str) -> None:
        allowed = {"pending", "approved", "rejected"}
        if review_state not in allowed:
            raise ValueError(f"Invalid review state: {review_state!r}")
        with self.cursor() as cur:
            cur.execute(
                "UPDATE contacts SET review_state = ? WHERE run_id = ? AND company_id = ? AND contact_key = ?",
                (review_state, run_id, company_id, contact_key_value),
            )

    def update_contact_fields(self, run_id: str, company_id: str, contact_key_value: str,
                              fields: dict[str, Any]) -> None:
        """Dashboards can edit contact details before approving. Whitelisted fields only."""
        allowed = {"name", "title", "department", "country", "linkedin_url", "work_email",
                   "email_status", "business_phone", "phone_status", "official_company_contact_page",
                   "role_relevance", "verification_notes"}
        sets = [f"{k} = ?" for k in fields if k in allowed]
        if not sets:
            return
        params = [fields[k] for k in fields if k in allowed] + [run_id, company_id, contact_key_value]
        with self.cursor() as cur:
            cur.execute(
                f"UPDATE contacts SET {', '.join(sets)} WHERE run_id = ? AND company_id = ? AND contact_key = ?",
                params,
            )

    def delete_contacts(self, run_id: str, company_id: str) -> None:
        with self.cursor() as cur:
            cur.execute("DELETE FROM contacts WHERE run_id = ? AND company_id = ?", (run_id, company_id))
            cur.execute("DELETE FROM proofs WHERE run_id = ? AND company_id = ?", (run_id, company_id))


def contact_key(name: str, email: str, linkedin: str) -> str:
    """Stable dedupe key for a contact within a company."""
    import hashlib

    raw = "|".join([(name or "").strip().lower(), (email or "").strip().lower(),
                    (linkedin or "").strip().lower().rstrip("/")])
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]