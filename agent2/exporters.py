"""Export discovered contacts to CSV, Excel and JSON.

All exporters emit one row per contact and always preserve the original
``company_id`` from Agent 1, so the contact file can be joined back to Agent
1's company records.

Export structure (per row):
    company_id, company_name, customer_id, name, title, department, country,
    linkedin_url, work_email, email_status, business_phone, phone_status,
    official_company_contact_page, source_identity_url, source_contact_url,
    role_relevance, source_urls, confidence_score, verification_notes,
    research_date, review_state, source_tags

``customer_id`` mirrors ``company_id`` for integrations that expect a
customer key on every row; both values are always the Agent 1 identifier.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd

from .models import CompanyContacts

# Column order for CSV/Excel. Every key must exist in the flat rows.
EXPORT_COLUMNS = [
    "company_id",
    "company_name",
    "customer_id",
    "name",
    "title",
    "department",
    "country",
    "linkedin_url",
    "work_email",
    "email_status",
    "business_phone",
    "phone_status",
    "official_company_contact_page",
    "source_identity_url",
    "source_contact_url",
    "role_relevance",
    "source_urls",
    "confidence_score",
    "verification_notes",
    "research_date",
    "review_state",
    "source_tags",
]


def rows_from_results(results: Sequence[CompanyContacts], *,
                      include_rejected: bool = True) -> list[dict[str, Any]]:
    """Flatten discovery results into one row per contact.

    Parameters
    ----------
    include_rejected : if False, rows whose review_state == 'rejected' are
        dropped (used by "export approved" in the dashboard).
    """
    rows: list[dict[str, Any]] = []
    for result in results:
        for contact in result.contacts:
            if not include_rejected and contact.review_state == "rejected":
                continue
            if contact.name == "":
                continue  # never export empty placeholder rows
            row = contact.to_row()
            row["company_id"] = result.company_id
            row["customer_id"] = result.company_id
            row["company_name"] = result.company_name
            rows.append(row)
    return rows


def _prepare(rows: Iterable[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(list(rows))
    if df.empty:
        for col in EXPORT_COLUMNS:
            df[col] = ""
    else:
        for col in EXPORT_COLUMNS:
            if col not in df.columns:
                df[col] = ""
        df = df[EXPORT_COLUMNS]
    return df


def export_csv(results: Sequence[CompanyContacts], path: str | Path, *,
               include_rejected: bool = True) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = _prepare(rows_from_results(results, include_rejected=include_rejected))
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def export_excel(results: Sequence[CompanyContacts], path: str | Path, *,
                 include_rejected: bool = True) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = _prepare(rows_from_results(results, include_rejected=include_rejected))
    df.to_excel(path, index=False, sheet_name="Contacts")
    return path


def export_json(results: Sequence[CompanyContacts], path: str | Path, *,
                include_rejected: bool = True) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "1.0",
        "exported_at": __import__("datetime").datetime.now().isoformat(timespec="seconds"),
        "companies": [
            {
                "company_id": r.company_id,
                "company_name": r.company_name,
                "contacts": [
                    c.model_dump() for c in r.contacts
                    if (include_rejected or c.review_state != "rejected") and c.name
                ],
            }
            for r in results
        ],
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def export_all(results: Sequence[CompanyContacts], output_dir: str | Path, *,
               include_rejected: bool = True) -> dict[str, Path]:
    """Export every format into ``output_dir`` and return the created paths."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return {
        "csv": export_csv(results, output_dir / "contacts.csv", include_rejected=include_rejected),
        "xlsx": export_excel(results, output_dir / "contacts.xlsx", include_rejected=include_rejected),
        "json": export_json(results, output_dir / "contacts.json", include_rejected=include_rejected),
    }