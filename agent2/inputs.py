"""Input adapters: load company records produced by Agent 1.

Agent 1's exact output format is not yet known to Agent 2, so this module
accepts JSON and CSV with graceful field tolerance:

- Any superset of the known keys is fine (extra fields are preserved on the
  model's ``existing_company_identifiers`` / extra dict).
- ``company_id`` is the *original identifier from Agent 1* and must survive
  round-trips; if it is missing it is generated deterministically from the
  company name so records remain joinable later.
- ``company_name`` is required; rows without it are reported in the load
  result so nothing silently disappears.

Minimal contract to connect Agent 1 automatically (see README):
    export the company data as JSON, one of:
      {"companies": [ {...}, ... ]}            (list wrapper)
      {"company": {...}}                       (single record)
      [ {...}, ... ]                           (raw list)
  or CSV with at least the columns:
      company_name, company_id
  plus any of the other documented fields.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from .models import Agent1Company

# Aliases Agent 1 might use for the same meaning. Keys are checked in order;
# the first non-empty value wins.
FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "company_id": ("company_id", "id", "companyIdentifier", "customer_id", "external_id"),
    "company_name": ("company_name", "name", "companyName", "company", "organization", "organization_name"),
    "website": ("website", "domain", "official_website", "url", "website_url", "site"),
    "country": ("country", "country_name"),
    "city": ("city", "city_name", "location"),
    "industry": ("industry", "sector", "vertical", "industry_name"),
    "company_size": ("company_size", "size", "employee_count", "employees", "companySize", "headcount"),
    "business_model": ("business_model", "businessModel", "business_type", "model"),
    "products_services": ("products_services", "products", "services", "productsAndServices", "offerings"),
    "technology_requirements": ("technology_requirements", "technology", "tech_stack", "technologies", "techRequirements"),
    "business_challenges": ("business_challenges", "challenges", "pain_points", "problems"),
    "reasons_potential_client": ("reasons_potential_client", "potential_client_reason", "reasons", "fit_reason", "why"),
    "company_research_source_urls": ("company_research_source_urls", "research_source_urls", "source_urls", "sources"),
    "existing_company_identifiers": ("existing_company_identifiers", "identifiers", "external_ids"),
}


@dataclass
class LoadResult:
    """Outcome of loading company data from a file or text."""

    companies: list[Agent1Company] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def _pick(row: dict[str, Any], field_name: str) -> Any:
    """Return the first non-empty value for ``field_name`` across its aliases."""
    for alias in FIELD_ALIASES.get(field_name, (field_name,)):
        if alias in row:
            val = row[alias]
            if isinstance(val, list):
                if val:
                    return val
            elif isinstance(val, str) and val.strip():
                return val.strip()
            elif isinstance(val, (int, float)) and val != 0:
                return val
            elif isinstance(val, dict) and val:
                return val
    if field_name == "company_research_source_urls":
        return []
    return ""


def make_company_id(name: str) -> str:
    """Deterministic id from a company name when Agent 1 supplied none."""
    base = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    if not base:
        base = "company"
    return f"{base}-{hashlib.sha1(name.encode('utf-8')).hexdigest()[:6]}"


def parse_company_dict(raw: dict[str, Any]) -> Agent1Company:
    """Map one raw dict (from JSON or a CSV row) onto an Agent1Company."""
    combined: dict[str, Any] = {}
    for field_name in FIELD_ALIASES:
        combined[field_name] = _pick(raw, field_name)
    # Explicitly-named (non-aliased) keys win over alias guessing.
    for key in list(FIELD_ALIASES) + [
        k for k in ("company_id", "company_name", "website", "country", "city")
        if k in raw
    ]:
        if key in raw and raw[key] not in (None, ""):
            combined[key] = raw[key]

    name = str(combined.get("company_name") or "").strip()
    combined["company_id"] = str(combined.get("company_id") or make_company_id(name or "unknown"))
    combined["company_name"] = name

    sources = combined.get("company_research_source_urls")
    if isinstance(sources, str) and sources:
        sources = [u.strip() for u in sources.split(",") if u.strip()]
    combined["company_research_source_urls"] = sources or []

    ids = combined.get("existing_company_identifiers")
    if isinstance(ids, str) and ids:
        ids = _parse_id_string(ids)
    combined["existing_company_identifiers"] = ids or {}

    # Preserve unknown top-level keys as extra context rather than dropping them.
    for key, value in raw.items():
        if key not in combined or combined[key] in ("", [], {}):
            if key not in combined:
                combined[key] = value

    return Agent1Company.model_validate(combined)


def _parse_id_string(raw: str) -> dict[str, str]:
    raw = raw.strip()
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return {str(k): str(v) for k, v in parsed.items()}
    except json.JSONDecodeError:
        pass
    result: dict[str, str] = {}
    for chunk in re.split(r"[;,]", raw):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "=" in chunk:
            k, v = chunk.split("=", 1)
            result[k.strip()] = v.strip()
        else:
            result["value"] = chunk
    return result


def load_companies_from_json_text(text: str) -> LoadResult:
    """Parse company data from a JSON string. See module docstring for shapes."""
    result = LoadResult()
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        result.errors.append(f"JSON parse error: {exc}")
        return result

    if isinstance(data, dict):
        if "companies" in data and isinstance(data["companies"], list):
            records = data["companies"]
        elif "company" in data:
            records = [data["company"]]
        elif "results" in data and isinstance(data["results"], list):
            records = data["results"]
        elif "data" in data and isinstance(data["data"], list):
            records = data["data"]
        else:
            records = [data]
    elif isinstance(data, list):
        records = data
    else:
        result.errors.append("Unsupported JSON shape: expected an object or a list of company objects.")
        return result

    for i, rec in enumerate(records):
        if not isinstance(rec, dict):
            result.errors.append(f"record {i}: expected an object, got {type(rec).__name__}")
            continue
        comp = parse_company_dict(rec)
        if not comp.company_name:
            result.errors.append(f"record {i}: missing company name")
            continue
        result.companies.append(comp)
    return result


def load_companies_from_json_file(path: str | Path) -> LoadResult:
    return load_companies_from_json_text(Path(path).read_text(encoding="utf-8-sig"))


def load_companies_from_csv_file(path: str | Path) -> LoadResult:
    """Load companies from a CSV file with a header row."""
    result = LoadResult()
    try:
        with open(path, newline="", encoding="utf-8-sig") as fh:
            reader = csv.DictReader(fh)
            rows = [dict(r) for r in reader]
    except OSError as exc:
        result.errors.append(f"Could not read file: {exc}")
        return result
    if not rows:
        result.errors.append("CSV file is empty (no data rows).")
        return result
    for raw in rows:
        comp = parse_company_dict(raw)
        if not comp.company_name:
            result.errors.append("one or more rows: missing company name (skipped)")
            continue
        result.companies.append(comp)
    return result


def load_companies_from_dataframe(df: pd.DataFrame) -> LoadResult:
    """Load companies from a pandas DataFrame (used by the dashboard)."""
    result = LoadResult()
    try:
        records = df.replace({pd.NA: None, float("nan"): None}).to_dict(orient="records")
    except Exception as exc:  # pragma: no cover - defensive
        result.errors.append(f"Could not read DataFrame: {exc}")
        return result
    for raw in records:
        comp = parse_company_dict(raw)
        if not comp.company_name:
            result.errors.append("one or more rows: missing company name (skipped)")
            continue
        result.companies.append(comp)
    return result


def load_companies(path: str | Path) -> LoadResult:
    """Load companies from a JSON or CSV file, detecting the format."""
    p = Path(path)
    if p.suffix.lower() == ".csv":
        return load_companies_from_csv_file(p)
    return load_companies_from_json_file(p)


def serialize_companies(companies: Sequence[Agent1Company]) -> list[dict[str, Any]]:
    """Round-trip companies back to plain dicts (useful for exports/tests)."""
    return [c.model_dump() for c in companies]