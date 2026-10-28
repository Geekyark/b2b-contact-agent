"""Discovery orchestration: turn Agent 1 company records into evidence-backed contacts."""

from __future__ import annotations

import json
import logging
from typing import Any, Callable, Optional

import anthropic
from pydantic import ValidationError

from .config import IT_SERVICES, Settings
from .extraction import (
    EXTRACTION_SYSTEM,
    EvidenceBuilder,
    ExtractionResult,
    ExtractedContact,
    build_user_prompt,
    now_iso,
)
from .models import (
    Agent1Company,
    CompanyContacts,
    Contact,
    Evidence,
    EmailStatus,
    PhoneStatus,
)
from .search import SearchResponse, build_search_provider, query_variants

logger = logging.getLogger(__name__)

# Streaming lets us use a large max_tokens without hitting HTTP timeouts.
MAX_TOKENS = 24576

DEFAULT_ROLES = ["CEO", "CTO", "IT Director"]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def run_discovery(companies: list[Agent1Company], settings: Settings, *,
                  anthropic_client: Optional[anthropic.Anthropic] = None,
                  search_provider: Any = None,
                  extraction_fn: Optional["Callable"] = None,
                  stop_on_error: bool = False) -> list[CompanyContacts]:
    """Run the full discovery pipeline over a batch of companies.

    Parameters
    ----------
    companies : company records from Agent 1 (input)
    settings : application settings (model, search provider, limits)
    anthropic_client : optional pre-built Anthropic client (used by tests)
    search_provider : optional pre-built search provider (used by tests)
    extraction_fn : optional extractor override; signature
        ``(company, evidence_json, settings) -> ExtractionResult``. Used for
        offline mock extraction; defaults to calling Claude.
    stop_on_error : if True, raise on the first per-company failure instead of
        recording it on that company's result
    """
    results: list[CompanyContacts] = []
    client = anthropic_client if extraction_fn is None else None
    if extraction_fn is None:
        client = anthropic_client or _default_client(settings)
    provider = search_provider or build_search_provider(settings)

    for comp in companies:
        try:
            contacts, proofs = _discover_one(
                comp, settings, client, provider,
                extraction_fn=extraction_fn,
            )
            result = CompanyContacts(
                company_id=comp.company_id, company_name=comp.company_name,
                contacts=contacts, error="",
            )
        except Exception as exc:
            logger.warning("Discovery failed for %s: %s", comp.company_name, exc)
            if stop_on_error:
                raise
            result = CompanyContacts(
                company_id=comp.company_id, company_name=comp.company_name,
                contacts=[], error=str(exc),
            )
        results.append(result)

    _close_provider(provider)
    return results


def _default_client(settings: Settings) -> anthropic.Anthropic:
    return anthropic.Anthropic(api_key=settings.anthropic_api_key or None)


def run_discovery_offline(companies: list[Agent1Company], settings: Settings) -> list[CompanyContacts]:
    """Deterministic offline discovery using the mock search provider and a
    rule-based extractor. Used by the demo and tests; never hits the network."""
    from .search import MockSearchProvider

    return run_discovery(companies, settings, search_provider=MockSearchProvider(),
                         extraction_fn=_mock_extract)


def _mock_extract(company: Agent1Company, evidence_json: str,
                  settings: Settings) -> ExtractionResult:
    """Offline extractor: returns fixed contacts whose cited source URLs match
    the MockSearchProvider's pages, so the grounding path is exercised for real."""
    from .extraction import ExtractedContact

    return ExtractionResult(contacts=[
        ExtractedContact(
            name="Ada Lovelace",
            title="Founder & CEO",
            department="Executive",
            country=company.country or "United Kingdom",
            linkedin_url="https://www.linkedin.com/in/ada-lovelace",
            work_email="",
            email_status="not_found",
            business_phone="",
            phone_status="not_found",
            role_relevance="Founder and CEO with budget authority for IT services engagements.",
            identity_source_url="https://examplecorp.dev/about",
            contact_source_url="https://examplecorp.dev/team",
            confidence=0.85,
            verification_notes="Profile published on official company site; email not published.",
        ),
        ExtractedContact(
            name="Grace Hopper",
            title="CTO",
            department="Technology",
            country=company.country or "United Kingdom",
            linkedin_url="https://www.linkedin.com/in/grace-hopper",
            work_email="grace@examplecorp.dev",
            email_status="published_in_source_not_independently_verified",
            business_phone="",
            phone_status="not_found",
            role_relevance="CTO owns engineering and vendor selection for technology services.",
            identity_source_url="https://examplecorp.dev/team",
            contact_source_url="https://examplecorp.dev/team",
            confidence=0.8,
            verification_notes="Email published on official team page; not independently verified.",
        ),
    ], notes="offline mock extraction")


def _close_provider(provider: Any) -> None:
    close = getattr(provider, "close", None)
    if close is not None:
        try:
            close()
        except Exception:  # pragma: no cover - never mask the discovery result
            pass


# ---------------------------------------------------------------------------
# One company
# ---------------------------------------------------------------------------
def _discover_one(comp: Agent1Company, settings: Settings,
                  client: Optional[anthropic.Anthropic],
                  provider: Any,
                  extraction_fn: Optional["Callable"] = None,
                  ) -> tuple[list[Contact], list[Evidence]]:
    """Discover and extract contacts for a single company."""
    queries = query_variants(comp.company_name, comp.website, comp.target_roles(),
                             settings.max_queries_per_company)

    search_responses: list[SearchResponse] = []
    for q in queries:
        try:
            resp = provider.search(q)
        except Exception as exc:  # a failing provider must not kill the batch
            logger.warning("Search failed for query %r: %s", q, exc)
            resp = SearchResponse(query=q, results=[], error=str(exc))
        search_responses.append(resp)

    available_urls: set[str] = set()
    snippet_by_url: dict[str, str] = {}
    for resp in search_responses:
        for r in resp.results:
            available_urls.add(r.url)
            snippet_by_url.setdefault(r.url, r.snippet)

    successful = [r for r in search_responses if r.results and not r.error]
    if not successful:
        errors = [r.error for r in search_responses if r.error]
        if errors:
            # Every query failed: surface it as an error on this company
            # instead of silently reporting "no contacts".
            raise RuntimeError("All searches failed: " + "; ".join(dict.fromkeys(errors))[:400])
        # No evidence at all: an empty result is correct. Agent 2 never
        # invents contacts to fill a target.
        return [], []

    evidence_json = json.dumps({
        "company": comp.to_context_dict(),
        "search_results": [
            {"query": resp.query, "results": [r.to_dict() for r in resp.results]}
            for resp in search_responses if resp.results
        ],
    }, indent=2, ensure_ascii=False)

    extraction = (
        extraction_fn(comp, evidence_json, settings)
        if extraction_fn is not None
        else _extract(client, settings, comp, evidence_json)
    )

    contacts: list[Contact] = []
    proofs: list[Evidence] = []
    for ec in extraction.contacts:
        contact, cproofs = _normalise_contact(ec, comp, available_urls, snippet_by_url)
        if contact is None:
            continue
        contacts.append(contact)
        proofs.extend(cproofs)

    contacts = _dedupe_contacts(contacts)
    return contacts[: settings.max_contacts_per_company], proofs


# ---------------------------------------------------------------------------
# Claude extraction
# ---------------------------------------------------------------------------
def _extract(client: anthropic.Anthropic, settings: Settings,
             comp: Agent1Company, evidence_json: str) -> ExtractionResult:
    """Call Claude to extract structured contacts from the search evidence."""
    system_text = EXTRACTION_SYSTEM.format(
        roles=", ".join(comp.target_roles() or DEFAULT_ROLES),
        min_contacts=settings.min_contacts_per_company,
        max_contacts=settings.max_contacts_per_company,
        services=", ".join(IT_SERVICES),
    )
    prompt = build_user_prompt(comp, evidence_json, settings.max_contacts_per_company)

    try:
        # Streaming path: the SDK requires streaming for requests with a large
        # max_tokens (non-streaming requests are capped), and structured output
        # is carried on the final ParsedMessage.
        stream = client.beta.messages.stream(
            model=settings.claude_model,
            max_tokens=MAX_TOKENS,
            output_format=ExtractionResult,
            system=[{
                "type": "text",
                "text": system_text,
                "cache_control": {"type": "ephemeral"},
            }],
            messages=[{"role": "user", "content": prompt}],
        )
        with stream as s:
            msg = s.get_final_message()
        if msg.stop_reason == "refusal":
            logger.warning("Claude refused extraction for %s: %s",
                           comp.company_name, getattr(msg, "stop_details", None))
            return ExtractionResult(contacts=[], notes="Model refused the request.")
        parsed = getattr(msg, "parsed_output", None)
        if not isinstance(parsed, ExtractionResult):
            logger.warning("Unexpected parse output for %s: %r", comp.company_name, type(parsed))
            return ExtractionResult(contacts=[], notes="Unexpected structured output.")
        return parsed
    except anthropic.BadRequestError as exc:  # e.g. model/tool mismatch
        logger.warning("Claude bad request for %s: %s", comp.company_name, exc)
        return ExtractionResult(contacts=[], notes=f"Claude bad request: {exc}")
    except anthropic.APIConnectionError as exc:  # network: recorded, batch continues
        logger.warning("Claude connection error for %s: %s", comp.company_name, exc)
        return ExtractionResult(contacts=[], notes=f"Claude connection error: {exc}")
    except anthropic.RateLimitError as exc:
        logger.warning("Claude rate limited for %s: %s", comp.company_name, exc)
        return ExtractionResult(contacts=[], notes="Rate limited; re-run later.")
    except anthropic.APIStatusError as exc:
        status = getattr(exc, "status_code", "?")
        logger.warning("Claude API error for %s (status %s): %s", comp.company_name, status, exc)
        return ExtractionResult(contacts=[], notes=f"Claude API error: {status}")
    except (ValidationError, json.JSONDecodeError) as exc:
        logger.warning("Malformed Claude output for %s: %s", comp.company_name, exc)
        return ExtractionResult(contacts=[], notes="Malformed model output.")
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("Unexpected Claude failure for %s", comp.company_name)
        return ExtractionResult(contacts=[], notes=f"Unexpected error: {exc}")


# ---------------------------------------------------------------------------
# Coercion and grounding
# ---------------------------------------------------------------------------
def _normalise_contact(ec: ExtractedContact, comp: Agent1Company,
                       available_urls: set[str],
                       snippet_by_url: dict[str, str]) -> tuple[Optional[Contact], list[Evidence]]:
    """Coerce one extracted contact into a Contact plus evidence rows."""
    if not (ec.name or "").strip():
        return None, []

    proofs = _build_evidence(ec, available_urls, snippet_by_url)

    # A person may only enter the record if their identity or role is tied to
    # a source the provider actually returned. An unsourced name (person or
    # title) is a hallucination and must never survive.
    if not any(p.fact_type in ("identity", "role") for p in proofs):
        return None, []

    contact = Contact(
        name=ec.name.strip(),
        title=ec.title.strip(),
        department=ec.department.strip(),
        country=ec.country.strip() or comp.country,
        linkedin_url=ec.linkedin_url.strip(),
        work_email=ec.work_email.strip(),
        business_phone=ec.business_phone.strip(),
        role_relevance=ec.role_relevance.strip(),
        verification_notes=ec.verification_notes.strip(),
        confidence_score=_clamp01(ec.confidence),
        research_date=now_iso(),
    )
    contact.email_status = _map_email_status(ec.email_status, contact.work_email)
    contact.phone_status = _map_phone_status(ec.phone_status, contact.business_phone)

    # Only values tied to a returned source survive; everything else is
    # dropped (a hallucinated LinkedIn/email/phone/title must never enter the db).
    contact.title = _sourced_value(contact.title, "role", proofs)
    contact.linkedin_url = _sourced_value(contact.linkedin_url, "linkedin", proofs)
    contact.work_email = _sourced_value(contact.work_email, "email", proofs)
    contact.business_phone = _sourced_value(contact.business_phone, "phone", proofs)
    if not contact.work_email:
        contact.email_status = EmailStatus.NOT_FOUND
    if not contact.business_phone:
        contact.phone_status = PhoneStatus.NOT_FOUND

    contact.source_identity_url = _sourced_value(ec.identity_source_url, "identity", proofs) or \
        _sourced_value(ec.identity_source_url, "role", proofs)
    contact.source_contact_url = (
        _sourced_value(ec.contact_source_url, "email", proofs)
        or _sourced_value(ec.contact_source_url, "phone", proofs)
        or _sourced_value(ec.contact_source_url, "linkedin", proofs)
        or _sourced_value(ec.contact_source_url, "relevance", proofs)
    )
    contact.source_urls = sorted({p.source_url for p in proofs if p.source_url})
    contact.source_tags = sorted({_tag_for_url(u) for u in contact.source_urls})

    if not contact.confidence_score:
        contact.confidence_score = _confidence_from_proofs(len(proofs), ec.confidence)
    return contact, proofs


def _build_evidence(ec: ExtractedContact, available_urls: set[str],
                    snippet_by_url: dict[str, str]) -> list[Evidence]:
    """Build grounded Evidence rows for one extracted contact."""
    eb = EvidenceBuilder(available_urls)
    proofs: list[Evidence] = []
    for fact_type, value, url in (
        ("identity", ec.name, ec.identity_source_url),
        ("role", ec.title, ec.identity_source_url),
        ("department", ec.department, ec.identity_source_url),
        ("country", ec.country, ec.identity_source_url),
        ("email", ec.work_email, ec.contact_source_url),
        ("phone", ec.business_phone, ec.contact_source_url),
        ("linkedin", ec.linkedin_url, ec.contact_source_url),
        ("relevance", ec.role_relevance, ec.contact_source_url or ec.identity_source_url),
    ):
        ev = eb.check(fact_type, value, url, snippet_by_url.get(url, ""))
        if ev is not None:
            proofs.append(ev)
    return proofs


def _sourced_value(value: str, fact_type: str, proofs: list[Evidence]) -> str:
    """Return the value only if an evidence row ties it to a supplied source."""
    value = (value or "").strip()
    if not value:
        return ""
    if any(p.fact_type == fact_type and p.value == value and p.source_url for p in proofs):
        return value
    return ""


def _clamp01(x: Any) -> float:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, x))


def _map_email_status(raw: str, email: str) -> EmailStatus:
    raw = (raw or "").strip().lower().replace("_", "_")
    if not email:
        return EmailStatus.NOT_FOUND
    for status in EmailStatus:
        if status.value == raw:
            if raw == EmailStatus.NOT_FOUND.value:
                return EmailStatus.NOT_FOUND
            return status
    # Unknown/implied status with a real email: published but unverified.
    return EmailStatus.PUBLISHED


def _map_phone_status(raw: str, phone: str) -> PhoneStatus:
    raw = (raw or "").strip().lower()
    if not phone:
        return PhoneStatus.NOT_FOUND
    for status in PhoneStatus:
        if status.value == raw:
            return status
    return PhoneStatus.PUBLISHED


def _tag_for_url(url: str) -> str:
    if not url:
        return ""
    if "linkedin.com" in url:
        return "linkedin"
    for key in ("/team", "/about", "/leadership", "/management", "/contact", "/company/"):
        if key in url:
            return "website"
    if any(k in url for k in ("press", "news", "release")):
        return "press"
    return "web"


def _confidence_from_proofs(n_proofs: int, base: Any) -> float:
    try:
        base = float(base)
    except (TypeError, ValueError):
        base = 0.0
    grounded = 0.35 + 0.12 * n_proofs
    confidence = (grounded + base) / 2 if base > 0 else grounded
    return max(0.0, min(1.0, confidence))


def _dedupe_contacts(contacts: list[Contact]) -> list[Contact]:
    """Remove duplicates within a company, keeping the most evidence-rich record.

    A contact is a duplicate when it shares an email, LinkedIn URL, or name
    with another contact of the same company.
    """
    keep: list[Contact] = []
    records: dict[tuple[str, ...], int] = {}

    for c in contacts:
        key = (
            (c.work_email or "").strip().lower(),
            (c.linkedin_url or "").strip().lower().rstrip("/"),
            (c.name or "").strip().lower(),
        )
        key = tuple(k for k in key if k)
        if not key:
            continue
        idx = records.get(key)
        if idx is None:
            records[key] = len(keep)
            keep.append(c)
        elif len(c.source_urls) > len(keep[idx].source_urls):
            keep[idx] = c
    return keep