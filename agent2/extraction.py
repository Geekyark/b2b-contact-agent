"""Evidence-backed contact extraction using Claude (official Anthropic SDK).

The pipeline for one company:

1. ``run_discovery`` (in ``discovery.py``) orchestrates: search (production)
   or mock evidence (tests) -> extract structured contacts with Claude ->
   build ``Contact`` records and proof rows.
2. Claude operates on the *returned search evidence only*. It reads the
   company profile (from Agent 1) purely to *choose which roles matter*; it
   is told - and the code enforces - that nothing in its internal knowledge
   is proof of anyone's current identity, role, email, phone or LinkedIn.

Every extracted contact field carries the source URL for identity and for
contact details. If Claude cannot tie a value to a returned source, the
field stays empty and the reason is recorded.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field

from .config import IT_SERVICES
from .models import Agent1Company, EmailStatus, Evidence, PhoneStatus

logger = logging.getLogger(__name__)

ROLE_TAXONOMY = [
    "CEO", "Founder", "COO", "CTO", "CIO", "CISO", "VP Engineering", "Director of Engineering",
    "Head of Engineering", "IT Director", "Head of IT", "Head of Technology", "Head of Digital Transformation",
    "Head of AI", "Data Director", "Cloud Engineering Director", "Cybersecurity Director",
    "Procurement Director", "Vendor Management Director", "Technology Partnerships Director",
    "Head of QA", "Head of Infrastructure", "Engineering Director",
]

# Verified values for these enum fields; anything else from the model is rejected.
ROLE_ALIASES = {r.lower(): r for r in ROLE_TAXONOMY}

EMAIL_STATUS_VALUES = {s.value: s for s in EmailStatus}
PHONE_STATUS_VALUES = {s.value: s for s in PhoneStatus}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# Extraction contract (structured outputs via client.messages.parse)
# ---------------------------------------------------------------------------
class ExtractedContact(BaseModel):
    """What Claude may return per person. Fields are optional strings or
    empty lists - Claude is told to leave a field empty rather than invent a
    value it cannot tie to a returned source."""

    name: str = Field(default="")
    title: str = Field(default="")
    department: str = Field(default="")
    country: str = Field(default="")
    linkedin_url: str = Field(default="")
    work_email: str = Field(default="")
    email_status: str = Field(default=EmailStatus.NOT_FOUND.value)
    business_phone: str = Field(default="")
    phone_status: str = Field(default=PhoneStatus.NOT_FOUND.value)
    role_relevance: str = Field(default="")
    identity_source_url: str = Field(default="")
    contact_source_url: str = Field(default="")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    verification_notes: str = Field(default="")


class ExtractionResult(BaseModel):
    contacts: list[ExtractedContact] = Field(default_factory=list)
    notes: str = Field(default="")


class EvidenceBuilder:
    """Collector that coerces extraction output into structured Evidence rows.

    Field values only become Evidence when they (a) have a value and (b) are
    tied to a source URL or flagged as manual/unavailable. Values without
    sources are dropped by the discovery step, never silently kept.
    """

    def __init__(self, available_urls: set[str]):
        self.available_urls = available_urls

    def check(self, fact_type: str, value: str, source_url: str, snippet: str = "") -> Optional[Evidence]:
        value = (value or "").strip()
        if not value:
            return None
        url = (source_url or "").strip()
        if url and url not in self.available_urls:
            # A URL that was not returned by the search provider is not
            # evidence; it is a hallucination and must not enter the record.
            return None
        return Evidence(fact_type=fact_type, value=value, source_url=url,
                        snippet=(snippet or "")[:500])


# ---------------------------------------------------------------------------
# Task prompts
# ---------------------------------------------------------------------------
EXTRACTION_SYSTEM = """You are a careful B2B decision-maker researcher for an IT services company. You never guess, invent, or rely on your internal knowledge about specific people. Everything you report must be grounded in the evidence sections a user provides, which come directly from a web search provider.

Rules:
1. Only report a person if the evidence explicitly names them and their role. Do not add people who are not in the evidence.
2. Extract the person's current title precisely as the evidence states it.
3. LinkedIn URLs and work emails must come verbatim from the evidence. If none exists for a person, leave the field empty.
4. Never guess or construct an email by pattern (e.g., firstname@company.com is never allowed).
5. Emails that look like generic contact boxes (hello@, info@, contact@, sales@, supporters@, press@, media@) are NOT a person's direct email - leave work_email empty and note it.
6. Phone numbers: only include if the evidence shows a direct business number for the person. A main reception line is acceptable only if it is the only business number available; mark phone_status accordingly.
7. Do not invent source URLs. Use only the exact URLs present in the evidence. If a field has no supporting source, leave both the field and its source empty.
8. If several sources conflict, prefer the official company pages and the most recent evidence; note the conflict in verification_notes.
9. For each person choose the most relevant role from this taxonomy where it matches: {roles}. If the person's actual title is a variant, use the actual title but keep it accurate. Department is one of: Executive, Technology, Security, Procurement/Finance, Customer Success, or blank if unknown.
10. email_status must be one of: published_in_source_not_independently_verified, verified_by_configured_provider, invalid, catch_all_or_uncertain, not_found. Use published_in_source_not_independently_verified when the email was published in a returned source but no verification provider is configured for this run. Use not_found when there is no email.
11. phone_status must be one of: published_in_source_not_independently_verified, verified_by_configured_provider, not_found.
12. Do NOT invent or extrapolate details for people who are not decision-makers. Prefer decision-makers (CEO, CTO, engineering, security, procurement, AI/data leads). Aim for {min_contacts} to {max_contacts} contacts, but never fabricate people to hit a target.
13. verification_notes should record exactly what is unverified or needs manual review (e.g., "LinkedIn URL not found in sources", "email is generic inbox, not direct", "role inferred from press release").
14. If the evidence contains nothing usable, return an empty contacts list - that is a correct, acceptable answer.
15. role_relevance: explain in one sentence why this specific person is relevant to an IT services seller offering: {services}."""


def build_user_prompt(company: Agent1Company, evidence_json: str, max_contacts: int = 5) -> str:
    roles = ", ".join(company.target_roles() or ["CEO", "CTO", "IT Director"])
    return f"""Company profile (from Agent 1 research, for role selection only):
{json.dumps(company.to_context_dict(), indent=2, ensure_ascii=False)}

Evidence from web search (the ONLY facts you may report):
<evidence>
{evidence_json}
</evidence>

Target role preference (from the company profile): {roles}

Now extract up to {max_contacts} decision-maker contacts from the evidence, following the system rules exactly. Contact details only from the evidence; blank = unknown. Output valid JSON matching the required schema."""