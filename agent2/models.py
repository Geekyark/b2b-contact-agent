"""Pydantic models for Agent 2.

The shapes here are the contract between modules:

- ``Agent1Company`` is the *input* record produced by Agent 1 (company
  research). It is deliberately tolerant: Agent 1's exact schema is unknown,
  so the loader maps whatever fields exist and this model accepts unknowns.
- ``Contact`` is the *output* record Agent 2 produces. Every field that can
  be left blank has a default of "" or None so Agent 2 never has to invent
  data - it just leaves the field empty and records why.
- ``VerificationStatus`` centralises the allowed status values so the UI,
  exporters and verification code all agree.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Verification status vocabulary
# ---------------------------------------------------------------------------
class EmailStatus(str, Enum):
    PUBLISHED = "published_in_source_not_independently_verified"
    VERIFIED = "verified_by_configured_provider"
    INVALID = "invalid"
    CATCH_ALL = "catch_all_or_uncertain"
    NOT_FOUND = "not_found"

    @classmethod
    def values(cls) -> list[str]:
        return [e.value for e in cls]


class PhoneStatus(str, Enum):
    PUBLISHED = "published_in_source_not_independently_verified"
    VERIFIED = "verified_by_configured_provider"
    NOT_FOUND = "not_found"

    @classmethod
    def values(cls) -> list[str]:
        return [e.value for e in cls]


class ContactReviewState(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


# ---------------------------------------------------------------------------
# Evidence: an atomic, citeable fact found by the search provider
# ---------------------------------------------------------------------------
class Evidence(BaseModel):
    """One source-backed fact about a person.

    ``source_url`` is a URL returned by the search provider (or marked
    ``manual / unavailable``). Agent 2 treats evidence items as the only way
    any personal detail enters a contact record - never the model's internal
    knowledge.
    """

    fact_type: str = Field(description="e.g. identity, role, email, phone, linkedin")
    value: str = Field(description="normalised value, e.g. an email address or title")
    source_url: str = Field(description="exact URL returned by the search provider")
    snippet: str = Field(default="", description="quote/snippet from the source, if any")

    def to_dict(self) -> dict:
        return {"fact_type": self.fact_type, "value": self.value,
                "source_url": self.source_url, "snippet": self.snippet}


# ---------------------------------------------------------------------------
# Input: company record produced by Agent 1
# ---------------------------------------------------------------------------
class Agent1Company(BaseModel):
    """A company record as received from Agent 1 (company research).

    Loaded by ``agent2.inputs`` from JSON or CSV. All fields default to ""
    because Agent 1's schema may vary; only the company name and identifier
    are required for Agent 2 to run.
    """

    model_config = ConfigDict(extra="allow")

    company_id: str = Field(default="", description="original identifier from Agent 1")
    company_name: str = Field(default="", description="official company name")
    website: str = Field(default="", description="official website / domain")
    country: str = Field(default="", description="country the company operates in")
    city: str = Field(default="", description="city, if known")
    industry: str = Field(default="", description="industry classification, if known")
    company_size: str = Field(default="", description="employee range or count, if known")
    business_model: str = Field(default="", description="business model, if known")
    products_services: str = Field(default="", description="products / services offered")
    technology_requirements: str = Field(default="", description="technology the company uses or needs")
    business_challenges: str = Field(default="", description="challenges the company faces")
    reasons_potential_client: str = Field(default="", description="why Agent 1 flagged this company")
    company_research_source_urls: list[str] = Field(default_factory=list)
    existing_company_identifiers: dict[str, str] = Field(default_factory=dict)

    @property
    def search_label(self) -> str:
        """A compact label for the model: name (website)."""
        return f"{self.company_name} ({self.website})" if self.website else self.company_name

    def target_roles(self) -> list[str]:
        """Suggested decision-maker roles based on the company's profile.

        Role selection is a heuristic; the Claude extraction step is asked to
        prefer these roles but may adjust based on the actual people it finds.
        """
        text = " ".join([
            self.industry, self.products_services, self.technology_requirements,
            self.business_challenges, self.business_model,
        ]).lower()
        size = self.company_size.lower()

        roles: list[str] = []
        if any(k in text for k in ("custom software", "software development", "web development", "mobile",
                                   "app development", "product engineering")):
            roles += ["CTO", "VP Engineering", "Engineering Director"]
        if any(k in text for k in ("cloud", "devops", "infrastructure", "kubernetes", "aws", "azure", "gcp")):
            roles += ["CTO", "Cloud Engineering Director", "Head of Infrastructure"]
        if any(k in text for k in ("ai", "artificial intelligence", "automation", "machine learning",
                                   "data science", "data")):
            roles += ["Head of AI", "Data Director", "CTO"]
        if any(k in text for k in ("cyber", "security", "compliance", "gdpr")):
            roles += ["CISO", "Security Director", "IT Director"]
        if any(k in text for k in ("qa", "testing", "quality assurance")):
            roles += ["Head of QA", "Engineering Director"]
        if any(k in text for k in ("outsourcing", "dedicated developer", "staff augmentation", "managed it")):
            roles += ["CTO", "Engineering Director", "Procurement Director", "Vendor Management Director"]
        if any(k in text for k in ("api", "integration", "excel", "sap", "salesforce", "erp")):
            roles += ["CTO", "Head of Technology", "IT Director"]
        if not roles:
            roles = ["CTO", "IT Director", "Head of Technology"]

        # Smaller companies: founders hold the buying power. Employee-range
        # heuristics are rough; everything is a preference, not a hard rule.
        if "1-" in size or "2-10" in size or "11-50" in size or "1-10" in size:
            head = ["Founder", "CEO"]
            roles = head + [r for r in roles if r not in head]
        if "5001" in size or "10,001" in size or "10000+" in size or "5000+" in size or "10001" in size:
            roles += ["Procurement Director", "Vendor Management Director"]

        seen: list[str] = []
        for r in roles:  # dedupe preserving order
            if r not in seen:
                seen.append(r)
        return seen[:6]

    def to_context_dict(self) -> dict[str, Any]:
        """Dict used in the prompt to Claude - only populated fields."""
        return {k: v for k, v in self.model_dump().items() if v not in ("", [], {})}


# ---------------------------------------------------------------------------
# Output: one decision-maker contact
# ---------------------------------------------------------------------------
class Contact(BaseModel):
    """A decision-maker and their publicly available professional details."""

    # Identity (evidence-backed)
    name: str = Field(default="")
    title: str = Field(default="")
    department: str = Field(default="")
    country: str = Field(default="")

    # Contact details (evidence-backed; "" when unavailable or unverifiable)
    linkedin_url: str = Field(default="")
    work_email: str = Field(default="")
    email_status: EmailStatus = Field(default=EmailStatus.NOT_FOUND)
    business_phone: str = Field(default="")
    phone_status: PhoneStatus = Field(default=PhoneStatus.NOT_FOUND)
    official_company_contact_page: str = Field(default="")
    source_identity_url: str = Field(default="")
    source_contact_url: str = Field(default="")

    # Relevance and provenance
    role_relevance: str = Field(default="", description="why this person matters for the IT services offered")
    source_urls: list[str] = Field(default_factory=list)
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0)
    verification_notes: str = Field(default="")
    research_date: str = Field(default="")

    # Review state set by the user in the dashboard (never by discovery)
    review_state: ContactReviewState = Field(default=ContactReviewState.PENDING)

    # Tracks explicitly set by the extraction step (import/paste / websearch /
    # web_fetch / manual). Persisted so the dashboard can show provenance.
    source_tags: list[str] = Field(default_factory=list)

    model_config = ConfigDict(use_enum_values=True)

    @property
    def has_email(self) -> bool:
        return self.email_status not in (EmailStatus.NOT_FOUND, EmailStatus.INVALID) and bool(self.work_email)

    @property
    def has_phone(self) -> bool:
        return self.phone_status != PhoneStatus.NOT_FOUND and bool(self.business_phone)

    def to_row(self) -> dict[str, Any]:
        """One flat row for CSV/Excel export (company fields joined upstream)."""
        row = self.model_dump()
        row["email_status"] = self.email_status.value if isinstance(self.email_status, EmailStatus) else self.email_status
        row["phone_status"] = self.phone_status.value if isinstance(self.phone_status, PhoneStatus) else self.phone_status
        row["review_state"] = self.review_state.value if isinstance(self.review_state, ContactReviewState) else self.review_state
        return row


# ---------------------------------------------------------------------------
# Company-level discovery result
# ---------------------------------------------------------------------------
class CompanyContacts(BaseModel):
    """All contacts found for one company, joined to its Agent 1 record."""

    company_id: str
    company_name: str
    contacts: list[Contact] = Field(default_factory=list)
    error: str = Field(default="")

    def to_row(self, contact: Contact) -> dict[str, Any]:
        row = contact.to_row()
        row["company_id"] = self.company_id
        row["company_name"] = self.company_name
        return row

    def to_rows(self) -> list[dict[str, Any]]:
        return [self.to_row(c) for c in self.contacts]


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------
_EMAIL_RE = re.compile(r"^[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}$")
_LINKEDIN_RE = re.compile(r"^(https?://)?([a-z]{2,3}\.)?linkedin\.com/", re.IGNORECASE)
_PHONE_RE = re.compile(r"^\+?[0-9][0-9 ()+\-\.]{4,}$")


def valid_email_shape(value: str) -> bool:
    return bool(_EMAIL_RE.match(value or ""))


def looks_like_linkedin(value: str) -> bool:
    return bool(_LINKEDIN_RE.match(value or ""))


def valid_phone_shape(value: str) -> bool:
    return bool(_PHONE_RE.match(value or ""))


def normalize_linkedin(value: str) -> str:
    """Normalise a LinkedIn URL to the https://www.linkedin.com/... form."""
    v = (value or "").strip()
    if not v:
        return ""
    v = v.split("?")[0].rstrip("/")
    if not v.lower().startswith("http"):
        v = "https://" + v
    return v if _LINKEDIN_RE.match(v) else ""